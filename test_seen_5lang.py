___author__ = "XI XUAN"

import argparse
import os
import sys
from pathlib import Path

# Enables running the script from root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dataset import MLAADFDDataset, five_lang_MLAADFDDataset 
from src.models.w2v2_aasist import SSL_AASIST,W2VAASIST,ResNet18,XLSR_ResNet18, ECAPA_TDNN

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Get metrics for multi-language evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/SSL-AASIST/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ResNet18/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/SSL-AASIST/result/",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ResNet18/LFCC_add_noise/result/",
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/LFCC-ECAPA-TDNN/LFCC_add_noise/result/",
        help="Where to write the results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of systems in the training dataset",
    )
    parser.add_argument(
        "--unseen_index",
        type=int,
        default=1, #{'en': 0, 'de': 1, 'fr': 2, 'it': 3, 'pl': 4, 'ru': 5}
        help="Index of the unseen language in the languages list (0-based)",
    )
    args = parser.parse_args()
    
    # 创建结果目录（如果不存在）
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        
    return args

def evaluate_model(model, dataset, device, args):
    """通用评估函数"""
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            feats, filename, labels = batch
            

            # feature processing
            feats = feats.unsqueeze(dim=1)#lfcc 
            feats = feats.transpose(2, 3).to(device)
            
            # Forward pass
            _, logits = model(feats)
            # logits = model(feats)#ASSIST
            logits = F.softmax(logits, dim=1)

            
            # 记录预测结果
            predicted = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def generate_report(all_preds, all_labels, lang_name=""):
    """生成分类报告"""
    if len(all_labels) == 0:
        return f"No valid labels found for {lang_name}!"
    
    # 过滤无效预测
    valid_labels = np.unique(all_labels)
    all_preds_filtered = np.array([p if p in valid_labels else -1 for p in all_preds])
    
    # 生成评估指标
    if len(valid_labels) == 0:
        return "No valid labels found. Classification report cannot be generated."
    
    macro_accuracy = balanced_accuracy_score(all_labels, all_preds_filtered)
    report_str = classification_report(
        all_labels, all_preds_filtered, 
        labels=valid_labels, 
        zero_division=1.0, 
        digits=6
    )
    report_str += f"\n\nMacro-Averaged Accuracy: {macro_accuracy:.6f}"
    return report_str

def main(args):
    # 初始化设备
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}..")

    # 加载模型
    print(f"Loading model from {args.model_path}")
    # model = ResNet18(num_classes=4)
    model = ECAPA_TDNN(C=192, num_classes=args.num_classes)
    # model = SSL_AASIST(args, device=device, class_num=4)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # # 定义语言数据集路径（使用LFCC_add_noise版本）
    languages = [
        {'code': 'en', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'de', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'fr', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'it', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'pl', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'ru', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc'},
    ]

    # 定义语言数据集路径（使用ssl版本）
    # languages = [
    #     {'code': 'en', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature'},
    #     {'code': 'de', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature'},
    #     {'code': 'fr', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature'},
    #     {'code': 'it', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature'},
    #     {'code': 'pl', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature'},
    #     {'code': 'ru', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature'},
    # ]

    # 检查索引有效性
    if args.unseen_index < 0 or args.unseen_index >= len(languages):
        raise ValueError(f"unseen_index must be between 0 and {len(languages)-1}")

    # 分割为已见和未见语言
    unseen_index = args.unseen_index
    seen_languages = languages[:unseen_index] + languages[unseen_index+1:]
    unseen_language = languages[unseen_index]

    print(f"\n{'='*50}")
    print(f"Using language '{unseen_language['code']}' as the unseen language (index: {unseen_index})")
    print(f"Seen languages: {[lang['code'] for lang in seen_languages]}")
    print(f"Unseen language: {unseen_language['code']}")
    print(f"{'='*50}\n")

    # 1. 处理合并的五语言测试集
    print("\n" + "="*50)
    print("Processing Seen 5 Languages Evaluation")

    # 提取 seen_languages 的路径
    seen_paths = [lang['path'] for lang in seen_languages]
    
    # ✅ 使用 * 解包列表为独立参数
    try:
        combined_dataset = five_lang_MLAADFDDataset(
            *seen_paths,  
            "eval", 
            mode="known", 
            max_samples=-1, 
            class_num=args.num_classes
        )
        print(f"Loaded combined dataset with {len(combined_dataset)} samples")
        
        # 执行评估
        all_preds, all_labels = evaluate_model(model, combined_dataset, device, args)
        report_str = generate_report(all_preds, all_labels, "seen 5 languages")
        
        # 保存结果
        result_path = os.path.join(args.results_path, "seen_5lang.txt")
        with open(result_path, "w") as f:
            f.write(report_str)
        print(f"\nSeen 5 languages results saved to: {result_path}")
        print("Classification Report:\n", report_str)
    except Exception as e:
        print(f"Error loading combined dataset: {str(e)}")

    # 2. 处理单语言测试集（未见语言）
    print("\n" + "="*50)
    print("Processing Unseen Single Language Evaluation")
    ru_lang = unseen_language
    ru_code = ru_lang['code']
    ru_base_path = ru_lang['path']
    ru_eval_path = os.path.join(ru_base_path, "eval")
    
    print(f"Loading {ru_code} dataset from: {ru_eval_path}")
    try:
        ru_dataset = MLAADFDDataset(ru_base_path, "eval", mode="known", 
                                  max_samples=-1, class_num=args.num_classes)
        all_preds, all_labels = evaluate_model(model, ru_dataset, device, args)
        report_str = generate_report(all_preds, all_labels, "unseen single language")
        
        # 保存结果
        result_path = os.path.join(args.results_path, "unseen_1lang.txt")
        with open(result_path, "w") as f:
            f.write(report_str)
        print(f"\nUnseen single language results saved to: {result_path}")
        print("Classification Report:\n", report_str)
        
    except Exception as e:
        print(f"Error loading unseen dataset: {str(e)}")

if __name__ == "__main__":
    args = parse_args()
    main(args)