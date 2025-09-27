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
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/SSL-AASIST/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ResNet18/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/SSL-AASIST/result/seen_speaker_results.txt",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ResNet18/LFCC_add_noise/result/unseen_speaker_results.txt",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/LFCC-ECAPA-TDNN/LFCC_add_noise/result/unseen_speaker_results.txt",
        help="Where to write the results (should be a .txt file)",
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/5lang/-ru/unseen_speaker/LFCC_add_noise/preprocess_lfcc",
        default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/5lang/-en/seen_speaker/Clean_no_feature",
        help="Path to the wav features directory",
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

    args.results_path = args.results_path.rstrip('/')


    if os.path.exists(args.results_path) and os.path.isdir(args.results_path):
        raise ValueError(f"The path '{args.results_path}' is a directory, but a file is expected.")

    results_dir = os.path.dirname(args.results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    return args

def evaluate_model(model, dataset, device, args):

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            feats, filename, labels = batch

            # feature processing
            feats = feats.unsqueeze(dim=1)  # lfcc 
            feats = feats.transpose(2, 3).to(device)
            
            # Forward pass
            _, logits = model(feats)
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
    # model = ECAPA_TDNN(C=192, num_classes=4)
    model = SSL_AASIST(args, device=device, class_num=4)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    wav_path = args.wav_path

    print(f"\n{'='*50}")
    print(f"Processing data from {wav_path}...")
    
    try:
        dataset = MLAADFDDataset(wav_path, "eval", mode="known", max_samples=-1, class_num=args.num_classes)
        
        if len(dataset) == 0:
            print("No data found! Exiting...")
            print(f"Please check if features exist in: {wav_path}")
            return
            
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print(f"Dataset path: {wav_path}")
        return

    print(f"Found {len(dataset)} samples for evaluation")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluation"):
            feats, filename, labels = batch
            
            # feature processing
            feats = feats.unsqueeze(dim=1)  # lfcc 
            feats = feats.transpose(2, 3).to(device)
            
            # Forward pass
            # _, logits = model(feats)
            logits = model(feats)#ASSIST
            logits = F.softmax(logits, dim=1)
            
            # Store predictions and labels
            predicted = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Generate evaluation metrics
    report_str = generate_report(all_preds, all_labels)

    report_path = args.results_path

    if os.path.isdir(report_path):
        raise ValueError(f"The path '{report_path}' is a directory, but a file is expected.")

    if os.path.exists(report_path) and os.path.isdir(report_path):
        raise ValueError(f"The path '{report_path}' already exists and is a directory.")

    with open(report_path, "w") as f:
        f.write(report_str)

    print("Classification report:")
    print(report_str)
    print(f"Results saved to {report_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)