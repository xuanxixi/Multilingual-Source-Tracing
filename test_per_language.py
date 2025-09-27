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

from src.datasets.dataset import MLAADFDDataset
from src.models.w2v2_aasist import AASIST_Model, hug_XLSR_AASIST, hug_w2v2_lang_AASIST, hug_w2v2_en_AASIST, w2v2_en_AASIST, SSL_AASIST,W2VAASIST,ResNet18,XLSR_ResNet18, ECAPA_TDNN

def parse_args():
    parser = argparse.ArgumentParser(description="Get metrics")
    parser.add_argument(
        "--model_path",
        type=str,
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/en_en/LFCC-ResNet18/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/SSL-AASIST/Clean_no_feature/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/pl_pl/w2v2-lv60-self-en-AASIST/Clean_no_feature/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/w2v2-lang-AASIST/Clean_no_feature/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/hug_XLSR-AASIST/Clean_no_feature/trained_models/anti-spoofing_feat_model.pth",
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/LFCC-AASIST/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        
        help="Path to trained model",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/en_en/LFCC-ResNet18/LFCC_add_noise/results/",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/SSL-AASIST/Clean_no_feature/results/",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/pl_pl/w2v2-lv60-self-en-AASIST/Clean_no_feature/results/",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/w2v2-lang-AASIST/Clean_no_feature/results/",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/hug_XLSR-AASIST/Clean_no_feature/results/",
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/ru_ru/LFCC-AASIST/LFCC_add_noise/results/",
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
    args = parser.parse_args()
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    return args

def main(args):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}..")

    print(f"Loading model from {args.model_path}")
    # model = ResNet18(num_classes=4)
    # model = ECAPA_TDNN(C=192, num_classes=args.num_classes)
    model = AASIST_Model(args, device=device, class_num=4)
    # model = SSL_AASIST(args, device=device, class_num=4)
    # model = hug_XLSR_AASIST(args, device=device, class_num=4)
    # model = w2v2_en_AASIST(args, device=device, class_num=4)
    # model = hug_w2v2_en_AASIST(args, device=device, class_num=4)
    # model = hug_w2v2_lang_AASIST(args, device=device, class_num=4)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Define language-specific evaluation paths 
    # languages = [
    #     {'code': 'en', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature'},
    #     {'code': 'de', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature'},
    #     {'code': 'fr', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature'},
    #     {'code': 'it', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature'},
    #     {'code': 'pl', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature'},
    #     {'code': 'ru', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature'},
    # ]
    languages = [
        {'code': 'en', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'de', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'fr', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'it', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'pl', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc'},
        {'code': 'ru', 'path': '/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc'},
    ]

    for lang in languages:
        lang_code = lang['code']
        base_path = lang['path']
        eval_path = os.path.join(base_path, "eval")  
        
        print(f"\n{'='*50}")
        print(f"Processing {lang_code.upper()} language data...")
        print(f"Looking for features in: {eval_path}")
        
        try:
            dataset = MLAADFDDataset(base_path, "eval", mode="known", max_samples=-1, class_num=args.num_classes)
            
            if len(dataset) == 0:
                print(f"No data found for {lang_code}! Skipping...")
                print(f"Please check if features exist in: {eval_path}")
                continue
                
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
            
        except Exception as e:
            print(f"Error loading dataset for {lang_code}: {str(e)}")
            print(f"Dataset path: {base_path}")
            continue

        print(f"Found {len(dataset)} samples for {lang_code} evaluation")
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluation on {lang_code}"):
                feats, filename, labels = batch
                
                # feature processing
                feats = feats.unsqueeze(dim=1)#lfcc 
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

        # Filter invalid predictions
        valid_labels = np.unique(all_labels)
        all_preds_filtered = np.array([p if p in valid_labels else -1 for p in all_preds])

        # Generate evaluation metrics
        if len(valid_labels) == 0:
            print(f"Warning: No valid labels found for {lang_code}!")
            report_str = "No valid labels found. Classification report cannot be generated."
        else:
            macro_accuracy = balanced_accuracy_score(all_labels, all_preds_filtered)
            report_str = classification_report(
                all_labels, all_preds_filtered, 
                labels=valid_labels, 
                zero_division=1.0, 
                digits=6
            )
            # report_str += f"\n\nMacro-Averaged Accuracy: {macro_accuracy:.6f}"

        # Save results
        report_path = os.path.join(args.results_path, f"eval_{lang_code}_results.txt")
        with open(report_path, "w") as f:
            f.write(report_str)
        
        print(f"Classification report for {lang_code} data:")
        print(report_str)
        print(f"Results saved to {report_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)