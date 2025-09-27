___author__ = "XI XUAN"

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import dataset and model
from src.datasets.dataset import family_MLAADFDDataset 
from src.models.w2v2_aasist import SSL_AASIST,W2VAASIST,ResNet18,XLSR_ResNet18, ECAPA_TDNN

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model across language families")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/family/Slavic/SSL-AASIST/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/family/Germanic/LFCC-ResNet18/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        # default="/home/xxuan/source_tracing_dataset/6l4m_exp/family/Slavic/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/anti-spoofing_feat_model.pth",
        help="Path to trained model (should contain 'Slavic')",#['Slavic', 'Romance', 'Germanic']
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/home/xxuan/source_tracing_dataset/6l4m_exp/family/results/",
        help="Base path to write results",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of system classes")
    return parser.parse_args()

def main(args):
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}..")

    # Define paths for each language family (each with two languages)
    # family_paths = {
    #     "Slavic": {
    #         "path1": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc",
    #         "path2": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc",
    #     },
    #     "Romance": {
    #         "path1": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc",
    #         "path2": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc",
    #     },
    #     "Germanic": {
    #         "path1": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc",
    #         "path2": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc",
    #     },
    # }

    family_paths = {
        "Slavic": {
            "path1": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature",
            "path2": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature",
        },
        "Romance": {
            "path1": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature",
            "path2": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature",
        },
        "Germanic": {
            "path1": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature",
            "path2": "/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature",
        },
    }

    # List of families to evaluate
    family_list = ['Slavic', 'Romance', 'Germanic']
    base_model_path = args.model_path
    base_results_path = args.results_path

    for family in family_list:
        print(f"\n{'=' * 60}")
        print(f"Evaluating model trained on {family} family...")

        # Update model and result paths
        # model_path = base_model_path.replace("Slavic", family)
        results_dir = os.path.join(base_results_path, family)
        os.makedirs(results_dir, exist_ok=True)

        # Load model
        print(f"Loading model from: {base_model_path}")
        # model = ResNet18(num_classes=4)
        # model = ECAPA_TDNN(C=192, num_classes=4)
        model = SSL_AASIST(args, device=device, class_num=4)
        try:
            state_dict = torch.load(base_model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model for {family}: {e}")
            continue
        model.to(device).eval()

        # Load combined dataset for the family
        paths = family_paths[family]
        print(f"Loading data from: {paths['path1']} and {paths['path2']}")
        try:
            dataset = family_MLAADFDDataset(
                path_to_features1=paths["path1"],
                path_to_features2=paths["path2"],
                part="eval",
                mode="known",
                max_samples=-1,
                class_num=args.num_classes
            )
            if len(dataset) == 0:
                print(f"No data found for family {family}. Skipping...")
                continue
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        except Exception as e:
            print(f"Error loading dataset for family {family}: {e}")
            continue

        # Run inference
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {family}"):
                feats, _, labels = batch


                # feature processing
                feats = feats.unsqueeze(dim=1)#lfcc 
                feats = feats.transpose(2, 3).to(device)
                
                # Forward pass
                # _, logits = model(feats)
                logits = model(feats)#ASSIST
                logits = F.softmax(logits, dim=1)


                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(predicted)
                all_labels.extend(labels.numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Filter out invalid predictions
        valid_labels = np.unique(all_labels)
        all_preds_filtered = np.array([p if p in valid_labels else -1 for p in all_preds])

        # Generate report
        if len(valid_labels) == 0:
            report_str = "No valid labels found."
        else:
            acc = balanced_accuracy_score(all_labels, all_preds_filtered)
            report = classification_report(
                all_labels,
                all_preds_filtered,
                labels=valid_labels,
                zero_division=1.0,
                digits=6
            )
            report_str = f"{report}\n"

        # Save results
        result_file = os.path.join(results_dir, "eval_results.txt")
        with open(result_file, "w") as f:
            f.write(report_str)

        print(f"Results saved to: {result_file}")
        print(report_str)

if __name__ == "__main__":
    args = parse_args()
    main(args)