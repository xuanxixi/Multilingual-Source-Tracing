"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.

___author__ = "XI XUAN"
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.sampler as torch_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.datasets.dataset import family_MLAADFDDataset, five_lang_MLAADFDDataset
from src.models.w2v2_aasist import SSL_AASIST,W2VAASIST,ResNet18,XLSR_ResNet18, ECAPA_TDNN


def log_results_to_file(file_path, epoch, train_loss, train_acc, val_loss, val_acc, train_gpu_mem, val_gpu_mem):
    with open(file_path, "a") as log_file:
        log_file.write(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%, Train GPU Mem: {train_gpu_mem:.4f} GiB, Val GPU Mem: {val_gpu_mem:.4f} GiB\n"
        )

def parse_args():
    parser = argparse.ArgumentParser("Training script parameters")

    # Paths to features and output
    parser.add_argument(
        "--path_to_features1",
        type=str,


        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature/",

        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc/",
        default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc/",

        help="Path to the previuosly extracted features",
    )

    parser.add_argument(
        "--path_to_features2",
        type=str,

        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature/",

        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc/",
        default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc/",


        help="Path to the previuosly extracted features",
    )

    parser.add_argument(
        "--path_to_features3",
        type=str,


        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature/",

        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc/",
        default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc/",

        help="Path to the previuosly extracted features",
    )

    parser.add_argument(
        "--path_to_features4",
        type=str,

        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature/",

        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc/",
        default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc/",


        help="Path to the previuosly extracted features",
    )

    parser.add_argument(
        "--path_to_features5",
        type=str,


        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/Clean_no_feature/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/Clean_no_feature/",

        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/en-en/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/de-de/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/fr-fr/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/it-it/LFCC_add_noise/preprocess_lfcc/",
        # default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/pl-pl/LFCC_add_noise/preprocess_lfcc/",
        default="/home/xxuan/source_tracing_dataset/6l4m/language_sub/ru-ru/LFCC_add_noise/preprocess_lfcc/",

        help="Path to the previuosly extracted features",
    )




    parser.add_argument(

        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/SSL-AASIST/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-pl/SSL-AASIST/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-it/SSL-AASIST/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-fr/SSL-AASIST/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/SSL-AASIST/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/SSL-AASIST/trained_models/"

        
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ResNet18/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-pl/LFCC-ResNet18/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-it/LFCC-ResNet18/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-fr/LFCC-ResNet18/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/LFCC-ResNet18/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/LFCC-ResNet18/LFCC_add_noise/trained_models/"

        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-pl/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-it/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-fr/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/"
        # "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/"
        "--out_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/LFCC-ECAPA-TDNN/LFCC_add_noise/trained_models/"

    
    )

    parser.add_argument(

        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/SSL-AASIST/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-pl/SSL-AASIST/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-it/SSL-AASIST/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-fr/SSL-AASIST/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/SSL-AASIST/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/SSL-AASIST/result/"
    
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ResNet18/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-pl/LFCC-ResNet18/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-it/LFCC-ResNet18/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-fr/LFCC-ResNet18/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/LFCC-ResNet18/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/LFCC-ResNet18/LFCC_add_noise/result/"

        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-ru/LFCC-ECAPA-TDNN/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-pl/LFCC-ECAPA-TDNN/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-it/LFCC-ECAPA-TDNN/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-fr/LFCC-ECAPA-TDNN/LFCC_add_noise/result/"
        # "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-de/LFCC-ECAPA-TDNN/LFCC_add_noise/result/"
        "--out_result_folder", type=str, default="/home/xxuan/source_tracing_dataset/6l4m_exp/five_lang/-en/LFCC-ECAPA-TDNN/LFCC_add_noise/result/"

    
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, help="random number seed", default=688)
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=1024,
        help="Feature dimension from the wav2vec model",
    )
    parser.add_argument(
        "--num_classes", type=int, default=24, help="Number of in domain classes"
    
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs for training"
    )
    parser.add_argument(
        # "--batch_size", type=int, default=128, help="Batch size for training"
        # "--batch_size", type=int, default=64, help="Batch size for training"#lfcc
        "--batch_size", type=int, default=4, help="Batch size for training"#ssl
    )
    parser.add_argument("--lr", type=float, 
                        default=0.0005, 
                        # default=0.001, 
                        help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="decay learning rate"
    )
    parser.add_argument("--interval", type=int, default=10, help="interval to decay lr")
    parser.add_argument("--beta_1", type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument("--beta_2", type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--num_workers", type=int, default=24, help="number of workers")
    parser.add_argument(
        "--base_loss",
        type=str,
        default="ce",
        choices=["ce", "bce"],
        help="Loss for basic training",
    )
    args = parser.parse_args()

    # Set seeds
    utils.set_seed(args.seed)

    # Path for output data
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_folder, "checkpoint")):
        os.makedirs(os.path.join(args.out_folder, "checkpoint"))



    # Save training arguments
    with open(os.path.join(args.out_folder, "args.json"), "w") as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=("\n", ":")))

    cuda = torch.cuda.is_available()
    print("Running on: ", "cuda" if cuda else "cpu")
    args.device = torch.device("cuda:1" if cuda else "cpu")
    return args

def train(args):
    
    # Load the train and dev data
    print("Loading training data...")
    training_set = five_lang_MLAADFDDataset(args.path_to_features1, args.path_to_features2, args.path_to_features3, args.path_to_features4, args.path_to_features5, "train", class_num=4)#
    print("\nLoading dev data...")
    dev_set = five_lang_MLAADFDDataset(args.path_to_features1, args.path_to_features2, args.path_to_features3, args.path_to_features4, args.path_to_features5, "dev", mode="known", class_num=4)#

    train_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))),
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(dev_set))),
    )

    # Setup the model

    # model = SSL_AASIST(args, device=args.device, class_num=4).to(args.device)
    # model = ResNet18(num_classes=4).to(args.device)

    model = ECAPA_TDNN(C=192, num_classes=4).to(args.device)

   
    print(f"Training a {type(model).__name__} model for {args.num_epochs} epochs")
    feat_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005,
    )
    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    prev_loss = 1e8
    # Main training loop
    for epoch_num in range(args.num_epochs):
        model.train()
        utils.adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        epoch_bar = tqdm(train_loader, desc=f"Epoch [{epoch_num+1}/{args.num_epochs}]")
        accuracy, train_loss = [], []
        train_gpu_mem = []
        for iter_num, batch in enumerate(epoch_bar):
            feat, audio, labels = batch
            # print("feat1",feat.shape)#feat1 torch.Size([128, 1, 199, 768])#lfcc: torch.Size([64, 80, 399])#ssl：torch.Size([64, 1, 1, 64000])
            # feat = feat.unsqueeze(dim=1).to(args.device)#ssl加上
            # feat = feat.unsqueeze(dim=1)#lfcc加上
            # print("feat2",feat.shape)#feat2 torch.Size([64, 1, 80, 399])

            #no-feature
            # feat = feat.unfold(dimension=1, size=400, step=400).to(args.device) # 分块 [64, 160, 400]
            # feat = feat.unsqueeze(dim=1)#resnet18
            # print("feat2",feat.shape)
            # feat = feat.transpose(1, 2).to(args.device)#ecapa-tdnn
            # print("feat3",feat.shape)#feat2 torch.Size([128, 1, 768, 199])

            #lfcc和SSL
            feat = feat.unsqueeze(dim=1).to(args.device)#lfcc加上
            feat = feat.transpose(2, 3).to(args.device)


            labels = labels.to(args.device)
            # print("labels",labels.shape)#labels torch.Size([128])


            #mix-label
            mix_feat, y_a, y_b, lam = utils.mixup_data(
                feat, labels, args.device, alpha=0.5
            )

            # print("mix_feat",mix_feat.shape)#mix_feat torch.Size([128, 1, 768, 199])
            # print("y_a",y_a.shape)#y_a torch.Size([128])
            # print("y_b",y_b.shape)# y_b torch.Size([128])
            


            targets_a = torch.cat([labels, y_a])
            targets_b = torch.cat([labels, y_b])
            # print("targets_a",targets_a.shape)# torch.Size([256])
            # print("targets_b",targets_b.shape)# torch.Size([256])
            feat = torch.cat([feat, mix_feat], dim=0)
            # print("feat3",feat.shape)#feat3 torch.Size([256, 1, 768, 199])

            # last_hidden, output, loss, prototype = model(feat)
            last_hidden, output = model(feat)#MLP
            # output = model(feat).to(args.device)#ssl

            # last_hidden, output, causal_reg = model(feat)#adapter

            # feats = last_hidden
            feat_outputs = output

            if args.base_loss == "bce":
                feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
            else:
                feat_loss = utils.regmix_criterion(
                    criterion, feat_outputs, targets_a, targets_b, lam
                )
                # 添加因果正则化损失
                # lambda_causal = 0.05  # 正则化权重从参数读取
                # feat_loss = feat_loss + lambda_causal * causal_reg

            score = F.softmax(feat_outputs, dim=1)
            predicted_classes = np.argmax(score.detach().cpu().numpy(), axis=1)
            correct_predictions = [
                1 for k in range(len(labels)) if predicted_classes[k] == labels[k]
            ]
            accuracy.append(sum(correct_predictions) / len(labels) * 100)
            train_loss.append(feat_loss.item())

            # Track GPU memory usage
            train_gpu_mem.append(torch.cuda.max_memory_allocated() / (1024 ** 3))

            epoch_bar.set_postfix(
                {
                    "train_loss": f"{sum(train_loss)/(iter_num+1):.4f}",
                    "acc": f"{sum(accuracy)/(iter_num+1):.4f}",
                    "gpu_mem": f"{train_gpu_mem[-1]:.4f} GiB",
                }
            )

            feat_optimizer.zero_grad()
            feat_loss.backward()
            feat_optimizer.step()

        train_loss_avg = sum(train_loss) / len(train_loss)
        train_acc_avg = sum(accuracy) / len(accuracy)
        train_gpu_mem_avg = sum(train_gpu_mem) / len(train_gpu_mem)

        # Epoch eval
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(dev_loader, desc=f"Validation for epoch {epoch_num+1}")
            accuracy, val_loss = [], []
            val_gpu_mem = []
            for iter_num, batch in enumerate(val_bar):
                feat, _, labels = batch
                # feat = feat.unsqueeze(dim=1).to(args.device)#ssl加上
                # feat = feat.unsqueeze(dim=1)#lfcc加上
                # feat = feat.transpose(2, 3).to(args.device)

                #no-feature
                # feat = feat.unfold(dimension=1, size=400, step=400).to(args.device) # 分块 [64, 160, 400]
                # feat = feat.unsqueeze(dim=1)#resnet18
                # print("feat2",feat.shape)
                # feat = feat.transpose(1, 2).to(args.device)#ecapa-tdnn
                # print("feat3",feat.shape)#feat2 torch.Size([128, 1, 768, 199])

                #lfcc和SSL
                feat = feat.unsqueeze(dim=1).to(args.device)#lfcc加上
                feat = feat.transpose(2, 3).to(args.device)


                labels = labels.to(args.device)

                # last_hidden, output, loss, prototype = model(feat)
                last_hidden, output = model(feat)#lfcc，w2v2
                # output = model(feat)#ssl
                # last_hidden, output, causal_reg = model(feat)#adapter

                # feats = last_hidden
                feat_outputs = output
                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)
                    # 添加因果正则化损失
                    # lambda_causal = 0.05  # 正则化权重从参数读取
                    # feat_loss = feat_loss + lambda_causal * causal_reg

                predicted_classes = np.argmax(score.detach().cpu().numpy(), axis=1)
                correct_predictions = [
                    1 for k in range(len(labels)) if predicted_classes[k] == labels[k]
                ]
                accuracy.append(sum(correct_predictions) / len(labels) * 100)

                val_loss.append(feat_loss.item())

                # Track GPU memory usage
                val_gpu_mem.append(torch.cuda.max_memory_allocated() / (1024 ** 3))

                val_bar.set_postfix(
                    {
                        "val_loss": f"{sum(val_loss)/(iter_num+1):.4f}",
                        "val_acc": f"{sum(accuracy)/(iter_num+1):.4f}",
                        "gpu_mem": f"{val_gpu_mem[-1]:.4f} GiB",
                    }
                )

            val_loss_avg = sum(val_loss) / len(val_loss)
            val_acc_avg = sum(accuracy) / len(accuracy)
            val_gpu_mem_avg = sum(val_gpu_mem) / len(val_gpu_mem)

            # Log results to file
            if not os.path.exists(args.out_result_folder):
                os.makedirs(args.out_result_folder)

            log_file_path = os.path.join(args.out_result_folder, "training_results.txt")
            log_results_to_file(log_file_path, epoch_num + 1, train_loss_avg, train_acc_avg, val_loss_avg, val_acc_avg, train_gpu_mem_avg, val_gpu_mem_avg)

            # Save model checkpoints
            if val_loss_avg < prev_loss:
                checkpoint_path = os.path.join(
                    args.out_folder, "anti-spoofing_feat_model.pth"
                )
                print(f"[INFO] Saving model with better val_loss to {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)
                prev_loss = val_loss_avg

            elif (epoch_num + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    args.out_folder,
                    "checkpoint",
                    "anti-spoofing_feat_model_%02d.pth" % (epoch_num + 1),
                )
                print(
                    f"[INFO] Saving intermediate model at epoch {epoch_num+1} to {checkpoint_path}"
                )
                torch.save(model.state_dict(), checkpoint_path)
            print("\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
