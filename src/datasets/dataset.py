"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.

___author__ = "XI XUAN"
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        meta_data: dict,
        basepath: str,
        class_mapping: dict,
        num_utter_per_class: int = 10,
        sr: int = 16_000,
        sample_length_s: float = 4,
        verbose: bool = False,
    ):
        super().__init__()
        self.class_mapping = class_mapping
        self.basepath = Path(basepath)
        self.sr = sr
        self.sample_length_s = sample_length_s
        self.num_utter_per_class = num_utter_per_class
        self.verbose = verbose
        self.samples, self.classes_in_subset = self._parse_samples(meta_data)

        if self.verbose:
            self._print_initialization_info()

    def _print_initialization_info(self):
        print("\n > Dataset initialization")
        print(f" | > Number of instances: {len(self.samples)}")
        print(f" | > Sequence length: {self.sample_length_s} s")
        print(f" | > Sampling rate: {self.sr}")
        print(f" | > Num Classes: {len(self.classes_in_subset)}")
        print(f" | > Classes: {list(self.classes_in_subset)}")

    def load_wav(self, file_path: str) -> np.ndarray:
        audio, sr = librosa.load(file_path, sr=None)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        return audio

    def _parse_samples(self, meta_data):
        class_to_utters = defaultdict(list)
        for sample in meta_data:
            path_ = self.basepath / sample["path"]
            assert path_.exists(), f"File does not exist: {path_}"
            class_name = sample["model_id"]
            class_to_utters[class_name].append(path_)

        inv_class_mapping = {v: k for k, v in self.class_mapping.items()}

        # skip classes with number of samples >= self.num_utter_per_class
        class_to_utters = {
            inv_class_mapping[k]: v
            for (k, v) in class_to_utters.items()
            if len(v) >= self.num_utter_per_class
        }

        classes = list(class_to_utters.keys())
        classes.sort()

        new_items = []
        for sample in meta_data:
            path_ = self.basepath / sample["path"]
            class_name = sample["model_name"]
            new_items.append(
                {
                    "wav_file_path": path_,
                    "class_name": class_name,
                    "class_id": self.class_mapping[class_name],
                }
            )

        return new_items, classes

    def __len__(self):
        return len(self.samples)

    def get_num_classes(self):
        return len(self.classes_in_subset)

    def get_class_list(self):
        return list(self.classes_in_subset)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def collate_fn(self, batch: torch.Tensor, return_metadata: bool = True) -> tuple:
        labels, feats, paths = [], [], []
        target_length = int(self.sample_length_s * self.sr)

        for item in batch:
            wav = self.load_wav(item["wav_file_path"])
            wav = self._process_wav(wav, target_length)
            feats.append(torch.from_numpy(wav).unsqueeze(0).float())
            labels.append(item["class_id"])
            paths.append(item["wav_file_path"])

        feats_tensor = torch.stack(feats)
        labels_tensor = torch.LongTensor(labels)
        return (
            (feats_tensor, labels_tensor, paths)
            if return_metadata
            else (feats_tensor, labels_tensor)
        )

    def _process_wav(self, wav: np.ndarray, target_length: int) -> np.ndarray:
        if wav.shape[0] >= target_length:
            offset = random.randint(0, wav.shape[0] - target_length)
            wav = wav[offset : offset + target_length]
        else:
            wav = np.pad(wav, (0, target_length - wav.shape[0]), mode="wrap")
        return wav


from typing_extensions import Tuple
class MLAADBaseDataset(Dataset):
    def __init__(
        self,
        meta_data: dict,
        basepath: str,
        class_mapping: dict,
        out_folder: str ,
        sr: int = 16_000,
        sample_length_s: float = 4,
        max_samples=-1,
        verbose: bool = True,
    ):
        super().__init__()
        self.class_mapping = {k: v[0] for k, v in class_mapping.items()}
        self.items = meta_data
        self.sample_length_s = sample_length_s
        self.basepath = basepath
        self.sr = sr
        self.verbose = verbose
        self.classes, self.items = self._parse_items()
        self.out_folder = out_folder

        # [TEMP] limit the number of samples per class for testing
        if max_samples > 0:
            counts = {k: 0 for k in self.classes}
            new_items = []
            for k in range(len(self.items)):
                if counts[self.items[k]["class_id"]] < max_samples:
                    new_items.append(self.items[k])
                    counts[self.items[k]["class_id"]] += 1

            self.items = new_items

        if self.verbose:
            self._print_initialization_info()

    # def _print_initialization_info(self):
    #     print("\n > DataLoader initialization")
    #     print(f" | > Number of instances : {len(self.items)}")
    #     print(f" | > Max sequence length: {self.sample_length_s} seconds")
    #     print(f" | > Num Classes: {len(self.classes)}")
    #     print(f" | > Classes: {self.classes}")

    def _print_initialization_info(self):
        # Collect the initialization information
        initialization_info = [
            "\n > DataLoader initialization",
            f" | > Number of instances : {len(self.items)}",
            f" | > Max sequence length: {self.sample_length_s} seconds",
            f" | > Num Classes: {len(self.classes)}",
            f" | > Classes: {self.classes}"
        ]

        # Print the information to the console
        for line in initialization_info:
            print(line)

        # Define the output folder and file path
        out_folder = self.out_folder
        # os.makedirs(out_folder, exist_ok=True)  # Ensure the folder exists
        stat_file_path = os.path.join(out_folder, "stat.txt")

        # Write the information to the file
        with open(stat_file_path, "a") as f:
            for line in initialization_info:
                f.write(line + "\n")

        print(f"Initialization information written to {stat_file_path}")

    def load_wav(self, file_path: str) -> np.ndarray:
        audio, sr = librosa.load(file_path, sr=None)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        return audio

    def _parse_items(self):
        class_to_utters = defaultdict(list)
        for item in self.items:
            path = Path(self.basepath) / item["path"]
            assert os.path.exists(path), f"File does not exist: {path}"
            class_id = self.class_mapping[item["model_name"]]
            class_to_utters[class_id].append(path)

        classes = sorted(class_to_utters.keys())
        new_items = [
            {
                "wav_file_path": Path(self.basepath) / item["path"],
                "class_id": self.class_mapping[item["model_name"]],
            }
            for item in self.items
        ]
        return classes, new_items

    def __len__(self):
        return len(self.items)

    def get_num_classes(self):
        return len(self.classes)

    def get_class_list(self):
        return self.classes

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

    # def collate_fn(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def collate_fn(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels, feats, files = [], [], []
        target_length = int(self.sample_length_s * self.sr)

        for item in batch:
            utter_path = item["wav_file_path"]
            class_id = item["class_id"]
            wav = self.load_wav(utter_path)
            wav = self._process_wav(wav, target_length)
            feats.append(torch.from_numpy(wav).unsqueeze(0).float())
            labels.append(class_id)
            files.append(item["wav_file_path"])
        return torch.stack(feats), torch.LongTensor(labels), files

    def _process_wav(self, wav: np.ndarray, target_length: int) -> np.ndarray:
        if wav.shape[0] >= target_length:
            offset = random.randint(0, wav.shape[0] - target_length)
            wav = wav[offset : offset + target_length]
        else:
            wav = np.pad(wav, (0, max(0, target_length - wav.shape[0])), mode="wrap")
        return wav

import glob

class MLAADFDDataset(Dataset):
    def __init__(self, path_to_features, part="train", mode="train", max_samples=-1, class_num=24):
        super().__init__()
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = path_to_features.strip()  # 去除路径中多余的空格
        self.ptf = os.path.join(path_to_features, self.part)
        self.ptf = self.ptf.strip()  # 去除路径中多余的空格
        self.all_files = glob.glob(os.path.join(self.ptf, "*.pt"))

        

        if mode == "known":
            # keep only known classes seen during training for F1 metrics
            self.all_files = [
                x for x in self.all_files if int(os.path.basename(x).split("_")[1]) < class_num#15#24
            ]

        if max_samples > 0:
            self.all_files = self.all_files[:max_samples]

        # Determine the set of labels
        self.labels = sorted(
            set([int(os.path.split(x)[1].split("_")[1]) for x in self.all_files])
        )
        self._print_info()

    def _print_info(self):
        print(f"Searching for features in folder: {self.ptf}")
        print(f"Found {len(self.all_files)} files...")
        print(f"Using {len(self.labels)} classes\n")
        print(
            "Seen classes: ",
            set([int(os.path.basename(x).split("_")[1]) for x in self.all_files]),
        )

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split("_")

        feature_tensor = torch.load(filepath)
        filename = "_".join(all_info[2:-1])
        label = int(all_info[1])

        return feature_tensor, filename, label


import os
import glob
from torch.utils.data import Dataset
import torch

class family_MLAADFDDataset(Dataset):
    def __init__(self, path_to_features1, path_to_features2, part="train", mode="train", max_samples=-1, class_num=4):
        super().__init__()
        self.part = part
        
        # 处理第一个路径
        self.ptf1 = path_to_features1.strip()
        self.ptf1 = os.path.join(self.ptf1, self.part)
        
        # 处理第二个路径
        self.ptf2 = path_to_features2.strip()
        self.ptf2 = os.path.join(self.ptf2, self.part)
        
        # 收集两个路径下的所有 .pt 文件
        self.all_files = []
        self.all_files.extend(glob.glob(os.path.join(self.ptf1, "*.pt")))
        self.all_files.extend(glob.glob(os.path.join(self.ptf2, "*.pt")))
        
        # 过滤已知类别（仅在 mode == "known" 时）
        if mode == "known":
            self.all_files = [
                x for x in self.all_files 
                if int(os.path.basename(x).split("_")[1]) < class_num
            ]
        
        # 限制最大样本数
        if max_samples > 0:
            self.all_files = self.all_files[:max_samples]

        # 提取唯一标签
        self.labels = sorted(
            set([int(os.path.basename(x).split("_")[1]) for x in self.all_files])
        )
        
        self._print_info()

    def _print_info(self):
        print(f"Searching for features in folders: {self.ptf1} and {self.ptf2}")
        print(f"Found {len(self.all_files)} files...")
        print(f"Using {len(self.labels)} classes\n")
        print("Seen classes: ", set([int(os.path.basename(x).split("_")[1]) for x in self.all_files]))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split("_")

        feature_tensor = torch.load(filepath)
        filename = "_".join(all_info[2:-1])
        label = int(all_info[1])

        return feature_tensor, filename, label




import os
import glob
import torch
from torch.utils.data import Dataset

class five_lang_MLAADFDDataset(Dataset):
    def __init__(self, path_to_features1, path_to_features2, path_to_features3, 
                 path_to_features4, path_to_features5, part="train", mode="train", 
                 max_samples=-1, class_num=4):
        super().__init__()
        self.part = part
        
        # 处理五个路径
        self.paths = []
        for p in [path_to_features1, path_to_features2, path_to_features3, 
                 path_to_features4, path_to_features5]:
            processed_path = p.strip()
            processed_path = os.path.join(processed_path, self.part)
            self.paths.append(processed_path)
        
        # 收集所有 .pt 文件
        self.all_files = []
        for path in self.paths:
            self.all_files.extend(glob.glob(os.path.join(path, "*.pt")))
        
        # 过滤已知类别（仅在 mode == "known" 时）
        if mode == "known":
            self.all_files = [
                x for x in self.all_files 
                if int(os.path.basename(x).split("_")[1]) < class_num
            ]
        
        # 限制最大样本数
        if max_samples > 0:
            self.all_files = self.all_files[:max_samples]

        # 提取唯一标签
        self.labels = sorted(
            set([int(os.path.basename(x).split("_")[1]) for x in self.all_files])
        )
        
        self._print_info()

    def _print_info(self):
        print(f"Searching for features in folders: {', '.join(self.paths)}")
        print(f"Found {len(self.all_files)} files...")
        print(f"Using {len(self.labels)} classes\n")
        print("Seen classes: ", set([int(os.path.basename(x).split("_")[1]) for x in self.all_files]))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split("_")

        feature_tensor = torch.load(filepath)
        filename = "_".join(all_info[2:-1])
        label = int(all_info[1])

        return feature_tensor, filename, label

    
