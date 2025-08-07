# 🌐 Multilingual Source Tracing of Speech Deepfakes: A First Benchmark

![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23F1C40F.svg?logo=Hugging%20Face&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Official PyTorch Implementation: Multilingual Source Tracing of Speech Deepfakes: A First Benchmark 🕵️‍♂️🔍

Authors: *Xi Xuan*, *Yang Xiao*, *Rohan Kumar Das*, *Tomi Kinnunen*

---

## 📌 Introduction

We present the **first multilingual benchmark** for **Source Tracing (ST) of speech deepfakes**, a task in identifying the origin of synthetic speech across diverse languages 🌍🗣️.

This work introduces a comprehensive evaluation framework that covers both **mono-lingual** and **cross-lingual** scenarios, enabling robust analysis of deepfake attribution in real-world, multilingual settings. 🔍

Furthermore, it is the **first study** to investigate the impact of **unseen languages** and **unseen speakers** on source tracing performance, paving the way for more generalizable and realistic deepfake detection systems 🚀.

Our benchmark is built on the MLAAD dataset and our work offers the first systematic evaluation of multilingual ST and lays a foundation for future research in this area. 

## 🚀 Getting Started

### 📥 Download Dataset

The baseline is based on the [MLAAD (Source Tracing Protocols) dataset](https://deepfake-total.com/sourcetracing).

To download the required resources, run:

```bash
python scripts/download_resources.py

The default scripts' arguments assume that all the required data is put into `data` dir in the project root directory.

### 🧰 Install Dependencies

Install all required dependencies from the `requirements.txt` file. The baseline was developed and tested using Python 3.11.

```bash
pip install -r requirements.txt
```

### 📂 Data augmentation and feature extraction

For data augmentation, download the 🎵 [MUSAN](https://www.openslr.org/17/) dataset, which provides four noise types (noise, music, babble, and reverberation). Each clean utterance is augmented to create five variants (original + 4 perturbed), enhancing model robustness in diverse acoustic environments.

### Step 1. Data augmentation and feature extraction

The first step of the tool reads the MCL-MLAAD data, augments it with four noise types (noise, music, babble, and reverberation) and extracts
the `wav2vec2-base` features needed to train the AASIST model.  Additional parameters can be set from the script,
such as max length, model, etc. 

```bash
python scripts/preprocess_dataset.py
```

Output will be written to `exp/preprocess_wav2vec2-base/`. You can change the path in the script. 

### Step 2. Train a AASIST model on top of the wav2vec2-base features

Using the augmented features, we then train an AASIST model for 30 epochs. The model is able to classify the samples
with respect to the source system. The class assignment will be written to `exp/label_assignment.txt`.

```bash
python train_refd.py
```

### Step 3. Get the classification metrics for the known (in-domain) classes

Given the trained model stored in `exp/trained_models/`, we can now compute its accuracy over known classes (those
seen during training time).

```bash
python scripts/get_classification_metrics.py
```

The script will limit the data in the `dev` and `eval` sets to the samples which are from the known systems 
(i.e. those also present in the training data) and compute their classification metrics.



# ✍️ Citation
If you find our work helpful, please use the following citations.
```  
@misc{xuan2025multilingualsourcetracingspeech,
      title={Multilingual Source Tracing of Speech Deepfakes: A First Benchmark}, 
      author={Xi Xuan and Yang Xiao and Rohan Kumar Das and Tomi Kinnunen},
      year={2025},
      eprint={2508.04143},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2508.04143}, 
}
```



