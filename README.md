# 🌐 Multilingual Source Tracing of Speech Deepfakes: A First Benchmark

![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23F1C40F.svg?logo=Hugging%20Face&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Official PyTorch Implementation: Multilingual Source Tracing of Speech Deepfakes: A First Benchmark 🕵️‍♂️🔍

Authors: Xi Xuan, Yang Xiao, Rohan Kumar Das, Tomi Kinnunen

---

## 📌 Abstract

Recent progress in generative AI has made it increasingly easy to create natural-sounding deepfake speech from just a few seconds of audio. While these tools support helpful applications, they also raise serious concerns by making it possible to generate convincing fake speech in many languages. Current research has largely focused on detecting fake speech, but little attention has been given to tracing the source models used to generate it. This paper introduces the first benchmark for multilingual speech deepfake source tracing, covering both mono- and crosslingual scenarios. We comparatively investigate DSP- and SSLbased modeling; examine how SSL representations fine-tuned on different languages impact cross-lingual generalization performance; and evaluate generalization to unseen languages and speakers. Our findings offer the first comprehensive insights into the challenges of identifying speech generation models when training and inference languages differ. 

## 🚀 Getting Started

### 📥 Download Dataset

The baseline is based on the [MLAAD (Source Tracing Protocols) dataset](https://deepfake-total.com/sourcetracing).

To download the required resources, run:

```bash
python scripts/download_resources.py
```
The default scripts' arguments assume that all the required data is put into `data` dir in the project root directory.

### Protocols


### 🧰 Install Dependencies

Install all required dependencies from the `requirements.txt` file. The baseline was developed and tested using Python 3.11.

```bash
pip install -r requirements.txt
```

### 📂 Data augmentation and feature extraction

For data augmentation, download the 🎵 [MUSAN](https://www.openslr.org/17/) dataset, which provides four noise types (noise, music, babble, and reverberation). Each clean utterance is augmented to create five variants (original + 4 perturbed), enhancing model robustness in diverse acoustic environments.

### Step 1. Data augmentation and feature extraction

The first step of the tool reads the MCL-MLAAD data, augments it with four noise types (noise, music, babble, and reverberation) and extracts
the DSP/SSL features needed to train the AASIST/ResNet/ECAPA-TDNN model.  Additional parameters can be set from the script,
such as max length, model, etc. 

```bash
python scripts/preprocess_dataset.py
```

Output will be written to `exp/preprocess_xxx-base/`. You can change the path in the script. 



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



