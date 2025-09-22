# 🌐 Multilingual Source Tracing of Speech Deepfakes: A First Benchmark

![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23F1C40F.svg?logo=Hugging%20Face&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


(Accepted at Interspeech SPSC 2025-Oral 🇳🇱🎙) 

[![arXiv](https://img.shields.io/badge/arXiv-2508.09294v1-b31b1b.svg)](https://www.arxiv.org/abs/2508.04143) [[slides](https://beaverton-my.sharepoint.com/personal/xixuan3_um_cityu_edu_hk/Documents/Interspeech25_SPSC_PPT_Xi_Xuan.pdf?CT=1756894314188&OR=ItemsView)]

📅 Presentation Details:
*   **🎙️ Speaker:** Xi Xuan
*   **🗓️ Date:** Saturday, August 16th, 2025
*   **🕚 Time:** 11:15 – 12:15
*   **📍 Location:** Aula Conference Center, TU Delft, Netherlands 🇳🇱

---

Official PyTorch Implementation: Multilingual Source Tracing of Speech Deepfakes: A First Benchmark 🕵️‍♂️🔍

Authors: Xi Xuan, Yang Xiao, Rohan Kumar Das, Tomi Kinnunen


## 📌 Abstract

Recent progress in generative AI has made it increasingly easy to create natural-sounding deepfake speech from just a few seconds of audio. While these tools support helpful applications, they also raise serious concerns by making it possible to generate convincing fake speech in many languages. Current research has largely focused on detecting fake speech, but little attention has been given to tracing the source models used to generate it. This paper introduces the first benchmark for multilingual speech deepfake source tracing, covering both mono- and cross-lingual scenarios. We comparatively investigate DSP- and SSL-based modeling; examine how SSL representations fine-tuned on different languages impact cross-lingual generalization performance; and evaluate generalization to unseen languages and speakers. Our findings offer the first comprehensive insights into the challenges of identifying speech generation models when training and inference languages differ. 

## 🚀 Getting Started

### 📥 Download Dataset

The baseline is based on the [MLAAD (Source Tracing Protocols) dataset](https://deepfake-total.com/sourcetracing).

To download the required resources, run:

```bash
python scripts/download_resources.py
```
The default scripts' arguments assume that all the required data is put into `data` dir in the project root directory.

### Protocol folders (mapping to paper Section 2.2)

**There are three folders under the Hugging Face dataset.**  
Each subsection in the paper maps to one of these folders:

- **2.2.1 Mono- & Cross-Lingual Protocol**  
  **2.2.2 Mono- & Cross Language Family Protocol**  
  **2.2.3 Seen & Unseen Languages Protocol**  
  → use files under **Languages Protocol**  
  🔗 [Languages Protocol](https://huggingface.co/datasets/xxuan-speech/Multilingual_Source_Tracing_Protocals/tree/main/Languages%20Protocol)

- **2.2.4 Seen & Unseen Speakers Protocol**  
  → use files under **Seen & Unseen Speakers Eval Protocol**  
  🔗 [Seen & Unseen Speakers Eval Protocol](https://huggingface.co/datasets/xxuan-speech/Multilingual_Source_Tracing_Protocals/tree/main/Seen%20%26%20Unseen%20Speakers%20Eval%20Protocol)

- **Complete protocol with pseudo-speaker labels (language + speaker-id)**  
  → generated via our pseudo-speaker labeling method; stored under **Languages+Speakers Protocol**  
  🔗 [Languages+Speakers Protocol](https://huggingface.co/datasets/xxuan-speech/Multilingual_Source_Tracing_Protocals/tree/main/Languages%2BSpeakers%20Protocol)


### DSP Models

All DSP front-ends include LFCC, implemented using [![torchaudio LFCC](https://img.shields.io/badge/torchaudio-LFCC-blue?logo=pytorch)](https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.LFCC.html).



### SSL Models

All SSL front-ends include two foundation models: wav2vec2.0 Large LV-60 and XLS-R-300M. Language-specific fine-tuned variants are based on large-xlsr-53, trained on six languages (en, de, fr, it, pl, ru).  

| Name              | Pretraining Dur. (h) | Pretraining Langs. | Fine-tuning Lang. | Datasets        | Hugging Face Link |
|-------------------|----------------------|--------------------|-------------------|-----------------|-------------------|
| **wav2vec2**      |                      |                    |                   |                 |                   |
| 1. large-lv60     | 53k                  | en                 | --                | LL              | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/facebook/wav2vec2-large-lv60) |
| 2. xls-r-300m     | 436k                 | many               | --                | CV, BBL, MLS, VP, VL | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/facebook/wav2vec2-xls-r-300m) |
| **Fine-tuned variants** |             |                    |                   |                 |                   |
| 3. large-xlsr-53-en | 56k                | many               | en                | CV-en           | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) |
| 4. large-xlsr-53-de | 56k                | many               | de                | CV-de           | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german) |
| 5. large-xlsr-53-fr | 56k                | many               | fr                | CV-fr           | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) |
| 6. large-xlsr-53-it | 56k                | many               | it                | CV-it           | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian) |
| 7. large-xlsr-53-pl | 56k                | many               | pl                | CV-pl           | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-polish) |
| 8. large-xlsr-53-ru | 56k                | many               | ru                | CV-ru           | [<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian) |

**Datasets Key**:  
- Librispeech (LL) [¹](https://ieeexplore.ieee.org/document/7178964)  
- CommonVoice (CV) [²](https://arxiv.org/abs/1912.06670)  
- BABEL (BBL) [³](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=31a13cefb42647e924e0d2778d341decc44c40e9)  
- Multilingual Librispeech (MLS) [⁴](https://arxiv.org/abs/2012.03411)  
- VoxPopuli (VP) [⁵](https://arxiv.org/abs/2101.00390)  
- VoxLingua107 (VL) [⁶](https://arxiv.org/abs/2011.12998)


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



