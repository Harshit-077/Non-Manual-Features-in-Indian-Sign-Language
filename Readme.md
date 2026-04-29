# 🤟 ISL Non-Manual Feature (NMF) Detection System

A deep learning + computer vision pipeline for detecting **Non-Manual Features (NMFs)** in **Indian Sign Language (ISL)** using:

- Facial expressions  
- Head movements  
- Eye & brow features  

This project combines multiple models and modalities to understand the **grammar of ISL**, where meaning is often conveyed through facial cues rather than just hand gestures.

---

## 🧠 Overview

In sign languages, **non-manual features (NMFs)** such as eyebrow raises, head tilts, and facial expressions play a crucial grammatical role.

This system:

1. Uses facial expression recognition (pretrained on AffectNet)  
2. Extracts facial landmarks using MediaPipe  
3. Learns temporal head movements from ISL videos  
4. Combines all signals via **late fusion**  

---

## 📦 Datasets Used

### 1. AffectNet
- ~440K facial images  
- 8 expression classes  
- Used for **pretraining expression model**

### 2. ISL-CSLTR
- ~700 videos  
- 100 ISL sentences  
- 7 signers  
- Used for **NMF learning and sequence modeling**

---

## 🏗️ Pipeline
1. Download datasets (Kaggle API)

2. Explore & visualize data

3. Extract frames from ISL videos

4. Extract facial landmarks (MediaPipe)
5. Train 3 parallel models:
- Expression CNN 
- Head Movement TCN 
- Eye/Brow MLP
6. Late fusion classifier
7. Evaluation & ablation
8. Export model (ONNX for deployment)


---

## ⚙️ Installation

```bash
pip install torch torchvision torchaudio
pip install mediapipe opencv-python-headless albumentations timm
pip install pandas matplotlib scikit-learn
```


