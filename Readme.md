# ISL Non-Manual Feature (NMF) Detection System

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
   - **Expression CNN** — EfficientNet-B0 on AffectNet (8 classes)
   - **Head Movement TCN** — Temporal Convolutional Network on pose sequences (5 classes)
   - **Eye/Brow MLP** — Multi-layer perceptron on landmark features (3 classes)
6. Late fusion classifier (future — requires ELAN annotations)
7. Evaluation & ablation
8. Export model (ONNX for deployment)

---

## 📁 Project Structure

```
├── isl_nfm.ipynb           # Main training & evaluation notebook
├── isl_nmf_web.py          # Flask web interface for real-time inference
├── isl_nmf/
│   ├── data/               # Datasets (AffectNet, ISL-CSLTR, landmarks)
│   ├── checkpoints/        # Trained model weights (.pt files)
│   ├── logs/               # Training plots & visualizations
│   ├── face_landmarker.task # MediaPipe face landmark model
│   └── onnx/               # Exported ONNX models (after running Sec 18)
└── Readme.md
```

---

## ⚙️ Installation

```bash
# Core ML stack
pip install torch torchvision torchaudio
pip install mediapipe opencv-python-headless albumentations timm
pip install pandas matplotlib scikit-learn seaborn

# For the web interface
pip install flask

# For ONNX export & inference
pip install onnxruntime
```

---

## 🚀 Usage

### Training (Jupyter Notebook)

**Option 1 — Google Colab (recommended):**  
Open `isl_nfm.ipynb` in Colab using the badge link in the notebook's first cell. Upload your `kaggle.json` for dataset downloads.

**Option 2 — Local:**
```bash
jupyter notebook isl_nfm.ipynb
```
Run cells sequentially. Sections 2-4 download data; Sections 10-13 train models.

### Web Interface (Real-time Inference)

```bash
python isl_nmf_web.py
# Open http://localhost:5050
```

The web app supports:
- **Webcam mode** — real-time NMF detection via browser camera
- **Upload mode** — analyse a single image

If ONNX models exist in `isl_nmf/onnx/`, the app uses them; otherwise it runs in **landmark-only demo mode** with heuristic classifiers.

---

## 📊 Current Results

| Model | Task | Val Accuracy | Notes |
|-------|------|-------------|-------|
| Expression CNN | 8-class facial expression | ~63% | AffectNet pretrained |
| Head Movement TCN | 5-class head movement | ~43% | Limited real data (34 seqs) |
| Eye/Brow MLP | 3-class brow state | 100%* | *Heuristic labels — see caveat below |

> **⚠️ Eye/Brow caveat:** The 100% val accuracy reflects the model learning the heuristic labeling function, not true NMF detection. Real evaluation requires manually annotated data.

> **⚠️ Head Movement caveat:** Only 34 real sequences from 50 videos are used. Using all 700 ISL-CSLTR videos (`max_videos=None` in Section 4) is recommended for meaningful training.

---

## 🔮 What's Next

- **Full ISL-CSLTR:** Change `max_videos=50` → `None` to use all 700 videos
- **ELAN annotations:** Manually annotate NMF tiers for ground-truth evaluation
- **Cross-attention fusion:** Replace late fusion with transformer-based cross-attention
- **Hand keypoints:** Add MediaPipe Hands for a complete ISL system
- **Signer-independent CV:** Leave-one-signer-out cross-validation for realistic metrics

---

## 📄 License

This project is for academic/research purposes.
