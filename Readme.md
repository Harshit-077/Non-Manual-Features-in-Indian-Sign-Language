# ISL Full Inference Pipeline

A real-time **Indian Sign Language (ISL)** recognition system that combines three trained models to detect facial expressions, head movements (Non-Manual Features / NMF), and full-sentence signs from live webcam or video input.

---

## Project Structure

```
.
├── training/
│   ├── isl_nmf_expr_hm.ipynb          # Training notebook: Expression CNN + Head Movement TCN
│   └── isl_pyt_200.ipynb              # Training notebook: ISL Sentence BiLSTM
│
├── isl_nmf/
│   └── onnx/
│       ├── expression_cnn.onnx    # Exported Expression CNN (EfficientNet-B0 + BiLSTM)
│       └── head_movement_tcn.onnx # Exported Head Movement TCN
│
├── models&config/
│   ├── isl_best_model.pt          # Trained ISL Sentence BiLSTM weights
│   └── label_encoder.pkl          # Sklearn LabelEncoder for sentence classes
│
├── hand_landmarker.task           # MediaPipe Hand Landmarker model
├── face_landmarker.task           # MediaPipe Face Landmarker model
├── pose_landmarker_full.task      # MediaPipe Pose Landmarker model
├── test.py                        # Main inference script (entry point)
└── requirements.txt


```

---

## Models

### 1. Expression CNN (`isl_nmf_expr_hm.ipynb`)
- **Architecture:** EfficientNet-B0 backbone + projection head + BiLSTM classifier
- **Task:** 8-class facial expression recognition (neutral, happy, sad, surprise, fear, disgust, anger, contempt)
- **Training Data:** AffectNet (~440K images) → fine-tuned on ISL-CSLTR face crops
- **Export:** ONNX (`expression_cnn.onnx`), input `(B, 3, 112, 112)`

### 2. Head Movement TCN (`isl_nmf_expr_hm.ipynb`)
- **Architecture:** Causal dilated Temporal Convolutional Network (TCN)
- **Task:** 5-class head movement recognition (nod, shake, tilt, forward, still)
- **Training Data:** ISL-CSLTR videos (MediaPipe head pose sequences) + synthetic augmentation
- **Export:** ONNX (`head_movement_tcn.onnx`), input `(B, T, 3)` — pitch/yaw/roll sequences

### 3. ISL Sentence BiLSTM (`isl_pyt_200.ipynb`)
- **Architecture:** 2× Bidirectional LSTM → BatchNorm → Dense(256) → Dense(128) → Softmax
- **Task:** Multi-class ISL sentence recognition (up to 200 classes from ISL-CSLTR)
- **Input:** Sequences of MediaPipe body keypoints — pose (33×3) + left hand (21×3) + right hand (21×3) = 225 features per frame
- **Saved as:** PyTorch `.pt` checkpoint

---

## Installation

```bash
pip install torch torchvision onnxruntime mediapipe opencv-python-headless numpy
```

For GPU inference with CUDA, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

### Run inference on webcam (default)
```bash
python test.py
```

### Run on a specific webcam index or video file
```bash
python test.py --video 1
python test.py --video path/to/video.mp4
```

### Disable individual model streams
```bash
python test.py --no-nmf         # Disable Expression + Head Movement
python test.py --no-sentence    # Disable ISL Sentence model
```

> **Note:** Passing both `--no-nmf` and `--no-sentence` at the same time will exit with an error.

### Controls
Press `Q` to quit the live inference window.

---

## Output Overlay

The live window displays three lines:

| Label | Source | Color |
|---|---|---|
| `Expression: happy (0.91)` | Expression CNN | Green |
| `Head: nod (0.87)` | Head Movement TCN | Green |
| `Sentence: thank_you (0.76)` | ISL Sentence BiLSTM | Yellow |

Predictions below the sentence confidence threshold (`0.40`) are shown with a `[low conf]` tag.

---

## Training Notebooks

### `isl_nmf_expr_hm.ipynb` — NMF Detection
Trains the Expression CNN and Head Movement TCN end-to-end:

1. Environment setup
2. Kaggle dataset download (AffectNet + ISL-CSLTR)
3. Dataset exploration and visualisation
4. Frame extraction from ISL-CSLTR videos
5. MediaPipe landmark extraction (face, pose, head pose)
6. Head movement sequence construction (sliding windows + synthetic augmentation)
7. Dataset classes and DataLoaders
8. Model architectures (ExpressionCNNLSTM + HeadMovementTCN)
9. Training utilities
10. Train Expression CNN on AffectNet
11. Fine-tune Expression CNN on ISL face crops
12. Train Head Movement TCN
13. Evaluation and confusion matrices
14. Ablation study
15. Real-time inference pipeline demo
16. Test on ISL-CSLTR frame samples
17. **ONNX export** (generates the `.onnx` files used by `test.py`)

### `isl_pyt_200.ipynb` — Sentence Classification
Trains the ISL Sentence BiLSTM end-to-end:

1. Dataset loading from `ISL_CSLRT_Corpus/Videos_Sentence_Level`
2. MediaPipe keypoint extraction with disk caching
3. Class imbalance audit (thin classes removed by default)
4. Label encoding and train/test split
5. PyTorch DataLoaders
6. BiLSTM model definition
7. Training loop with early stopping and ReduceLROnPlateau
8. Model checkpoint saving (`isl_best_model.pt`, `label_encoder.pkl`)
9. Evaluation and training history plots

---

## Datasets

| Dataset | Purpose | Kaggle Slug |
|---|---|---|
| AffectNet | Expression CNN pretraining | `mstjebashazida/affectnet` |
| ISL-CSLTR | Head movement + sentence training | `drblack00/isl-csltr-indian-sign-language-dataset` |

To download via Kaggle API, place `kaggle.json` in `~/.kaggle/` and run the download cells in `isl_nmf_expr_hm.ipynb` Section 2.

---

## Configuration (test.py)

Key constants at the top of `test.py`:

| Constant | Default | Description |
|---|---|---|
| `SEQ_LEN` | `30` | Frames per temporal window |
| `NUM_KEYPOINTS` | `225` | Body keypoint vector length |
| `FRAME_SKIP` | `5` | Sample every Nth frame for the sentence model |
| `SENTENCE_CONF` | `0.40` | Minimum confidence to display a sentence prediction |

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- ONNX Runtime
- MediaPipe ≥ 0.10
- OpenCV
- NumPy

GPU (CUDA) is supported automatically; the pipeline falls back to CPU if unavailable.
