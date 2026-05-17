# ISL Full Inference Pipeline

A real-time **Indian Sign Language (ISL)** recognition system that combines three trained models to detect facial expressions, head movements (Non-Manual Features / NMF), and full-sentence signs from live webcam or video input.

---

## Project Structure

```
.
├── isl_nmf/
│   ├── checkpoints/
│   │   ├── expr_best.pt                # Expression model checkpoint
│   │   └── hm_best.pt                  # Head movement model checkpoint
│   └── onnx/
│       ├── expression_resnet_tcn.onnx  # Exported Expression model (B, T, C, H, W)
│       └── head_movement_tcn.onnx      # Exported Head Movement TCN (B, T, 3)
│
├── isl_nmf_expr_hm.ipynb               # Training: Expression ResNetTCN + Head Movement TCN
├── isl_pyt_200_tcn.ipynb               # Training: ISL Sentence TCN
│
├── isl_best_model.pt                   # ISL Sentence TCN weights
├── label_encoder.pkl                   # Sklearn LabelEncoder for sentence classes
│
├── hand_landmarker.task                # MediaPipe Hand Landmarker model
├── face_landmarker.task                # MediaPipe Face Landmarker model (auto-downloaded)
├── pose_landmarker_full.task           # MediaPipe Pose Landmarker model
│
├── isl_combined_inference.py           # Combined inference — all 3 models simultaneously
├── isl_inference.py                    # Sentence-only inference
├── isl_nmf_inference.py                # Expression + Head Movement inference only
└── requirements.txt
```

---

## Models

### 1. Expression ResNetTCN (`isl_nmf_expr_hm.ipynb`)
- **Architecture:** ResNet-18 backbone → linear projection (512 → 256) → 3-block causal dilated TCN (256 → 256 → 128) → GlobalAvgPool → Dense(128) → Softmax
- **Task:** 8-class facial expression recognition (neutral, happy, sad, surprise, fear, disgust, anger, contempt)
- **Training Data:** AffectNet (~440K images) → fine-tuned on ISL-CSLTR face crops
- **Checkpoint:** `isl_nmf/checkpoints/expr_best.pt`
- **ONNX export:** `expression_resnet_tcn.onnx`, input `(B, T, C, H, W)` — T=1 at inference

### 2. Head Movement TCN (`isl_nmf_expr_hm.ipynb`)
- **Architecture:** 2-block causal dilated TCN (32 → 64) → GlobalAvgPool → Dense(64) → Softmax
- **Task:** 5-class head movement recognition (nod, shake, tilt, forward, still)
- **Training Data:** ISL-CSLTR videos (MediaPipe head pose sequences) + synthetic augmentation
- **Checkpoint:** `isl_nmf/checkpoints/hm_best.pt`
- **ONNX export:** `head_movement_tcn.onnx`, input `(B, T, 3)` — pitch/yaw/roll sequences

### 3. ISL Sentence TCN (`isl_pyt_200_tcn.ipynb`)
- **Architecture:** 4-block causal dilated TCN (128 → 128 → 256 → 256) → GlobalAvgPool → BatchNorm → Dense(256) → Dense(128) → Softmax
- **Task:** Multi-class ISL sentence recognition (up to 200 classes from ISL-CSLRT)
- **Input:** Sequences of MediaPipe body keypoints — pose (33×3) + left hand (21×3) + right hand (21×3) = 225 features per frame, sampled every 5 frames into a 30-frame window
- **Checkpoint:** `isl_best_model.pt` + `label_encoder.pkl`

---

## Installation

```bash
# PyTorch with CUDA (recommended — check pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# All other dependencies
pip install -r requirements.txt
```

For CPU-only inference, skip the CUDA torch install and just run `pip install -r requirements.txt`.

---

## Usage

### Combined inference — all 3 models at once (recommended)

```bash
# Webcam
python isl_combined_inference.py --source webcam

# Video file
python isl_combined_inference.py --source video --path "path/to/video.mp4"

# Custom checkpoint paths
python isl_combined_inference.py --source webcam \
    --sent-model  isl_best_model.pt \
    --sent-encoder label_encoder.pkl \
    --expr-ckpt   isl_nmf/checkpoints/expr_best.pt \
    --hm-ckpt     isl_nmf/checkpoints/hm_best.pt
```

### Sentence-only inference

```bash
python isl_inference.py --source webcam
python isl_inference.py --source video --path "path/to/video.mp4"
```

### Expression + Head Movement only

```bash
python isl_nmf_inference.py --source webcam
python isl_nmf_inference.py --source video --path "path/to/video.mp4"
```

### Controls (all scripts)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset all buffers (combined), or sentence buffer (sentence-only) |

---

## Output Overlay

The combined inference window displays three rows in a semi-transparent panel:

| Label | Stream | Color |
|-------|--------|-------|
| `SIGN: thank_you (76.3%)` | Sentence TCN | Green |
| `EXPR: neutral (84.1%) [Statement]` | Expression ResNetTCN | Cyan |
| `HEAD: nod (71.2%) [Affirm / end]` | Head Movement TCN | Orange |

Two progress bars at the bottom show how full the sentence buffer and head movement buffer are.

---

## Short Video Handling

The sentence model requires a 30-frame keypoint window (150 video frames at `FRAME_SKIP=5`). For videos shorter than this, the extracted keyframes are **looped round-robin** to fill the buffer before prediction fires. The console prints `[buffer filled by looping]` when this happens.

---

## Training Notebooks

### `isl_nmf_expr_hm.ipynb` — NMF Detection

Trains the Expression ResNetTCN and Head Movement TCN end-to-end:

1. Environment setup
2. Kaggle dataset download (AffectNet + ISL-CSLTR)
3. Dataset exploration and visualisation
4. Frame extraction from ISL-CSLTR videos
5. MediaPipe landmark extraction (face, pose, head pose)
6. Head movement sequence construction (sliding windows + synthetic augmentation)
7. Dataset classes and DataLoaders
8. Model architectures (ExpressionResNetTCN + HeadMovementTCN)
9. Training utilities
10. Train Expression ResNetTCN on AffectNet (backbone frozen → unfrozen at epoch 6)
11. Fine-tune on ISL face crops (confidence-filtered pseudo-labels)
12. Train Head Movement TCN
13. Evaluation and confusion matrices
14. Ablation study
15. Real-time inference pipeline demo
16. Test on ISL-CSLTR frame samples
17. ONNX export

### `isl_pyt_200_tcn.ipynb` — Sentence Classification

Trains the ISL Sentence TCN end-to-end:

1. Dataset loading from `ISL_CSLRT_Corpus/Videos_Sentence_Level`
2. MediaPipe keypoint extraction with disk caching
3. Class imbalance audit (thin classes removed by default)
4. Label encoding and train/test split
5. PyTorch DataLoaders
6. TCN model definition (4-block causal dilated TCN)
7. Training loop with early stopping and ReduceLROnPlateau
8. Model checkpoint saving (`isl_best_model.pt`, `label_encoder.pkl`)
9. Evaluation and training history plots

---

## Datasets

| Dataset | Purpose | Kaggle Slug |
|---------|---------|-------------|
| AffectNet | Expression pretraining | `mstjebashazida/affectnet` |
| ISL-CSLTR | Head movement + sentence training | `drblack00/isl-csltr-indian-sign-language-dataset` |

To download via Kaggle API, place `kaggle.json` in `~/.kaggle/` and run the download cells in `isl_nmf_expr_hm.ipynb` Section 2.

---

## Configuration

Key constants at the top of each inference script:

| Constant | Default | Description |
|----------|---------|-------------|
| `SENT_SEQ_LEN` | `30` | Keyframe windows per sentence prediction |
| `SENT_NUM_KP` | `225` | Body keypoint vector length per frame |
| `SENT_FRAME_SKIP` | `5` | Sample every Nth frame for the sentence model |
| `NMF_SEQ_LEN` | `30` | Frames per head movement window |

---

## Requirements

- Python 3.10+
- PyTorch 2.x + torchvision
- MediaPipe ≥ 0.10
- OpenCV
- Albumentations
- NumPy

GPU (CUDA) is detected automatically; all scripts fall back to CPU if unavailable.
`face_landmarker.task` is downloaded automatically on first run if not present.
