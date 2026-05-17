"""
ISL Combined Inference — Sentence + Expression + Head Movement
==============================================================
Three models running simultaneously on every frame:

  Stream 1 — ISL Sentence Classifier (TCN over hand+pose keypoints)
             Predicts which ISL sentence is being signed.
             Sampled every FRAME_SKIP=5 frames; predicts once buffer fills (30 frames).

  Stream 2 — Expression (ResNet-18 + TCN)
             Detects facial expression from face crop every frame.

  Stream 3 — Head Movement (CausalTCN over pitch/yaw/roll)
             Detects head movement type from a rolling 30-frame pose buffer.

Usage:
    # Webcam
    python isl_combined_inference.py --source webcam

    # Video file
    python isl_combined_inference.py --source video --path "path/to/video.mp4"

Required files (same folder as this script unless overridden):
    isl_best_model.pt                    ← sentence TCN checkpoint
    label_encoder.pkl                    ← sentence label encoder
    isl_nmf/checkpoints/expr_best.pt     ← expression checkpoint
    isl_nmf/checkpoints/hm_best.pt       ← head movement checkpoint
    hand_landmarker.task                 ← MediaPipe hand model
    pose_landmarker_full.task            ← MediaPipe pose model
    face_landmarker.task                 ← MediaPipe face model (auto-downloaded)

Controls (OpenCV window):
    Q  — quit
    R  — reset all buffers
"""

import argparse
import pickle
import sys
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================
# SHARED CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sentence model ────────────────────────────────────────────────────────────
SENT_SEQ_LEN     = 30
SENT_NUM_KP      = 225        # pose(33*3) + left(21*3) + right(21*3)
SENT_FRAME_SKIP  = 5
SENT_MODEL_PATH  = "isl_best_model.pt"
SENT_ENCODER_PATH = "label_encoder.pkl"
HAND_MODEL       = "hand_landmarker.task"
POSE_MODEL       = "pose_landmarker_full.task"

# ── NMF models ────────────────────────────────────────────────────────────────
NMF_SEQ_LEN   = 30
EXPR_CKPT     = Path("isl_nmf/checkpoints/expr_best.pt")
HM_CKPT       = Path("isl_nmf/checkpoints/hm_best.pt")
FACE_MODEL    = "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

EXPR_NAMES = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]
HM_NAMES   = ["nod", "shake", "tilt", "forward", "still"]

GRAMMAR = {
    "neutral":  "Statement",
    "happy":    "Affirmative",
    "surprise": "Wh-question",
    "fear":     "Wh-question",
    "sad":      "Negation",
    "disgust":  "Negation",
    "anger":    "Negation",
    "contempt": "Negation",
    "nod":      "Affirm / end",
    "shake":    "Negation",
    "tilt":     "Conditional",
    "forward":  "Emphasis",
    "still":    "No head NMF",
}

# Image transform for expression model (matches val_tf in notebook)
val_tf = A.Compose([
    A.Resize(112, 112),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ============================================================
# MODEL DEFINITIONS
# ============================================================

# ── Stream 1: ISL Sentence TCN ───────────────────────────────────────────────

class SentTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1    = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  padding=padding, dilation=dilation)
        self.conv2    = nn.Conv1d(out_channels, out_channels, kernel_size,
                                  padding=padding, dilation=dilation)
        self.relu1    = nn.ReLU()
        self.relu2    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.dropout1(self.relu1(self.conv1(x)[:, :, :x.size(2)]))
        out = self.dropout2(self.relu2(self.conv2(out)[:, :, :x.size(2)]))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class ISLSentenceModel(nn.Module):
    def __init__(self, input_size, num_classes, num_channels=None,
                 kernel_size=3, dropout=0.2, fc_dropout1=0.4, fc_dropout2=0.3):
        super().__init__()
        if num_channels is None:
            num_channels = [128, 128, 256, 256]
        layers, in_ch = [], input_size
        for i, out_ch in enumerate(num_channels):
            layers.append(SentTemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
            in_ch = out_ch
        self.tcn   = nn.Sequential(*layers)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.bn    = nn.BatchNorm1d(in_ch)
        self.fc1   = nn.Linear(in_ch, 256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(fc_dropout1)
        self.fc2   = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(fc_dropout2)
        self.out   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = self.bn(x)
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.out(x)


# ── Stream 2: Expression ResNet-18 + TCN ─────────────────────────────────────

class ExprTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, drop=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.drop  = nn.Dropout(drop)
        self.act   = nn.GELU()
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        T = x.size(-1)
        o = self.act(self.norm1(self.conv1(x)[..., :T]))
        o = self.drop(self.act(self.norm2(self.conv2(o)[..., :T])))
        return o + self.res(x)


class ExpressionResNetTCN(nn.Module):
    def __init__(self, num_classes=8, feat_dim=256,
                 tcn_channels=(256, 256, 128), tcn_kernel=3):
        super().__init__()
        rn = tvm.resnet18(weights=None)
        bb_dim = rn.fc.in_features
        rn.fc  = nn.Identity()
        self.backbone = rn
        self.proj = nn.Sequential(
            nn.Linear(bb_dim, feat_dim), nn.LayerNorm(feat_dim),
            nn.GELU(), nn.Dropout(0.3),
        )
        tcn_layers, in_ch = [], feat_dim
        for i, out_ch in enumerate(tcn_channels):
            tcn_layers.append(ExprTemporalBlock(in_ch, out_ch,
                                                kernel=tcn_kernel,
                                                dilation=2**i, drop=0.2))
            in_ch = out_ch
        self.tcn  = nn.Sequential(*tcn_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(128, num_classes)

    def _encode_frames(self, x):
        return self.proj(self.backbone(x))

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        feats   = self._encode_frames(x.reshape(B * T, C, H, W)).view(B, T, -1)
        tcn_out = self.tcn(feats.permute(0, 2, 1))
        embed   = self.pool(tcn_out).squeeze(-1)
        return self.classifier(self.head(embed)), embed


# ── Stream 3: Head Movement CausalTCN ────────────────────────────────────────

class CausalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, drop=0.3):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.drop  = nn.Dropout(drop)
        self.act   = nn.GELU()
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        T = x.size(-1)
        o = self.act(self.norm1(self.conv1(x)[..., :T]))
        o = self.drop(self.act(self.norm2(self.conv2(o)[..., :T])))
        return o + self.res(x)


class HeadMovementTCN(nn.Module):
    def __init__(self, in_ch=3, num_classes=5, channels=(32, 64), feat_dim=64):
        super().__init__()
        blocks, prev = [], in_ch
        for i, ch in enumerate(channels):
            blocks.append(CausalBlock(prev, ch, dilation=2**i))
            prev = ch
        self.tcn  = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(prev, feat_dim), nn.LayerNorm(feat_dim),
            nn.GELU(), nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        f = self.pool(self.tcn(x)).squeeze(-1)
        return self.classifier(self.head(f)), f


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def _merge_weight_norm(sd):
    bases = {k[:-2] for k in sd if k.endswith("_g")}
    new_sd = {}
    for base in bases:
        g = sd[base + "_g"]
        v = sd[base + "_v"]
        norm = v.view(v.size(0), -1).norm(dim=1)
        norm = norm.view(v.size(0), *([1] * (v.dim() - 1)))
        new_sd[base] = g * (v / norm)
    for k, val in sd.items():
        if not (k.endswith("_g") or k.endswith("_v")):
            new_sd[k] = val
    return new_sd


def _load_sd(path):
    """Load state dict, unwrap 'model' key if present, fix weight_norm."""
    ck = torch.load(path, map_location=DEVICE, weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if any(k.endswith("_g") for k in sd):
        print(f"    weight_norm keys detected — merging...")
        sd = _merge_weight_norm(sd)
    val_acc = ck.get("val_acc", "?") if isinstance(ck, dict) else "?"
    return sd, val_acc


def load_all_models(sent_model_path, sent_encoder_path,
                    expr_ckpt, hm_ckpt):
    # ── Sentence model ────────────────────────────────────────────────────────
    with open(sent_encoder_path, "rb") as f:
        encoder = pickle.load(f)

    sent_sd, sent_acc = _load_sd(sent_model_path)
    if any(k.endswith("_g") for k in sent_sd):
        sent_sd = _merge_weight_norm(sent_sd)
    num_sent_classes = sent_sd["out.weight"].shape[0]

    sent_model = ISLSentenceModel(
        input_size=SENT_NUM_KP, num_classes=num_sent_classes
    ).to(DEVICE)
    sent_model.load_state_dict(sent_sd)
    sent_model.eval()
    print(f"✓ Sentence model  ({num_sent_classes} classes, val_acc={sent_acc})")

    # ── Expression model ──────────────────────────────────────────────────────
    expr_model = ExpressionResNetTCN(
        num_classes=len(EXPR_NAMES), feat_dim=256, tcn_channels=(256, 256, 128)
    ).to(DEVICE)
    if Path(expr_ckpt).exists():
        sd, acc = _load_sd(expr_ckpt)
        expr_model.load_state_dict(sd)
        print(f"✓ Expression model  ({len(EXPR_NAMES)} classes, val_acc={acc})")
    else:
        print(f"  ⚠ No checkpoint at {expr_ckpt} — random weights.")
    expr_model.eval()

    # ── Head movement model ───────────────────────────────────────────────────
    hm_model = HeadMovementTCN(
        num_classes=len(HM_NAMES), channels=(32, 64), feat_dim=64
    ).to(DEVICE)
    if Path(hm_ckpt).exists():
        sd, acc = _load_sd(hm_ckpt)
        hm_model.load_state_dict(sd)
        print(f"✓ Head movement model  ({len(HM_NAMES)} classes, val_acc={acc})")
    else:
        print(f"  ⚠ No checkpoint at {hm_ckpt} — random weights.")
    hm_model.eval()

    return sent_model, encoder, expr_model, hm_model


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

def ensure_face_model():
    path = Path(FACE_MODEL)
    if not path.exists():
        print(f"Downloading {FACE_MODEL}...")
        urllib.request.urlretrieve(FACE_MODEL_URL, path)
        print(f"✓ Downloaded {FACE_MODEL}")
    return path


def build_all_detectors():
    BaseOptions       = mp.tasks.BaseOptions
    ImageMode         = mp.tasks.vision.RunningMode.IMAGE
    VideoMode         = mp.tasks.vision.RunningMode.VIDEO

    # Hand landmarker (IMAGE mode — no timestamp needed)
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    hand_det = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=ImageMode, num_hands=2,
        )
    )

    # Pose landmarker (IMAGE mode)
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    pose_det = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=ImageMode,
        )
    )

    # Face landmarker (VIDEO mode — needs monotonic timestamps)
    face_model_path = ensure_face_model()
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    face_det = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_model_path)),
            running_mode=VideoMode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )

    print("✓ MediaPipe detectors ready (hand / pose / face)")
    return hand_det, pose_det, face_det


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_keypoints(frame, hand_det, pose_det):
    """Hand + pose keypoints for sentence model."""
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_det.detect(mp_image)
    pose = np.zeros(33 * 3, dtype=np.float32)
    if pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in pose_result.pose_landmarks[0]]).flatten()

    hand_result = hand_det.detect(mp_image)
    left_hand  = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)
    if hand_result.hand_landmarks:
        for idx, lms in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
            if handedness == "Left":
                left_hand = arr
            else:
                right_hand = arr

    return np.concatenate([pose, left_hand, right_hand]).astype(np.float32)


def extract_face_features(frame, face_det, frame_ts_ms):
    """Head pose + face bbox for NMF streams."""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_det.detect_for_video(mp_img, frame_ts_ms)

    if not result.face_landmarks:
        return None

    pts = np.array([[p.x, p.y, p.z]
                    for p in result.face_landmarks[0]], dtype=np.float32)
    chin, forehead = pts[199], pts[10]
    l_ear, r_ear   = pts[234], pts[454]

    pitch = float(np.degrees(np.arctan2(
        chin[1] - forehead[1], abs(chin[2] - forehead[2]) + 1e-6)))
    yaw   = float(np.degrees(np.arctan2(
        r_ear[0] - l_ear[0], abs(r_ear[2] - l_ear[2]) + 1e-6)))
    roll  = float(np.degrees(np.arctan2(
        r_ear[1] - l_ear[1], abs(r_ear[0] - l_ear[0]) + 1e-6)))

    h, w = frame.shape[:2]
    x1 = max(0, int(pts[:, 0].min() * w) - 15)
    y1 = max(0, int(pts[:, 1].min() * h) - 15)
    x2 = min(w, int(pts[:, 0].max() * w) + 15)
    y2 = min(h, int(pts[:, 1].max() * h) + 15)

    return {
        "head_pose": np.array([pitch, yaw, roll], dtype=np.float32),
        "face_bbox": (x1, y1, x2, y2),
    }


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_sentence(buffer, sent_model, encoder):
    seq = np.array(list(buffer), dtype=np.float32)
    x   = torch.tensor(seq).unsqueeze(0).to(DEVICE)
    logits = sent_model(x)
    probs  = torch.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    label = encoder.inverse_transform([idx.item()])[0]
    return label, conf.item()


@torch.no_grad()
def predict_expression(face_crop_bgr, expr_model):
    rgb   = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img_t = val_tf(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    probs = F.softmax(expr_model(img_t)[0], dim=1)[0]
    idx   = probs.argmax().item()
    return EXPR_NAMES[idx], probs[idx].item()


@torch.no_grad()
def predict_head_movement(pose_buffer, hm_model):
    seq   = np.stack(list(pose_buffer))
    seq   = (seq - seq.mean(0)) / (seq.std(0) + 1e-6)
    t     = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs = F.softmax(hm_model(t)[0], dim=1)[0]
    idx   = probs.argmax().item()
    return HM_NAMES[idx], probs[idx].item()


# ============================================================
# OVERLAY DRAWING
# ============================================================

def draw_overlay(frame, sent_result, expr_result, hm_result,
                 sent_buf_len, hm_buf_len):
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Panel background (top-left) ───────────────────────────────────────────
    panel_h = 115
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # ── Sentence prediction ───────────────────────────────────────────────────
    if sent_result:
        label, conf = sent_result
        text = f"SIGN: {label}  ({conf*100:.1f}%)"
        cv2.putText(out, text, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 128), 2)
    else:
        cv2.putText(out, "SIGN: collecting...", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)

    # ── Expression ────────────────────────────────────────────────────────────
    if expr_result:
        label, conf = expr_result
        text = f"EXPR: {label}  ({conf*100:.1f}%)  [{GRAMMAR.get(label, '')}]"
        cv2.putText(out, text, (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    else:
        cv2.putText(out, "EXPR: no face", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    # ── Head movement ─────────────────────────────────────────────────────────
    if hm_result:
        label, conf = hm_result
        text = f"HEAD: {label}  ({conf*100:.1f}%)  [{GRAMMAR.get(label, '')}]"
        cv2.putText(out, text, (12, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 50), 2)
    else:
        cv2.putText(out, "HEAD: buffering...", (12, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    # ── Buffer progress bars (bottom) ─────────────────────────────────────────
    bar_y_base = h - 55
    bar_w      = int(w * 0.45)
    bar_h      = 12

    # Sentence buffer (green)
    s_filled = int(bar_w * min(sent_buf_len, SENT_SEQ_LEN) / SENT_SEQ_LEN)
    cv2.rectangle(out, (12, bar_y_base), (12 + bar_w, bar_y_base + bar_h), (40, 40, 40), -1)
    cv2.rectangle(out, (12, bar_y_base), (12 + s_filled, bar_y_base + bar_h), (0, 200, 100), -1)
    cv2.putText(out, f"Sign buf {sent_buf_len}/{SENT_SEQ_LEN}",
                (12, bar_y_base - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # Head movement buffer (orange) — right side
    hm_x = 12 + bar_w + 20
    hm_filled = int(bar_w * min(hm_buf_len, NMF_SEQ_LEN) / NMF_SEQ_LEN)
    cv2.rectangle(out, (hm_x, bar_y_base), (hm_x + bar_w, bar_y_base + bar_h), (40, 40, 40), -1)
    cv2.rectangle(out, (hm_x, bar_y_base), (hm_x + hm_filled, bar_y_base + bar_h), (50, 140, 255), -1)
    cv2.putText(out, f"Head buf {hm_buf_len}/{NMF_SEQ_LEN}",
                (hm_x, bar_y_base - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # ── Controls ──────────────────────────────────────────────────────────────
    cv2.putText(out, "Q: quit   R: reset buffers",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    return out


# ============================================================
# MAIN LOOP
# ============================================================

def run(source, video_path,
        sent_model, encoder,
        expr_model, hm_model,
        hand_det, pose_det, face_det):

    if source == "webcam":
        cap = cv2.VideoCapture(0)
        print("✓ Webcam opened — Q to quit, R to reset")
    else:
        p = Path(video_path)
        if not p.exists():
            sys.exit(f"[ERROR] File not found: {video_path}")
        cap = cv2.VideoCapture(str(p))
        print(f"✓ Video opened: {p.name}")

    if not cap.isOpened():
        sys.exit("[ERROR] Could not open video source.")

    # Buffers
    sent_buf        = deque(maxlen=SENT_SEQ_LEN)
    pose_buf        = deque(maxlen=NMF_SEQ_LEN)
    frame_idx       = 0
    frame_ts        = 0       # monotonically increasing ms timestamp for face landmarker
    sent_kp_cache   = []      # all keypoints extracted from the video so far
    sent_loop_idx   = 0       # index into cache for looped fill

    # Last predictions
    sent_result = None
    expr_result = None
    hm_result   = None

    print("Running — press Q to quit, R to reset all buffers\n")

    while True:
        ret, frame = cap.read()
        video_ended = not ret

        if video_ended:
            if source == "video":
                # Video ran out of frames.
                # If we haven't filled the sentence buffer yet, loop the cached
                # keyframes round-robin until it reaches SEQ_LEN, then predict.
                if sent_kp_cache and len(sent_buf) < SENT_SEQ_LEN:
                    while len(sent_buf) < SENT_SEQ_LEN:
                        sent_buf.append(sent_kp_cache[sent_loop_idx % len(sent_kp_cache)])
                        sent_loop_idx += 1
                    sent_result = predict_sentence(sent_buf, sent_model, encoder)
                    lbl, conf = sent_result
                    print(f"  SIGN  → {lbl} ({conf*100:.1f}%)  [buffer filled by looping]")

                # Reset for next play-through
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                sent_buf.clear()
                pose_buf.clear()
                frame_idx     = 0
                sent_loop_idx = 0
                # frame_ts NOT reset — MediaPipe VIDEO mode requires monotonic ts
                continue
            else:
                break

        frame_idx += 1

        # ── Stream 1: sentence keypoints (every FRAME_SKIP frames) ───────────
        if frame_idx % SENT_FRAME_SKIP == 0:
            kp = extract_keypoints(frame, hand_det, pose_det)
            sent_kp_cache.append(kp)   # always cache every extracted keypoint
            sent_buf.append(kp)
            if len(sent_buf) == SENT_SEQ_LEN:
                sent_result = predict_sentence(sent_buf, sent_model, encoder)
                lbl, conf = sent_result
                print(f"  SIGN  → {lbl} ({conf*100:.1f}%)")

        # ── Streams 2 & 3: face features (every frame) ───────────────────────
        face_feat = extract_face_features(frame, face_det, frame_ts)
        frame_ts += 33   # always increment, never reset

        if face_feat is not None:
            # Expression — single frame prediction
            x1, y1, x2, y2 = face_feat["face_bbox"]
            if x2 > x1 and y2 > y1:
                crop        = frame[y1:y2, x1:x2]
                expr_result = predict_expression(crop, expr_model)

            # Head movement — rolling buffer prediction
            pose_buf.append(face_feat["head_pose"])
            if len(pose_buf) == NMF_SEQ_LEN:
                hm_result = predict_head_movement(pose_buf, hm_model)
                lbl, conf = hm_result
                expr_lbl  = expr_result[0] if expr_result else "-"
                print(f"  EXPR  → {expr_lbl}  |  HEAD → {lbl} ({conf*100:.1f}%)")

        # ── Draw & display ────────────────────────────────────────────────────
        display = draw_overlay(
            frame, sent_result, expr_result, hm_result,
            len(sent_buf), len(pose_buf)
        )
        cv2.imshow("ISL Combined Inference", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            sent_buf.clear()
            pose_buf.clear()
            sent_kp_cache.clear()
            frame_idx     = 0
            sent_loop_idx = 0
            sent_result   = None
            expr_result   = None
            hm_result     = None
            print("  All buffers reset.")

    cap.release()
    cv2.destroyAllWindows()
    face_det.close()
    print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Combined Inference")
    parser.add_argument("--source", choices=["webcam", "video"], required=True)
    parser.add_argument("--path",   type=str, default=None,
                        help="Path to video file (required when --source video)")
    parser.add_argument("--sent-model",   default=SENT_MODEL_PATH)
    parser.add_argument("--sent-encoder", default=SENT_ENCODER_PATH)
    parser.add_argument("--expr-ckpt",    default=str(EXPR_CKPT))
    parser.add_argument("--hm-ckpt",      default=str(HM_CKPT))
    args = parser.parse_args()

    if args.source == "video" and args.path is None:
        parser.error("--path is required when --source is video")

    print(f"Device: {DEVICE}\n")

    # Load models
    sent_model, encoder, expr_model, hm_model = load_all_models(
        args.sent_model, args.sent_encoder,
        args.expr_ckpt,  args.hm_ckpt,
    )

    # Load MediaPipe detectors
    hand_det, pose_det, face_det = build_all_detectors()

    # Run
    run(
        args.source, args.path,
        sent_model, encoder,
        expr_model, hm_model,
        hand_det, pose_det, face_det,
    )
