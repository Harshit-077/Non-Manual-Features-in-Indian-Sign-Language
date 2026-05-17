"""
ISL NMF Inference — Expression + Head Movement
================================================
Two-stream NMF detection for Indian Sign Language:
  Stream 1: ExpressionResNetTCN  — 8 facial expressions (AffectNet classes)
  Stream 2: HeadMovementTCN      — 5 head movement classes (nod/shake/tilt/forward/still)

Usage:
    # Webcam
    python isl_nmf_inference.py --source webcam

    # Video file
    python isl_nmf_inference.py --source video --path "path/to/video.mp4"

Required files (same folder as this script):
    isl_nmf/checkpoints/expr_best.pt      ← saved by training notebook
    isl_nmf/checkpoints/hm_best.pt        ← saved by training notebook
    face_landmarker.task                   ← MediaPipe model
"""

import argparse
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
# CONFIG  (must match training notebook)
# ============================================================

SEQ_LEN    = 30
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

EXPR_CKPT = Path("isl_nmf/checkpoints/expr_best.pt")
HM_CKPT   = Path("isl_nmf/checkpoints/hm_best.pt")

EXPR_NAMES = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]
HM_NAMES   = ["nod", "shake", "tilt", "forward", "still"]

GRAMMAR = {
    "neutral":  "Statement / no NMF",
    "happy":    "Affirmative marker",
    "surprise": "Wh-question (who/what/where)",
    "fear":     "Wh-question (secondary)",
    "sad":      "Negation / emotional tone",
    "disgust":  "Negation",
    "anger":    "Negation / intensity",
    "contempt": "Negation",
    "nod":      "Yes / affirmative / sentence end",
    "shake":    "No / negation",
    "tilt":     "Conditional / topic marker",
    "forward":  "Emphasis / assertion",
    "still":    "Neutral / no head NMF",
}

# ============================================================
# IMAGE TRANSFORM  (matches val_tf in notebook)
# ============================================================

val_tf = A.Compose([
    A.Resize(112, 112),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ============================================================
# MODEL DEFINITIONS  (identical to notebook)
# ============================================================

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
        rn = tvm.resnet18(weights=None)   # weights loaded from checkpoint
        bb_dim = rn.fc.in_features        # 512
        rn.fc  = nn.Identity()
        self.backbone = rn

        self.proj = nn.Sequential(
            nn.Linear(bb_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.3),
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
            nn.Linear(in_ch, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(128, num_classes)

    def _encode_frames(self, x):
        return self.proj(self.backbone(x))

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)                          # (B,1,C,H,W)
        B, T, C, H, W = x.shape
        feats   = self._encode_frames(x.reshape(B * T, C, H, W)).view(B, T, -1)
        tcn_out = self.tcn(feats.permute(0, 2, 1))
        embed   = self.pool(tcn_out).squeeze(-1)
        embed   = self.head(embed)
        return self.classifier(embed), embed


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

    def forward(self, x):                  # x: (B, T, 3)
        x = x.transpose(1, 2)              # (B, 3, T)
        f = self.pool(self.tcn(x)).squeeze(-1)
        e = self.head(f)
        return self.classifier(e), e


# ============================================================
# MODEL LOADING
# ============================================================

def load_models(expr_ckpt: Path, hm_ckpt: Path):
    # Expression
    model_expr = ExpressionResNetTCN(
        num_classes=len(EXPR_NAMES), feat_dim=256, tcn_channels=(256, 256, 128)
    ).to(DEVICE)

    if expr_ckpt.exists():
        ck = torch.load(expr_ckpt, map_location=DEVICE, weights_only=True)
        sd = ck["model"] if "model" in ck else ck
        model_expr.load_state_dict(sd)
        val_acc = ck.get("val_acc", "?")
        print(f"✓ Expression model loaded  (val_acc={val_acc})")
    else:
        print(f"  ⚠ No checkpoint at {expr_ckpt} — running with random weights.")

    # Head movement
    model_hm = HeadMovementTCN(
        num_classes=len(HM_NAMES), channels=(32, 64), feat_dim=64
    ).to(DEVICE)

    if hm_ckpt.exists():
        ck = torch.load(hm_ckpt, map_location=DEVICE, weights_only=True)
        sd = ck["model"] if "model" in ck else ck
        model_hm.load_state_dict(sd)
        val_acc = ck.get("val_acc", "?")
        print(f"✓ Head movement model loaded  (val_acc={val_acc})")
    else:
        print(f"  ⚠ No checkpoint at {hm_ckpt} — running with random weights.")

    model_expr.eval()
    model_hm.eval()
    return model_expr, model_hm


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

def ensure_face_model():
    path = Path(FACE_MODEL_PATH)
    if not path.exists():
        print(f"Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(FACE_MODEL_URL, path)
        print(f"✓ Downloaded: {path}")
    return path


def build_face_landmarker(model_path: Path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

    opts = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(opts)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_face_features(frame_bgr, face_landmarker, frame_ts_ms: int):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_img, frame_ts_ms)

    if not result.face_landmarks:
        return None

    pts = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)

    # Head pose from key landmarks
    chin, forehead   = pts[199], pts[10]
    l_ear, r_ear     = pts[234], pts[454]
    pitch = float(np.degrees(np.arctan2(
        chin[1] - forehead[1], abs(chin[2] - forehead[2]) + 1e-6)))
    yaw   = float(np.degrees(np.arctan2(
        r_ear[0] - l_ear[0], abs(r_ear[2] - l_ear[2]) + 1e-6)))
    roll  = float(np.degrees(np.arctan2(
        r_ear[1] - l_ear[1], abs(r_ear[0] - l_ear[0]) + 1e-6)))

    # Face bounding box with padding
    h, w = frame_bgr.shape[:2]
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
def predict_expression(model, face_crop_bgr):
    rgb   = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img_t = val_tf(image=rgb)["image"].unsqueeze(0).to(DEVICE)   # (1,C,H,W)
    probs = F.softmax(model(img_t)[0], dim=1)[0]
    idx   = probs.argmax().item()
    return EXPR_NAMES[idx], probs[idx].item()


@torch.no_grad()
def predict_head_movement(model, pose_buffer):
    seq = np.stack(pose_buffer)                                   # (T, 3)
    seq = (seq - seq.mean(0)) / (seq.std(0) + 1e-6)
    t   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs = F.softmax(model(t)[0], dim=1)[0]
    idx   = probs.argmax().item()
    return HM_NAMES[idx], probs[idx].item()


# ============================================================
# OVERLAY DRAWING
# ============================================================

def draw_overlay(frame, expr_result, hm_result, pose_buf_len):
    out = frame.copy()
    h, w = out.shape[:2]

    # Expression
    if expr_result:
        label, conf = expr_result
        grammar     = GRAMMAR.get(label, "")
        text        = f"Expr: {label}  ({conf*100:.1f}%)"
        cv2.rectangle(out, (8, 8), (8 + 320, 36), (0, 0, 0), -1)
        cv2.putText(out, text, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 80), 2)

    # Head movement
    if hm_result:
        label, conf = hm_result
        grammar     = GRAMMAR.get(label, "")
        text        = f"Head: {label}  ({conf*100:.1f}%)"
        cv2.rectangle(out, (8, 42), (8 + 320, 70), (0, 0, 0), -1)
        cv2.putText(out, text, (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 140, 255), 2)

    # Grammar interpretation
    grammars = []
    if expr_result:  grammars.append(GRAMMAR.get(expr_result[0], ""))
    if hm_result:    grammars.append(GRAMMAR.get(hm_result[0], ""))
    grammars = list(dict.fromkeys(g for g in grammars if g))
    if grammars:
        isl_text = "ISL: " + " | ".join(grammars[:2])
        cv2.putText(out, isl_text, (12, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Head movement buffer progress bar
    bar_w  = int(w * 0.55)
    bar_h  = 14
    bar_x  = 12
    bar_y  = h - 45
    filled = int(bar_w * min(pose_buf_len, SEQ_LEN) / SEQ_LEN)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (80, 140, 255), -1)
    cv2.putText(out, f"HM buffer {pose_buf_len}/{SEQ_LEN}",
                (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Controls
    cv2.putText(out, "Q: quit   R: reset HM buffer",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    return out


# ============================================================
# MAIN LOOP
# ============================================================

def run(source: str, video_path: str, model_expr, model_hm):
    face_model_path = ensure_face_model()
    face_landmarker = build_face_landmarker(face_model_path)
    print("✓ MediaPipe face landmarker ready")

    if source == "webcam":
        cap = cv2.VideoCapture(0)
        print("✓ Webcam opened — Q to quit, R to reset head movement buffer")
    else:
        path = Path(video_path)
        if not path.exists():
            sys.exit(f"[ERROR] File not found: {video_path}")
        cap = cv2.VideoCapture(str(path))
        print(f"✓ Video opened: {path.name}")

    if not cap.isOpened():
        sys.exit("[ERROR] Could not open video source.")

    pose_buf    = deque(maxlen=SEQ_LEN)
    frame_ts_ms = 0
    expr_result = None
    hm_result   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            if source == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                pose_buf.clear()
                # NOTE: frame_ts_ms is NOT reset — MediaPipe requires monotonically
                # increasing timestamps across the entire session, even across loops.
                continue
            break

        feat = extract_face_features(frame, face_landmarker, frame_ts_ms)
        frame_ts_ms += 33   # ~30 fps

        if feat is not None:
            # Expression — every frame (fast, single-frame TCN)
            x1, y1, x2, y2 = feat["face_bbox"]
            if x2 > x1 and y2 > y1:
                crop        = frame[y1:y2, x1:x2]
                expr_result = predict_expression(model_expr, crop)
                label, conf = expr_result
                print(f"  expr: {label} ({conf*100:.1f}%)", end="")

            # Head movement — once buffer is full
            pose_buf.append(feat["head_pose"])
            if len(pose_buf) == SEQ_LEN:
                hm_result   = predict_head_movement(model_hm, list(pose_buf))
                label, conf = hm_result
                print(f"  |  head: {label} ({conf*100:.1f}%)", end="")

            print()

        display = draw_overlay(frame, expr_result, hm_result, len(pose_buf))
        cv2.imshow("ISL NMF Inference", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            pose_buf.clear()
            hm_result = None
            print("  Head movement buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL NMF Inference")
    parser.add_argument("--source", choices=["webcam", "video"], required=True)
    parser.add_argument("--path",   type=str, default=None,
                        help="Path to video file (required when --source video)")
    parser.add_argument("--expr-ckpt", type=str, default=str(EXPR_CKPT),
                        help=f"Expression checkpoint path (default: {EXPR_CKPT})")
    parser.add_argument("--hm-ckpt",   type=str, default=str(HM_CKPT),
                        help=f"Head movement checkpoint path (default: {HM_CKPT})")
    args = parser.parse_args()

    if args.source == "video" and args.path is None:
        parser.error("--path is required when --source is video")

    print(f"Device: {DEVICE}")
    model_expr, model_hm = load_models(Path(args.expr_ckpt), Path(args.hm_ckpt))
    run(args.source, args.path, model_expr, model_hm)
