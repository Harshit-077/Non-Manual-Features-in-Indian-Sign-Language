"""
ISL Sentence-Level Inference
==============================
Usage:
    # Webcam
    python isl_inference.py --source webcam

    # Video file
    python isl_inference.py --source video --path "path/to/your/video.mp4"

Requirements:
    - isl_best_model.pt        (saved by training notebook)
    - label_encoder.pkl        (saved by training notebook)
    - hand_landmarker.task     (MediaPipe model file)
    - pose_landmarker_full.task(MediaPipe model file)
"""

import argparse
import pickle
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# CONFIG  (must match training notebook)
# ============================================================

SEQ_LEN      = 30
NUM_KEYPOINTS = 225   # pose(33*3) + left(21*3) + right(21*3)
FRAME_SKIP   = 5

HAND_MODEL   = "hand_landmarker.task"
POSE_MODEL   = "pose_landmarker_full.task"
MODEL_PATH   = "isl_best_model.pt"
ENCODER_PATH = "label_encoder.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL DEFINITION  (must match training notebook exactly)
# ============================================================

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
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
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class ISLModel(nn.Module):
    def __init__(self, input_size, num_classes, num_channels=None,
                 kernel_size=3, dropout=0.2, fc_dropout1=0.4, fc_dropout2=0.3):
        super().__init__()

        if num_channels is None:
            num_channels = [128, 128, 256, 256]

        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.gap  = nn.AdaptiveAvgPool1d(1)

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


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def _merge_weight_norm(sd):
    """
    Checkpoints saved with nn.utils.weight_norm store weight_g + weight_v
    instead of weight. Merge them back into a single weight tensor so the
    state dict can be loaded into a plain Conv1d model.
    """
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


def load_model(encoder_path, model_path):
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)

    # Handle checkpoints saved with weight_norm
    if any(k.endswith("_g") for k in ckpt):
        print("  warning: weight_norm keys detected - merging into plain weights...")
        ckpt = _merge_weight_norm(ckpt)

    # Infer num_classes from checkpoint (avoids encoder mismatch)
    num_classes = ckpt["out.weight"].shape[0]
    if num_classes != len(encoder.classes_):
        print(f"  warning: Checkpoint has {num_classes} classes, "
              f"encoder has {len(encoder.classes_)} - using checkpoint value.")

    model = ISLModel(input_size=NUM_KEYPOINTS, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded ({num_classes} classes, device={DEVICE})")
    return model, encoder


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

def build_detectors():
    BaseOptions         = mp.tasks.BaseOptions
    VisionRunningMode   = mp.tasks.vision.RunningMode
    HandLandmarker      = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    PoseLandmarker      = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )
    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=VisionRunningMode.IMAGE,
    )
    hand_det = HandLandmarker.create_from_options(hand_opts)
    pose_det = PoseLandmarker.create_from_options(pose_opts)
    print("✓ MediaPipe detectors ready")
    return hand_det, pose_det


# ============================================================
# KEYPOINT EXTRACTION  (identical to training)
# ============================================================

def extract_keypoints(frame, hand_det, pose_det):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_det.detect(mp_image)
    pose = np.zeros(33 * 3)
    if pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in pose_result.pose_landmarks[0]]).flatten()

    hand_result = hand_det.detect(mp_image)
    left_hand  = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)
    if hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            arr = np.array([[lm.x, lm.y, lm.z]
                            for lm in hand_landmarks]).flatten()
            if handedness == "Left":
                left_hand = arr
            else:
                right_hand = arr

    return np.concatenate([pose, left_hand, right_hand]).astype(np.float32)


# ============================================================
# INFERENCE ON A FIXED-LENGTH BUFFER
# ============================================================

def predict(buffer, model, encoder):
    """buffer: list of SEQ_LEN keypoint arrays → (label, confidence)"""
    seq = np.array(buffer, dtype=np.float32)  # (SEQ_LEN, NUM_KEYPOINTS)

    # Pad or trim just in case
    if len(seq) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(seq), NUM_KEYPOINTS), dtype=np.float32)
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:SEQ_LEN]

    x = torch.tensor(seq).unsqueeze(0).to(DEVICE)   # (1, SEQ_LEN, NUM_KEYPOINTS)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    label = encoder.inverse_transform([idx.item()])[0]
    return label, conf.item()


# ============================================================
# DRAW OVERLAY
# ============================================================

def draw_overlay(frame, label, confidence, buffer_len, collecting):
    h, w = frame.shape[:2]

    # Progress bar background
    bar_w = int(w * 0.6)
    bar_h = 18
    bar_x, bar_y = 20, h - 50
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    filled = int(bar_w * min(buffer_len, SEQ_LEN) / SEQ_LEN)
    color  = (0, 200, 100) if collecting else (100, 100, 200)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  color, -1)
    cv2.putText(frame, f"Buffer {buffer_len}/{SEQ_LEN}",
                (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Prediction label
    if label:
        text  = f"{label}  ({confidence*100:.1f}%)"
        scale = 1.0
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(frame, (18, 18), (tw + 28, th + 30), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 20 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 128), thick)

    # Controls hint
    cv2.putText(frame, "Q: quit   R: reset buffer",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


# ============================================================
# MAIN LOOP
# ============================================================

def run(source, video_path, model, encoder, hand_det, pose_det):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        print("✓ Webcam opened — press Q to quit, R to reset buffer")
    else:
        path = Path(video_path)
        if not path.exists():
            sys.exit(f"[ERROR] Video file not found: {video_path}")
        cap = cv2.VideoCapture(str(path))
        print(f"✓ Video opened: {path.name} — press Q to quit")

    if not cap.isOpened():
        sys.exit("[ERROR] Could not open video source.")

    buffer     = deque(maxlen=SEQ_LEN)
    frame_idx  = 0
    last_label = ""
    last_conf  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            # For video files: loop back to start
            if source == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                buffer.clear()
                frame_idx = 0
                continue
            else:
                break

        frame_idx += 1

        # Only extract keypoints every FRAME_SKIP frames (matches training)
        if frame_idx % FRAME_SKIP == 0:
            kp = extract_keypoints(frame, hand_det, pose_det)
            buffer.append(kp)

            # Predict once we have a full window
            if len(buffer) == SEQ_LEN:
                last_label, last_conf = predict(list(buffer), model, encoder)
                print(f"  → {last_label}  ({last_conf*100:.1f}%)")

        # Draw and show
        display = draw_overlay(
            frame.copy(), last_label, last_conf,
            len(buffer), collecting=(len(buffer) < SEQ_LEN)
        )
        cv2.imshow("ISL Inference", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            buffer.clear()
            frame_idx  = 0
            last_label = ""
            last_conf  = 0.0
            print("  Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Sentence Inference")
    parser.add_argument("--source", choices=["webcam", "video"], required=True,
                        help="Input source: 'webcam' or 'video'")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to video file (required when --source video)")
    args = parser.parse_args()

    if args.source == "video" and args.path is None:
        parser.error("--path is required when --source is 'video'")

    model, encoder = load_model(ENCODER_PATH, MODEL_PATH)
    hand_det, pose_det = build_detectors()
    run(args.source, args.path, model, encoder, hand_det, pose_det)
