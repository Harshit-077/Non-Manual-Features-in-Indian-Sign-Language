"""
test_isl_pipeline.py
====================
ISL Full Inference Test Script
Combines:
  - Expression CNN (ONNX) from isl_nmf_expr_hm.ipynb
  - Head Movement TCN (ONNX) from isl_nmf_expr_hm.ipynb
  - ISL Sentence BiLSTM (PyTorch) from isl_pyt_200.ipynb
  - Live webcam inference logic from test.ipynb

Usage:
  python test_isl_pipeline.py [--video 0] [--no-sentence] [--no-nmf]

Requirements:
  pip install torch torchvision onnxruntime mediapipe opencv-python-headless numpy

Expected file layout:
  isl_nmf/onnx/expression_cnn.onnx
  isl_nmf/onnx/head_movement_tcn.onnx
  models&config/label_encoder.pkl
  models&config/isl_best_model.pt
  face_landmarker.task
  pose_landmarker_full.task
  hand_landmarker.task
"""

import argparse
import pickle
import sys
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SEQ_LEN       = 30        # frames per temporal window
NUM_KEYPOINTS = 225       # 33*3 pose + 21*3 left hand + 21*3 right hand
FRAME_SKIP    = 5         # sample every Nth frame for the sentence model
SENTENCE_CONF = 0.40      # minimum confidence to display a sentence prediction

EXPR_CLASSES = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]
HEAD_CLASSES = ["nod", "shake", "tilt", "still"]

MEDIAPIPE_MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

# ---------------------------------------------------------------------------
# ISL Sentence BiLSTM  (mirrors isl_pyt_200.ipynb exactly)
# ---------------------------------------------------------------------------

class ISLModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout1: float = 0.3,
        dropout2: float = 0.4,
        dropout3: float = 0.3,
    ):
        super().__init__()
        self.bilstm1   = nn.LSTM(input_size, hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.dropout1  = nn.Dropout(dropout1)
        self.bilstm2   = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc1        = nn.Linear(hidden_size * 2, 256)
        self.relu1      = nn.ReLU()
        self.dropout2   = nn.Dropout(dropout2)
        self.fc2        = nn.Linear(256, 128)
        self.relu2      = nn.ReLU()
        self.dropout3   = nn.Dropout(dropout3)
        self.out        = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out    = self.dropout1(out)
        out, _ = self.bilstm2(out)
        out    = out[:, -1, :]          # last timestep
        out    = self.batch_norm(out)
        out    = self.dropout2(self.relu1(self.fc1(out)))
        out    = self.dropout3(self.relu2(self.fc2(out)))
        return self.out(out)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize and normalise a face crop for the expression ONNX model."""
    face = cv2.resize(face_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))           # HWC -> CHW
    return np.expand_dims(face, axis=0)             # (1, 3, 112, 112)


def extract_head_features(landmarks) -> np.ndarray:
    """Compute 3-DOF head pose from face landmarks (pitch, yaw, roll)."""
    nose      = landmarks[1]
    left_eye  = landmarks[33]
    right_eye = landmarks[263]
    pitch = nose.y - 0.5
    yaw   = nose.x - 0.5
    roll  = right_eye.y - left_eye.y
    return np.array([pitch, yaw, roll], dtype=np.float32)


def extract_body_keypoints(mp_image, pose_detector, hand_detector) -> np.ndarray:
    """
    Extract pose (33×3) + left hand (21×3) + right hand (21×3) keypoints.
    Returns a float32 vector of length 225.
    """
    pose_result = pose_detector.detect(mp_image)
    pose = np.zeros(33 * 3, dtype=np.float32)
    if pose_result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]
        ).flatten()

    hand_result = hand_detector.detect(mp_image)
    left_hand   = np.zeros(21 * 3, dtype=np.float32)
    right_hand  = np.zeros(21 * 3, dtype=np.float32)
    if hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
            ).flatten()
            if handedness == "Left":
                left_hand = arr
            else:
                right_hand = arr

    return np.concatenate([pose, left_hand, right_hand])


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _download_if_missing(filename: str) -> bool:
    path = Path(filename)
    if path.exists():
        return True
    url = MEDIAPIPE_MODELS.get(filename)
    if url is None:
        return False
    print(f"  Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"  ✓ Saved {filename}")
        return True
    except Exception as exc:
        print(f"  ✗ Failed to download {filename}: {exc}")
        return False


def load_onnx_sessions(use_gpu: bool = True):
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )

    expr_path = "isl_nmf/onnx/expression_cnn.onnx"
    hm_path   = "isl_nmf/onnx/head_movement_tcn.onnx"

    for p in (expr_path, hm_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {p}\n"
                "Run isl_nmf_expr_hm.ipynb Section 17 (ONNX Export) first."
            )

    expr_session = ort.InferenceSession(expr_path, providers=providers)
    hm_session   = ort.InferenceSession(hm_path,   providers=providers)
    print(f"✓ ONNX NMF models loaded (providers: {providers})")
    return expr_session, hm_session


def load_sentence_model(device: torch.device):
    encoder_path = Path("models&config/label_encoder.pkl")
    weights_path = Path("models&config/isl_best_model.pt")

    for p in (encoder_path, weights_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run isl_pyt_200.ipynb to train and save the sentence model."
            )

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Read num_classes directly from the checkpoint so the architecture always
    # matches the saved weights, regardless of how many classes the encoder has.
    state_dict  = torch.load(weights_path, map_location=device, weights_only=True)
    num_classes = state_dict["out.weight"].shape[0]

    if num_classes != len(label_encoder.classes_):
        print(
            f"  ⚠ num_classes mismatch: checkpoint={num_classes}, "
            f"encoder={len(label_encoder.classes_)}. Using checkpoint value."
        )

    model = ISLModel(
        input_size=NUM_KEYPOINTS,
        hidden_size=128,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ ISL sentence model loaded — {num_classes} classes")
    return model, label_encoder


def load_mediapipe_landmarkers():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    for filename in MEDIAPIPE_MODELS:
        ok = _download_if_missing(filename)
        if not ok:
            raise FileNotFoundError(
                f"MediaPipe model file '{filename}' is missing and could not be downloaded.\n"
                "Download it manually from the URLs listed in MEDIAPIPE_MODELS."
            )

    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_full.task"),
            running_mode=vision.RunningMode.IMAGE,
        )
    )
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )
    print("✓ All MediaPipe landmarkers loaded")
    return face_landmarker, pose_detector, hand_detector


# ---------------------------------------------------------------------------
# Per-frame prediction helpers
# ---------------------------------------------------------------------------

def predict_expression(face_crop: np.ndarray, session) -> tuple[str, float]:
    inp    = preprocess_face(face_crop)
    logits = session.run(["expr_logits"], {"face_image": inp})[0]
    probs  = softmax(logits[0])
    idx    = int(np.argmax(probs))
    return EXPR_CLASSES[idx], float(probs[idx])


def predict_head_movement(
    pose_buffer: deque, session
) -> tuple[str | None, float]:
    if len(pose_buffer) < SEQ_LEN:
        return None, 0.0
    seq    = np.expand_dims(np.array(pose_buffer, dtype=np.float32), axis=0)
    logits = session.run(["hm_logits"], {"pose_seq": seq})[0]
    probs  = softmax(logits[0])
    idx    = int(np.argmax(probs))
    return HEAD_CLASSES[idx], float(probs[idx])


def predict_sentence(
    sentence_buffer: deque,
    isl_model: nn.Module,
    label_encoder,
    device: torch.device,
) -> tuple[str | None, float]:
    if len(sentence_buffer) < SEQ_LEN:
        return None, 0.0
    seq = np.array(sentence_buffer, dtype=np.float32)          # (30, 225)
    t   = torch.tensor(seq).unsqueeze(0).to(device)            # (1, 30, 225)
    with torch.no_grad():
        logits = isl_model(t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx   = int(np.argmax(probs))
    label = label_encoder.inverse_transform([idx])[0]
    return label, float(probs[idx])


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    expr_display: str,
    hm_display: str,
    sentence_display: str,
) -> np.ndarray:
    lines  = [expr_display, hm_display, sentence_display]
    colors = [(0, 255, 0), (0, 255, 0), (0, 200, 255)]
    for i, (text, color) in enumerate(zip(lines, colors)):
        cv2.putText(
            frame, text, (20, 40 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
        )
    return frame


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    video_source: int | str = 0,
    use_nmf: bool = True,
    use_sentence: bool = True,
):
    import mediapipe as mp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    expr_session = hm_session = None
    isl_model    = label_encoder = None

    if use_nmf:
        expr_session, hm_session = load_onnx_sessions()

    if use_sentence:
        isl_model, label_encoder = load_sentence_model(device)

    face_landmarker, pose_detector, hand_detector = load_mediapipe_landmarkers()

    # ── Buffers ───────────────────────────────────────────────────────────────
    pose_buffer     = deque(maxlen=SEQ_LEN)
    sentence_buffer = deque(maxlen=SEQ_LEN)
    frame_counter   = 0

    # Persistent display strings
    expr_display     = "Expression: ---"
    hm_display       = "Head: ---"
    sentence_display = "Sentence: ---"

    # ── Open capture ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    print("Press Q to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_counter += 1

            # ── FACE: expression + head movement ─────────────────────────────
            if use_nmf:
                face_result = face_landmarker.detect(mp_image)

                if face_result.face_landmarks:
                    landmarks = face_result.face_landmarks[0]

                    # Bounding box from landmarks
                    xs = [lm.x for lm in landmarks]
                    ys = [lm.y for lm in landmarks]
                    x1 = max(0, int(min(xs) * w) - 20)
                    y1 = max(0, int(min(ys) * h) - 20)
                    x2 = min(w, int(max(xs) * w) + 20)
                    y2 = min(h, int(max(ys) * h) + 20)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        label, conf  = predict_expression(face_crop, expr_session)
                        expr_display = f"Expression: {label} ({conf:.2f})"

                    head_pose = extract_head_features(landmarks)
                    pose_buffer.append(head_pose)

                    hm_label, hm_conf = predict_head_movement(pose_buffer, hm_session)
                    if hm_label is not None:
                        hm_display = f"Head: {hm_label} ({hm_conf:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ── BODY KEYPOINTS → sentence model (every FRAME_SKIP-th frame) ──
            if use_sentence and frame_counter % FRAME_SKIP == 0:
                keypoints = extract_body_keypoints(mp_image, pose_detector, hand_detector)
                sentence_buffer.append(keypoints)

                sent_label, sent_conf = predict_sentence(
                    sentence_buffer, isl_model, label_encoder, device
                )
                if sent_label is not None:
                    tag = "" if sent_conf >= SENTENCE_CONF else " [low conf]"
                    sentence_display = f"Sentence: {sent_label} ({sent_conf:.2f}){tag}"

            # ── Overlay ───────────────────────────────────────────────────────
            frame = draw_overlay(frame, expr_display, hm_display, sentence_display)
            cv2.imshow("ISL Full Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        pose_detector.close()
        hand_detector.close()
        print("\n✓ Done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ISL Full Inference Pipeline")
    parser.add_argument(
        "--video",
        default=0,
        help="Webcam index (default 0) or path to a video file",
    )
    parser.add_argument(
        "--no-sentence",
        action="store_true",
        help="Disable the ISL sentence BiLSTM model",
    )
    parser.add_argument(
        "--no-nmf",
        action="store_true",
        help="Disable the NMF expression + head movement models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert --video to int if it looks like a device index
    video_source = args.video
    try:
        video_source = int(video_source)
    except ValueError:
        pass  # it's a file path — leave as string

    if args.no_sentence and args.no_nmf:
        print("Error: both --no-sentence and --no-nmf passed — nothing to run.")
        sys.exit(1)

    run_inference(
        video_source=video_source,
        use_nmf=not args.no_nmf,
        use_sentence=not args.no_sentence,
    )
