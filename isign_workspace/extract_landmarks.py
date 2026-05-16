"""
extract_landmarks.py
=====================
Extracts face + hand + pose landmarks from ISL videos.
Compatible with MediaPipe 0.10+

Usage:
  python extract_landmarks.py --video_dir isign_workspace/cislr_videos --out_dir isign_workspace/poses
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def _lm_to_array(landmarks, n_points):
    if landmarks is None:
        return np.zeros((n_points, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def extract_landmarks_from_video(video_path, detectors):
    face_det, hand_det, pose_det = detectors
    cap = cv2.VideoCapture(str(video_path))
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Face landmarks
        face_result = face_det.detect(mp_image)
        if face_result.face_landmarks:
            face_arr = _lm_to_array(face_result.face_landmarks[0], 478)
        else:
            face_arr = np.zeros((478, 3), dtype=np.float32)

        # Hand landmarks
        hand_result = hand_det.detect(mp_image)
        left_arr  = np.zeros((21, 3), dtype=np.float32)
        right_arr = np.zeros((21, 3), dtype=np.float32)
        if hand_result.hand_landmarks:
            for i, handedness in enumerate(hand_result.handedness):
                label = handedness[0].category_name  # 'Left' or 'Right'
                arr = _lm_to_array(hand_result.hand_landmarks[i], 21)
                if label == "Left":
                    left_arr = arr
                else:
                    right_arr = arr

        # Pose landmarks
        pose_result = pose_det.detect(mp_image)
        if pose_result.pose_landmarks:
            pose_arr = _lm_to_array(pose_result.pose_landmarks[0], 33)
        else:
            pose_arr = np.zeros((33, 3), dtype=np.float32)

        frames_data.append({
            "face":       face_arr,
            "left_hand":  left_arr,
            "right_hand": right_arr,
            "pose":       pose_arr,
        })

    cap.release()
    return frames_data


def build_detectors():
    # Face
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(Path(__file__).parent / "face_landmarker.task")
        ),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    face_det = mp_vision.FaceLandmarker.create_from_options(face_opts)

    # Hand
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(Path(__file__).parent / "hand_landmarker.task")
        ),
        num_hands=2,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)

    # Pose
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(Path(__file__).parent / "pose_landmarker_lite.task")
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)

    return face_det, hand_det, pose_det


def download_models(model_dir):
    import urllib.request
    models = {
        "face_landmarker.task":        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "hand_landmarker.task":        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "pose_landmarker_lite.task":   "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    }
    for fname, url in models.items():
        out = model_dir / fname
        if not out.exists():
            print(f"  Downloading {fname} ...")
            urllib.request.urlretrieve(url, out)
            print(f"  Saved {fname}")
        else:
            print(f"  {fname} already exists, skipping.")


def main(video_dir, out_dir):
    video_dir = Path(video_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Models go in the same folder as this script
    model_dir = Path(__file__).parent
    print("Downloading MediaPipe models if needed...")
    download_models(model_dir)

    video_files = list(video_dir.glob("**/*.mp4"))
    print(f"Found {len(video_files)} videos\n")

    print("Building detectors...")
    detectors = build_detectors()
    print("Detectors ready\n")

    for video_path in tqdm(video_files, desc="Extracting landmarks"):
        out_path = out_dir / (video_path.stem + ".npz")
        if out_path.exists():
            continue

        try:
            frames = extract_landmarks_from_video(video_path, detectors)
            if not frames:
                continue

            np.savez_compressed(
                out_path,
                face       = np.stack([f["face"]       for f in frames]),
                left_hand  = np.stack([f["left_hand"]  for f in frames]),
                right_hand = np.stack([f["right_hand"] for f in frames]),
                pose       = np.stack([f["pose"]       for f in frames]),
            )
        except Exception as e:
            tqdm.write(f"  Error on {video_path.name}: {e}")

    print(f"\nDone! Landmarks saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="isign_workspace/cislr_videos")
    parser.add_argument("--out_dir",   default="isign_workspace/poses")
    args = parser.parse_args()
    main(args.video_dir, args.out_dir)
