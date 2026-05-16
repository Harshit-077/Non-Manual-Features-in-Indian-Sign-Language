"""
iSign Dataset Download & Setup Script
======================================
For: ISL Non-Manual Feature Detection Research
Setup: Laptop with Integrated NVIDIA GPU (low VRAM, lightweight config)

What this script does:
  1. Installs required packages
  2. Downloads iSign dataset from HuggingFace
  3. Verifies the dataset structure
  4. Prepares a lightweight folder layout for your pipeline
  5. Samples a few videos so you can inspect them

Run:
  pip install -r requirements_isign.txt
  python isign_setup.py
"""

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — change these paths if needed
# ─────────────────────────────────────────────
BASE_DIR        = Path("isign_workspace")
RAW_DIR         = BASE_DIR / "raw"
VIDEO_DIR       = BASE_DIR / "videos"
POSE_DIR        = BASE_DIR / "poses"
SAMPLES_DIR     = BASE_DIR / "samples"
METADATA_DIR    = BASE_DIR / "metadata"

HF_DATASET_ID   = "Exploration-Lab/iSign"
MAX_SAMPLE_VIDS = 10   # How many sample videos to pull for inspection


# ─────────────────────────────────────────────
# STEP 0 — Install dependencies
# ─────────────────────────────────────────────
def install_dependencies():
    packages = [
        "datasets",          # HuggingFace datasets
        "huggingface_hub",   # HF hub utilities
        "tqdm",              # Progress bars
        "pandas",            # Metadata handling
        "opencv-python",     # Video reading/inspection
        "Pillow",            # Image utilities
    ]
    print("\n[Step 0] Installing dependencies...")
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"]
        )
    print("  ✓ Dependencies ready\n")


# ─────────────────────────────────────────────
# STEP 1 — Create workspace folder structure
# ─────────────────────────────────────────────
def create_workspace():
    print("[Step 1] Creating workspace structure...")
    for d in [RAW_DIR, VIDEO_DIR, POSE_DIR, SAMPLES_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Workspace created at: {BASE_DIR.resolve()}\n")


# ─────────────────────────────────────────────
# STEP 2 — Download iSign from HuggingFace
# ─────────────────────────────────────────────
def get_hf_token():
    """
    Get HuggingFace token from env var or prompt user.
    iSign is a gated dataset — you must accept the terms on HuggingFace first.
    Visit: https://huggingface.co/datasets/Exploration-Lab/iSign
    Click 'Agree and access repository', then get your token from:
    https://huggingface.co/settings/tokens
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        print("  Using HF_TOKEN from environment variable.")
        return token

    print("\n  iSign is a GATED dataset on HuggingFace.")
    print("  Before running this, you need to:")
    print("  1. Visit: https://huggingface.co/datasets/Exploration-Lab/iSign")
    print("  2. Click 'Agree and access repository'")
    print("  3. Get your token from: https://huggingface.co/settings/tokens")
    print()
    token = input("  Paste your HuggingFace token here (or press Enter to skip): ").strip()
    return token if token else None


def download_isign():
    from datasets import load_dataset

    print("[Step 2] Downloading iSign dataset from HuggingFace...")
    print("  NOTE: Full dataset is large. Downloading metadata + pose splits first.")
    print("  Videos can be downloaded separately via OneDrive links (see README below).\n")

    token = get_hf_token()
    if not token:
        print("  Skipping dataset download — no token provided.")
        print("  You can re-run after setting: set HF_TOKEN=your_token (Windows)")
        return None

    try:
        dataset = load_dataset(
            HF_DATASET_ID,
            streaming=True,   # lightweight — doesn't download all at once
            token=token
        )

        print("  Dataset connected. Available splits:")
        for split in dataset.keys():
            print(f"      - {split}")

        splits_info = {"splits": list(dataset.keys())}
        with open(METADATA_DIR / "splits.json", "w", encoding="utf-8") as f:
            json.dump(splits_info, f, indent=2)

        return dataset

    except Exception as e:
        print(f"  Error loading dataset: {e}")
        print("  Make sure you accepted the dataset terms on HuggingFace first.")
        return None


# ─────────────────────────────────────────────
# STEP 3 — Inspect & save sample metadata
# ─────────────────────────────────────────────
def inspect_and_sample(dataset):
    import pandas as pd
    from tqdm import tqdm

    if dataset is None:
        print("[Step 3] Skipping — dataset not loaded.\n")
        return

    print("[Step 3] Inspecting dataset structure and saving samples...")

    all_samples = []
    for split_name, split_data in dataset.items():
        print(f"\n  Split: {split_name}")
        samples = []
        for i, example in enumerate(tqdm(split_data, desc=f"  Reading {split_name}", total=MAX_SAMPLE_VIDS)):
            if i >= MAX_SAMPLE_VIDS:
                break
            samples.append(example)
            all_samples.append({"split": split_name, **{
                k: str(v)[:200] for k, v in example.items()  # truncate long fields
            }})

        # Save sample rows as JSON for inspection
        sample_path = METADATA_DIR / f"sample_{split_name}.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(samples[:5], f, indent=2, default=str)
        print(f"  ✓ Saved {len(samples)} sample rows → {sample_path}")

        # Print field names from first example
        if samples:
            print(f"  Fields available: {list(samples[0].keys())}")

    # Save combined overview CSV
    if all_samples:
        df = pd.DataFrame(all_samples)
        df.to_csv(METADATA_DIR / "dataset_overview.csv", index=False, encoding="utf-8")
        print(f"\n  Overview saved -> {METADATA_DIR / 'dataset_overview.csv'}")


# ─────────────────────────────────────────────
# STEP 4 — Write OneDrive download helper
# ─────────────────────────────────────────────
def write_video_download_info():
    """
    iSign videos are hosted on OneDrive (not directly on HuggingFace).
    This writes out the links and instructions clearly.
    """
    info = """
# iSign Video Download Instructions
# ====================================
# The raw ISL videos are hosted on OneDrive by IIT Kanpur.
# Download them manually using the links below, then place them
# in the isign_workspace/videos/ folder.

# --- Task 1: ISL Video to English Translation ---
# ISL Videos:
#   https://onedrive.live.com/?authkey=%21ALT%2D9g%5F2oEHaJHU&id=A668F45668274EE0%2123784&cid=A668F45668274EE0
# English Translations (text):
#   https://onedrive.live.com/view.aspx?resid=A668F45668274EE0!52991&cid=a668f45668274ee0&authkey=!ALT-9g_2oEHaJHU

# --- Task 2: ISL Pose to English (RECOMMENDED -- lighter than raw video) ---
# ISL Pose data (pre-extracted landmarks!):
#   https://onedrive.live.com/?authkey=%21ALT%2D9g%5F2oEHaJHU&id=A668F45668274EE0%2123785&cid=A668F45668274EE0

# --- Task 3: CISLR -- Isolated Sign Recognition (HuggingFace) ---
# Directly loadable (no token needed):
#   from datasets import load_dataset
#   ds = load_dataset("IIT-K/CISLR")

# RECOMMENDATION FOR LOW-VRAM LAPTOPS:
#   Start with Task 2 (ISLPose) -- poses are pre-extracted,
#   so you skip the heavy MediaPipe step and go straight to
#   training your temporal model.
#   Then use Task 3 (CISLR) for isolated sign recognition experiments.
"""
    info_path = BASE_DIR / "VIDEO_DOWNLOAD_README.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(info)
    print(f"\n[Step 4] Video download instructions saved -> {info_path}")


# ─────────────────────────────────────────────
# STEP 5 — Write next-step pipeline scripts
# ─────────────────────────────────────────────
def write_next_step_scripts():
    """Writes skeleton scripts for MediaPipe + OpenFace steps."""

    # --- MediaPipe landmark extraction skeleton ---
    mediapipe_script = '''"""
extract_landmarks.py
=====================
Step 2 of the ISL NMF pipeline:
Extract face + hand + pose landmarks from ISL videos using MediaPipe Holistic.

Optimized for low-VRAM laptops:
- Processes videos frame by frame (no batching)
- Saves landmarks as compressed .npz (not raw video)
- Uses model_complexity=1 (lighter than 2, still accurate)

Usage:
  python extract_landmarks.py --video_dir isign_workspace/videos --out_dir isign_workspace/poses
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_landmarks_from_video(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        frame_landmarks = {
            "face":       _lm_to_array(results.face_landmarks, 478),
            "left_hand":  _lm_to_array(results.left_hand_landmarks, 21),
            "right_hand": _lm_to_array(results.right_hand_landmarks, 21),
            "pose":       _lm_to_array(results.pose_landmarks, 33),
        }
        frames_data.append(frame_landmarks)

    cap.release()
    return frames_data

def _lm_to_array(landmark_list, n_points):
    """Convert MediaPipe landmark list to (N, 3) numpy array."""
    if landmark_list is None:
        return np.zeros((n_points, 3), dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark],
        dtype=np.float32
    )

def main(video_dir, out_dir):
    video_dir = Path(video_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(video_dir.glob("**/*.mp4"))
    print(f"Found {len(video_files)} videos")

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,        # lighter for integrated GPU
        enable_segmentation=False, # saves memory
        refine_face_landmarks=True
    ) as holistic:
        for video_path in tqdm(video_files, desc="Extracting landmarks"):
            out_path = out_dir / (video_path.stem + ".npz")
            if out_path.exists():
                continue  # skip already processed

            try:
                frames = extract_landmarks_from_video(video_path, holistic)
                # Stack into arrays: shape (T, N_points, 3)
                np.savez_compressed(
                    out_path,
                    face       = np.stack([f["face"]       for f in frames]),
                    left_hand  = np.stack([f["left_hand"]  for f in frames]),
                    right_hand = np.stack([f["right_hand"] for f in frames]),
                    pose       = np.stack([f["pose"]       for f in frames]),
                )
            except Exception as e:
                print(f"  Error on {video_path.name}: {e}")

    print(f"\\nDone! Landmarks saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="isign_workspace/videos")
    parser.add_argument("--out_dir",   default="isign_workspace/poses")
    args = parser.parse_args()
    main(args.video_dir, args.out_dir)
'''

    with open(BASE_DIR / "extract_landmarks.py", "w", encoding="utf-8") as f:
        f.write(mediapipe_script)
    print(f"  Landmark extraction script -> {BASE_DIR / 'extract_landmarks.py'}")


# ─────────────────────────────────────────────
# STEP 6 — Write requirements file
# ─────────────────────────────────────────────
def write_requirements():
    reqs = """# ISL NMF Pipeline — Requirements
# Install with: pip install -r requirements_isign.txt

# Core data
datasets>=2.18.0
huggingface_hub>=0.22.0
pandas>=2.0.0
tqdm>=4.66.0

# Vision & landmarks
opencv-python>=4.9.0
mediapipe>=0.10.0
Pillow>=10.0.0
numpy>=1.26.0

# Visualization (optional but useful)
matplotlib>=3.8.0
"""
    req_path = BASE_DIR / "requirements_isign.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(reqs)
    print(f"  Requirements file -> {req_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  iSign Dataset Setup — ISL NMF Research Pipeline")
    print("=" * 55)

    install_dependencies()
    create_workspace()
    dataset = download_isign()
    inspect_and_sample(dataset)
    write_video_download_info()
    write_next_step_scripts()
    write_requirements()

    print("\n" + "=" * 55)
    print("  SETUP COMPLETE")
    print("=" * 55)
    print(f"""
Next Steps:
-----------
1. Check metadata:
     isign_workspace/metadata/

2. Download ISL videos or poses (see):
     isign_workspace/VIDEO_DOWNLOAD_README.txt

3. RECOMMENDED for your laptop — start with ISLPose (Task 2):
     Pre-extracted poses, no heavy GPU work needed yet

4. Once videos are downloaded, extract landmarks:
     pip install mediapipe
     python isign_workspace/extract_landmarks.py

5. Then we'll build the NMF classifier on top!
""")


if __name__ == "__main__":
    main()
