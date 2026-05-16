"""
download_cislr_videos.py
=========================
Downloads CISLR video clips from YouTube using yt-dlp.
Clips are trimmed to exact duration using ffmpeg.

Requirements:
    pip install yt-dlp datasets
    ffmpeg must be installed and on PATH
    Download ffmpeg from: https://ffmpeg.org/download.html
    Or via winget: winget install ffmpeg
"""

import os
import subprocess
import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

TOKEN      = "token"
VIDEO_DIR  = Path("isign_workspace/cislr_videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

FAILED_LOG = Path("isign_workspace/failed_downloads.txt")
CATEGORIES = ["emotion", "gesture", "behavior", "action"]  # 271? clips total


# ── Install yt-dlp if needed ──────────────────────────────
def install_ytdlp():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])


# ── Parse uid into video_id and clip_number ───────────────
def parse_uid(uid):
    # uid format: '<youtube_id>_<clip_number>'
    # youtube IDs can contain underscores too, so split from the RIGHT once
    # e.g. '-NIz596Y27Q_1' -> video_id='-NIz596Y27Q', clip=1
    parts = uid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return uid, 1


# ── Download and trim a single clip ──────────────────────
def download_clip(video_id, clip_num, duration, gloss, out_dir):
    out_path = out_dir / f"{video_id}_{clip_num}.mp4"
    if out_path.exists():
        return True  # already downloaded

    url = f"https://www.youtube.com/watch?v={video_id}"
    tmp_path = out_dir / f"_tmp_{video_id}.mp4"

    try:
        # Step 1 — download full video (lowest quality to save time/space)
        dl_cmd = [
            "yt-dlp",
            "-f", "worst[ext=mp4]/worst",
            "--cookies", "cookies.txt",
            "--remote-components", "ejs:github",
            "-o", str(tmp_path),
            "--quiet",
            "--no-warnings",
            url
        ]
        result = subprocess.run(dl_cmd, capture_output=True, timeout=300)
        if result.returncode != 0 or not tmp_path.exists():
            return False

        # Step 2 — trim to clip duration using ffmpeg
        # CISLR clips are sequential — clip N starts at (N-1)*duration
        start_time = (clip_num - 1) * duration
        trim_cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", str(tmp_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",                # overwrite if exists
            "-loglevel", "error",
            str(out_path)
        ]
        subprocess.run(trim_cmd, capture_output=True, timeout=60)

        return out_path.exists()

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


# ── Main ─────────────────────────────────────────────────
def main():
    install_ytdlp()

    print("Loading CISLR dataset...")
    cislr = load_dataset("IIT-K/CISLR", token=TOKEN)["test"]
    df = pd.DataFrame(cislr)
    print(f"Total clips: {len(df)}")

    # Filter to target categories
    df = df[df["category"].isin(CATEGORIES)].reset_index(drop=True)
    print(f"Downloading categories {CATEGORIES}: {len(df)} clips total")
    for cat in CATEGORIES:
        n = len(df[df["category"] == cat])
        print(f"  {cat}: {n} clips")

    failed = []
    success = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        video_id, clip_num = parse_uid(row["uid"])
        ok = download_clip(
            video_id=video_id,
            clip_num=clip_num,
            duration=row["duration"],
            gloss=row["gloss"],
            out_dir=VIDEO_DIR
        )
        if ok:
            success += 1
        else:
            failed.append(row["uid"])

    # Save failed downloads for retry
    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(failed))

    print(f"\nDone!")
    print(f"  Downloaded : {success}/{len(df)}")
    print(f"  Failed     : {len(failed)}")
    print(f"  Videos at  : {VIDEO_DIR.resolve()}")
    if failed:
        print(f"  Failed IDs : {FAILED_LOG.resolve()}")


if __name__ == "__main__":
    main()
