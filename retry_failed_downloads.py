"""
retry_failed_downloads.py
==========================
Retries failed CISLR downloads with better format handling.
Shows exact reason for each failure.
"""

import subprocess
import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

TOKEN     = "your_hf_token_here"
VIDEO_DIR = Path("isign_workspace/cislr_videos")
FAILED_LOG = Path("isign_workspace/failed_downloads.txt")
FAILED_LOG2 = Path("isign_workspace/failed_downloads_round2.txt")


def parse_uid(uid):
    parts = uid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return uid, 1


def download_clip(video_id, clip_num, duration, out_dir):
    out_path = out_dir / f"{video_id}_{clip_num}.mp4"
    if out_path.exists():
        return True, "already exists"

    url = f"https://www.youtube.com/watch?v={video_id}"
    tmp_path = out_dir / f"_tmp_{video_id}.mp4"

    # Try multiple format options in order
    format_attempts = [
        "worst[ext=mp4]/worst",
        "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "best",
    ]

    for fmt in format_attempts:
        try:
            dl_cmd = [
                "yt-dlp",
                "-f", fmt,
                "--cookies", "cookies.txt",
                "--remote-components", "ejs:github",
                "-o", str(tmp_path),
                "--no-warnings",
                "--extractor-retries", "3",
                url
            ]
            result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                error = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
                continue  # try next format

            if not tmp_path.exists():
                continue

            # Trim clip
            start_time = (clip_num - 1) * duration
            trim_cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-i", str(tmp_path),
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y",
                "-loglevel", "error",
                str(out_path)
            ]
            subprocess.run(trim_cmd, capture_output=True, timeout=120)

            if out_path.exists():
                return True, "ok"

        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # All formats failed — get the actual reason
    try:
        check = subprocess.run(
            ["yt-dlp", "--no-download", "--print", "%(title)s", url],
            capture_output=True, text=True, timeout=30
        )
        if "Private video" in check.stderr:
            return False, "private video"
        elif "age" in check.stderr.lower():
            return False, "age restricted"
        elif "not available" in check.stderr.lower():
            return False, "not available"
        else:
            last_line = check.stderr.strip().splitlines()[-1] if check.stderr else "unknown"
            return False, last_line
    except Exception:
        return False, "unknown"


def main():
    if not FAILED_LOG.exists():
        print("No failed_downloads.txt found. Nothing to retry.")
        return

    failed_uids = FAILED_LOG.read_text(encoding="utf-8").strip().splitlines()
    failed_uids = [u for u in failed_uids if u]
    print(f"Retrying {len(failed_uids)} failed downloads...\n")

    # Load metadata for duration info
    cislr = load_dataset("IIT-K/CISLR", token=TOKEN)["test"]
    df = pd.DataFrame(cislr).set_index("uid")

    results = {"ok": [], "failed": {}}

    for uid in tqdm(failed_uids, desc="Retrying"):
        if uid not in df.index:
            results["failed"][uid] = "uid not in dataset"
            continue

        row = df.loc[uid]
        video_id, clip_num = parse_uid(uid)
        ok, reason = download_clip(video_id, clip_num, row["duration"], VIDEO_DIR)

        if ok:
            results["ok"].append(uid)
        else:
            results["failed"][uid] = reason
            tqdm.write(f"  FAIL [{uid}]: {reason}")

    # Save still-failing
    with open(FAILED_LOG2, "w", encoding="utf-8") as f:
        for uid, reason in results["failed"].items():
            f.write(f"{uid}\t{reason}\n")

    print(f"\nResults:")
    print(f"  Recovered : {len(results['ok'])}")
    print(f"  Still failing: {len(results['failed'])}")

    if results["failed"]:
        print(f"\nFailure reasons:")
        reasons = {}
        for reason in results["failed"].values():
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x  {reason}")

    total = 8 + len(results["ok"])
    print(f"\nTotal downloaded so far: {total}/30")


if __name__ == "__main__":
    main()
