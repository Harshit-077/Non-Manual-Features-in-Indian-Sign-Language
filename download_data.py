import os
from pathlib import Path

TOKEN = "token"  # replace with your HuggingFace token
SAVE_DIR = Path("isign_workspace/metadata")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. iSign CSV ──────────────────────────────────────────
print("Downloading iSign_v1.1.csv ...")
from huggingface_hub import hf_hub_download

csv_path = hf_hub_download(
    repo_id="Exploration-Lab/iSign",
    filename="iSign_v1.1.csv",
    repo_type="dataset",
    token=TOKEN,
    local_dir=SAVE_DIR,
)
print(f"Saved to: {csv_path}")

import pandas as pd
df = pd.read_csv(csv_path)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))


# ── 2. CISLR ─────────────────────────────────────────────
print("\nDownloading CISLR ...")
from datasets import load_dataset

cislr = load_dataset("IIT-K/CISLR", token=TOKEN)
print(cislr)
print("\nSample entry:")
print(cislr["test"][0])
