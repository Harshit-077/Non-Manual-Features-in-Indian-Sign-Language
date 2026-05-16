"""
nmf_classifier.py
==================
BiLSTM-based NMF classifier for ISL emotion signs.
Input  : MediaPipe landmark sequences (.npz files)
Output : Predicted emotion/gloss label

Dependencies:
    pip install scikit-learn torch tqdm numpy pandas matplotlib seaborn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
POSES_DIR   = Path("isign_workspace/poses")
CISLR_TOKEN = "token"
MAX_FRAMES  = 150       # pad/truncate all sequences to this length
BATCH_SIZE  = 8
EPOCHS      = 50
LR          = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
CATEGORIES  = ["emotion", "gesture", "behavior", "action"]  # filter to these categories (None = all)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# STEP 1 — Load labels from CISLR
# ─────────────────────────────────────────────
def load_labels():
    from datasets import load_dataset
    print("Loading CISLR labels...")
    cislr = load_dataset("IIT-K/CISLR", token=CISLR_TOKEN)["test"]
    df = pd.DataFrame(cislr)

    # Build lookup: both exact uid and uid without suffix -> npz path
    npz_files = {}
    for p in POSES_DIR.glob("*.npz"):
        npz_files[p.stem] = p
        # also index by base video id (without _N suffix)
        base = p.stem.rsplit("_", 1)[0]
        if base not in npz_files:
            npz_files[base] = p

    records = []
    for _, row in df.iterrows():
        uid = row["uid"]
        if uid in npz_files:
            records.append({
                "uid"      : uid,
                "npz_path" : npz_files[uid],
                "gloss"    : row["gloss"],
                "category" : row["category"],
            })

    matched_df = pd.DataFrame(records)
    # Group glosses into broader NMF classes
    NMF_GROUPS = {
        "angry": "negative", "hate": "negative", "fear": "negative",
        "guilty": "negative", "sad": "negative", "regret": "negative",
        "mean": "negative", "ruthless": "negative", "selfish": "negative",
        "shock": "negative", "upset": "negative", "worry": "negative",
        "miss": "negative", "tease": "negative",
        "happy": "positive", "trust": "positive", "kind": "positive",
        "calm": "positive", "thank you": "positive", "love": "positive",
        "hug": "positive", "respect": "positive", "free": "positive",
        "tolerate": "positive", "social": "positive",
        "surprise": "aroused", "nervous": "aroused", "shy": "aroused",
        "confused": "aroused", "interest": "aroused", "attitude": "aroused",
        "confident": "assertive", "hardworking": "assertive", "clever": "assertive",
        "busy": "assertive", "initiative": "assertive", "strict": "assertive",
        "cunning": "assertive", "dream": "assertive", "patience": "assertive",
        "minor": "assertive", "manner": "assertive", "model": "assertive",
        "sign": "gesture", "okay": "gesture", "tally": "gesture",
        "reception": "gesture", "ethics": "gesture",
        # action category — mapped by movement/expression type
        "run": "action", "walk": "action", "jump": "action", "eat": "action",
        "drink": "action", "sleep": "action", "sit": "action", "stand": "action",
        "write": "action", "read": "action", "speak": "action", "listen": "action",
        "watch": "action", "play": "action", "work": "action", "study": "action",
        "cook": "action", "clean": "action", "wash": "action", "drive": "action",
        "swim": "action", "dance": "action", "sing": "action", "draw": "action",
        "fight": "action", "help": "action", "give": "action", "take": "action",
        "open": "action", "close": "action", "push": "action", "pull": "action",
        "cut": "action", "buy": "action", "sell": "action", "pay": "action",
        "call": "action", "meet": "action", "ask": "action", "answer": "action",
        "teach": "action", "learn": "action", "think": "action", "know": "action",
        "forget": "action", "remember": "action", "understand": "action",
        "division": "action", "subtract": "action", "calculate": "action",
        "destroy": "action", "build": "action", "make": "action", "break": "action",
        "throw": "action", "catch": "action", "kick": "action", "hit": "action",
        "climb": "action", "fall": "action", "carry": "action", "lift": "action",
        "pour": "action", "fill": "action", "mix": "action", "stir": "action",
        "point": "action", "show": "action", "hide": "action", "find": "action",
        "search": "action", "wait": "action", "stop": "action", "start": "action",
        "finish": "action", "continue": "action", "return": "action", "leave": "action",
        "arrive": "action", "travel": "action", "visit": "action", "enter": "action",
        "exit": "action", "turn": "action", "move": "action", "bring": "action",
        "send": "action", "receive": "action", "share": "action", "keep": "action",
        "put": "action", "remove": "action", "change": "action", "fix": "action",
        "repair": "action", "save": "action", "print": "action", "type": "action",
        "upload": "action", "download": "action", "install": "action",
        "rehabilitation": "action", "agenda": "action", "jump": "action",
    }
    matched_df["nmf_group"] = matched_df["gloss"].map(NMF_GROUPS).fillna("other")
    matched_df = matched_df[matched_df["nmf_group"] != "other"].reset_index(drop=True)
    print(f"NMF groups: {matched_df['nmf_group'].value_counts().to_dict()}")
    print(f"Matched {len(matched_df)} npz files to CISLR labels")
    print(f"Labels: {sorted(matched_df['gloss'].unique())}")
    return matched_df


# ─────────────────────────────────────────────
# STEP 2 — Feature extraction from landmarks
# ─────────────────────────────────────────────
def extract_features(npz_path):
    """
    Extract NMF-relevant features from a landmark npz file.
    We focus on face landmarks (NMF) + pose (head movement).
    Returns array of shape (T, feature_dim)
    """
    data = np.load(npz_path)

    face       = data["face"]        # (T, 478, 3)
    left_hand  = data["left_hand"]   # (T, 21, 3)
    right_hand = data["right_hand"]  # (T, 21, 3)
    pose       = data["pose"]        # (T, 33, 3)

    T = face.shape[0]

    # --- NMF-relevant face landmark indices ---
    # Eyebrows: 33 key points
    LEFT_EYEBROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    # Eyes
    LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]
    # Mouth
    MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
             291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    # Nose tip
    NOSE = [1, 2, 98, 327]

    # Combine selected face indices
    face_idx = LEFT_EYEBROW + RIGHT_EYEBROW + LEFT_EYE + RIGHT_EYE + MOUTH + NOSE
    face_sel = face[:, face_idx, :].reshape(T, -1)   # (T, N_face*3)

    # Head pose from pose landmarks (nose=0, left ear=7, right ear=8)
    head_points = pose[:, [0, 7, 8], :].reshape(T, -1)  # (T, 9)

    # Hand presence (binary: is hand detected or all zeros?)
    lh_presence = (left_hand.sum(axis=(1, 2)) != 0).astype(np.float32).reshape(T, 1)
    rh_presence = (right_hand.sum(axis=(1, 2)) != 0).astype(np.float32).reshape(T, 1)

    # Hand landmark summary (wrist + fingertips only to keep it light)
    # Wrist=0, thumb tip=4, index=8, middle=12, ring=16, pinky=20
    FINGERTIPS = [0, 4, 8, 12, 16, 20]
    lh_key = left_hand[:, FINGERTIPS, :].reshape(T, -1)   # (T, 18)
    rh_key = right_hand[:, FINGERTIPS, :].reshape(T, -1)  # (T, 18)

    # Concatenate all features
    features = np.concatenate([
        face_sel,      # face NMF landmarks
        head_points,   # head pose
        lh_presence,   # left hand detected?
        rh_presence,   # right hand detected?
        lh_key,        # left hand fingertips
        rh_key,        # right hand fingertips
    ], axis=1)

    return features.astype(np.float32)


def pad_or_truncate(seq, max_len):
    T, F = seq.shape
    if T >= max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - T, F), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


# ─────────────────────────────────────────────
# STEP 3 — Dataset
# ─────────────────────────────────────────────
class NMFDataset(Dataset):
    def __init__(self, df, label_encoder, max_frames=MAX_FRAMES):
        self.samples = []
        for _, row in df.iterrows():
            features = extract_features(row["npz_path"])
            features = pad_or_truncate(features, max_frames)
            label    = label_encoder.transform([row["nmf_group"]])[0]
            self.samples.append((
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# STEP 4 — BiLSTM Model
# ─────────────────────────────────────────────
class NMFClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last timestep
        out = self.dropout(out)
        return self.fc(out)


# ─────────────────────────────────────────────
# STEP 5 — Training
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels


# ─────────────────────────────────────────────
# STEP 6 — Plot helpers
# ─────────────────────────────────────────────
def plot_training(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs,   label="Val Acc")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.savefig("isign_workspace/training_curves.png")
    print("Training curves saved to isign_workspace/training_curves.png")


def plot_confusion(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("isign_workspace/confusion_matrix.png")
    print("Confusion matrix saved to isign_workspace/confusion_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    # Load and match labels
    df = load_labels()
    if len(df) == 0:
        print("No matched files found. Check POSES_DIR and CISLR data.")
        return

    # Encode labels
    le = LabelEncoder()
    le.fit(df["nmf_group"])
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}\n")

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42,
                                         stratify=df["nmf_group"])
    print(f"Train: {len(train_df)}  Val: {len(val_df)}\n")

    # Build datasets
    print("Building datasets and extracting features...")
    train_ds = NMFDataset(train_df, le)
    val_ds   = NMFDataset(val_df,   le)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Get input size from first sample
    input_size = train_ds[0][0].shape[1]
    print(f"Feature dim per frame: {input_size}")
    print(f"Sequence length: {MAX_FRAMES}\n")

    # Build model
    model = NMFClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(DEVICE)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0

    print("Training...\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, preds, labels = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "isign_workspace/best_nmf_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f}")

    print(f"\nBest Val Accuracy: {best_val_acc:.3f}")

    # Final evaluation
    print("\nClassification Report:")
    present = sorted(set(labels) | set(preds))
    present_names = le.inverse_transform(present)
    print(classification_report(labels, preds, labels=present, target_names=present_names))

    # Plots
    plot_training(train_losses, val_losses, train_accs, val_accs)
    plot_confusion(labels, preds, present_names)

    print("\nDone! Model saved to isign_workspace/best_nmf_model.pt")


if __name__ == "__main__":
    main()
