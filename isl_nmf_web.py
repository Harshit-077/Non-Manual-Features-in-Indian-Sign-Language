"""
ISL NMF Detection — Web Interface
==================================
Run:  python isl_nmf_web.py
Open: http://localhost:5050

Requires the same Python environment as the training notebook plus:
    pip install flask

The app loads trained ONNX models (if available) or runs in demo/mock mode.
"""

import os, io, time, base64, json, platform
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify, Response

# ── MediaPipe Tasks ─────────────────────────────────────────────────────────
from mediapipe import tasks
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

# ── Optional ONNX Runtime ────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("[WARN] onnxruntime not found — running in landmark-only mode.")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path("isl_nmf")
MODEL_DIR = BASE / "onnx"
FACE_LM_MODEL = BASE / "face_landmarker.task"

EXPR_NAMES = ["neutral","happy","sad","surprise","fear","disgust","anger","contempt"]
HM_NAMES   = ["nod","shake","tilt","forward","still"]
EB_NAMES   = ["neutral","raised_brows","furrowed_brows"]
GRAMMAR    = {
    "neutral":        "Statement / no NMF",
    "happy":          "Affirmative marker",
    "surprise":       "Wh-question (who/what/where)",
    "fear":           "Wh-question (secondary)",
    "sad":            "Negation / emotional tone",
    "disgust":        "Negation",
    "anger":          "Negation / intensity",
    "contempt":       "Negation",
    "nod":            "Yes / affirmative / sentence end",
    "shake":          "No / negation",
    "tilt":           "Conditional / topic marker",
    "forward":        "Emphasis / assertion",
    "raised_brows":   "Wh-question facial marker",
    "furrowed_brows": "Negation / yes-no question",
    "still":          "No movement",
}

LM_GROUPS = {
    "left_brow":  [70, 63, 105, 66, 107],
    "right_brow": [336, 296, 334, 293, 300],
    "left_eye":   [33, 160, 158, 133, 153, 144],
    "right_eye":  [362, 385, 387, 263, 373, 380],
}

SEQ_LEN = 30

# ── Load FaceLandmarker ───────────────────────────────────────────────────────
def download_face_model():
    import urllib.request
    FACE_LM_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if not FACE_LM_MODEL.exists():
        print("Downloading face_landmarker.task …")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task",
            FACE_LM_MODEL,
        )
        print("Downloaded.")

def make_face_landmarker(running_mode=vision.RunningMode.IMAGE):
    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_LM_MODEL)),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)

# ── Load ONNX models ─────────────────────────────────────────────────────────
def load_onnx(name):
    path = MODEL_DIR / name
    if not path.exists() or not HAS_ORT:
        return None
    providers = (
        ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if platform.system() == "Darwin"
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    try:
        sess = ort.InferenceSession(str(path), providers=providers)
        print(f"  Loaded {name}")
        return sess
    except Exception as e:
        print(f"  [WARN] Could not load {name}: {e}")
        return None

# ── Feature extraction ────────────────────────────────────────────────────────
def eye_aspect_ratio(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def extract_features(frame_bgr, face_lm):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb)
    result = face_lm.detect(mp_img)
    if not result.face_landmarks:
        return None
    pts = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)

    nose, chin   = pts[1],  pts[199]
    l_ear, r_ear = pts[234], pts[454]
    forehead     = pts[10]
    pitch = float(np.degrees(np.arctan2(chin[1] - forehead[1], abs(chin[2] - forehead[2]) + 1e-6)))
    yaw   = float(np.degrees(np.arctan2(r_ear[0] - l_ear[0],  abs(r_ear[2] - l_ear[2]) + 1e-6)))
    roll  = float(np.degrees(np.arctan2(r_ear[1] - l_ear[1],  abs(r_ear[0] - l_ear[0]) + 1e-6)))

    l_eye_pts = pts[LM_GROUPS["left_eye"]]
    r_eye_pts = pts[LM_GROUPS["right_eye"]]
    ear_l = eye_aspect_ratio(l_eye_pts)
    ear_r = eye_aspect_ratio(r_eye_pts)
    l_brow_y = pts[LM_GROUPS["left_brow"]][:, 1].mean()
    r_brow_y = pts[LM_GROUPS["right_brow"]][:, 1].mean()
    brow_raise_l = float(l_eye_pts[:, 1].mean() - l_brow_y)
    brow_raise_r = float(r_eye_pts[:, 1].mean() - r_brow_y)
    furrow_dist  = float(abs(pts[336][0] - pts[107][0]))

    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(pts[:, 0].min() * w) - 15)
    y1 = max(0, int(pts[:, 1].min() * h) - 15)
    x2 = min(w, int(pts[:, 0].max() * w) + 15)
    y2 = min(h, int(pts[:, 1].max() * h) + 15)

    return {
        "landmarks":     pts,
        "head_pose":     np.array([pitch, yaw, roll], dtype=np.float32),
        "eye_features":  np.array([ear_l, ear_r, (ear_l + ear_r) / 2], dtype=np.float32),
        "brow_features": np.array([brow_raise_l, brow_raise_r,
                                    (brow_raise_l + brow_raise_r) / 2, furrow_dist], dtype=np.float32),
        "face_bbox":     (x1, y1, x2, y2),
    }

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

class InferencePipeline:
    def __init__(self):
        download_face_model()
        self.face_lm    = make_face_landmarker(vision.RunningMode.IMAGE)
        self.expr_sess  = load_onnx("expression_cnn.onnx")
        self.hm_sess    = load_onnx("head_movement_tcn.onnx")
        self.eb_sess    = load_onnx("eyebrow_mlp.onnx")
        self.pose_buf   = deque(maxlen=SEQ_LEN)
        self.has_models = any([self.expr_sess, self.hm_sess, self.eb_sess])
        print(f"Pipeline ready. ONNX models loaded: {self.has_models}")

    def infer(self, frame_bgr):
        feat = extract_features(frame_bgr, self.face_lm)
        if feat is None:
            return {"face": False}

        out = {"face": True, "landmarks": feat["landmarks"].tolist(),
               "head_pose": feat["head_pose"].tolist(),
               "face_bbox": list(feat["face_bbox"])}

        self.pose_buf.append(feat["head_pose"])

        # ── Expression ───────────────────────────────────────────────────────
        if self.expr_sess:
            x1, y1, x2, y2 = feat["face_bbox"]
            if x2 > x1 and y2 > y1:
                crop = frame_bgr[y1:y2, x1:x2]
                crop = cv2.resize(crop, (112, 112))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                crop = ((crop - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
                logits = self.expr_sess.run(["expr_logits"], {"face_image": crop})[0][0]
                probs  = softmax(logits)
                cls    = int(probs.argmax())
                out["expression"] = {"label": EXPR_NAMES[cls], "conf": float(probs[cls]),
                                      "probs": probs.tolist(), "names": EXPR_NAMES}
        else:
            # Demo mode: use head/brow heuristics to pick expression
            ear = float(feat["eye_features"][2])
            br  = float(feat["brow_features"][2])
            fur = float(feat["brow_features"][3])
            if br > 0.075 and ear > 0.28:
                label, conf = "surprise", 0.72
            elif fur < 0.11 or ear < 0.22:
                label, conf = "disgust", 0.65
            else:
                label, conf = "neutral", 0.80
            probs = [0.05] * 8
            probs[EXPR_NAMES.index(label)] = conf
            out["expression"] = {"label": label, "conf": conf,
                                  "probs": probs, "names": EXPR_NAMES}

        # ── Head movement ─────────────────────────────────────────────────────
        if len(self.pose_buf) == SEQ_LEN:
            seq = np.stack(list(self.pose_buf))
            seq = (seq - seq.mean(0)) / (seq.std(0) + 1e-6)
            if self.hm_sess:
                inp = seq[None].astype(np.float32)
                logits = self.hm_sess.run(["hm_logits"], {"pose_seq": inp})[0][0]
                probs  = softmax(logits)
            else:
                std_p, std_y, std_r = seq[:,0].std(), seq[:,1].std(), seq[:,2].std()
                total = std_p + std_y + std_r
                if total < 0.3:
                    idx = 4
                else:
                    idx = int(np.argmax([std_p, std_y, std_r, 0, 0]))
                probs = np.full(5, 0.05); probs[idx] = 0.75
            cls = int(probs.argmax())
            out["head_movement"] = {"label": HM_NAMES[cls], "conf": float(probs[cls]),
                                     "probs": probs.tolist(), "names": HM_NAMES}

        # ── Eye / brow ────────────────────────────────────────────────────────
        eb_vec = np.concatenate([feat["eye_features"], feat["brow_features"]])[None].astype(np.float32)
        if self.eb_sess:
            logits = self.eb_sess.run(["eb_logits"], {"eb_feats": eb_vec})[0][0]
            probs  = softmax(logits)
        else:
            ear = float(feat["eye_features"][2])
            br  = float(feat["brow_features"][2])
            fur = float(feat["brow_features"][3])
            if ear > 0.30 and br > 0.07:
                probs = np.array([0.05, 0.85, 0.10])
            elif ear < 0.25 or fur < 0.11:
                probs = np.array([0.05, 0.10, 0.85])
            else:
                probs = np.array([0.80, 0.10, 0.10])
        cls = int(probs.argmax())
        out["eye_brow"] = {"label": EB_NAMES[cls], "conf": float(probs[cls]),
                            "probs": probs.tolist(), "names": EB_NAMES}

        # ── Grammar ───────────────────────────────────────────────────────────
        grammar_hits = []
        for key in ["expression", "head_movement", "eye_brow"]:
            if key in out:
                g = GRAMMAR.get(out[key]["label"], "")
                if g and g not in grammar_hits:
                    grammar_hits.append(g)
        out["grammar"] = grammar_hits

        # ── Pose buffer length for UI progress indicator ─────────────────────
        out["pose_buf_len"] = len(self.pose_buf)

        # ── Raw feature values for the dashboard ─────────────────────────────
        out["raw"] = {
            "pitch":       round(float(feat["head_pose"][0]), 2),
            "yaw":         round(float(feat["head_pose"][1]), 2),
            "roll":        round(float(feat["head_pose"][2]), 2),
            "ear":         round(float(feat["eye_features"][2]), 4),
            "brow_raise":  round(float(feat["brow_features"][2]), 4),
            "furrow":      round(float(feat["brow_features"][3]), 4),
        }
        return out

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
pipeline = None   # initialised lazily on first request

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline()
    return pipeline

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ISL NMF · Non-Manual Feature Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #080c10;
    --surface:   #0e1318;
    --border:    #1e2832;
    --accent:    #00e5a0;
    --accent2:   #0090ff;
    --accent3:   #ff5c5c;
    --text:      #d4dce8;
    --muted:     #5a6878;
    --card:      #111820;
    --glow:      rgba(0,229,160,.15);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; font-size: 13px; overflow-x: hidden; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }

  /* ── grid layout ── */
  .shell { display: grid; grid-template-rows: 56px 1fr; min-height: 100vh; }
  header { display: flex; align-items: center; gap: 16px; padding: 0 28px; border-bottom: 1px solid var(--border); background: var(--surface); position: sticky; top: 0; z-index: 99; }
  .logo-mark { width: 28px; height: 28px; background: var(--accent); clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%); flex-shrink: 0; }
  .logo-text { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 15px; letter-spacing: .02em; color: #fff; }
  .logo-text span { color: var(--accent); }
  .badge { margin-left: auto; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; padding: 4px 10px; border: 1px solid var(--border); border-radius: 3px; color: var(--muted); }

  main { display: grid; grid-template-columns: 1fr 340px; gap: 0; height: calc(100vh - 56px); }

  /* ── left: camera panel ── */
  .cam-panel { display: flex; flex-direction: column; border-right: 1px solid var(--border); }
  .cam-toolbar { display: flex; align-items: center; gap: 12px; padding: 14px 20px; border-bottom: 1px solid var(--border); background: var(--surface); }
  .cam-toolbar h2 { font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); flex: 1; }
  .btn { cursor: pointer; border: none; background: none; font-family: 'DM Mono', monospace; font-size: 12px; padding: 7px 16px; border-radius: 4px; transition: all .18s; letter-spacing: .04em; }
  .btn-primary { background: var(--accent); color: #080c10; font-weight: 500; }
  .btn-primary:hover { background: #00ffc0; }
  .btn-danger  { background: transparent; border: 1px solid var(--accent3); color: var(--accent3); }
  .btn-danger:hover { background: var(--accent3); color: #fff; }
  .btn-ghost   { background: transparent; border: 1px solid var(--border); color: var(--text); }
  .btn-ghost:hover { border-color: var(--accent); color: var(--accent); }

  .cam-wrap { flex: 1; position: relative; display: flex; align-items: center; justify-content: center; background: #05080b; overflow: hidden; }
  #videoEl  { max-width: 100%; max-height: 100%; object-fit: contain; transform: scaleX(-1); }
  #canvasEl { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; transform: scaleX(-1); }
  .cam-placeholder { display: flex; flex-direction: column; align-items: center; gap: 16px; color: var(--muted); }
  .cam-placeholder svg { opacity: .3; }
  .cam-placeholder p { font-size: 12px; letter-spacing: .06em; }

  .upload-row { padding: 14px 20px; border-top: 1px solid var(--border); background: var(--surface); display: flex; align-items: center; gap: 12px; }
  .upload-row label { font-size: 11px; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
  #fileInput { display: none; }
  .file-btn { cursor: pointer; padding: 6px 14px; border: 1px dashed var(--border); border-radius: 4px; font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); transition: all .18s; }
  .file-btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ── right: results panel ── */
  .res-panel { display: flex; flex-direction: column; overflow-y: auto; background: var(--surface); }
  .section { padding: 18px 20px; border-bottom: 1px solid var(--border); }
  .section-title { font-family: 'Syne', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }

  /* status dot */
  .status-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
  .dot.live { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 1.4s ease infinite; }
  .dot.warn { background: #ffae00; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .status-text { font-size: 11px; color: var(--muted); }

  /* prediction cards */
  .pred-card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; margin-bottom: 10px; position: relative; overflow: hidden; transition: border-color .2s; }
  .pred-card.active { border-color: var(--accent); }
  .pred-card::before { content: ''; position: absolute; inset: 0; background: var(--glow); opacity: 0; transition: opacity .3s; pointer-events: none; }
  .pred-card.active::before { opacity: 1; }
  .pred-label { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 18px; color: #fff; line-height: 1; margin-bottom: 4px; }
  .pred-sub { font-size: 10px; color: var(--muted); letter-spacing: .06em; margin-bottom: 10px; }
  .pred-grammar { font-size: 10px; color: var(--accent); margin-bottom: 10px; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  .bar-name { width: 96px; font-size: 10px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
  .bar-track { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 2px; background: var(--accent2); transition: width .35s ease; }
  .bar-fill.top { background: var(--accent); }
  .bar-val { width: 36px; text-align: right; font-size: 10px; color: var(--muted); }
  .conf-badge { position: absolute; top: 12px; right: 12px; font-size: 10px; font-weight: 500; color: var(--accent); }

  /* raw metrics */
  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .metric { background: var(--card); border: 1px solid var(--border); border-radius: 5px; padding: 10px 12px; }
  .metric-name { font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
  .metric-val  { font-family: 'DM Mono', monospace; font-size: 16px; color: #fff; }
  .metric-unit { font-size: 9px; color: var(--muted); }

  /* grammar strip */
  .grammar-strip { display: flex; flex-wrap: wrap; gap: 6px; }
  .grammar-tag { font-size: 10px; padding: 4px 10px; border-radius: 3px; border: 1px solid var(--accent); color: var(--accent); letter-spacing: .04em; background: rgba(0,229,160,.06); }

  /* landmark canvas overlay */
  .lm-dot { fill: rgba(0,229,160,.7); }

  /* history log */
  .log { font-size: 11px; color: var(--muted); max-height: 130px; overflow-y: auto; line-height: 1.8; }
  .log-entry { display: flex; gap: 8px; }
  .log-time { color: var(--accent); width: 56px; flex-shrink: 0; }
  .log-msg { color: var(--text); }

  /* no-face state */
  .no-face { text-align: center; padding: 20px 0; color: var(--muted); font-size: 11px; }
  .no-face-icon { font-size: 28px; margin-bottom: 8px; opacity: .4; }
  
  /* mode tabs */
  .tabs { display: flex; gap: 0; }
  .tab { flex: 1; padding: 9px 0; text-align: center; font-size: 11px; letter-spacing: .06em; cursor: pointer; border-bottom: 2px solid transparent; color: var(--muted); transition: all .18s; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* image preview */
  #imgPreview { max-width: 100%; max-height: 360px; object-fit: contain; border-radius: 4px; display: none; }

  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>
<div class="shell">

<header>
  <div class="logo-mark"></div>
  <div class="logo-text">ISL <span>NMF</span> Detector</div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:12px;">
    <span id="modeBadge" class="badge">LANDMARK ONLY</span>
    <span class="badge">Indian Sign Language</span>
  </div>
</header>

<main>

  <!-- ── Left: camera / upload ── -->
  <div class="cam-panel">
    <div class="cam-toolbar">
      <h2>Input</h2>
      <div class="tabs" style="flex:1;max-width:220px;">
        <div class="tab active" id="tabCam" onclick="switchMode('cam')">Webcam</div>
        <div class="tab" id="tabImg" onclick="switchMode('img')">Upload</div>
      </div>
      <button class="btn btn-primary" id="startBtn" onclick="startCam()">Start Camera</button>
      <button class="btn btn-danger"  id="stopBtn"  onclick="stopCam()" style="display:none">Stop</button>
    </div>

    <div class="cam-wrap" id="camWrap">
      <div class="cam-placeholder" id="placeholder">
        <svg width="64" height="64" viewBox="0 0 64 64" fill="none"><rect x="8" y="16" width="48" height="36" rx="4" stroke="#5a6878" stroke-width="2"/><circle cx="32" cy="34" r="10" stroke="#5a6878" stroke-width="2"/><circle cx="32" cy="34" r="4" fill="#5a6878"/><path d="M22 16l4-6h12l4 6" stroke="#5a6878" stroke-width="2"/></svg>
        <p>Click "Start Camera" or upload an image</p>
      </div>
      <video id="videoEl" autoplay playsinline style="display:none"></video>
      <canvas id="canvasEl"></canvas>
      <img id="imgPreview" alt="Uploaded frame">
    </div>

    <div class="upload-row" id="uploadRow" style="display:none">
      <label>Upload image</label>
      <input type="file" id="fileInput" accept="image/*" onchange="handleFile(event)">
      <div class="file-btn" onclick="document.getElementById('fileInput').click()">Choose file…</div>
      <span id="fileName" style="font-size:11px;color:var(--muted);flex:1;"></span>
      <button class="btn btn-ghost" id="analyseBtn" style="display:none" onclick="analyseImage()">Analyse →</button>
    </div>
  </div>

  <!-- ── Right: results ── -->
  <div class="res-panel">

    <div class="section">
      <div class="section-title">Status</div>
      <div class="status-row">
        <div class="dot" id="statusDot"></div>
        <span class="status-text" id="statusText">Idle — start camera or upload image</span>
      </div>
      <div class="status-row" style="margin-bottom:0">
        <div class="dot" id="fpsDot" style="background:var(--border)"></div>
        <span class="status-text" id="fpsText">—</span>
      </div>
    </div>

    <!-- Expression -->
    <div class="section">
      <div class="section-title">Expression</div>
      <div id="exprCard" class="pred-card">
        <div class="no-face"><div class="no-face-icon">👤</div>No face detected</div>
      </div>
    </div>

    <!-- Head movement -->
    <div class="section">
      <div class="section-title">Head Movement</div>
      <div id="hmCard" class="pred-card">
        <div class="no-face"><div class="no-face-icon">↕</div>Collecting frames…</div>
      </div>
    </div>

    <!-- Eye / brow -->
    <div class="section">
      <div class="section-title">Eye / Brow</div>
      <div id="ebCard" class="pred-card">
        <div class="no-face"><div class="no-face-icon">👁</div>No face detected</div>
      </div>
    </div>

    <!-- ISL Grammar -->
    <div class="section">
      <div class="section-title">ISL Grammar Function</div>
      <div class="grammar-strip" id="grammarStrip">
        <span style="font-size:11px;color:var(--muted)">—</span>
      </div>
    </div>

    <!-- Raw metrics -->
    <div class="section">
      <div class="section-title">Raw Features</div>
      <div class="metrics-grid" id="metricsGrid">
        <div class="metric"><div class="metric-name">Pitch</div><div class="metric-val" id="mPitch">—</div></div>
        <div class="metric"><div class="metric-name">Yaw</div><div class="metric-val" id="mYaw">—</div></div>
        <div class="metric"><div class="metric-name">Roll</div><div class="metric-val" id="mRoll">—</div></div>
        <div class="metric"><div class="metric-name">EAR</div><div class="metric-val" id="mEar">—</div></div>
        <div class="metric"><div class="metric-name">Brow Raise</div><div class="metric-val" id="mBrow">—</div></div>
        <div class="metric"><div class="metric-name">Furrow</div><div class="metric-val" id="mFurrow">—</div></div>
      </div>
    </div>

    <!-- Log -->
    <div class="section" style="border-bottom:none">
      <div class="section-title">Activity Log</div>
      <div class="log" id="actLog"></div>
    </div>

  </div>
</main>
</div>

<script>
let stream = null, rafId = null, mode = 'cam';
let lastExpr = '', lastHM = '', lastEB = '';
let frameCount = 0, fpsTimer = 0, fps = 0;
let imageBlob = null;

const video    = document.getElementById('videoEl');
const canvas   = document.getElementById('canvasEl');
const ctx      = canvas.getContext('2d');
const imgPrev  = document.getElementById('imgPreview');
const placeholder = document.getElementById('placeholder');

function switchMode(m) {
  mode = m;
  document.getElementById('tabCam').classList.toggle('active', m === 'cam');
  document.getElementById('tabImg').classList.toggle('active', m === 'img');
  document.getElementById('startBtn').style.display = m === 'cam' ? '' : 'none';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('uploadRow').style.display = m === 'img' ? '' : 'none';
  if (m === 'cam') stopCam();
  else { stopCam(); placeholder.style.display = 'flex'; }
}

async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480, facingMode:'user' } });
    video.srcObject = stream;
    video.style.display = '';
    placeholder.style.display = 'none';
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = '';
    setStatus('live', 'Live · Webcam active');
    loop();
  } catch(e) {
    setStatus('warn', 'Camera access denied');
    log('Error: ' + e.message);
  }
}

function stopCam() {
  cancelAnimationFrame(rafId); rafId = null;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.style.display = 'none';
  document.getElementById('startBtn').style.display = '';
  document.getElementById('stopBtn').style.display = 'none';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setStatus('idle', 'Idle');
}

function loop() {
  rafId = requestAnimationFrame(loop);
  if (video.readyState < 2) return;
  const now = performance.now();
  frameCount++;
  if (now - fpsTimer > 1000) {
    fps = Math.round(frameCount * 1000 / (now - fpsTimer));
    frameCount = 0; fpsTimer = now;
    document.getElementById('fpsText').textContent = fps + ' fps  ·  inference running';
    document.getElementById('fpsDot').style.background = fps > 10 ? 'var(--accent)' : '#ffae00';
  }
  // Throttle inference to ~10fps to keep UI smooth
  if (frameCount % 3 !== 0) return;
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  canvas.toBlob(blob => sendFrame(blob), 'image/jpeg', 0.7);
}

function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  imageBlob = file;
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('analyseBtn').style.display = '';
  const url = URL.createObjectURL(file);
  imgPrev.src = url; imgPrev.style.display = ''; placeholder.style.display = 'none';
  canvas.style.display = 'none';
}

function analyseImage() {
  if (!imageBlob) return;
  setStatus('live', 'Analysing image…');
  sendFrame(imageBlob, true);
}

let inferring = false;
async function sendFrame(blob, isImage=false) {
  if (inferring) return;
  inferring = true;
  try {
    const form = new FormData();
    form.append('frame', blob, 'frame.jpg');
    const res  = await fetch('/infer', { method:'POST', body: form });
    const data = await res.json();
    updateUI(data, isImage);
  } catch(e) {
    console.warn('infer error', e);
  } finally {
    inferring = false;
  }
}

function updateUI(d, isImage=false) {
  if (!d.face) {
    document.getElementById('statusDot').className = 'dot warn';
    if (!stream) setStatus('warn', 'No face detected in image');
    drawLandmarks([]);
    return;
  }
  document.getElementById('statusDot').className = 'dot live';
  if (isImage) setStatus('live', 'Face detected · Analysis complete');

  drawLandmarks(d.landmarks || []);

  if (d.expression) {
    renderPredCard('exprCard', d.expression, d.expression.label !== lastExpr);
    if (d.expression.label !== lastExpr) { log('Expression: ' + d.expression.label); lastExpr = d.expression.label; }
  }
  if (d.head_movement) {
    renderPredCard('hmCard', d.head_movement, d.head_movement.label !== lastHM);
    if (d.head_movement.label !== lastHM) { log('Head: ' + d.head_movement.label); lastHM = d.head_movement.label; }
  } else {
    const bufLen = d.pose_buf_len || 0;
    document.getElementById('hmCard').innerHTML = '<div class="no-face"><div class="no-face-icon">↕</div>Collecting frames… ' + bufLen + '/30</div>';
  }
  if (d.eye_brow) {
    renderPredCard('ebCard', d.eye_brow, d.eye_brow.label !== lastEB);
    if (d.eye_brow.label !== lastEB) { log('Eye/Brow: ' + d.eye_brow.label); lastEB = d.eye_brow.label; }
  }

  const gs = document.getElementById('grammarStrip');
  if (d.grammar && d.grammar.length) {
    gs.innerHTML = d.grammar.map(g => `<span class="grammar-tag">${g}</span>`).join('');
  } else {
    gs.innerHTML = '<span style="font-size:11px;color:var(--muted)">—</span>';
  }

  if (d.raw) {
    document.getElementById('mPitch').textContent  = d.raw.pitch + '°';
    document.getElementById('mYaw').textContent    = d.raw.yaw + '°';
    document.getElementById('mRoll').textContent   = d.raw.roll + '°';
    document.getElementById('mEar').textContent    = d.raw.ear;
    document.getElementById('mBrow').textContent   = d.raw.brow_raise;
    document.getElementById('mFurrow').textContent = d.raw.furrow;
  }
}

function renderPredCard(id, pred, changed) {
  const top = pred.probs.indexOf(Math.max(...pred.probs));
  const rows = pred.names.map((n, i) => {
    const pct = Math.round(pred.probs[i] * 100);
    return `<div class="bar-row">
      <div class="bar-name">${n}</div>
      <div class="bar-track"><div class="bar-fill ${i===top?'top':''}" style="width:${pct}%"></div></div>
      <div class="bar-val">${pct}%</div>
    </div>`;
  }).join('');
  document.getElementById(id).innerHTML = `
    <div class="pred-label">${pred.label}</div>
    <div class="conf-badge">${Math.round(pred.conf*100)}%</div>
    <div style="height:8px"></div>
    ${rows}`;
  if (changed) {
    const card = document.getElementById(id);
    card.classList.add('active');
    setTimeout(() => card.classList.remove('active'), 800);
  }
}

const W = 640, H = 480;
function drawLandmarks(lms) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!lms || !lms.length) return;
  const cw = canvas.width || W, ch = canvas.height || H;
  ctx.fillStyle = 'rgba(0,229,160,0.55)';
  lms.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x * cw, y * ch, 1.2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function setStatus(type, text) {
  const dot  = document.getElementById('statusDot');
  const span = document.getElementById('statusText');
  dot.className = 'dot ' + (type === 'live' ? 'live' : type === 'warn' ? 'warn' : '');
  span.textContent = text;
}

function log(msg) {
  const el = document.getElementById('actLog');
  const now = new Date();
  const t = now.toTimeString().slice(0, 8);
  const div = document.createElement('div');
  div.className = 'log-entry';
  div.innerHTML = `<span class="log-time">${t}</span><span class="log-msg">${msg}</span>`;
  el.insertBefore(div, el.firstChild);
  if (el.children.length > 40) el.removeChild(el.lastChild);
}

// Init
switchMode('cam');
document.getElementById('modeBadge').textContent = '{{ mode }}';
</script>
</body>
</html>
"""

@app.route("/")
def index():
    mode_text = "ONNX MODELS" if (HAS_ORT and any([
        (MODEL_DIR / "expression_cnn.onnx").exists(),
        (MODEL_DIR / "head_movement_tcn.onnx").exists(),
    ])) else "LANDMARK ONLY"
    return render_template_string(HTML, mode=mode_text)

@app.route("/infer", methods=["POST"])
def infer():
    if "frame" not in request.files:
        return jsonify({"error": "no frame"}), 400
    data = request.files["frame"].read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"face": False})
    result = get_pipeline().infer(img)
    return jsonify(result)

@app.route("/status")
def status():
    p = get_pipeline()
    return jsonify({
        "onnx_available": HAS_ORT,
        "models": {
            "expression":    p.expr_sess is not None,
            "head_movement": p.hm_sess is not None,
            "eye_brow":      p.eb_sess is not None,
        },
        "platform": platform.system(),
        "python":   platform.python_version(),
    })

if __name__ == "__main__":
    print("\n" + "="*54)
    print("  ISL NMF Detection — Web Interface")
    print("="*54)
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  ONNX RT  : {'yes' if HAS_ORT else 'no (pip install onnxruntime)'}")
    print(f"  Models   : {MODEL_DIR}")
    print(f"  Open     : http://localhost:5050")
    print("="*54 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
