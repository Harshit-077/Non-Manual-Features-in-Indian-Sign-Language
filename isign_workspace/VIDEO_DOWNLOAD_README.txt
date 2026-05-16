
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
