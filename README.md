# SmartRupee – AI Currency Authenticity Scanner

Flask web app that screens a single banknote photo: it aligns the note, inspects the four corners plus central design regions against verified references (ORB + SSIM), runs a MobileNetV2 genuine/counterfeit classifier, and reports GREEN / RED / YELLOW per region and an overall verdict (LIKELY GENUINE, POTENTIALLY COUNTERFEIT, INCONCLUSIVE). No denomination detection.

**Important:** it is a visual screening aid. Tactile, UV, infrared, watermark and colour-shift features need physical verification. Nothing is hard-coded: with no references or model, the app says so and returns INCONCLUSIVE instead of guessing.

## Setup (Windows + VS Code)
1. Install Python 3.10–3.12 (tick "Add Python to PATH") and VS Code with the Python extension.
2. `File > Open Folder` → this folder. Open a terminal (`Ctrl+``), PowerShell:
```
python -m venv venv
venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```
3. In VS Code press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `venv`.

## Data you must provide
- `data/references/` – verified genuine note photos (needed for ORB/SSIM region checks).
- `dataset/{train,validation,test}/{genuine,fake}/` – verified, labelled images needed for training, validation and held-out evaluation. Keep each physical note in one split only.

## Train, evaluate, run
```
python train.py          # saves models/currency_model.keras + class_names.json
python evaluate.py       # held-out test metrics -> reports/evaluation.json
python app.py            # open http://127.0.0.1:5000
```
The dashboard shows the test accuracy only after `evaluate.py` has produced it. Camera capture works on `localhost`; remote devices need HTTPS.

## How it works
`smartrupee/engine.py`: quality check (blur/brightness/glare/resolution) → denoise → note detection and perspective warp to 1200×520 → region extraction with visibility checks → ORB (ratio test + RANSAC inliers) and SSIM against each reference (best match kept; test crop is homography-aligned when possible) → difference maps → MobileNetV2 → verdict rules. The scan animation follows the backend's real stage via `/api/status/<id>`.

## Tuning and limits
- Thresholds in `CFG` (engine.py) are conservative starting heuristics, not validated values. Calibrate them with your labelled data and check them with `evaluate.py` before relying on results.
- Region positions are fractions of the aligned note (`REGIONS`); adjust them to your note design.
- Photograph the note upright (not upside-down) on a plain contrasting background; 180° rotation is not auto-corrected.
- Never treat a result as the sole basis for accepting or rejecting a note; report suspected counterfeits to a bank or the police.
