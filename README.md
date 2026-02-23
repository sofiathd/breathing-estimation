# Breathing Measurement From Video Using Dense Point Tracking

This project extracts a breathing waveform from video frames and evaluates it against **COSMED** ground-truth respiration data.  
It supports running batches of experiments (camera × subject × take) with multiple ROI modes and optional PCA aggregation strategy versus a median vertical displacement strategy.

---

## What this project does

Given a dataset of extracted video frames (per subject/take/camera), the pipeline:

1. Detects or builds an **ROI** (chest / abdomen / segmentation-based)
2. Tracks motion using **CoTracker**
3. Converts motion into a breathing proxy waveform
4. Filters + estimates respiratory rate (RR)
5. Aligns and compares against **COSMED** respiration
6. Calibrates proxy -> tidal volume (VT) using linear regression + blocked cross-validation
7. Saves time-series, breath-level metrics, plots, and a global summary report

---

## Key features

- Batch pipeline runner (`run_pipeline.py`)
- Multiple ROI modes:
  - `chest`
  - `abdomen`
  - `segmented` (DeepLab-based)
- Optional PCA stabilization mode
- Automatic evaluation against COSMED ground truth
- Robust output folder structure
- Resume support (skips finished experiments)
- Global report generation with snapshots

---

## Project structure

```bash
breathing-ma3/
  run_pipeline.py
  config.yaml
  requirements.txt
  README.md
  src/
    pipeline/
      batch.py
      specs.py
    estimator/
      cotracker_rf.py
    cv/
      roi_detection.py
      image.py
    signals/
      preprocess.py
      rr.py
      quality.py
      events.py
      plots.py
    vt_calib/
      linear.py
      blocked_cv.py
    cosmed.py
    io_utils.py
    models.py
```
---

## Quickstart

Run the pipeline:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

python run_pipeline.py --config config.yaml
```

Notes:
- Edit `config.yaml` to point `paths.frames_root_base`, `paths.results_dir`, and `paths.cosmed` to your local data locations.
- This repository assumes OpenPifPaf is available for ROI extraction.

---

## COSMED Ground Truth

COSMED signals are loaded from an Excel file per participant.
