# Soccer Penalty Direction Prediction — Integrated Project Guide

## What is included

This package combines the two source archives supplied by the project owner and integrates a **single-frame MobileNetV3-Small** comparison into the existing **MobileNetV3-Small + GRU** pipeline. It preserves the original manifest, cropping, cross-validation, pose-logistic and majority baselines, training, checkpoints, and prediction interface. The original experiments remain accessible through `python -m src.experiments`.

**Validation performed in the build environment:** Python compilation succeeded and `pytest -q` reported **9 passed, 1 skipped**. The skipped test depends on a locally unavailable fixture or dependency; inspect the pytest output on your machine. **No training on the user's real dataset was performed**, since raw video data and processed clips were not uploaded. The source package is not a substitute for actual experimental results.

## Install without destroying the existing repository

1. Save `Soccer_Penalty_Integrated_Project.zip` to your Mac (usually `~/Downloads`).
2. In Terminal:

```bash
cd ~/Downloads
unzip -q Soccer_Penalty_Integrated_Project.zip -d Soccer_Penalty_Integrated_Project
cd Soccer_Penalty_Integrated_Project
python3 install_into_existing.py --project ~/Desktop/Soccer-Penalty-Direction-Prediction
```

The installer copies only source, tests, README, and the guide. It backs up any overwritten files in a timestamped folder **outside** the repository and never touches `data/`, `results/`, `.git/`, or local model weights. It does not delete files. Review `git diff` after installation.

## Activate the environment and verify

```bash
cd ~/Desktop/Soccer-Penalty-Direction-Prediction
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate soccer-penalty
python -m compileall -q src
pytest -q
python -m src.compare_single_frame --help
```

If libraries are missing, run `python -m pip install -r requirements.txt` in the activated environment. For the pretrained backbone, the first run may need internet access to retrieve Torchvision ImageNet weights.

## Required local dataset

These files must exist on the Mac; they are not included in the package:

- `data/processed/manifest.jsonl`
- `data/processed/clips/clips.jsonl`
- all MP4 files referenced by `clips.jsonl`

The model uses 16 pre-contact frames at each horizon, and the single-frame model selects the **last frame only**. Both models retain the original ImageNet normalization and 224x224 crop. The binary study excludes the single `center` example; it **does not establish three-class performance**.

## Optional quick smoke test (NOT a scientific result)

```bash
python -m src.compare_single_frame \
  --device mps --fold 0 --epochs 1 --batch-size 2 \
  --no-pretrained --architecture both \
  --output results/metrics/professor_smoke_v2
```

The smoke test uses randomly initialized frozen encoders. **Never report its scores as trained scientific results.**

## Full professor-requested paired comparison

```bash
python -m src.compare_single_frame \
  --device mps --epochs 15 --batch-size 4 --seed 42 \
  --architecture both \
  --output results/metrics/professor_comparison_v2
```

This trains **30 model/fold/horizon combinations** (2 architectures x 5 folds x 3 horizons) using the same common eligible kicks, same train/test folds, same within-fold validation selection, frozen ImageNet backbone, optimizer schedule, random seed, and augmentation. The GRU head has different capacity than the single-frame linear head, so the experiment tests the overall model configurations, not an isolated causal effect of sequence information. Both models use the same 16-frame video decoding path, so latency includes decoding all 16 frames and is **not pure model inference latency**.

On an 8GB MacBook Air, expect this to take significant time. The program prints epoch progress and writes checkpoints as it proceeds. If interrupted, rerun **the exact same command with `--resume` added**. Do not change seed, epochs, batch size, source manifests, or architecture when resuming in the same output directory. Use a fresh directory for a new configuration.

Outputs include:

- `run_config.json`: dataset hashes and experiment parameters
- `folds.json`: kick IDs in each group-safe fold
- `fold_results.json`: per-fold predictions for each model and baseline
- `summary.json`: pooled metrics, completeness flag, paired GRU minus single-frame differences
- `comparison.csv`: human-readable results table
- `macro_f1_comparison.png`: visualization of model and baseline macro-F1
- `models/{gru,single_frame}/h{horizon}/fold_{fold}/`: checkpoints and test metrics

**Only interpret complete five-fold results with `complete_requested_run: true` and `scientific_pretrained_weights: true`.** A run with `--fold 0` is partial, regardless of its scores.

## Predict using a trained checkpoint

The original `src.predict` entry point now supports both architecture tags and `--device mps`:

```bash
python -m src.predict \
  --video results/demo/penalty_2-2_unseen_0.2s.mp4 \
  --checkpoint results/metrics/professor_comparison_v2/models/single_frame/h0.2/fold_0/best.pt \
  --horizon 0.2 --device mps --kicker-box 320 180 160 240
```

The kicker-box numbers above are **illustrative**, not verified coordinates for your clip. Supply a valid `CX CY W H` box or omit the option to use automatic person detection (which requires Ultralytics and compatible detector weights). A fold checkpoint is for demonstration only; it is not a deployment model trained on all available kicks. The original GRU final-training function remains available.

## Research and reporting cautions

- Directions are from the **goalkeeper's perspective**.
- Keep the same pre-contact cutoff and exact preprocessing for both models.
- Player/match-connected kicks remain in the same fold, preventing those forms of leakage. Audit unknown or missing player/match identifiers; they cannot guarantee independence.
- Accuracy and macro-F1 are the primary outcomes. Report all three horizons and all baselines, including unfavorable comparisons.
- Avoid asserting a statistical advantage based solely on a few percentage points with a small dataset. Differences are paired on the same held-out kicks, but repeated fold training and the small sample limit inference.
- The previous published GRU metrics were generated separately; use the **new paired run** for the fairest direct comparison.
- Do not report single-frame accuracy until the new full training finishes.

## Cleanup performed

The distributed ZIP omits generated `__pycache__`, compiled `.pyc`, local `install.log`, the old `src/training/train.py.backup`, and redundant `.gitkeep` placeholders in populated folders. **Nothing is deleted from the Mac repository by the installer.** Keep all original data, results, demo videos, and model checkpoints. Archive stale local logs and backup files manually only after validating the new installation.
