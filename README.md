# Soccer Penalty Direction Prediction

This project predicts kick direction from the kicker's movement **before ball contact**. The intended output is left, center, or right from the goalkeeper's perspective. The core model uses a frozen pretrained MobileNetV3-Small image encoder, a GRU, and a classification head. The comparison models are a training-fold majority classifier and pose keypoints with logistic regression.

## Important dataset finding

The [Mendeley version 2 release](https://data.mendeley.com/datasets/brx9bsxnpx/2) contains 11,140 CSV rows, but only **one** unique kick labeled center. The released CSV has 132 unique `kick_id` values for 133 videos: one `15-3` sequence was labeled `16-3`, and `4-1` has duplicate tracking rows. The indexer records these repairs and rejects rows without a single final `last_touch=1`, mismatched video metadata, or invalid boxes. In the local audit, 126 kicks indexed and 120 had all three 16-frame windows (66 left, 53 right, 1 center). Counts may change after any manual corrections.

Five-fold three-class evaluation is therefore **not valid with this release alone**. The experiment command stops with an explicit error. `--binary` runs an exploratory left/right experiment and excludes the single center kick. Obtain and label additional center kicks before claiming a three-class result. SoccerNet is a later data-expansion option, not a dependency of this pipeline.

## Setup

Use Python 3.10 or newer. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the downloaded Mendeley archive into the repository or pass its full path. Source data, generated clips, metrics, and checkpoints are ignored by Git.

```bash
python -m src.data.unpack --archive "Women’s Soccer Penalty Kick Data Captured by Compu.zip"
python -m src.data.manifest --csv data/raw/penalty_pose_keypoints.csv \
  --videos data/raw/videos --output data/processed/manifest.jsonl
python -m src.data.audit --manifest data/processed/manifest.jsonl \
  --output data/processed/contact_audit.jpg
python -m src.data.preprocess --manifest data/processed/manifest.jsonl \
  --output data/processed/clips
```

Review `data/processed/contact_audit.jpg` against the source videos before accepting contact and direction labels. The audit JSON lists excluded records and CSV repairs. If a different release uses different column names, supply `--columns columns.json`; if video names differ, supply `--video-map video_map.json` mapping kick IDs to paths under `--videos`.

Batch cropping creates 16-frame, 224 × 224 MP4 clips for 1.0, 0.5, and 0.2 seconds before contact. It pads crops at image edges, logs per-video failures in `crop_report.json`, and skips valid completed clips on rerun. `clips.jsonl` includes only kicks available at all three horizons, so comparisons use the same kicks. Neither the contact frame nor any later frame enters a model clip.

## Experiments

On the NVIDIA machine, copy the repository and ignored `data/processed/` files, install requirements, and check CUDA with `python -c "import torch; print(torch.cuda.is_available())"`. Then run the current **binary exploratory** study:

```bash
python -m src.experiments --device cuda --binary \
  --output results/metrics/binary_main
```

The default uses five shared player/match-grouped folds, three horizons, 15 epochs per fold, pretrained MobileNetV3-Small weights, and a final model trained on all eligible kicks for the median best epoch count. Initial pretrained weights may need downloading on the GPU machine. Every fold writes `last.pt` after each completed epoch, plus `best.pt` based on inner-validation macro-F1. Final training also saves a resumable `final_<horizon>_last.pt` and deployable `final_<horizon>.pt`.

To continue after an interruption, run the **same command** with `--resume`:

```bash
python -m src.experiments --device cuda --binary \
  --output results/metrics/binary_main --resume
```

Resume validates the fold, class order, data IDs, seed, batch size, and learning rate before restoring model, optimizer, scheduler, and random states. It starts at the next epoch; work in an unfinished epoch is repeated. Use a separate output directory for changed experiments. `--fold 0 --skip-final` restricts a trial to one fold. `--no-pretrained` is only for offline smoke tests; it is not the intended scientific run. `--baselines-only --binary --device cpu` runs the two lightweight baselines locally.

`results/metrics/.../summary.json` contains pooled cross-validation metrics and confusion matrices for each horizon. Batch-one *model* latency in this file includes clip loading and model inference. The separate prediction command reports full local latency including decoding and cropping. Do not compare those as if they measure the same operation.

## Local CPU inference

Copy a deployable checkpoint for the desired horizon from the GPU machine to the local machine, for example with `scp` or removable storage. The raw input clip must **already end at that horizon before ball contact**; the command uses its last 16 frames. The tool cannot verify this timing from an arbitrary video.

To make an uncropped clip from an indexed dataset video, run:

```bash
python -m src.data.trim --kick-id 3-9 --horizon 0.2 \
  --output data/processed/demo_3-9_0.2.mp4
```

```bash
python -m src.predict --video path/to/precontact.mov \
  --checkpoint path/to/final_0.2.pt --horizon 0.2 --device cpu
```

The CPU path uses YOLOv8n to find the kicker. Its weights may need to be downloaded once or copied locally; pass `--detector-weights path/to/yolov8n.pt` if needed. If multiple people cannot be distinguished reliably, the command errors and accepts `--kicker-box CX CY W H` in source-video pixels. A binary checkpoint returns only left/right probabilities. A future three-class checkpoint will return all three.

Run local checks with `pytest -q`. The tests cover contact-safe frame selection, edge crops, batch reruns, grouped splits, epoch resume, and CPU checkpoint inference without requiring the full data archive.

## Current limitations

- The contact-audit sheet is for human verification; the code cannot certify label semantics from video automatically.
- One center kick is insufficient for credible three-class training or cross-validation. The released data supports exploratory left/right experiments only.
- Automatic kicker selection requires an available YOLO weight file and can fail on crowded or ball-obscured footage. The explicit box option is the fallback.
- End-to-end CPU latency and GPU training metrics must be measured on their intended machines; the local development environment is not a substitute for the NVIDIA run.
