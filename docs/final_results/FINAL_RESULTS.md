# Final experimental results — soccer penalty direction prediction

## Scope and provenance

Exploratory **binary left/right** prediction from pre-contact video, with labels defined from the goalkeeper’s perspective. The available dataset contained only one center kick, so these results **do not validate three-class prediction**.

- Dataset: 119 eligible kicks (66 left, 53 right).
- Evaluation: five grouped cross-validation folds, pooled out-of-fold predictions.
- Pre-contact horizons: 1.0, 0.5 and 0.2 seconds.
- Compared: majority, pose logistic regression, MobileNetV3-Small + GRU, and single-frame MobileNetV3-Small.
- ImageNet-pretrained frozen visual encoder; 15 training epochs per fold for the paired neural comparison.
- Source-code commit at publication: `47a9824`.
- Original raw videos, processed clips and large model checkpoints are not published in this lightweight results folder.

## Pooled five-fold results

| Horizon | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| 1.0s | majority | 55.46% | 35.68% |
| 1.0s | pose logistic | 36.97% | 36.69% |
| 1.0s | gru | 45.38% | 44.61% |
| 1.0s | single frame | 47.90% | 46.22% |
| 0.5s | majority | 55.46% | 35.68% |
| 0.5s | pose logistic | 57.14% | 57.13% |
| 0.5s | gru | 50.42% | 50.29% |
| 0.5s | single frame | 53.78% | 53.31% |
| 0.2s | majority | 55.46% | 35.68% |
| 0.2s | pose logistic | 63.87% | 63.36% |
| 0.2s | gru | 53.78% | 51.46% |
| 0.2s | single frame | 51.26% | 49.99% |

## Paired GRU minus single-frame differences

| Horizon | Accuracy difference (percentage points) | Macro-F1 difference (percentage points) |
|---|---:|---:|
| 1.0s | -2.52 | -1.61 |
| 0.5s | -3.36 | -3.01 |
| 0.2s | +2.52 | +1.48 |

## Interpretation and limitations

The pose-logistic baseline performed best at 0.2s (63.87% accuracy, 63.36% macro-F1). The GRU exceeded the single-frame model only at 0.2s, by approximately 2.52 accuracy percentage points and 1.48 macro-F1 percentage points; this is not evidence of statistical significance.

The GRU and single-frame heads have different capacities, so this is a comparison of complete model configurations rather than a controlled isolation of temporal information. The small dataset, uncertain generalization to new players/matches, and lack of a viable center class limit conclusions. Model timing in the paired experiment includes decoding 16 frames and must not be described as network-only inference latency.

## Files

`summary.json`, `comparison.csv`, `folds.json`, `run_config.json`, `macro_f1_comparison.png`, and `SHA256SUMS.txt` are published alongside this report. The full 283 MB experiment folder, including fold-level predictions and checkpoints, remains local.

