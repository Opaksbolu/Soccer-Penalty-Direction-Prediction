"""Paired, player/match-group-safe MobileNetV3-GRU vs single-frame experiment.

Outputs are isolated from the original project and may be resumed. A run with
--no-pretrained is only a software smoke test, never a scientific result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.common import HORIZONS, read_jsonl
from src.evaluation.metrics import classification_metrics, save_json
from src.evaluation.splits import group_keys, make_folds
from src.experiments import _baseline_predictions
from src.training.train import train_fold

CLASSES = ("left", "right")
ARCHITECTURES = ("gru", "single_frame")
BASELINES = ("majority", "pose_logistic")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(output: Path, config: dict, resume: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Output contains a different experiment configuration. Use a new --output directory.")
        if not resume:
            raise FileExistsError("Run directory already exists. Use --resume or choose a new --output.")
    else:
        if any(output.iterdir()):
            raise FileExistsError("Output directory is not empty; choose a new directory.")
        save_json(path, config)


def aggregate(results: dict, output: Path, *, requested: tuple[str, ...], complete: bool, pretrained: bool) -> dict:
    summary = {}
    for horizon in HORIZONS:
        for model in (*BASELINES, *requested):
            prefix = f"{horizon:.1f}_{model}_fold_"
            matching = [v for k, v in results.items() if k.startswith(prefix)]
            if not matching:
                continue
            pooled = [p for value in matching for p in value["predictions"]]
            metric = classification_metrics([p["truth"] for p in pooled],
                                            [p["prediction"] for p in pooled], CLASSES)
            metric["folds"] = len(matching)
            metric["n_test_predictions"] = len(pooled)
            metric["complete_five_folds"] = len(matching) == 5
            if model in ARCHITECTURES:
                metric["mean_seconds_per_kick_including_decode"] = float(np.mean([
                    v["prediction_latency_seconds"] for v in matching]))
            summary[f"{horizon:.1f}_{model}"] = metric
    paired = {}
    for horizon in HORIZONS:
        a = summary.get(f"{horizon:.1f}_gru")
        b = summary.get(f"{horizon:.1f}_single_frame")
        if a and b and a["complete_five_folds"] and b["complete_five_folds"]:
            paired[f"{horizon:.1f}"] = {
                "accuracy_gru_minus_single_frame": a["accuracy"] - b["accuracy"],
                "macro_f1_gru_minus_single_frame": a["macro_f1"] - b["macro_f1"],
            }
    report = {"scientific_pretrained_weights": pretrained,
              "complete_requested_run": complete,
              "summary": summary, "paired_differences": paired,
              "caveat": "Inference timing includes decoding all 16 frames, not isolated model latency."}
    save_json(output / "summary.json", report)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon_seconds", "model", "folds", "n_test_predictions",
                         "accuracy", "macro_f1", "seconds_per_kick_including_decode"])
        for key, metric in sorted(summary.items()):
            horizon, model = key.split("_", 1)
            writer.writerow([horizon, model, metric["folds"], metric["n_test_predictions"],
                             metric["accuracy"], metric["macro_f1"],
                             metric.get("mean_seconds_per_kick_including_decode", "")])
    fig, ax = plt.subplots(figsize=(9, 5))
    models = [m for m in (*BASELINES, *requested) if any(f"{h:.1f}_{m}" in summary for h in HORIZONS)]
    if models:
        x = np.arange(len(HORIZONS))
        width = 0.8 / len(models)
        for index, model in enumerate(models):
            values = [summary.get(f"{h:.1f}_{model}", {}).get("macro_f1", np.nan) for h in HORIZONS]
            ax.bar(x - .4 + width * (index + .5), values, width, label=model)
        ax.set_xticks(x, [f"{h:.1f}s" for h in HORIZONS])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Pooled macro-F1")
        ax.set_xlabel("Seconds before contact")
        ax.legend()
        ax.set_title("Penalty direction: matched folds and horizons")
        fig.tight_layout()
        fig.savefig(output / "macro_f1_comparison.png", dpi=170)
    plt.close(fig)
    return report


def run(clips_path: Path, manifest_path: Path, output: Path, *, device: str = "mps",
        epochs: int = 15, batch_size: int = 4, seed: int = 42,
        pretrained: bool = True, resume: bool = False, selected_fold: int | None = None,
        architectures: tuple[str, ...] = ARCHITECTURES) -> dict:
    if not architectures or not set(architectures) <= set(ARCHITECTURES):
        raise ValueError("Select at least one supported architecture")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    clips_path, manifest_path, output = map(lambda p: Path(p).resolve(),
                                             (clips_path, manifest_path, output))
    config = {"clips_sha256": digest(clips_path), "manifest_sha256": digest(manifest_path),
              "device": device, "epochs": epochs, "batch_size": batch_size,
              "seed": seed, "pretrained": pretrained, "fold": selected_fold,
              "architectures": list(architectures), "class_order": list(CLASSES),
              "horizons": list(HORIZONS), "folds": 5, "comparison_version": 2}
    prepare(output, config, resume)
    clips = read_jsonl(clips_path)
    manifest_rows = read_jsonl(manifest_path)
    manifest = {r["kick_id"]: r for r in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate kick IDs in manifest")
    by_horizon = {}
    for h in HORIZONS:
        rows = [r for r in clips if float(r["horizon"]) == h]
        by_horizon[h] = {r["kick_id"]: r for r in rows}
        if len(by_horizon[h]) != len(rows):
            raise ValueError(f"Duplicate clips at horizon {h}")
    common = set.intersection(*(set(by_horizon[h]) for h in HORIZONS))
    eligible = [manifest[k] for k in sorted(common)
                if k in manifest and manifest[k]["direction"] in CLASSES]
    if len(eligible) < 10 or min(sum(r["direction"] == c for r in eligible) for c in CLASSES) < 5:
        raise ValueError("Insufficient eligible binary kicks for five-fold CV")
    for h in HORIZONS:
        for row in eligible:
            clip = by_horizon[h][row["kick_id"]]
            if clip["direction"] != row["direction"]:
                raise ValueError(f"Direction mismatch for {row['kick_id']}")
    folds_path = output / "folds.json"
    if folds_path.exists():
        fold_info = json.loads(folds_path.read_text(encoding="utf-8"))
        if fold_info["seed"] != seed:
            raise ValueError("Existing fold seed mismatch")
        folds = fold_info["folds"]
    else:
        folds = make_folds(eligible, folds_path, seed=seed)
    expected = {r["kick_id"] for r in eligible}
    groups = dict(zip([r["kick_id"] for r in eligible], group_keys(eligible)))
    for fold in folds:
        train_ids, test_ids = set(fold["train"]), set(fold["test"])
        if train_ids & test_ids or train_ids | test_ids != expected:
            raise AssertionError("Fold partition mismatch")
        if {groups[i] for i in train_ids} & {groups[i] for i in test_ids}:
            raise AssertionError("Player/match leakage in folds")
    results_path = output / "fold_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    selected = [f for f in folds if selected_fold is None or f["fold"] == selected_fold]
    for h in HORIZONS:
        for fold in selected:
            training = [by_horizon[h][k] for k in fold["train"]]
            testing = [by_horizon[h][k] for k in fold["test"]]
            truth = [CLASSES.index(r["direction"]) for r in testing]
            for baseline in BASELINES:
                key = f"{h:.1f}_{baseline}_fold_{fold['fold']}"
                if key not in results:
                    predictions = _baseline_predictions(baseline, training, testing, manifest, CLASSES)
                    metrics = classification_metrics(truth, predictions, CLASSES)
                    metrics["predictions"] = [{"kick_id": r["kick_id"], "truth": int(a), "prediction": int(p)}
                                               for r, a, p in zip(testing, truth, predictions)]
                    results[key] = metrics
                    save_json(results_path, results)
            for architecture in architectures:
                key = f"{h:.1f}_{architecture}_fold_{fold['fold']}"
                folder = output / "models" / architecture / f"h{h:.1f}" / f"fold_{fold['fold']}"
                if key in results:
                    continue
                metric_file = folder / "test_metrics.json"
                if metric_file.exists():
                    if not resume:
                        raise FileExistsError(f"Existing metrics: {metric_file}")
                    results[key] = json.loads(metric_file.read_text(encoding="utf-8"))
                else:
                    if folder.exists() and any(folder.iterdir()) and not resume:
                        raise FileExistsError(f"Existing partial checkpoint: {folder}")
                    results[key] = train_fold(
                        training, testing, fold["fold"], h, folder,
                        device_name=device, epochs=epochs, batch_size=batch_size,
                        seed=seed, pretrained=pretrained,
                        resume=resume and (folder / "last.pt").is_file(),
                        class_order=CLASSES, architecture=architecture)
                save_json(results_path, results)
    complete = all(f"{h:.1f}_{arch}_fold_{fold['fold']}" in results
                   for h in HORIZONS for arch in architectures for fold in folds)
    return aggregate(results, output, requested=architectures, complete=complete, pretrained=pretrained)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("data/processed/clips/clips.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/professor_comparison_v2"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Smoke test only, NOT scientific evaluation")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--architecture", choices=("both", "gru", "single_frame"), default="both")
    args = parser.parse_args()
    architectures = ARCHITECTURES if args.architecture == "both" else (args.architecture,)
    print(json.dumps(run(args.clips, args.manifest, args.output,
                         device=args.device, epochs=args.epochs, batch_size=args.batch_size,
                         seed=args.seed, pretrained=not args.no_pretrained,
                         resume=args.resume, selected_fold=args.fold,
                         architectures=architectures), indent=2))


if __name__ == "__main__":
    main()
