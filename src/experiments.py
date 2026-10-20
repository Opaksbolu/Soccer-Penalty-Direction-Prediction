"""Run common-fold baselines, video CV, and final model training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.common import CLASSES, HORIZONS, read_jsonl
from src.data.dataset import pose_vector
from src.evaluation.metrics import classification_metrics, save_json
from src.evaluation.splits import make_folds
from src.training.train import train_final, train_fold


def _baseline_predictions(name: str, train: list[dict], test: list[dict],
                          manifest: dict[str, dict], class_order: tuple[str, ...]) -> list[int]:
    y_train = [class_order.index(row["direction"]) for row in train]
    if name == "majority":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(np.zeros((len(train), 1)), y_train)
        return model.predict(np.zeros((len(test), 1))).astype(int).tolist()
    if name == "pose_logistic":
        x_train = np.stack([pose_vector(manifest[row["kick_id"]], row["frame_indices"]) for row in train])
        x_test = np.stack([pose_vector(manifest[row["kick_id"]], row["frame_indices"]) for row in test])
        model = make_pipeline(SimpleImputer(strategy="constant", fill_value=0),
                              StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(x_train, y_train)
        return model.predict(x_test).astype(int).tolist()
    raise ValueError(name)


def run(clips_path: Path, manifest_path: Path, output: Path, *, device: str,
        epochs: int, batch_size: int, seed: int, resume: bool, pretrained: bool,
        selected_fold: int | None = None, baselines_only: bool = False,
        skip_final: bool = False, binary: bool = False) -> dict:
    clips = read_jsonl(clips_path)
    manifest = {row["kick_id"]: row for row in read_jsonl(manifest_path)}
    class_order = ("left", "right") if binary else CLASSES
    per_horizon = {h: {row["kick_id"]: row for row in clips if row["horizon"] == h} for h in HORIZONS}
    common = set.intersection(*(set(rows) for rows in per_horizon.values()))
    records = [manifest[id] for id in sorted(common) if manifest[id]["direction"] in class_order]
    eligible_ids = {row["kick_id"] for row in records}
    if not records:
        raise ValueError("No kicks have all three cropped horizons")
    counts = {label: sum(row["direction"] == label for row in records) for label in class_order}
    if min(counts.values()) < 5:
        raise ValueError(f"Five-fold {'binary' if binary else 'three-class'} CV requires at least five kicks per class; found {counts}. "
                         "Use --binary for the current Mendeley release; obtain more center kicks for the project target.")
    folds = make_folds(records, output / "folds.json", seed=seed)
    results: dict[str, dict] = defaultdict(dict)
    for horizon in HORIZONS:
        rows = per_horizon[horizon]
        for fold in folds:
            if selected_fold is not None and fold["fold"] != selected_fold:
                continue
            training = [rows[id] for id in fold["train"]]
            testing = [rows[id] for id in fold["test"]]
            truth = [class_order.index(row["direction"]) for row in testing]
            for baseline in ("majority", "pose_logistic"):
                predictions = _baseline_predictions(baseline, training, testing, manifest, class_order)
                metrics = classification_metrics(truth, predictions, class_order)
                metrics["predictions"] = [{"kick_id": row["kick_id"], "truth": int(a),
                                           "prediction": int(p)} for row, a, p in zip(testing, truth, predictions)]
                results[f"{horizon:.1f}"][f"{baseline}_fold_{fold['fold']}"] = metrics
            if not baselines_only:
                fold_dir = output / "models" / f"h{horizon:.1f}" / f"fold_{fold['fold']}"
                metrics = train_fold(training, testing, fold["fold"], horizon, fold_dir,
                                     device_name=device, epochs=epochs, batch_size=batch_size,
                                     seed=seed, pretrained=pretrained,
                                     resume=resume and (fold_dir / "last.pt").is_file(),
                                     class_order=class_order)
                results[f"{horizon:.1f}"][f"video_fold_{fold['fold']}"] = metrics
            save_json(output / "fold_results.json", results)
    summary = {}
    for horizon, by_model in results.items():
        for model_name in ("majority", "pose_logistic", "video"):
            relevant = [value for name, value in by_model.items() if name.startswith(model_name + "_fold_")]
            if not relevant:
                continue
            pooled = [prediction for value in relevant for prediction in value["predictions"]]
            summary[f"{horizon}_{model_name}"] = classification_metrics(
                [item["truth"] for item in pooled], [item["prediction"] for item in pooled], class_order)
            summary[f"{horizon}_{model_name}"]["folds"] = len(relevant)
            if model_name == "video":
                summary[f"{horizon}_{model_name}"]["mean_batch_one_model_latency_seconds"] = float(
                    np.mean([value["prediction_latency_seconds"] for value in relevant]))
    save_json(output / "summary.json", summary)
    if not baselines_only and not skip_final and selected_fold is None:
        for horizon in HORIZONS:
            best_epochs = []
            for fold in folds:
                path = output / "models" / f"h{horizon:.1f}" / f"fold_{fold['fold']}" / "best.pt"
                import torch
                best_epochs.append(torch.load(path, map_location="cpu", weights_only=False)["epoch"] + 1)
            chosen_epochs = max(1, int(np.median(best_epochs)))
            train_final([per_horizon[horizon][id] for id in sorted(eligible_ids)], horizon,
                        output / "models", chosen_epochs, device, batch_size, seed, pretrained,
                        class_order, resume)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("data/processed/clips/clips.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/main"))
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="For offline smoke tests only")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--binary", action="store_true", help="Exploratory left/right run; excludes the single center kick")
    args = parser.parse_args()
    print(json.dumps(run(args.clips, args.manifest, args.output, device=args.device,
                         epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                         resume=args.resume, pretrained=not args.no_pretrained,
                         selected_fold=args.fold, baselines_only=args.baselines_only,
                         skip_final=args.skip_final, binary=args.binary), indent=2))


if __name__ == "__main__":
    main()
