"""Train one fold and horizon, saving resumable epoch checkpoints."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from src.data.common import CLASSES
from src.data.dataset import ClipDataset
from src.evaluation.metrics import classification_metrics
from src.evaluation.splits import group_keys
from src.models.video import MobileNetGRU
from .checkpoint import atomic_save, load_checkpoint, restore_rng, rng_state


def device_from_name(name: str) -> torch.device:
    if name not in ("cuda", "cpu", "mps"):
        raise ValueError("Choose --device cuda, cpu, or mps")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable on this machine")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable on this machine")
    return torch.device(name)


def validation_partition(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    labels = np.asarray([row["direction"] for row in records])
    groups = np.asarray(group_keys(records))
    n = min(5, int(min(np.unique(labels, return_counts=True)[1])))
    if n < 2:
        raise ValueError("Need at least two training examples in every class for validation")
    if len(set(groups)) < n:
        raise ValueError("Not enough independent groups for validation")
    shared = len(set(groups)) < len(groups)
    splitter = (StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed) if shared
                else StratifiedKFold(n_splits=n, shuffle=True, random_state=seed))
    train_idx, val_idx = next(iter(splitter.split(records, labels, groups) if shared
                                   else splitter.split(records, labels)))
    if {groups[i] for i in train_idx} & {groups[i] for i in val_idx}:
        raise AssertionError("validation group leakage")
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


def _loader(rows: list[dict], batch_size: int, shuffle: bool, augment: bool,
            generator: torch.Generator | None = None,
            class_order: tuple[str, ...] = CLASSES) -> DataLoader:
    return DataLoader(ClipDataset(rows, augment=augment, class_order=class_order), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, generator=generator)


@torch.no_grad()
def predict(model: nn.Module, rows: list[dict], device: torch.device,
            batch_size: int = 8, class_order: tuple[str, ...] = CLASSES) -> tuple[list[int], float]:
    model.eval()
    predictions = []
    loader = _loader(rows, batch_size, False, False, class_order=class_order)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for frames, _ in loader:
        logits = model(frames.to(device))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, (time.perf_counter() - start) / max(1, len(rows))


def train_fold(train_rows: list[dict], test_rows: list[dict], fold: int,
               horizon: float, output: Path, *, device_name: str = "cuda",
               epochs: int = 15, batch_size: int = 4, learning_rate: float = 1e-3,
               seed: int = 42, pretrained: bool = True, resume: bool = False,
               class_order: tuple[str, ...] = CLASSES) -> dict:
    device = device_from_name(device_name)
    if not train_rows or not test_rows:
        raise ValueError("empty train or test partition")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    fit_rows, val_rows = validation_partition(train_rows, seed)
    config = {"fold": fold, "horizon": horizon, "batch_size": batch_size,
              "learning_rate": learning_rate, "seed": seed, "architecture": "mobilenet_v3_small_gru",
              "hidden_size": 128, "dropout": 0.3, "class_order": list(class_order),
              "fit_ids": sorted(row["kick_id"] for row in fit_rows),
              "val_ids": sorted(row["kick_id"] for row in val_rows),
              "test_ids": sorted(row["kick_id"] for row in test_rows)}
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output / "last.pt", output / "best.pt"
    model = MobileNetGRU(pretrained=pretrained and not resume, num_classes=len(class_order)).to(device)
    counts = np.bincount([class_order.index(row["direction"]) for row in fit_rows], minlength=len(class_order))
    weights = torch.tensor(len(fit_rows) / (len(class_order) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(list(model.gru.parameters()) + list(model.head.parameters()),
                                  lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    generator = torch.Generator().manual_seed(seed)
    start_epoch, best_score = 0, -1.0
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"No checkpoint at {last_path}")
        saved = load_checkpoint(last_path, device)
        if saved["config"] != config:
            raise ValueError("Checkpoint configuration or data split differs from requested run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["loader_generator"])
        restore_rng(saved["rng"])
        start_epoch, best_score = saved["epoch"] + 1, saved["best_validation_macro_f1"]
    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        for frames, labels in _loader(fit_rows, batch_size, True, True, generator, class_order):
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        val_predictions, _ = predict(model, val_rows, device, batch_size, class_order)
        val_truth = [class_order.index(row["direction"]) for row in val_rows]
        score = classification_metrics(val_truth, val_predictions, class_order)["macro_f1"]
        improved = score > best_score
        best_score = max(best_score, score)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "fold": fold,
                      "config": config, "rng": rng_state(), "loader_generator": generator.get_state(),
                      "best_validation_macro_f1": best_score}
        atomic_save(checkpoint, last_path)
        if improved:
            atomic_save({"model": model.state_dict(), "config": config, "epoch": epoch,
                         "best_validation_macro_f1": score}, best_path)
        print(json.dumps({"fold": fold, "horizon": horizon, "epoch": epoch,
                          "loss": float(np.mean(losses)), "validation_macro_f1": score}), flush=True)
    if not best_path.is_file():
        raise RuntimeError("No best checkpoint saved; increase --epochs or check --resume")
    model.load_state_dict(load_checkpoint(best_path, device)["model"])
    predictions, latency = predict(model, test_rows, device, batch_size=1, class_order=class_order)
    truth = [class_order.index(row["direction"]) for row in test_rows]
    metrics = classification_metrics(truth, predictions, class_order)
    metrics.update({"fold": fold, "horizon": horizon, "prediction_latency_seconds": latency,
                    "predictions": [{"kick_id": row["kick_id"], "truth": int(actual),
                                     "prediction": int(predicted)}
                                    for row, actual, predicted in zip(test_rows, truth, predictions)]})
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_final(rows: list[dict], horizon: float, output: Path, epochs: int,
                device_name: str = "cuda", batch_size: int = 4, seed: int = 42,
                pretrained: bool = True, class_order: tuple[str, ...] = CLASSES,
                resume: bool = False) -> Path:
    """Train deployment weights on all audited kicks for a CV-selected epoch count."""
    device = device_from_name(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = output / f"final_{horizon:.1f}.pt"
    last_path = output / f"final_{horizon:.1f}_last.pt"
    config = {"horizon": horizon, "architecture": "mobilenet_v3_small_gru",
              "class_order": list(class_order), "image_size": 224, "frames": 16,
              "epochs": epochs, "training_kick_ids": sorted(row["kick_id"] for row in rows),
              "batch_size": batch_size, "seed": seed}
    if resume and path.is_file():
        saved = load_checkpoint(path, device)
        if saved["config"] == config:
            return path
    model = MobileNetGRU(pretrained=pretrained and not (resume and last_path.is_file()),
                         num_classes=len(class_order)).to(device)
    optimizer = torch.optim.AdamW(list(model.gru.parameters()) + list(model.head.parameters()),
                                  lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    counts = np.bincount([class_order.index(row["direction"]) for row in rows], minlength=len(class_order))
    weights = torch.tensor(len(rows) / (len(class_order) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(seed)
    start_epoch = 0
    if resume and last_path.is_file():
        saved = load_checkpoint(last_path, device)
        if saved["config"] != config:
            raise ValueError("Final-training checkpoint configuration differs from requested run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["loader_generator"])
        restore_rng(saved["rng"])
        start_epoch = saved["epoch"] + 1
    for epoch in range(start_epoch, epochs):
        model.train()
        for frames, labels in _loader(rows, batch_size, True, True, generator, class_order):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(frames.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch,
                     "fold": "final", "config": config, "rng": rng_state(),
                     "loader_generator": generator.get_state(),
                     "best_validation_macro_f1": None}, last_path)
        print(json.dumps({"stage": "final", "horizon": horizon, "epoch": epoch}), flush=True)
    atomic_save({"model": model.state_dict(), "config": config}, path)
    return path
