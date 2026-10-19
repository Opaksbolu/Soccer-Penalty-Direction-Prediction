"""Consistent three-class metrics and artifact writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data.common import CLASSES


def classification_metrics(truth: list[int], predictions: list[int],
                           class_order: tuple[str, ...] = CLASSES) -> dict:
    labels = list(range(len(class_order)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1": float(np.mean(f1)),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(class_order)},
            "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist()}


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
