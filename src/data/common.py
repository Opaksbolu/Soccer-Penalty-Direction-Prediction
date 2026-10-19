"""Shared data conventions. Directions are always from the goalkeeper's view."""

from __future__ import annotations

import json
from pathlib import Path

CLASSES = ("left", "center", "right")
HORIZONS = (1.0, 0.5, 0.2)
FRAMES = 16
IMAGE_SIZE = 224
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def portable_path(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(value)


def horizon_key(horizon: float) -> str:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}; choose {HORIZONS}")
    return f"{horizon:.1f}"


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def normalize_direction(value: object) -> str:
    text = str(value).strip().lower()
    labels = {"l": "left", "left": "left", "c": "center", "centre": "center",
              "center": "center", "middle": "center", "r": "right", "right": "right"}
    if text not in labels:
        raise ValueError(f"Unknown direction: {value!r}")
    return labels[text]
