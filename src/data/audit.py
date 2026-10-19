"""Render a contact and direction review sheet from the indexed source videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .common import project_path, read_jsonl


def _frame(video: str, index: int, box: list[float] | None) -> np.ndarray:
    capture = cv2.VideoCapture(str(project_path(video)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, image = capture.read()
    capture.release()
    if not ok:
        image = np.zeros((180, 320, 3), dtype=np.uint8)
    elif box:
        cx, cy, width, height = box
        p1 = (int(cx - width / 2), int(cy - height / 2))
        p2 = (int(cx + width / 2), int(cy + height / 2))
        cv2.rectangle(image, p1, p2, (0, 255, 255), 3)
    return cv2.resize(image, (320, 180))


def contact_sheet(manifest: Path, output: Path, count: int = 12) -> Path:
    rows = read_jsonl(manifest)
    rows.sort(key=lambda row: (row["direction"] != "center", row["kick_id"]))
    if len(rows) > count:
        positions = np.linspace(0, len(rows) - 1, count, dtype=int)
        chosen = [rows[0]] + [rows[i] for i in positions[1:]]
    else:
        chosen = rows
    strips = []
    for row in chosen:
        contact = int(row["contact_frame"])
        box = row["frames"][-1]["box_cxcywh"]
        panels = []
        for offset in (-5, 0, 5):
            image = _frame(row["video"], contact + offset, box if offset == 0 else None)
            cv2.putText(image, f"{row['kick_id']} {row['direction']} frame {contact + offset}",
                        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
            panels.append(image)
        strips.append(np.hstack(panels))
    if not strips:
        raise ValueError("No indexed kicks")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack(strips)):
        raise OSError(f"Could not write {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/contact_audit.jpg"))
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    print(contact_sheet(args.manifest, args.output, args.count))


if __name__ == "__main__":
    main()
