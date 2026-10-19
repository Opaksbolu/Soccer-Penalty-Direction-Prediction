"""Create contact-safe, kicker-centered video clips in a resumable batch."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .common import FRAMES, HORIZONS, IMAGE_SIZE, horizon_key, portable_path, project_path, read_jsonl, write_jsonl


def select_indices(contact: int, fps: float, horizon: float, count: int = FRAMES) -> list[int]:
    """Last input frame is at least `horizon` seconds before contact."""
    if contact < 0 or fps <= 0 or horizon <= 0 or count <= 0:
        raise ValueError("invalid contact frame, fps, horizon, or sequence length")
    end = contact - math.ceil(horizon * fps)
    first = end - count + 1
    if first < 0 or end >= contact:
        raise ValueError(f"insufficient pre-contact footage: first={first}, contact={contact}")
    return list(range(first, end + 1))


def square_crop(frame: np.ndarray, box: list[float], size: int = IMAGE_SIZE,
                margin: float = 1.25) -> np.ndarray:
    cx, cy, width, height = map(float, box)
    if not np.isfinite([cx, cy, width, height]).all() or min(width, height) <= 0:
        raise ValueError("invalid crop box")
    side = max(width, height) * margin
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty crop")
    output = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    height_px, width_px = frame.shape[:2]
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, width_px), min(y1, height_px)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        raise ValueError("crop box outside video frame")
    output[src_y0 - y0:src_y1 - y0, src_x0 - x0:src_x1 - x0] = frame[src_y0:src_y1, src_x0:src_x1]
    return cv2.resize(output, (size, size), interpolation=cv2.INTER_LINEAR)


def _valid_clip(path: Path, expected_frames: int = FRAMES) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    opened = capture.isOpened()
    capture.release()
    return opened and count == expected_frames


def crop_one(record: dict, horizon: float, output_dir: Path, overwrite: bool = False) -> dict:
    key = horizon_key(horizon)
    destination = output_dir / key / f"{record['kick_id']}.mp4"
    metadata = destination.with_suffix(".json")
    indices = select_indices(record["contact_frame"], record["fps"], horizon)
    frame_map = {int(row["index"]): row for row in record["frames"]}
    missing = [index for index in indices if index not in frame_map]
    if missing:
        raise ValueError(f"missing CSV boxes for frames {missing}")
    if not overwrite and _valid_clip(destination) and metadata.is_file():
        saved = json.loads(metadata.read_text())
        if (saved.get("frame_indices") == indices and
                project_path(saved.get("source_video", "")).resolve() == project_path(record["video"]).resolve()):
            saved["source_video"] = record["video"]
            saved["clip"] = portable_path(destination)
            metadata.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            return {**saved, "status": "skipped"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    capture = cv2.VideoCapture(str(project_path(record["video"])))
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(record["fps"]), (IMAGE_SIZE, IMAGE_SIZE))
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise OSError("could not open source video or output encoder")
        capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        for index in indices:
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"could not read video frame {index}")
            writer.write(square_crop(frame, frame_map[index]["box_cxcywh"]))
    finally:
        capture.release()
        writer.release()
    if not _valid_clip(temporary):
        temporary.unlink(missing_ok=True)
        raise OSError("encoded clip did not contain 16 frames")
    os.replace(temporary, destination)
    item = {"kick_id": record["kick_id"], "direction": record["direction"], "horizon": horizon,
            "clip": portable_path(destination), "source_video": record["video"],
            "frame_indices": indices, "contact_frame": record["contact_frame"],
            "player_id": record.get("player_id"), "match_id": record.get("match_id"),
            "camera_direction": record.get("camera_direction"), "source": record.get("source")}
    metadata.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return {**item, "status": "created"}


def crop_batch(manifest: Path, output_dir: Path, overwrite: bool = False) -> dict:
    records = read_jsonl(manifest)
    successful, failed = [], []
    for record in tqdm(records, desc="Cropping kicks"):
        for horizon in HORIZONS:
            try:
                successful.append(crop_one(record, horizon, output_dir, overwrite))
            except (ValueError, OSError, KeyError) as exc:
                failed.append({"kick_id": record.get("kick_id"), "horizon": horizon, "reason": str(exc)})
    # Paired evaluation uses only kicks available at all three horizons.
    counts: dict[str, int] = {}
    for row in successful:
        counts[row["kick_id"]] = counts.get(row["kick_id"], 0) + 1
    paired = [row for row in successful if counts[row["kick_id"]] == len(HORIZONS)]
    write_jsonl(output_dir / "clips.jsonl", paired)
    report = {"source_kicks": len(records), "paired_kicks": len(set(row["kick_id"] for row in paired)),
              "created": sum(row["status"] == "created" for row in successful),
              "skipped": sum(row["status"] == "skipped" for row in successful), "failures": failed}
    (output_dir / "crop_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/clips"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(crop_batch(args.manifest, args.output, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
