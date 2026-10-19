"""Load cropped sequences and pose vectors for shared experiments."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import CLASSES, FRAMES, IMAGE_SIZE, project_path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_clip(path: str) -> torch.Tensor:
    capture = cv2.VideoCapture(str(project_path(path)))
    frames = []
    try:
        while len(frames) < FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    finally:
        capture.release()
    if len(frames) != FRAMES:
        raise ValueError(f"Expected {FRAMES} frames in {path}, found {len(frames)}")
    pixels = np.stack(frames).astype(np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy())


class ClipDataset(Dataset):
    def __init__(self, records: list[dict], augment: bool = False,
                 class_order: tuple[str, ...] = CLASSES):
        self.records = records
        self.augment = augment
        self.class_order = class_order

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        frames = load_clip(row["clip"])
        if self.augment:
            # Photometric augmentation preserves the left/right label.
            frames = frames * torch.empty(1).uniform_(0.9, 1.1).item()
        return frames, self.class_order.index(row["direction"])


def pose_vector(manifest_record: dict, frame_indices: list[int]) -> np.ndarray:
    by_index = {int(row["index"]): row for row in manifest_record["frames"]}
    vectors = []
    for index in frame_indices:
        row = by_index[index]
        pose = row.get("pose", [])
        if len(pose) != 24:
            raise ValueError("pose baseline requires 12 x/y joint pairs")
        box = row["box_cxcywh"]
        values = np.asarray([np.nan if v is None else v for v in pose], dtype=np.float32)
        # Published data may use box-centered or absolute coordinates.
        if manifest_record.get("pose_columns", [""])[0].endswith("_x"):
            values[0::2] -= box[0]
            values[1::2] -= box[1]
        values[0::2] /= box[2]
        values[1::2] /= box[3]
        vectors.extend(values.tolist())
    return np.asarray(vectors, dtype=np.float32)
