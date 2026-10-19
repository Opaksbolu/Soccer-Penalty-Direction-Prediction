import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.common import read_jsonl, write_jsonl
from src.data.manifest import build_manifest
from src.data.preprocess import crop_batch, select_indices, square_crop
from src.evaluation.splits import group_keys, make_folds
from src.models.video import MobileNetGRU
from src.predict import predict_clip
from src.training.checkpoint import atomic_save


def video(path: Path, frames: int = 80) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((64, 64, 3), index % 200, dtype=np.uint8)
        writer.write(image)
    writer.release()


def test_contact_safe_selection_and_edge_crop():
    for horizon in (1.0, 0.5, 0.2):
        indices = select_indices(79, 25.0, horizon)
        assert len(indices) == 16
        assert max(indices) <= 79 - int(np.ceil(horizon * 25))
        assert max(indices) < 79
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    result = square_crop(image, [2, 2, 20, 20])
    assert result.shape == (224, 224, 3)
    assert np.any(result == 0) and np.any(result == 100)
    with pytest.raises(ValueError, match="insufficient"):
        select_indices(20, 25, 1.0)


def test_manifest_and_batch_rerun(tmp_path):
    import pandas as pd

    raw = tmp_path / "raw"
    raw.mkdir()
    video(raw / "kick1.mp4")
    csv = raw / "poses.csv"
    rows = []
    for index in range(80):
        row = {"kick_id": "kick1", "kick_direction": "L", "frame": index,
               "bbox_cx": 3, "bbox_cy": 3, "bbox_w": 20, "bbox_h": 20}
        for joint in range(5, 17):
            row[f"kp_{joint}_xc"] = float(joint)
            row[f"kp_{joint}_yc"] = float(joint)
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, index=False)
    manifest = tmp_path / "processed" / "manifest.jsonl"
    audit = build_manifest(csv, raw, manifest)
    assert audit["indexed_kicks"] == 1
    record = read_jsonl(manifest)[0]
    assert record["contact_frame"] == 79
    output = tmp_path / "clips"
    first = crop_batch(manifest, output)
    assert first["created"] == 3 and first["paired_kicks"] == 1
    second = crop_batch(manifest, output)
    assert second["skipped"] == 3
    for row in read_jsonl(output / "clips.jsonl"):
        assert len(row["frame_indices"]) == 16
        assert max(row["frame_indices"]) < record["contact_frame"]


def test_player_and_match_connections_stay_in_one_fold(tmp_path):
    records = []
    for index in range(30):
        records.append({"kick_id": str(index), "direction": ("left", "center", "right")[index % 3],
                        "player_id": f"p{index}", "match_id": f"m{index}"})
    records[0]["player_id"] = records[3]["player_id"]
    records[3]["match_id"] = records[6]["match_id"]
    groups = group_keys(records)
    assert groups[0] == groups[3] == groups[6]
    folds = make_folds(records, tmp_path / "folds.json", folds=3)
    for fold in folds:
        train = {int(i) for i in fold["train"]}
        test = {int(i) for i in fold["test"]}
        assert all((i in train) == (j in train) for i, j in ((0, 3), (3, 6)))
        assert not train & test


def test_cpu_checkpoint_prediction(tmp_path):
    clip = tmp_path / "precontact.mp4"
    video(clip, 16)
    model = MobileNetGRU(pretrained=False)
    checkpoint = tmp_path / "model.pt"
    atomic_save({"model": model.state_dict(), "config": {"horizon": 0.2,
                 "class_order": ["left", "center", "right"]}}, checkpoint)
    result = predict_clip(clip, checkpoint, 0.2, kicker_box=[32, 32, 40, 40])
    assert result["prediction"] in ("left", "center", "right")
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
    assert result["device"] == "cpu"
    with pytest.raises(ValueError, match="horizon"):
        predict_clip(clip, checkpoint, 0.5, kicker_box=[32, 32, 40, 40])
