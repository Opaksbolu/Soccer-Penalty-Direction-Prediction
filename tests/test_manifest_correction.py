"""Regression tests for the known Mendeley dataset identifier correction."""

from pathlib import Path

import pytest

from src.data.common import read_jsonl
from src.data.manifest import build_manifest


def test_mendeley_player_identifier_correction(tmp_path):
    # Verify the known identifier correction against the original dataset.
    csv_path = Path(
        "data/raw/penalty_kicks/penalty_pose_keypoints.csv"
    )
    video_dir = Path("data/raw/videos")

    if not csv_path.is_file() or not video_dir.is_dir():
        pytest.skip("Original Mendeley dataset is not available locally")

    output = tmp_path / "manifest.jsonl"

    audit = build_manifest(
        csv_path=csv_path,
        video_dir=video_dir,
        output=output,
    )

    records = {
        row["kick_id"]: row
        for row in read_jsonl(output)
    }

    # Confirm the two sequences remain separate.
    assert "15-3" in records
    assert "16-3" in records

    corrected = records["15-3"]
    original = records["16-3"]

    # Check the corrected sequence.
    assert str(corrected["player_id"]) == "15"
    assert corrected["direction"] == "left"
    assert corrected["video"].endswith("15-03.mov")
    assert abs(corrected["fps"] - 22.963) < 0.01

    # Check the genuine 16-3 sequence.
    assert str(original["player_id"]) == "16"
    assert original["direction"] == "right"
    assert original["video"].endswith("16-03.mov")
    assert abs(original["fps"] - 27.416) < 0.01

    # Confirm the expected repair is documented.
    repairs = [
        repair
        for repair in audit["repairs"]
        if repair.get("original_kick_id") == "16-3"
        and repair.get("corrected_kick_id") == "15-3"
    ]

    assert len(repairs) == 1
    assert repairs[0]["rows"] == 78

    # Verify the expected dataset size remains unchanged.
    assert audit["indexed_kicks"] == 126
