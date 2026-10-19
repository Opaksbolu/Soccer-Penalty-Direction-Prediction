"""Unpack the Mendeley release, including its nested video archive, into data/raw."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def _copy_if_needed(source, target: Path, size: int) -> bool:
    if target.is_file() and target.stat().st_size == size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with partial.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    partial.replace(target)
    return True


def unpack(archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as outer:
        csv_names = [name for name in outer.namelist() if name.endswith("penalty_pose_keypoints.csv")]
        video_names = [name for name in outer.namelist() if name.endswith("Women_Players_Videos.zip")]
        if len(csv_names) != 1 or len(video_names) != 1:
            raise ValueError("Archive must contain one pose CSV and one Women_Players_Videos.zip")
        csv_info = outer.getinfo(csv_names[0])
        with outer.open(csv_info) as source:
            csv_created = _copy_if_needed(source, output / "penalty_pose_keypoints.csv", csv_info.file_size)
        zip_info = outer.getinfo(video_names[0])
        inner_path = output / "Women_Players_Videos.zip"
        with outer.open(zip_info) as source:
            inner_created = _copy_if_needed(source, inner_path, zip_info.file_size)
    count = 0
    with zipfile.ZipFile(inner_path) as inner:
        for info in inner.infolist():
            if not info.filename.startswith("Women_Players_Videos/") or not info.filename.lower().endswith(".mov"):
                continue
            name = Path(info.filename).name
            if name in ("", ".", ".."):
                continue
            with inner.open(info) as source:
                count += _copy_if_needed(source, output / "videos" / name, info.file_size)
    return {"csv_created": csv_created, "nested_zip_created": inner_created,
            "videos_created": count, "video_count": len(list((output / "videos").glob("*.mov")))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(unpack(args.archive, args.output))


if __name__ == "__main__":
    main()
