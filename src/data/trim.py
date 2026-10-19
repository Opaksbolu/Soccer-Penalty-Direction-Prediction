"""Export an uncropped, contact-safe clip for local inference demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .common import read_jsonl, project_path
from .preprocess import select_indices


def trim_one(record: dict, horizon: float, output: Path) -> Path:
    indices = select_indices(record["contact_frame"], record["fps"], horizon)
    capture = cv2.VideoCapture(str(project_path(record["video"])))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(record["fps"]), (width, height))
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise OSError("Could not open source or output video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        for index in indices:
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Could not decode frame {index}")
            writer.write(frame)
    except BaseException:
        writer.release()
        capture.release()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    capture.release()
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--kick-id", required=True)
    parser.add_argument("--horizon", type=float, choices=(1.0, 0.5, 0.2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = next((row for row in read_jsonl(args.manifest) if row["kick_id"] == args.kick_id), None)
    if record is None:
        raise ValueError(f"Unknown kick id {args.kick_id}")
    print(trim_one(record, args.horizon, args.output))


if __name__ == "__main__":
    main()
