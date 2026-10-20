"""Index the released frame-level CSV against local videos.

Column overrides exist because the released CSV is not bundled with this repository.
The final CSV row for each kick is treated as ball contact, as described by the
dataset authors; audit images must be reviewed before using the manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import normalize_direction, portable_path, write_jsonl

ALIASES = {
    "kick": ("kick_id", "kickid", "penalty_id", "video_id"),
    "direction": ("kick_direction", "direction", "shot_direction"),
    "frame": ("frame", "frame_id", "frame_idx", "frame_number"),
    "bbox_cx": ("bbox_cx", "box_cx"),
    "bbox_cy": ("bbox_cy", "box_cy"),
    "bbox_w": ("bbox_w", "box_w"),
    "bbox_h": ("bbox_h", "box_h"),
    "player": ("player_id", "player_name", "kicker_id"),
    "match": ("match_id", "game_id", "match"),
    "camera": ("camera_direction", "camera_view"),
    "contact": ("last_touch", "contact", "ball_contact"),
    "fps": ("fps", "frame_rate"),
}


def column(df: pd.DataFrame, name: str, overrides: dict[str, str], required: bool = True) -> str | None:
    options = (overrides[name],) if name in overrides else ALIASES[name]
    matches = {str(col).lower(): str(col) for col in df.columns}
    for option in options:
        if option.lower() in matches:
            return matches[option.lower()]
    if required:
        raise ValueError(f"Missing {name} column. Tried {options}; available: {list(df.columns)}")
    return None


def _safe_id(value: object) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text) or text in (".", ".."):
        raise ValueError(f"Unsafe kick id {value!r}")
    return text


def _canonical_id(value: str) -> str:
    parts = value.split("-")
    return "-".join(str(int(part)) for part in parts) if all(part.isdigit() for part in parts) else value.lower()


def _video_index(video_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in video_dir.rglob("*"):
        if path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            index.setdefault(_canonical_id(path.stem), []).append(path.resolve())
    return index


def _pose_columns(df: pd.DataFrame) -> list[str]:
    # Prefer the published box-centered coordinates; fall back to absolute x/y.
    names = {str(c).lower(): str(c) for c in df.columns}
    for suffix in ("xc", "x"):
        cols = []
        for joint in range(5, 17):
            other = "yc" if suffix == "xc" else "y"
            x = names.get(f"kp_{joint}_{suffix}")
            y = names.get(f"kp_{joint}_{other}")
            if not x or not y:
                break
            cols.extend([x, y])
        if len(cols) == 24:
            return cols
    return []


def build_manifest(csv_path: Path, video_dir: Path, output: Path,
                   video_map: Path | None = None, overrides: dict[str, str] | None = None) -> dict:
    overrides = overrides or {}
    df = pd.read_csv(csv_path)
    kick_col = column(df, "kick", overrides)
    direction_col = column(df, "direction", overrides)
    frame_col = column(df, "frame", overrides, required=False)
    box_cols = [column(df, key, overrides) for key in ("bbox_cx", "bbox_cy", "bbox_w", "bbox_h")]
    player_col = column(df, "player", overrides, required=False)
    match_col = column(df, "match", overrides, required=False)
    camera_col = column(df, "camera", overrides, required=False)
    contact_col = column(df, "contact", overrides, required=False)
    fps_col = column(df, "fps", overrides, required=False)
    pose_cols = _pose_columns(df)
    mapped = json.loads(video_map.read_text()) if video_map else {}
    videos = _video_index(video_dir)
    records, exclusions, repairs = [], [], []
    # Version 2 contains one mislabeled 15-3 sequence embedded in kick_id 16-3.
    # Its 22.963 FPS matches video 15-03; the true 16-3 rows are 27.416 FPS.
    if "16-3" in df[kick_col].astype(str).values and fps_col:
        repair_mask = (
            (df[kick_col].astype(str) == "16-3")
            & (
                pd.to_numeric(df[fps_col], errors="coerce")
                .sub(22.963)
                .abs()
                < 0.002
            )
        )

        if repair_mask.any():
            repaired_rows = int(repair_mask.sum())

            # Correct the kick identifier using the matching video FPS.
            df.loc[repair_mask, kick_col] = "15-3"

            # The video filename and sequence numbering support player 15.
            # This is an inferred metadata correction, not identity verification.
            if player_col:
                df.loc[repair_mask, player_col] = 15
            elif "kicker" in df.columns:
                df.loc[repair_mask, "kicker"] = 15

            repairs.append({
                "original_kick_id": "16-3",
                "corrected_kick_id": "15-3",
                "rows": repaired_rows,
                "reason": (
                    "Separated merged sequences using video FPS; "
                    "player 15 inferred from video filename and sequence numbering"
                ),
            })
    for raw_id, original_group in df.groupby(kick_col, sort=False):
        try:
            kick_id = _safe_id(raw_id)
            if frame_col and original_group.duplicated(subset=[frame_col]).any():
                comparable = original_group.drop(columns=["kicker_track_id_main"], errors="ignore")
                for _, duplicate_rows in comparable.groupby(frame_col):
                    if len(duplicate_rows.drop_duplicates()) != 1:
                        raise ValueError("conflicting annotations for the same frame")
            group = original_group.drop_duplicates(subset=[frame_col] if frame_col else None).copy()
            if len(group) < len(original_group):
                repairs.append({"kick_id": kick_id, "removed_duplicate_tracking_rows": len(original_group) - len(group)})
            matches = sorted(videos.get(_canonical_id(kick_id), []))
            if kick_id in mapped:
                video = (video_dir / mapped[kick_id]).resolve()
                if not video.is_file() or not video.is_relative_to(video_dir.resolve()):
                    raise ValueError("mapped video is missing or outside video directory")
            elif len(matches) == 1:
                video = matches[0]
            else:
                raise ValueError(f"Expected one video named {kick_id}, found {len(matches)}; use --video-map")
            directions = {normalize_direction(v) for v in group[direction_col].dropna()}
            if len(directions) != 1:
                raise ValueError(f"inconsistent direction labels: {directions}")
            group = group.copy()
            if frame_col:
                group = group.sort_values(frame_col)
                raw_frames = pd.to_numeric(group[frame_col], errors="raise").astype(int).tolist()
                frame_base = 1 if min(raw_frames) == 1 else 0
                indices = [v - frame_base for v in raw_frames]
            else:
                indices = list(range(len(group)))
                frame_base = 0
            if len(indices) != len(set(indices)) or min(indices) < 0:
                raise ValueError("duplicate or invalid frame indices")
            capture = cv2.VideoCapture(str(video))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if not (fps > 0 and total > max(indices)):
                raise ValueError(f"video metadata mismatch: fps={fps}, frames={total}, last CSV frame={max(indices)}")
            if fps_col and abs(float(group[fps_col].iloc[0]) - fps) > 0.5:
                raise ValueError(f"CSV/video frame-rate mismatch: {group[fps_col].iloc[0]} vs {fps}")
            if contact_col:
                touch = pd.to_numeric(group[contact_col], errors="coerce").fillna(0).to_numpy()
                if int(np.sum(touch == 1)) != 1 or indices[int(np.argmax(touch))] != indices[-1]:
                    raise ValueError("requires exactly one last_touch=1 at the final CSV frame")
            frames = []
            for (_, row), index in zip(group.iterrows(), indices):
                box = [float(row[col]) for col in box_cols]
                if not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
                    raise ValueError(f"invalid box at frame {index}")
                pose = [float(row[col]) if pd.notna(row[col]) else None for col in pose_cols]
                frames.append({"index": index, "box_cxcywh": box, "pose": pose})
            def identifier(col: str | None) -> str | None:
                values = group[col].dropna().astype(str).unique() if col else []
                return str(values[0]) if len(values) == 1 else None
            records.append({"kick_id": kick_id, "source": "mendeley-v2", "video": portable_path(video),
                            "direction": next(iter(directions)), "fps": fps, "video_frames": total,
                            "contact_frame": indices[-1], "frame_base": frame_base,
                            "player_id": identifier(player_col) or (identifier("kicker") if "kicker" in group else None),
                            "match_id": identifier(match_col),
                            "camera_direction": identifier(camera_col), "pose_columns": pose_cols,
                            "frames": frames})
        except (ValueError, TypeError, KeyError) as exc:
            exclusions.append({"kick_id": str(raw_id), "reason": str(exc)})
    write_jsonl(output, records)
    audit = {"csv_rows": len(df), "indexed_kicks": len(records), "excluded_kicks": exclusions,
             "class_counts": {label: sum(r["direction"] == label for r in records) for label in ("left", "center", "right")},
             "repairs": repairs,
             "column_mapping": {key: overrides.get(key, ALIASES[key]) for key in ALIASES},
             "contact_rule": "require last_touch=1 on final CSV frame when available; requires visual audit"}
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--video-map", type=Path, help="JSON mapping kick_id to path relative to --videos")
    parser.add_argument("--columns", type=Path, help="JSON mapping logical column names to CSV columns")
    args = parser.parse_args()
    audit = build_manifest(args.csv, args.videos, args.output, args.video_map,
                           json.loads(args.columns.read_text()) if args.columns else None)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
