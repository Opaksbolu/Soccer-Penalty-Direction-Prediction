"""Predict from an uncropped clip already ending before ball contact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.common import CLASSES, FRAMES, IMAGE_SIZE
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.data.preprocess import square_crop
from src.models.video import MobileNetGRU
from src.training.checkpoint import load_checkpoint
from src.training.train import device_from_name


def _read_last_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < FRAMES:
        capture.release()
        raise ValueError(f"Clip needs at least {FRAMES} frames")
    capture.set(cv2.CAP_PROP_POS_FRAMES, total - FRAMES)
    frames = []
    try:
        for _ in range(FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise OSError("Cannot decode the last 16 frames")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    left = max(a[0], b[0]); top = max(a[1], b[1])
    right = min(a[2], b[2]); bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(intersection / max(1.0, area_a + area_b - intersection))


def _xyxy_to_cxcywh(box: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = box.tolist()
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def detect_kicker_boxes(frames: list[np.ndarray], device: str,
                        detector_weights: str = "yolov8n.pt") -> list[list[float]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics for automatic kicker detection, or pass --kicker-box") from exc
    detector = YOLO(detector_weights)
    observations = []
    for frame in frames:
        result = detector.predict(frame, device=device, verbose=False, conf=0.25)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        people = boxes[classes == 0]
        balls = boxes[classes == 32]
        observations.append((people, balls))
    people, balls = observations[-1]
    if len(people) == 0:
        raise ValueError("No person detected in final frame; pass --kicker-box")
    if len(people) == 1:
        chosen = people[0]
    elif len(balls):
        ball_center = (balls[0, :2] + balls[0, 2:]) / 2
        distances = [np.linalg.norm((box[:2] + box[2:]) / 2 - ball_center) for box in people]
        order = np.argsort(distances)
        if len(order) > 1 and distances[order[1]] < distances[order[0]] * 1.25:
            raise ValueError("Ambiguous kicker near ball; pass --kicker-box")
        chosen = people[order[0]]
    else:
        raise ValueError("Multiple people and no ball detected; pass --kicker-box")
    selected = [chosen]
    for candidates, _ in reversed(observations[:-1]):
        if len(candidates) == 0:
            raise ValueError("Kicker tracking failed; pass --kicker-box")
        scores = [_iou(chosen, candidate) for candidate in candidates]
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            raise ValueError("Kicker tracking ambiguous; pass --kicker-box")
        chosen = candidates[best]
        selected.append(chosen)
    return [_xyxy_to_cxcywh(box) for box in reversed(selected)]


def predict_clip(video: Path, checkpoint: Path, horizon: float, *, device_name: str = "cpu",
                 kicker_box: list[float] | None = None, detector_weights: str = "yolov8n.pt") -> dict:
    device = device_from_name(device_name)
    start = time.perf_counter()
    frames = _read_last_frames(video)
    boxes = ([kicker_box] * FRAMES if kicker_box is not None else
             detect_kicker_boxes(frames, device_name, detector_weights))
    crops = [square_crop(frame, box, IMAGE_SIZE) for frame, box in zip(frames, boxes)]
    pixels = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in crops]).astype(np.float32) / 255
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    inputs = torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).unsqueeze(0).to(device)
    saved = load_checkpoint(checkpoint, device)
    config = saved["config"]
    class_order = tuple(config["class_order"])
    if abs(config["horizon"] - horizon) > 1e-6 or not set(class_order).issubset(CLASSES):
        raise ValueError("Checkpoint horizon or class order does not match request")
    model = MobileNetGRU(pretrained=False, hidden_size=config.get("hidden_size", 128),
                         dropout=config.get("dropout", 0.3), num_classes=len(class_order)).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0].cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"prediction": class_order[int(np.argmax(probabilities))],
            "probabilities": dict(zip(class_order, probabilities)), "horizon_seconds": horizon,
            "device": device_name, "end_to_end_latency_seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--horizon", type=float, choices=(1.0, 0.5, 0.2), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--kicker-box", type=float, nargs=4, metavar=("CX", "CY", "W", "H"))
    parser.add_argument("--detector-weights", default="yolov8n.pt")
    args = parser.parse_args()
    print(json.dumps(predict_clip(args.video, args.checkpoint, args.horizon,
                                  device_name=args.device, kicker_box=args.kicker_box,
                                  detector_weights=args.detector_weights), indent=2))


if __name__ == "__main__":
    main()
