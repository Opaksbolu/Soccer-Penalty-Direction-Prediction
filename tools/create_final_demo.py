import cv2
import json
import numpy as np
from pathlib import Path

ROOT = Path("results/demo")
SOURCE = ROOT / "penalty_2-2_unseen_0.2s.mp4"
RESULT = ROOT / "unseen_prediction_step69.json"
OUTPUT = ROOT / "soccer_penalty_final_demo.mp4"

data = json.loads(RESULT.read_text(encoding="utf-8"))
prediction = data["prediction_result"]
probabilities = prediction["probabilities"]

assert data["kick_id"] == "2-2"
assert data["dataset_split"] == "held_out_test"

WIDTH, HEIGHT = 1280, 720
FPS = 24
DURATION = 60
TOTAL_FRAMES = FPS * DURATION

capture = cv2.VideoCapture(str(SOURCE))
assert capture.isOpened(), "Cannot open source video"

source_frames = []

while True:
    ok, frame = capture.read()
    if not ok:
        break
    source_frames.append(frame)

capture.release()

assert len(source_frames) == 16

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (WIDTH, HEIGHT)
)

assert writer.isOpened(), "Cannot create output video"

def text(frame, message, x, y, scale=0.8, color=(255, 255, 255)):
    cv2.putText(
        frame,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )

def background():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (35, 24, 16)
    return frame

def heading(frame, title, subtitle=None):
    text(frame, title, 65, 90, 1.15)
    cv2.line(frame, (65, 115), (1215, 115), (180, 170, 90), 3)

    if subtitle:
        text(frame, subtitle, 65, 155, 0.65, (200, 200, 200))

def video_frame(index):
    original = source_frames[index % len(source_frames)]
    h, w = original.shape[:2]

    scale = min(1100 / w, 470 / h)
    resized = cv2.resize(
        original,
        (int(w * scale), int(h * scale))
    )

    canvas = background()

    rh, rw = resized.shape[:2]
    x = (WIDTH - rw) // 2
    y = 175

    canvas[y:y + rh, x:x + rw] = resized
    return canvas

for frame_number in range(TOTAL_FRAMES):
    seconds = frame_number / FPS

    if seconds < 8:
        frame = background()
        heading(frame, "SOCCER PENALTY DIRECTION PREDICTION")
        text(frame, "Deep Learning Semester Project", 65, 230, 0.9)
        text(frame, "MobileNetV3-Small + GRU", 65, 290, 0.8)
        text(frame, "Pre-contact video classification", 65, 350, 0.75)
        text(frame, "Left vs. Right", 65, 410, 0.75)

    elif seconds < 18:
        frame = background()
        heading(frame, "THE PROBLEM")
        text(frame, "Can a model predict penalty direction", 65, 240, 0.9)
        text(frame, "before the player contacts the ball?", 65, 290, 0.9)
        text(frame, "Input: 16 video frames", 65, 390, 0.8)
        text(frame, "Prediction horizon: 0.2 seconds", 65, 440, 0.8)
        text(frame, "Evaluation: player-group-safe five-fold CV", 65, 490, 0.7)

    elif seconds < 32:
        index = int((seconds - 18) * 12) % 16
        frame = video_frame(index)
        heading(
            frame,
            "HELD-OUT TEST VIDEO",
            "Kick 2-2 | Fold 0 | 16 pre-contact frames"
        )
        text(frame, "The model does not see the ball being kicked.", 65, 660, 0.7)

    elif seconds < 44:
        frame = background()
        heading(frame, "MODEL PREDICTION")
        text(frame, "Predicted direction: LEFT", 65, 240, 1.0)
        text(
            frame,
            f"Left: {probabilities['left'] * 100:.2f}%",
            65, 340, 0.95
        )
        text(
            frame,
            f"Right: {probabilities['right'] * 100:.2f}%",
            65, 400, 0.95
        )
        text(frame, "Model outputs are nearly evenly split.", 65, 490, 0.7)
        text(frame, "Annotation-assisted kicker crop", 65, 550, 0.65)

    elif seconds < 51:
        frame = background()
        heading(frame, "ACTUAL OUTCOME")
        text(frame, "Actual direction: RIGHT", 65, 250, 1.0)
        text(frame, "Prediction: INCORRECT", 65, 340, 1.0, (100, 130, 255))
        text(frame, "This was an unseen test kick.", 65, 430, 0.8)

    else:
        frame = background()
        heading(frame, "EXPERIMENTAL FINDINGS")
        text(frame, "Deep model accuracy (0.2s): 49.58%", 65, 235, 0.85)
        text(frame, "Pose logistic accuracy (0.2s): 63.87%", 65, 305, 0.85)
        text(frame, "Majority baseline accuracy: 55.46%", 65, 375, 0.85)
        text(frame, "Key finding: The pose baseline performed better.", 65, 480, 0.72)
        text(frame, "Results are from five-fold evaluation.", 65, 535, 0.7)

    cv2.rectangle(
        frame,
        (0, HEIGHT - 9),
        (int(WIDTH * frame_number / TOTAL_FRAMES), HEIGHT),
        (180, 170, 90),
        -1
    )

    writer.write(frame)

writer.release()

check = cv2.VideoCapture(str(OUTPUT))

frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
fps = check.get(cv2.CAP_PROP_FPS)

check.release()

assert frames == TOTAL_FRAMES
assert abs(fps - FPS) < 0.1

print("=" * 65)
print("STEP 71: FINAL DEMONSTRATION VIDEO")
print("=" * 65)

print("Output:", OUTPUT)
print("Duration:", frames / fps, "seconds")
print("Resolution:", f"{WIDTH}x{HEIGHT}")
print("Frames:", frames)
print("FPS:", fps)
print("Size:", round(OUTPUT.stat().st_size / (1024 * 1024), 2), "MB")

print("\nPASS: Video successfully generated")
print("PASS: Duration is 60 seconds")
print("PASS: Uses held-out test footage")
print("PASS: Prediction and actual outcome disclosed")
print("PASS: Experimental results included")

print("\nSTEP 71 PASSED")
