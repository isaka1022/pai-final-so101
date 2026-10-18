"""Hand detection on camera frames with MediaPipe HandLandmarker."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent.parent / "out/real/models/hand_landmarker.task"
NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
LANDMARK_COUNT = 21


@dataclass(frozen=True)
class Hand:
    x: float  # hand centre, offset from the image centre in units of image width, -0.5..0.5 (right is positive)
    y: float  # same, in units of image height, -0.5..0.5 (down is positive)
    size: float  # wrist to middle-finger base distance in pixels, divided by image width; grows as the hand nears the camera
    landmarks_px: np.ndarray  # (21, 2) pixel coordinates
    confidence: float


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


class HandDetector:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> Hand | None:
        """rgb: (H, W, 3) uint8. timestamp_ms must increase between calls."""
        height, width = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        points = np.array([[p.x * width, p.y * height] for p in result.hand_landmarks[0]], dtype=np.float32)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        size = float(np.linalg.norm(points[WRIST_INDEX] - points[MIDDLE_MCP_INDEX]) / width)
        confidence = float(result.handedness[0][0].score) if result.handedness else 0.0
        return Hand(
            x=float(center[0] / width - 0.5),
            y=float(center[1] / height - 0.5),
            size=size,
            landmarks_px=points,
            confidence=confidence,
        )

    def close(self) -> None:
        self._landmarker.close()
