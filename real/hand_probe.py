"""Detect a hand in wrist-camera frames and show it in rerun. No arm involved."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand import Hand, HandDetector  # noqa: E402

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
FPS_WINDOW = 30
VIEW_JPEG_QUALITY = 80


def log_view(rr, frame_rgb: np.ndarray, hand: Hand | None) -> None:
    height, width = frame_rgb.shape[:2]
    rr.log("camera", rr.Image(frame_rgb).compress(jpeg_quality=VIEW_JPEG_QUALITY))
    if hand is None:
        rr.log("camera/hand/landmarks", rr.Clear(recursive=False))
        rr.log("camera/hand/center", rr.Clear(recursive=False))
        return
    rr.log("camera/hand/landmarks", rr.Points2D(hand.landmarks_px, radii=3, colors=[0, 255, 0]))
    center = [(hand.x + 0.5) * width, (hand.y + 0.5) * height]
    rr.log("camera/hand/center", rr.Points2D([center], radii=8, colors=[255, 0, 0]))


def describe(hand: Hand | None, fps: float) -> str:
    if hand is None:
        return f"no hand            fps={fps:5.1f}"
    return f"x={hand.x:+.3f} y={hand.y:+.3f} size={hand.size:.3f} conf={hand.confidence:.2f} fps={fps:5.1f}"


def run_image(path: str, detector: HandDetector) -> int:
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"cannot read image: {path}", file=sys.stderr)
        return 1
    hand = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0)
    print(describe(hand, 0.0))
    return 0


def run_camera(args: argparse.Namespace, detector: HandDetector) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"cannot open camera {args.camera}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    rr = None
    if not args.no_view:
        import rerun as rr

        rr.init("hand_probe", spawn=True)

    frames = detected = 0
    sizes: list[float] = []
    recent: list[float] = []
    start = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - start < args.seconds:
            ok, bgr = cap.read()
            if not ok:
                print("\ncamera read failed", file=sys.stderr)
                break
            now = time.monotonic()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hand = detector.detect(rgb, int((now - start) * 1000))
            frames += 1
            recent = (recent + [now])[-FPS_WINDOW:]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
            if hand is not None:
                detected += 1
                sizes.append(hand.size)
            if rr is not None:
                log_view(rr, rgb, hand)
            print(f"\r{describe(hand, fps)}\033[K", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    elapsed = time.monotonic() - start
    print()
    print(f"frames: {frames}")
    print(f"detected: {detected / frames:.1%}" if frames else "detected: n/a")
    print(f"mean fps: {frames / elapsed:.1f}" if elapsed > 0 else "mean fps: n/a")
    print(f"hand size: {min(sizes):.3f} - {max(sizes):.3f}" if sizes else "hand size: n/a")
    return 0 if frames else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=int(os.environ.get("WRIST_CAMERA", 0)))
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds (default: until Ctrl+C)")
    parser.add_argument("--no-view", action="store_true", help="do not open rerun")
    parser.add_argument("--image", help="detect on this still image instead of the camera")
    args = parser.parse_args()

    detector = HandDetector()
    try:
        return run_image(args.image, detector) if args.image else run_camera(args, detector)
    finally:
        detector.close()


if __name__ == "__main__":
    sys.exit(main())
