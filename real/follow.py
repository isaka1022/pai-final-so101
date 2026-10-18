"""Turn the SO-101 toward a hand seen by the wrist camera (shoulder_pan follows x, wrist_flex follows y)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand import Hand  # noqa: E402
from view import log_camera, start_viewer  # noqa: E402

PAN_KEY = "shoulder_pan.pos"
FLEX_KEY = "wrist_flex.pos"
# Units follow the robot config: degrees (SOFollowerConfig.use_degrees defaults to True).
PAN_SIGN = -1.0  # sign of the pan motion that brings a hand on the right back to the centre; verify on the arm
FLEX_SIGN = 1.0  # same for a hand below the centre; verify on the arm
GAIN_DEG = 20.0  # degrees of target change per unit of image offset
DEADZONE = 0.06  # offsets smaller than this (fraction of the image) do not move the arm
MAX_STEP_DEG = 1.0  # largest target change per control step
MAX_RANGE_DEG = 25.0  # targets stay within the start pose +- this
MAX_RELATIVE_TARGET_DEG = 3.0  # robot-side cap on goal minus present position
MIN_CONFIDENCE = 0.7
MIN_SIZE = 0.05  # smaller hands are too far away or false detections
MAX_SIZE = 0.30  # larger hands are too close to the camera
CONFIRM_FRAMES = 5  # consecutive valid detections before the arm starts moving

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
FPS_WINDOW = 30
WRIST_CAMERA_NAME = "wrist"
DRY_RUN_START_POSE = 0.0


class FollowController:
    def __init__(self) -> None:
        self._start: dict[str, float] | None = None
        self._streak = 0
        self._tracked = False

    def update(self, hand: Hand | None, current: dict[str, float]) -> tuple[dict[str, float], str]:
        if self._start is None:
            self._start = {PAN_KEY: current[PAN_KEY], FLEX_KEY: current[FLEX_KEY]}
        target = dict(current)

        too_close = hand is not None and hand.confidence >= MIN_CONFIDENCE and hand.size > MAX_SIZE
        valid = hand is not None and hand.confidence >= MIN_CONFIDENCE and MIN_SIZE <= hand.size <= MAX_SIZE
        if not valid:
            self._streak = 0
            if too_close:
                return target, "too_close"
            return target, "lost" if self._tracked else "idle"

        self._streak += 1
        if self._streak < CONFIRM_FRAMES:
            return target, "lost" if self._tracked else "idle"

        self._tracked = True
        target[PAN_KEY] = self._step(PAN_KEY, current, PAN_SIGN, hand.x)
        target[FLEX_KEY] = self._step(FLEX_KEY, current, FLEX_SIGN, hand.y)
        return target, "tracking"

    def _step(self, key: str, current: dict[str, float], sign: float, offset: float) -> float:
        if abs(offset) < DEADZONE:
            return current[key]
        delta = float(np.clip(sign * GAIN_DEG * offset, -MAX_STEP_DEG, MAX_STEP_DEG))
        low = self._start[key] - MAX_RANGE_DEG
        high = self._start[key] + MAX_RANGE_DEG
        return float(np.clip(current[key] + delta, low, high))


def log_view(rr, frame_rgb: np.ndarray, hand: Hand | None, target: dict[str, float]) -> None:
    log_camera(rr, frame_rgb, hand)
    rr.log("target/shoulder_pan", rr.Scalars(target[PAN_KEY]))
    rr.log("target/wrist_flex", rr.Scalars(target[FLEX_KEY]))


def describe(state: str, hand: Hand | None, target: dict[str, float], fps: float) -> str:
    offset = f"x={hand.x:+.3f} y={hand.y:+.3f} size={hand.size:.3f}" if hand else "no hand                      "
    return f"{state:<9} {offset} pan={target[PAN_KEY]:+7.2f} flex={target[FLEX_KEY]:+7.2f} fps={fps:5.1f}"


def connect_robot(port: str, camera: int):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    config = SO101FollowerConfig(
        port=port,
        id="follower",
        max_relative_target=MAX_RELATIVE_TARGET_DEG,
        cameras={
            WRIST_CAMERA_NAME: OpenCVCameraConfig(
                index_or_path=camera, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=CAMERA_FPS
            )
        },
    )
    robot = make_robot_from_config(config)
    robot.connect()
    return robot


def run(args: argparse.Namespace) -> int:
    from hand import HandDetector

    cap = None
    robot = None
    if args.dry_run:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"cannot open camera {args.camera}", file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        joints = {PAN_KEY: DRY_RUN_START_POSE, FLEX_KEY: DRY_RUN_START_POSE}
    else:
        robot = connect_robot(args.follower_port, args.camera)

    rr = None if args.no_view else start_viewer("follow")

    detector = HandDetector()
    controller = FollowController()
    recent: list[float] = []
    start = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - start < args.seconds:
            if cap is not None:
                ok, bgr = cap.read()
                if not ok:
                    print("\ncamera read failed", file=sys.stderr)
                    return 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                observation = robot.get_observation()
                rgb = observation[WRIST_CAMERA_NAME]
                joints = {key: observation[key] for key in observation if key.endswith(".pos")}
            now = time.monotonic()
            hand = detector.detect(rgb, int((now - start) * 1000))
            target, state = controller.update(hand, joints)
            if robot is not None:
                robot.send_action(target)
            joints = target
            recent = (recent + [now])[-FPS_WINDOW:]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
            if rr is not None:
                log_view(rr, rgb, hand, target)
            print(f"\r{describe(state, hand, target, fps)}\033[K", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        detector.close()
        if cap is not None:
            cap.release()
        if robot is not None:
            robot.disconnect()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=int(os.environ.get("WRIST_CAMERA", 0)))
    parser.add_argument("--follower-port", help="serial port of the follower arm (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="do not touch the arm; use a virtual pose")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds (default: until Ctrl+C)")
    parser.add_argument("--no-view", action="store_true", help="do not open rerun")
    args = parser.parse_args()
    if not args.dry_run and not args.follower_port:
        parser.error("--follower-port is required unless --dry-run is given")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
