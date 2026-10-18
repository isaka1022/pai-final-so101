import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import follow  # noqa: E402
from follow import FLEX_KEY, PAN_KEY, FollowController  # noqa: E402
from hand import Hand  # noqa: E402

START = {PAN_KEY: 10.0, FLEX_KEY: -5.0, "gripper.pos": 30.0}
GOOD_SIZE = 0.15


def make_hand(x=0.3, y=0.0, size=GOOD_SIZE, confidence=0.95) -> Hand:
    return Hand(x=x, y=y, size=size, landmarks_px=np.zeros((21, 2), dtype=np.float32), confidence=confidence)


def run(hand: Hand | None, frames: int, pose=START):
    controller = FollowController()
    current = dict(pose)
    state = ""
    for _ in range(frames):
        current, state = controller.update(hand, current)
    return current, state


def test_right_offset_moves_pan_in_sign_direction():
    pose, state = run(make_hand(x=0.3), follow.CONFIRM_FRAMES + 3)
    assert state == "tracking"
    assert np.sign(pose[PAN_KEY] - START[PAN_KEY]) == follow.PAN_SIGN
    assert pose["gripper.pos"] == START["gripper.pos"]


def test_single_frame_does_not_move():
    pose, state = run(make_hand(), 1)
    assert pose == START
    assert state == "idle"


@pytest.mark.parametrize(
    "hand",
    [
        make_hand(size=0.01),
        make_hand(confidence=0.3),
        make_hand(size=0.6),
    ],
    ids=["too_small", "low_confidence", "too_close"],
)
def test_rejected_detection_does_not_move(hand):
    pose, _ = run(hand, follow.CONFIRM_FRAMES * 4)
    assert pose == START


def test_too_close_reports_state():
    _, state = run(make_hand(size=0.6), follow.CONFIRM_FRAMES * 2)
    assert state == "too_close"


def test_step_and_range_limits():
    controller = FollowController()
    current = dict(START)
    for _ in range(500):
        nxt, _ = controller.update(make_hand(x=0.5, y=0.5), current)
        assert abs(nxt[PAN_KEY] - current[PAN_KEY]) <= follow.MAX_STEP_DEG + 1e-9
        assert abs(nxt[FLEX_KEY] - current[FLEX_KEY]) <= follow.MAX_STEP_DEG + 1e-9
        current = nxt
    assert abs(current[PAN_KEY] - START[PAN_KEY]) == pytest.approx(follow.MAX_RANGE_DEG)
    assert abs(current[FLEX_KEY] - START[FLEX_KEY]) == pytest.approx(follow.MAX_RANGE_DEG)


def test_deadzone_does_not_move():
    pose, state = run(make_hand(x=follow.DEADZONE / 2, y=-follow.DEADZONE / 2), follow.CONFIRM_FRAMES * 2)
    assert state == "tracking"
    assert pose == START
