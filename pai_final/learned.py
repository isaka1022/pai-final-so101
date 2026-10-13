"""Wrist-camera grasp segment: start above a noisy target, descend, close and lift.

The learned policy sees only the wrist image and the joint angles. Simulator
state is read for scripted demos, for placing the start pose and for the
success check, never inside the policy loop.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from lerobot_env_so101.policy import DEFAULT_ACTION_SCALE
from lerobot_env_so101.scripted.pick_place import CAREFUL_SPEED, Phase, PickPlaceController
from lerobot_env_so101.scripted.reach_close import GRIPPER_CLOSED

from pai_final.env import CUBE_HALF_SIZE, PickPlaceSceneEnv

GRASP_PHASES = (Phase.HOVER, Phase.DESCEND, Phase.CLOSE, Phase.LIFT)
IMAGE_SIZE = 96
START_NOISE = 0.02
SEGMENT_MAX_STEPS = 100
HOVER_MAX_STEPS = 60
DEMO_ACTION_NOISE = 0.05

# The hand has descended below DESCENT_Z and risen back above LIFT_DONE_Z.
DESCENT_Z = 0.05
LIFT_DONE_Z = 0.08

# Success: cube centre this far above the table, within HOLD_RADIUS of the grasp
# point, after the arm has held still for SETTLE_STEPS.
LIFT_SUCCESS_RISE = 0.03
HOLD_RADIUS = 0.03
SETTLE_STEPS = 10

# Once the jaw moves less than this per step it has stopped closing.
GRIPPER_SETTLED_DELTA = 0.005

# Action deltas below this are snapped to exactly zero so the environment holds position.
ZERO_SNAP = 0.05

ActFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SegmentResult:
    success: bool
    steps: int
    finished: bool


def snap_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    action[:3] = np.clip(action[:3], -1.0, 1.0)
    action[3] = np.clip(action[3], 0.0, 1.0)
    if np.linalg.norm(action[:3]) < ZERO_SNAP:
        action[:3] = 0.0
    return action


def start_segment(env: PickPlaceSceneEnv, seed: int) -> tuple[str, np.ndarray]:
    """Reset with random cubes, pick a target and hover over it with a noisy xy offset.

    Returns the target cube name and its true centre.
    """
    env.reset(seed=seed, randomize_cubes=True)
    rng = np.random.default_rng([seed, 1])
    cube = env.cube_names[int(rng.integers(len(env.cube_names)))]
    true_pos = env.cube_pos(cube)
    hover_over(env, true_pos[:2] + rng.uniform(-START_NOISE, START_NOISE, 2))
    return cube, true_pos


def hover_over(env: PickPlaceSceneEnv, xy: np.ndarray) -> None:
    """Move the grasp point above world ``xy`` at hover height; reads no simulator state but the arm's."""
    target = np.asarray([*xy, CUBE_HALF_SIZE])
    controller = PickPlaceController(DEFAULT_ACTION_SCALE)
    steps = 0
    while controller.phase is Phase.HOVER and steps < HOVER_MAX_STEPS:
        env.step(controller.act(env.ik_point_pos, target, target))
        steps += 1


def is_grasp_finished(hand_z: float, lowest_z: float) -> bool:
    return lowest_z < DESCENT_Z and hand_z > LIFT_DONE_Z


def is_cube_held(env: PickPlaceSceneEnv, cube: str) -> bool:
    """Evaluation only: the cube is lifted off the table and still next to the grasp point."""
    pos = env.cube_pos(cube)
    lifted = pos[2] - CUBE_HALF_SIZE > LIFT_SUCCESS_RISE
    return bool(lifted and np.linalg.norm(pos - env.ik_point_pos) < HOLD_RADIUS)


def observe(env: PickPlaceSceneEnv) -> tuple[np.ndarray, np.ndarray]:
    return env.render("wrist", IMAGE_SIZE), env.joint_angles()


def run_policy(env: PickPlaceSceneEnv, act: ActFn, on_step: Callable[[], None] | None = None) -> tuple[int, bool]:
    """Closed loop from the current hover pose until the finish rule or the step limit; ``(steps, finished)``."""
    lowest_z = np.inf
    finished = False
    steps = 0
    while steps < SEGMENT_MAX_STEPS and not finished:
        image, joints = observe(env)
        env.step(snap_action(act(image, joints)))
        steps += 1
        if on_step is not None:
            on_step()
        hand_z = float(env.ik_point_pos[2])
        lowest_z = min(lowest_z, hand_z)
        finished = is_grasp_finished(hand_z, lowest_z)
    return steps, finished


def run_segment(env: PickPlaceSceneEnv, cube: str, act: ActFn) -> SegmentResult:
    """``run_policy``, then hold still and judge whether ``cube`` is held."""
    steps, finished = run_policy(env, act)
    hold = np.asarray([0.0, 0.0, 0.0, GRIPPER_CLOSED], dtype=np.float32)
    for _ in range(SETTLE_STEPS):
        env.step(hold)
    return SegmentResult(is_cube_held(env, cube), steps, finished)


def collect_demo(env: PickPlaceSceneEnv, seed: int) -> tuple[dict[str, np.ndarray], bool]:
    """Scripted demo from a noisy hover: images, joints and noise-free action labels, plus success."""
    cube, true_pos = start_segment(env, seed)
    rng = np.random.default_rng([seed, 2])
    controller = PickPlaceController(DEFAULT_ACTION_SCALE)
    goal = true_pos
    images, joints, actions = [], [], []
    previous_gripper = np.inf
    while len(actions) < SEGMENT_MAX_STEPS:
        label = controller.act(env.ik_point_pos, env.cube_pos(cube), goal)
        if controller.phase not in GRASP_PHASES:
            break
        image, joint = observe(env)
        executed = label.copy()
        # A still jaw looks the same whether the script is holding or about to lift, so label it "lift".
        if controller.phase is Phase.CLOSE and abs(joint[-1] - previous_gripper) < GRIPPER_SETTLED_DELTA:
            label = label.copy()
            label[2] = CAREFUL_SPEED
        previous_gripper = joint[-1]
        images.append(image)
        joints.append(joint)
        actions.append(label)
        # Zero deltas (the close hold) are executed verbatim to keep the hold-position lock.
        if np.any(executed[:3] != 0.0):
            executed[:3] = np.clip(executed[:3] + rng.normal(0.0, DEMO_ACTION_NOISE, 3), -1.0, 1.0)
        env.step(executed)
    hold = np.asarray([0.0, 0.0, 0.0, GRIPPER_CLOSED], dtype=np.float32)
    for _ in range(SETTLE_STEPS):
        env.step(hold)
    episode = {
        "images": np.stack(images),
        "joints": np.stack(joints),
        "actions": np.stack(actions),
    }
    return episode, is_cube_held(env, cube)
