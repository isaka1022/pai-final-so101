"""Run the upstream scripted pick-and-place controller to completion."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from lerobot_env_so101.policy import DEFAULT_ACTION_SCALE
from lerobot_env_so101.scripted.pick_place import (
    CAREFUL_SPEED,
    GOAL_TOLERANCE,
    GRASP_HEIGHT,
    OPEN_STEPS,
    PHASE_TIMEOUT_STEPS,
    Phase,
    PickPlaceController,
    phase_target,
)
from lerobot_env_so101.scripted.reach_close import GRIPPER_CLOSED, GRIPPER_OPEN

from pai_final.env import CUBE_HALF_SIZE, PickPlaceSceneEnv
from pai_final.learned import ActFn, hover_over, run_policy

MAX_STEPS = 300
SETTLE_STEPS = 10


@dataclass(frozen=True)
class PickPlaceResult:
    cube: str
    place: str
    success: bool
    error_m: float
    steps: int


def pick_and_place(
    env: PickPlaceSceneEnv,
    cube: str | np.ndarray,
    place_xy: np.ndarray,
    place_name: str | None = None,
    on_step: Callable[[Phase], None] | None = None,
) -> PickPlaceResult:
    """Pick ``cube`` (a name, or a world xy resolved to the nearest cube) and put it at ``place_xy``.

    Runs through RETREAT, then holds still for ``SETTLE_STEPS`` so the success
    check sees a resting cube. ``place_name`` defaults to the marker nearest ``place_xy``.
    """
    if not isinstance(cube, str):
        cube = env.nearest_cube(cube)
    place_xy = np.asarray(place_xy, dtype=np.float64)
    if place_name is None:
        place_name = min(env.place_names, key=lambda n: np.linalg.norm(env.place_xy(n) - place_xy))
    goal = np.asarray([*place_xy, CUBE_HALF_SIZE])

    controller = PickPlaceController(DEFAULT_ACTION_SCALE)
    steps = 0
    while steps < MAX_STEPS and not controller.done:
        action = controller.act(env.ik_point_pos, env.cube_pos(cube), goal)
        phase = controller.phase
        env.step(action)
        steps += 1
        if on_step is not None:
            on_step(phase)
    for _ in range(SETTLE_STEPS):
        env.step(controller.act(env.ik_point_pos, env.cube_pos(cube), goal))
    return PickPlaceResult(cube, place_name, env.is_placed(cube, place_name), env.place_error(cube, place_name), steps)


CARRY_PHASES = (Phase.TRAVERSE, Phase.LOWER, Phase.OPEN, Phase.RETREAT)
_CARRY_GRIPPER = {Phase.TRAVERSE: GRIPPER_CLOSED, Phase.LOWER: GRIPPER_CLOSED, Phase.OPEN: GRIPPER_OPEN, Phase.RETREAT: GRIPPER_OPEN}


def carry_and_release(
    env: PickPlaceSceneEnv, place_xy: np.ndarray, on_step: Callable[[], None] | None = None
) -> int:
    """Traverse, lower, open and retreat from the hand pose alone; returns the steps taken.

    The cube is assumed to hang ``GRASP_HEIGHT`` below the grasp point, as in the scripted grasp.
    """
    goal = np.asarray([*place_xy, CUBE_HALF_SIZE])
    steps = 0
    for phase in CARRY_PHASES:
        phase_steps = 0
        while True:
            grasp = env.ik_point_pos
            if phase is Phase.OPEN:
                finished, delta = phase_steps >= OPEN_STEPS, np.zeros(3)
            else:
                held = grasp - np.asarray([0.0, 0.0, GRASP_HEIGHT])
                error = phase_target(phase, grasp, held, goal, CUBE_HALF_SIZE) - grasp
                finished = np.linalg.norm(error) < GOAL_TOLERANCE or phase_steps >= PHASE_TIMEOUT_STEPS
                limit = CAREFUL_SPEED if phase is Phase.LOWER else 1.0
                delta = np.clip(error / DEFAULT_ACTION_SCALE, -limit, limit)
            if finished:
                break
            env.step(np.asarray([*delta, _CARRY_GRIPPER[phase]], dtype=np.float32))
            phase_steps += 1
            steps += 1
            if on_step is not None:
                on_step()
    for _ in range(SETTLE_STEPS):
        env.step(np.asarray([0.0, 0.0, 0.0, GRIPPER_OPEN], dtype=np.float32))
    return steps


def grasp_and_place_learned(
    env: PickPlaceSceneEnv,
    pick_xy: np.ndarray,
    place_xy: np.ndarray,
    act: ActFn,
    on_step: Callable[[], None] | None = None,
) -> int:
    """Hover over ``pick_xy``, let the policy ``act`` grasp and lift, then carry to ``place_xy`` and release.

    Reads only the arm state and the two world points, never a cube position.
    """
    hover_over(env, np.asarray(pick_xy))
    grasp_steps, _ = run_policy(env, act, on_step)
    return grasp_steps + carry_and_release(env, np.asarray(place_xy, dtype=np.float64), on_step)


def pick_and_place_learned(
    env: PickPlaceSceneEnv,
    pick_xy: np.ndarray,
    place_xy: np.ndarray,
    act: ActFn,
    on_step: Callable[[], None] | None = None,
) -> PickPlaceResult:
    """``grasp_and_place_learned``, judging the cube nearest ``pick_xy`` (the only cube state read, outside the control loop)."""
    cube = env.nearest_cube(pick_xy)
    steps = grasp_and_place_learned(env, pick_xy, place_xy, act, on_step)
    place_name = min(env.place_names, key=lambda n: np.linalg.norm(env.place_xy(n) - place_xy))
    return PickPlaceResult(cube, place_name, env.is_placed(cube, place_name), env.place_error(cube, place_name), steps)
