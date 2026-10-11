"""Run the upstream scripted pick-and-place controller to completion."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from lerobot_env_so101.policy import DEFAULT_ACTION_SCALE
from lerobot_env_so101.scripted.pick_place import Phase, PickPlaceController

from pai_final.env import CUBE_HALF_SIZE, PickPlaceSceneEnv

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
