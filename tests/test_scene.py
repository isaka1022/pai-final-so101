import numpy as np
import pytest

from pai_final.env import CUBE_HALF_SIZE, PickPlaceSceneEnv
from pai_final.skill import pick_and_place


@pytest.fixture
def env():
    env = PickPlaceSceneEnv()
    env.reset(seed=0)
    yield env
    env.close()


def test_pick_and_place_succeeds(env):
    result = pick_and_place(env, "cube_red", env.place_xy("place_cyan"), "place_cyan")
    assert result.success


def test_teleported_cube_is_not_success(env):
    env.data.jnt("cube_red").qpos[:3] = [*env.place_xy("place_cyan"), CUBE_HALF_SIZE]
    zero_action = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(20):
        env.step(zero_action)
    assert env.place_error("cube_red", "place_cyan") < 0.01
    assert not env.is_placed("cube_red", "place_cyan")


def test_park_and_unpark_leave_cubes_in_place(env):
    before = {n: env.cube_pos(n) for n in env.cube_names}
    env.park()
    env.unpark()
    for name, pos in before.items():
        assert np.linalg.norm(env.cube_pos(name) - pos) < 0.001
