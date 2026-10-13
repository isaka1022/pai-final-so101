import sys

import numpy as np
import pytest

from pai_final.env import CUBE_MIN_SPACING, PickPlaceSceneEnv
from pai_final.learned import collect_demo, is_cube_held, start_segment
from pai_final.skill import grasp_and_place_learned

SEED = 3


@pytest.fixture
def env():
    env = PickPlaceSceneEnv()
    yield env
    env.close()


def test_randomized_reset_keeps_cubes_apart(env):
    env.reset(seed=SEED, randomize_cubes=True)
    xy = [env.cube_pos(name)[:2] for name in env.cube_names]
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(xy) for b in xy[i + 1 :]]
    assert min(gaps) >= CUBE_MIN_SPACING


def test_wrist_frame_has_requested_size(env):
    env.reset(seed=SEED)
    assert env.render("wrist", 96).shape == (96, 96, 3)


def test_scripted_demo_lifts_cube_but_hovering_does_not(env):
    _, demo_success = collect_demo(env, SEED)
    assert demo_success
    cube, _ = start_segment(env, SEED)
    assert not is_cube_held(env, cube)


def test_learned_path_never_reads_cube_state(env):
    env.reset(seed=0)
    callers = set()
    original = env.cube_pos

    def spy(name):
        callers.add(sys._getframe(1).f_code.co_name)
        return original(name)

    env.cube_pos = spy
    calls = iter(range(1000))

    def scripted_act(_image, _joints):
        k = next(calls)
        if k < 12:
            return np.asarray([0.0, 0.0, -0.4, 1.0])
        return np.asarray([0.0, 0.0, 0.4 if k > 22 else 0.0, 0.0])

    place_xy = env.place_xy("place_cyan")
    grasp_and_place_learned(env, np.asarray([0.18, -0.08]), place_xy, scripted_act)
    assert callers <= {"step", "_observe"}  # env bookkeeping only
    assert np.linalg.norm(env.ik_point_pos[:2] - place_xy) < 0.01
