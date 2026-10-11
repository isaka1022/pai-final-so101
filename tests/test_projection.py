import numpy as np
import pytest

from pai_final.env import CUBE_HALF_SIZE, RENDER_HEIGHT, RENDER_WIDTH, PickPlaceSceneEnv
from pai_final.projection import NORMALISED_RANGE, pixel_to_world_xy, world_to_pixel

DOMINANCE = 100  # lighting washes colours out, so test channel dominance rather than absolute level
CUBE_RGB_RULES = {
    "cube_red": lambda r, g, b: (r - np.maximum(g, b)) > DOMINANCE,
    "cube_green": lambda r, g, b: (g - np.maximum(r, b)) > DOMINANCE,
    "cube_blue": lambda r, g, b: (b - np.maximum(r, g)) > DOMINANCE,
}
CENTROID_TOLERANCE_M = 0.005
FRONT_CAMERA_Z = 0.5  # front looks downward, so a plane above it is never hit


@pytest.fixture
def env():
    env = PickPlaceSceneEnv()
    env.reset(seed=0)
    yield env
    env.close()


def test_colour_centroid_matches_cube_position(env):
    frame = env.render("top", park=True).astype(int)
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    for name, rule in CUBE_RGB_RULES.items():
        rows, cols = np.nonzero(rule(r, g, b))
        assert rows.size > 0, name
        # Pixel indices are centres, so +0.5 gives continuous image coordinates.
        point = [
            (rows.mean() + 0.5) / RENDER_HEIGHT * NORMALISED_RANGE,
            (cols.mean() + 0.5) / RENDER_WIDTH * NORMALISED_RANGE,
        ]
        xy = pixel_to_world_xy(
            env.model, env.data, "top", point, CUBE_HALF_SIZE * 2, RENDER_HEIGHT, RENDER_WIDTH
        )
        error = xy - env.cube_pos(name)[:2]
        print(f"{name}: centroid error mm (x, y) = {error * 1000}")
        assert np.linalg.norm(error) < CENTROID_TOLERANCE_M, (name, error)


def test_roundtrip(env):
    xyz = env.cube_pos("cube_red") + [0, 0, CUBE_HALF_SIZE]
    point = world_to_pixel(env.model, env.data, "top", xyz, RENDER_HEIGHT, RENDER_WIDTH)
    xy = pixel_to_world_xy(env.model, env.data, "top", point, xyz[2], RENDER_HEIGHT, RENDER_WIDTH)
    assert np.allclose(xy, xyz[:2], atol=1e-3)


def test_plane_behind_ray_raises(env):
    with pytest.raises(ValueError):
        pixel_to_world_xy(env.model, env.data, "front", [0, 500], FRONT_CAMERA_Z + 0.1, RENDER_HEIGHT, RENDER_WIDTH)
