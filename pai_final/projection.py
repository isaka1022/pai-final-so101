"""Map normalised image points of a MuJoCo camera to and from world points on a horizontal plane."""

import mujoco
import numpy as np

NORMALISED_RANGE = 1000.0
PARALLEL_EPS = 1e-9


def _camera(model: mujoco.MjModel, data: mujoco.MjData, camera: str):
    cam = model.camera(camera)
    return data.cam_xpos[cam.id], data.cam_xmat[cam.id].reshape(3, 3), float(model.cam_fovy[cam.id])


def _focal_px(fovy_deg: float, height: int) -> float:
    return (height / 2) / np.tan(np.radians(fovy_deg) / 2)


def pixel_to_world_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str,
    point_yx: tuple[float, float],
    plane_z: float,
    height: int,
    width: int,
) -> np.ndarray:
    """World xy where the ray through ``point_yx`` (``[y, x]``, 0-1000 of image height/width) meets z = ``plane_z``."""
    pos, rot, fovy = _camera(model, data, camera)
    y_norm, x_norm = point_yx
    f = _focal_px(fovy, height)
    right = x_norm / NORMALISED_RANGE * width - width / 2
    down = y_norm / NORMALISED_RANGE * height - height / 2
    # Camera frame: +x right, +y up, looks along -z.
    direction = rot @ np.array([right / f, -down / f, -1.0])
    if abs(direction[2]) < PARALLEL_EPS:
        raise ValueError(f"ray from camera {camera!r} is parallel to the plane z={plane_z}")
    t = (plane_z - pos[2]) / direction[2]
    if t <= 0:
        raise ValueError(f"ray from camera {camera!r} does not reach the plane z={plane_z}")
    return (pos + t * direction)[:2]


def world_to_pixel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str,
    xyz,
    height: int,
    width: int,
) -> np.ndarray:
    """Normalised ``[y, x]`` (0-1000) of world point ``xyz`` in ``camera``."""
    pos, rot, fovy = _camera(model, data, camera)
    local = rot.T @ (np.asarray(xyz, dtype=np.float64) - pos)
    if local[2] >= 0:
        raise ValueError(f"point is behind camera {camera!r}")
    f = _focal_px(fovy, height)
    right = f * local[0] / -local[2]
    down = -f * local[1] / -local[2]
    return np.array(
        [(down + height / 2) / height * NORMALISED_RANGE, (right + width / 2) / width * NORMALISED_RANGE]
    )
