"""Multi-cube pick-and-place environment on top of ``SO101GymEnv``."""

import re
import tempfile
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from gymnasium import spaces
from lerobot_env_so101 import GymRenderingSpec, SO101GymEnv
from lerobot_env_so101.gripper import ctrl_to_normalized
from lerobot_env_so101.mujoco_env import ASSETS_DIR
from lerobot_env_so101.pick_place_env import PLACE_SUCCESS_DISTANCE
from lerobot_env_so101.policy import DEFAULT_ACTION_SCALE

SCENE_XML = Path(__file__).parent / "scene.xml"
ROBOT_XML_NAME = "so101_new_calib.xml"
ROBOT_MESH_DIR_NAME = "assets"

CUBE_HALF_SIZE = 0.02
CUBE_START_XY = {
    "cube_red": (0.18, -0.08),
    "cube_green": (0.18, 0.0),
    "cube_blue": (0.18, 0.08),
}
PLACE_NAMES = ("place_yellow", "place_cyan", "place_black")

MIN_CARRY_RISE = 0.03
PLACE_MAX_CUBE_SPEED = 0.01
PLACE_OPEN_FRACTION = 0.5

# Shoulder pan that swings the arm sideways past the workspace edge for an unobstructed view.
PARK_PAN = 1.8
PARK_MOVE_STEPS = 20
PARK_SETTLE_STEPS = 10

RENDER_HEIGHT = 480
RENDER_WIDTH = 640

# Camera clipped to the gripper body, looking down the jaws (pose relative to that body).
WRIST_CAMERA = '<camera name="wrist" pos="0.0 0.055 -0.045" euler="-0.57 0 0" fovy="70"/>'
GRIPPER_BODY_PATTERN = re.compile(r'<body name="gripper"[^>]*>')

# Cube spawn region for randomized resets, measured against the scripted grasp,
# and the minimum centre distance that keeps cubes out of the jaws' way.
CUBE_RANDOM_X = (0.15, 0.25)
CUBE_RANDOM_Y = (-0.12, 0.12)
CUBE_MIN_SPACING = 0.07


class PickPlaceSceneEnv(SO101GymEnv):
    """SO-101 with three coloured cubes and three flat place markers.

    Success for a (cube, place) pair needs the cube to have been lifted at some
    point, so pushing or teleporting it onto the marker does not count.
    """

    def __init__(self, seed: int = 0, control_dt: float = 0.1, physics_dt: float = 0.002):
        # MuJoCo resolves include and meshdir relative to the main MJCF, so the
        # scene is loaded from a temp dir that links to the upstream robot files.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            robot_xml = (ASSETS_DIR / ROBOT_XML_NAME).read_text()
            (tmp_dir / ROBOT_XML_NAME).write_text(
                GRIPPER_BODY_PATTERN.sub(lambda m: m.group(0) + WRIST_CAMERA, robot_xml, count=1)
            )
            (tmp_dir / ROBOT_MESH_DIR_NAME).symlink_to(ASSETS_DIR / ROBOT_MESH_DIR_NAME)
            scene_path = tmp_dir / SCENE_XML.name
            scene_path.write_text(SCENE_XML.read_text())
            super().__init__(
                xml_path=scene_path,
                seed=seed,
                control_dt=control_dt,
                physics_dt=physics_dt,
                render_spec=GymRenderingSpec(height=RENDER_HEIGHT, width=RENDER_WIDTH),
                action_scale=DEFAULT_ACTION_SCALE,
                top_down_ik=True,
                jaw_pads=True,
            )
        self.cube_names = tuple(CUBE_START_XY)
        self.place_names = PLACE_NAMES
        self._gripper_qpos_adr = self._model.joint("gripper").qposadr[0]
        self._was_lifted = dict.fromkeys(self.cube_names, False)
        self._extra_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.observation_space = spaces.Dict(
            {"environment_state": spaces.Box(-np.inf, np.inf, (3 * len(self.cube_names),), np.float32)}
        )

    def reset(self, seed=None, randomize_cubes: bool = False, **kwargs):
        """Reset the arm and cubes; ``randomize_cubes`` samples non-overlapping cube positions from ``seed``."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()
        starts = self._sample_cube_xy(np.random.default_rng(seed)) if randomize_cubes else CUBE_START_XY
        for name, xy in starts.items():
            self._data.jnt(name).qpos[:] = [*xy, CUBE_HALF_SIZE, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self._model, self._data)
        self._was_lifted = dict.fromkeys(self.cube_names, False)
        return self._observe(), {}

    def step(self, action):
        self.apply_action(action)
        for name in self.cube_names:
            if self.cube_pos(name)[2] - CUBE_HALF_SIZE > MIN_CARRY_RISE:
                self._was_lifted[name] = True
        return self._observe(), 0.0, False, False, {}

    def _sample_cube_xy(self, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
        points: list[np.ndarray] = []
        while len(points) < len(self.cube_names):
            point = rng.uniform((CUBE_RANDOM_X[0], CUBE_RANDOM_Y[0]), (CUBE_RANDOM_X[1], CUBE_RANDOM_Y[1]))
            if all(np.linalg.norm(point - other) >= CUBE_MIN_SPACING for other in points):
                points.append(point)
        return {name: (float(p[0]), float(p[1])) for name, p in zip(self.cube_names, points)}

    def _observe(self):
        state = np.concatenate([self.cube_pos(name) for name in self.cube_names])
        return {"environment_state": state.astype(np.float32)}

    def cube_pos(self, name: str) -> np.ndarray:
        return self._data.body(name).xpos.copy()

    def place_xy(self, name: str) -> np.ndarray:
        return self._model.geom(name).pos[:2].copy()

    def nearest_cube(self, xy) -> str:
        """Name of the cube whose centre is closest to the world-frame ``xy``."""
        xy = np.asarray(xy, dtype=np.float64)
        return min(self.cube_names, key=lambda n: np.linalg.norm(self.cube_pos(n)[:2] - xy))

    def place_error(self, cube: str, place: str) -> float:
        return float(np.linalg.norm(self.cube_pos(cube)[:2] - self.place_xy(place)))

    def is_placed(self, cube: str, place: str) -> bool:
        """Lifted earlier, within ``PLACE_SUCCESS_DISTANCE`` of the marker, jaw open, cube at rest."""
        dof_adr = self._model.joint(cube).dofadr[0]
        speed = np.linalg.norm(self._data.qvel[dof_adr : dof_adr + 3])
        gripper_range = self._model.actuator("gripper").ctrlrange
        opening = ctrl_to_normalized(self._data.qpos[self._gripper_qpos_adr], gripper_range)
        return bool(
            self._was_lifted[cube]
            and self.place_error(cube, place) < PLACE_SUCCESS_DISTANCE
            and opening > PLACE_OPEN_FRACTION
            and speed < PLACE_MAX_CUBE_SPEED
        )

    def render(self, camera: str = "front", size: int | None = None) -> np.ndarray:
        """Render ``camera``; ``size`` gives a square frame instead of the default 480x640."""
        if size is None:
            if self._viewer is None:
                self._viewer = mujoco.Renderer(self._model, self._render_specs.height, self._render_specs.width)
            renderer = self._viewer
        else:
            if (size, size) not in self._extra_renderers:
                self._extra_renderers[(size, size)] = mujoco.Renderer(self._model, size, size)
            renderer = self._extra_renderers[(size, size)]
        renderer.update_scene(self._data, camera=camera)
        return renderer.render()

    def close(self) -> None:
        for renderer in self._extra_renderers.values():
            renderer.close()
        self._extra_renderers.clear()
        super().close()

    def joint_angles(self) -> np.ndarray:
        """Five arm joint angles followed by the gripper angle, in radians."""
        arm = self._data.qpos[self._arm_dof_ids]
        return np.append(arm, self._data.qpos[self._gripper_qpos_adr]).astype(np.float32)

    def park(self, on_step: Callable[[], None] | None = None) -> None:
        """Swing the arm sideways out of the top camera's view, simulating the motion."""
        target = self._home_position.copy()
        target[self._arm_joint_index("shoulder_pan")] = PARK_PAN
        self._move_arm(target, on_step)

    def unpark(self, on_step: Callable[[], None] | None = None) -> None:
        """Bring the arm back to the home pose and resynchronise the IK target."""
        self._move_arm(self._home_position, on_step)
        self._target_ee_pos = self.ik_point_pos
        self._last_action_was_zero = False

    def _arm_joint_index(self, name: str) -> int:
        return int(np.nonzero(self._arm_dof_ids == self._model.joint(name).id)[0][0])

    def _move_arm(self, target_q: np.ndarray, on_step: Callable[[], None] | None) -> None:
        """Interpolate the arm actuator targets to ``target_q`` in joint space, then hold while it settles."""
        start = self._data.ctrl[self._arm_ctrl_ids].copy()
        for i in range(PARK_MOVE_STEPS + PARK_SETTLE_STEPS):
            fraction = min(1.0, (i + 1) / PARK_MOVE_STEPS)
            self._data.ctrl[self._arm_ctrl_ids] = start + fraction * (target_q - start)
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
            if on_step is not None:
                on_step()
