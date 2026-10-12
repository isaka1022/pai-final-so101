"""One language-instructed pick-and-place episode: observe, plan with a VLM answer, execute, record."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from pai_final.env import CUBE_HALF_SIZE, PickPlaceSceneEnv
from pai_final.planner import MODEL_ID, NORMALISED_RANGE, ask_vlm, parse_points
from pai_final.projection import pixel_to_world_xy
from pai_final.skill import pick_and_place

SEED = 0
OBSERVE_CAMERA = "top"
RECORD_CAMERA = "front"
FPS = 10
OVERVIEW_SECONDS = 2
# Pick points land on the cube's top face; place points on the marker surface (box half-height 0.001, centred at 0.001).
PICK_PLANE_Z = 2 * CUBE_HALF_SIZE
PLACE_PLANE_Z = 0.002
MARKER_RADIUS = 7
PICK_COLOUR = (255, 0, 255)
PLACE_COLOUR = (255, 140, 0)


def annotate(frame: np.ndarray, pick_yx, place_yx) -> np.ndarray:
    """Copy of ``frame`` with the pick and place points marked and labelled."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    height, width = frame.shape[:2]
    for label, (y, x), colour in (("pick", pick_yx, PICK_COLOUR), ("place", place_yx, PLACE_COLOUR)):
        cx, cy = x / NORMALISED_RANGE * width, y / NORMALISED_RANGE * height
        draw.ellipse([cx - MARKER_RADIUS, cy - MARKER_RADIUS, cx + MARKER_RADIUS, cy + MARKER_RADIUS], outline=colour, width=3)
        draw.text((cx + MARKER_RADIUS + 3, cy - MARKER_RADIUS), label, fill=colour)
    return np.asarray(image)


def run_episode(
    instruction: str,
    out_video: Path,
    raw_response: str | None = None,
    ask: Callable[[str, str], str] = ask_vlm,
) -> dict:
    """Run one episode and write ``out_video``, its observation PNG and a ``.json`` record beside it.

    With ``raw_response`` the VLM is not called and that answer is used instead.
    """
    out_video.parent.mkdir(parents=True, exist_ok=True)
    env = PickPlaceSceneEnv()
    try:
        env.reset(seed=SEED)
        frames: list[np.ndarray] = []
        record = lambda: frames.append(env.render(RECORD_CAMERA))  # noqa: E731
        record()
        env.park(record)
        top = env.render(OBSERVE_CAMERA)
        height, width = top.shape[:2]
        image_path = out_video.with_name(out_video.stem + "_top.png")
        Image.fromarray(top).save(image_path)

        if raw_response is None:
            raw_response = ask(str(image_path), instruction)
        pick_yx, place_yx = parse_points(raw_response)
        to_world = lambda point, z: pixel_to_world_xy(env.model, env.data, OBSERVE_CAMERA, point, z, height, width)  # noqa: E731
        pick_xy, place_xy = to_world(pick_yx, PICK_PLANE_Z), to_world(place_yx, PLACE_PLANE_Z)
        cube = env.nearest_cube(pick_xy)

        overview = annotate(top, pick_yx, place_yx)
        frames.extend([overview] * (OVERVIEW_SECONDS * FPS))
        env.unpark(record)
        result = pick_and_place(env, cube, place_xy, on_step=lambda _phase: record())
        record()
        env.park(record)
        frames.extend([env.render(OBSERVE_CAMERA)] * (OVERVIEW_SECONDS * FPS))
        imageio.mimsave(out_video, frames, fps=FPS, macro_block_size=1)
        final_cube_xy = {name: env.cube_pos(name)[:2].tolist() for name in env.cube_names}
    finally:
        env.close()

    record_data = {
        "instruction": instruction,
        "model": MODEL_ID,
        "raw_response": raw_response,
        "pick_yx": list(pick_yx),
        "place_yx": list(place_yx),
        "camera": OBSERVE_CAMERA,
        "image_size": {"height": height, "width": width},
        "seed": SEED,
        "pick_world_xy": pick_xy.tolist(),
        "place_world_xy": place_xy.tolist(),
        "result": asdict(result),
        "final_cube_xy": final_cube_xy,
    }
    out_video.with_suffix(".json").write_text(json.dumps(record_data, indent=2, ensure_ascii=False) + "\n")
    return record_data
