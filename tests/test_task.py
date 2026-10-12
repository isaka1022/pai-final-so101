from pathlib import Path

import numpy as np

from pai_final.env import CUBE_START_XY
from pai_final.task import run_episode

FIXTURE = Path(__file__).parent / "fixtures" / "red_to_cyan_response.txt"
UNMOVED_TOLERANCE_M = 0.002
PLACED_AFTER_PARK_M = 0.03


def test_replay_places_red_on_cyan_and_leaves_others(tmp_path):
    record = run_episode("put the red cube on the cyan square", tmp_path / "out.mp4", FIXTURE.read_text())
    assert (tmp_path / "out.mp4").stat().st_size > 0
    assert (record["result"]["cube"], record["result"]["place"]) == ("cube_red", "place_cyan")
    assert record["result"]["success"]
    assert np.linalg.norm(np.array(record["final_cube_xy"]["cube_red"]) - record["place_world_xy"]) < PLACED_AFTER_PARK_M
    for name in ("cube_green", "cube_blue"):
        assert np.linalg.norm(np.array(record["final_cube_xy"][name]) - CUBE_START_XY[name]) < UNMOVED_TOLERANCE_M
