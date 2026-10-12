"""Run two pick-and-place tasks in one episode and dump frames to out/."""

from pathlib import Path

from lerobot_env_so101.scripted.pick_place import Phase
from PIL import Image

from pai_final.env import PickPlaceSceneEnv
from pai_final.skill import pick_and_place

OUT_DIR = Path("out")
TASKS = (("cube_red", "place_cyan"), ("cube_blue", "place_yellow"))


def save(env: PickPlaceSceneEnv, name: str, camera: str = "front") -> None:
    Image.fromarray(env.render(camera)).save(OUT_DIR / f"{name}.png")


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    env = PickPlaceSceneEnv()
    env.reset(seed=0)
    save(env, "initial_front")
    env.park()
    save(env, "initial_top", "top")
    env.unpark()

    failures = 0
    for index, (cube, place) in enumerate(TASKS, start=1):
        captured = False

        def on_step(phase: Phase) -> None:
            nonlocal captured
            if phase is Phase.TRAVERSE and not captured:
                save(env, f"task{index}_lifted")
                captured = True

        result = pick_and_place(env, cube, env.place_xy(place), place, on_step)
        save(env, f"task{index}_done")
        env.park()
        save(env, f"task{index}_done_top", "top")
        env.unpark()
        print(f"{cube} -> {place}: success={result.success} error={result.error_m * 1000:.1f} mm steps={result.steps}")
        failures += not result.success
    env.close()
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
