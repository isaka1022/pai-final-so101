"""Collect scripted wrist-camera grasp demos (successful episodes only) into an npz."""

import argparse
import time
from pathlib import Path

import numpy as np

from pai_final.env import PickPlaceSceneEnv
from pai_final.learned import collect_demo

DEFAULT_OUT = Path("out/learned/demos.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0, help="episode i uses seed + i")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    env = PickPlaceSceneEnv()
    started = time.time()
    chunks: dict[str, list[np.ndarray]] = {"images": [], "joints": [], "actions": []}
    lengths, kept = [], 0
    for i in range(args.episodes):
        episode, success = collect_demo(env, args.seed + i)
        if not success:
            continue
        kept += 1
        for key, values in chunks.items():
            values.append(episode[key])
        lengths.append(len(episode["actions"]))
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **{key: np.concatenate(values) for key, values in chunks.items()},
        episode_ends=np.cumsum(lengths),
    )
    print(f"saved {args.out}: {kept}/{args.episodes} successful episodes, {sum(lengths)} steps, "
          f"{time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
