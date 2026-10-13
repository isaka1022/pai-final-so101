"""Closed-loop evaluation of a grasp policy on unseen cube layouts."""

import argparse
import time
from pathlib import Path

from pai_final.env import PickPlaceSceneEnv
from pai_final.grasp_net import LearnedPolicy, pick_device
from pai_final.learned import run_segment, start_segment

DEFAULT_POLICY = Path("out/learned/policy.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10000, help="episode i uses seed + i")
    args = parser.parse_args()

    device = pick_device()
    policy = LearnedPolicy(args.policy, device)
    env = PickPlaceSceneEnv()
    started = time.time()
    successes, unfinished = 0, 0
    for i in range(args.episodes):
        cube, _ = start_segment(env, args.seed + i)
        result = run_segment(env, cube, policy)
        successes += result.success
        unfinished += not result.finished
    env.close()
    mode = "image+joints" if policy.config.use_image else "joints only"
    print(f"{mode}: {successes}/{args.episodes} success ({unfinished} hit the step limit), "
          f"device {device}, {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
