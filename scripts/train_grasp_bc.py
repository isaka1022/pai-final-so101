"""Train the wrist-camera grasp policy by one-step behaviour cloning (MSE)."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pai_final.grasp_net import (
    GraspNet,
    NetConfig,
    pick_device,
    save_checkpoint,
    to_image_tensor,
)

DEFAULT_DEMOS = Path("out/learned/demos.npz")
DEFAULT_OUT = Path("out/learned/policy.pt")
VAL_FRACTION = 0.1
SHIFT_PIXELS = 4
CLOSING_JAW_RANGE = (0.5, 1.65)
CLOSING_WEIGHT = 5.0


def random_shift(images: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = images.shape
    padded = F.pad(images, (SHIFT_PIXELS,) * 4, mode="replicate")
    dx, dy = np.random.randint(0, 2 * SHIFT_PIXELS + 1, 2)
    return padded[:, :, dy : dy + height, dx : dx + width]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-image", action="store_true", help="joint-angle-only policy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.demos)
    ends = data["episode_ends"]
    n_episodes = len(ends)
    starts = np.concatenate([[0], ends[:-1]])
    episode_of = np.repeat(np.arange(n_episodes), ends - starts)
    val_episodes = np.random.default_rng(args.seed).permutation(n_episodes)[: max(1, int(n_episodes * VAL_FRACTION))]
    is_val = np.isin(episode_of, val_episodes)

    joints = data["joints"]
    joint_mean = joints[~is_val].mean(axis=0)
    joint_std = joints[~is_val].std(axis=0) + 1e-6
    device = pick_device()
    config = NetConfig(use_image=not args.no_image, image_size=data["images"].shape[1])

    images_t = torch.as_tensor(data["images"], device=device) if config.use_image else None
    joints_t = torch.as_tensor((joints - joint_mean) / joint_std, device=device).float()
    targets_t = torch.as_tensor(data["actions"], device=device)
    jaw = joints[:, -1]
    in_closing = (jaw > CLOSING_JAW_RANGE[0]) & (jaw < CLOSING_JAW_RANGE[1])
    weights_t = torch.as_tensor(np.where(in_closing, CLOSING_WEIGHT, 1.0).astype(np.float32), device=device)

    def weighted_mse(pred: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return ((pred - targets_t[idx]).square().mean(dim=1) * weights_t[idx]).sum() / weights_t[idx].sum()

    train_idx = np.nonzero(~is_val)[0]
    val_idx = torch.as_tensor(np.nonzero(is_val)[0], device=device)

    model = GraspNet(config).to(device)
    # Materialise the lazy linear layer before building the optimizer.
    dummy_image = to_image_tensor(data["images"][:1], device) if config.use_image else None
    model(dummy_image, joints_t[:1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    def batch_inputs(idx: torch.Tensor, augment: bool):
        image = None
        if config.use_image:
            image = images_t[idx].permute(0, 3, 1, 2).float() / 255.0
            if augment:
                image = random_shift(image)
        return image, joints_t[idx]

    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.as_tensor(np.random.permutation(train_idx), device=device)
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            loss = weighted_mse(model(*batch_inputs(idx, True)), idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = weighted_mse(model(*batch_inputs(val_idx, False)), val_idx).item()
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train {total / len(order):.5f}  val {val_loss:.5f}", flush=True)

    save_checkpoint(args.out, model, joint_mean, joint_std, {**vars(args), "demos": str(args.demos), "val_episodes": val_episodes.tolist()})
    print(f"saved {args.out} on {device}; {len(train_idx)} train / {len(val_idx)} val steps; {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
