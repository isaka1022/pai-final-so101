"""Behaviour-cloning network for the wrist-camera grasp segment (requires torch)."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pai_final.learned import IMAGE_SIZE

JOINT_DIM = 6
ACTION_DIM = 4
HIDDEN = 256


@dataclass(frozen=True)
class NetConfig:
    use_image: bool = True
    image_size: int = IMAGE_SIZE


class GraspNet(nn.Module):
    """Small CNN over the wrist image plus an MLP over joints, predicting the next action."""

    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config
        feature_dim = 0
        if config.use_image:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(HIDDEN), nn.ReLU(),
            )
            feature_dim += HIDDEN
        self.joint_mlp = nn.Sequential(nn.Linear(JOINT_DIM, 64), nn.ReLU())
        feature_dim += 64
        self.head = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM),
        )

    def forward(self, image: torch.Tensor | None, joints: torch.Tensor) -> torch.Tensor:
        """``image`` is float (B, 3, H, W) in [0, 1]; ``joints`` is normalised (B, 6)."""
        features = [self.joint_mlp(joints)]
        if self.config.use_image:
            features.append(self.cnn(image))
        return self.head(torch.cat(features, dim=1))


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_image_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 (B, H, W, 3) to float (B, 3, H, W) in [0, 1]."""
    return torch.as_tensor(images, device=device).permute(0, 3, 1, 2).float() / 255.0


def save_checkpoint(path: Path, model: GraspNet, joint_mean, joint_std, train_args: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "joint_mean": torch.as_tensor(np.asarray(joint_mean, dtype=np.float32)),
            "joint_std": torch.as_tensor(np.asarray(joint_std, dtype=np.float32)),
            "train_args": {k: str(v) if isinstance(v, Path) else v for k, v in train_args.items()},
        },
        path,
    )


class LearnedPolicy:
    """Closed-loop policy from a checkpoint: ``(image, joints) -> action``, next action."""

    def __init__(self, path: Path, device: torch.device):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.config = NetConfig(**ckpt["config"])
        self.model = GraspNet(self.config)
        self.model(torch.zeros(1, 3, self.config.image_size, self.config.image_size), torch.zeros(1, JOINT_DIM))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.joint_mean = ckpt["joint_mean"].numpy()
        self.joint_std = ckpt["joint_std"].numpy()
        self.train_args = ckpt["train_args"]

    @torch.no_grad()
    def __call__(self, image: np.ndarray, joints: np.ndarray) -> np.ndarray:
        joints_t = torch.as_tensor((joints - self.joint_mean) / self.joint_std, device=self.device)[None].float()
        image_t = to_image_tensor(image[None], self.device) if self.config.use_image else None
        return self.model(image_t, joints_t)[0].cpu().numpy()
