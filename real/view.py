"""rerun viewer shared by the wrist-camera scripts."""
from __future__ import annotations

import numpy as np

from hand import Hand

# Past this the viewer drops its oldest data; unbounded, a long session fills RAM and then swap on disk.
VIEWER_MEMORY_LIMIT = "1GiB"
JPEG_QUALITY = 80


def start_viewer(app_id: str):
    import rerun as rr

    rr.init(app_id)
    rr.spawn(memory_limit=VIEWER_MEMORY_LIMIT)
    return rr


def log_camera(rr, frame_rgb: np.ndarray, hand: Hand | None) -> None:
    height, width = frame_rgb.shape[:2]
    rr.log("camera", rr.Image(frame_rgb).compress(jpeg_quality=JPEG_QUALITY))
    if hand is None:
        rr.log("camera/hand/landmarks", rr.Clear(recursive=False))
        rr.log("camera/hand/center", rr.Clear(recursive=False))
        return
    rr.log("camera/hand/landmarks", rr.Points2D(hand.landmarks_px, radii=3, colors=[0, 255, 0]))
    center = [(hand.x + 0.5) * width, (hand.y + 0.5) * height]
    rr.log("camera/hand/center", rr.Points2D([center], radii=8, colors=[255, 0, 0]))
