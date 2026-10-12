"""Join run.py recordings into one video, with an instruction caption band and a title card per clip."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SPEED_LABEL = "1x"
CARD_SECONDS = 2
BAND_HEIGHT = 56
BAND_COLOUR = (20, 20, 20)
TEXT_COLOUR = (255, 255, 255)
BAND_FONT_SIZE = 28
CARD_FONT_SIZE = 44
MARGIN = 16
BAND_LABEL_WIDTH = 60
MIN_FONT_SIZE = 12
FONT_CANDIDATES = (
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def find_font(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [Path(p) for p in FONT_CANDIDATES]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("no CJK font found; searched:\n" + "\n".join(f"  {p}" for p in candidates) + "\npass --font <path>")


def fit_font(font_path: Path, size: int, text: str, max_width: float) -> ImageFont.FreeTypeFont:
    """Largest font up to ``size`` whose rendering of ``text`` fits ``max_width``."""
    while True:
        font = ImageFont.truetype(str(font_path), size)
        if font.getlength(text) <= max_width or size <= MIN_FONT_SIZE:
            return font
        size -= 1


def with_band(frame: np.ndarray, text: str, font_path: Path) -> np.ndarray:
    height, width = frame.shape[:2]
    canvas = Image.new("RGB", (width, height + BAND_HEIGHT), BAND_COLOUR)
    canvas.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(canvas)
    mid = height + BAND_HEIGHT // 2
    font = fit_font(font_path, BAND_FONT_SIZE, text, width - 2 * MARGIN - BAND_LABEL_WIDTH)
    draw.text((MARGIN, mid), text, fill=TEXT_COLOUR, font=font, anchor="lm")
    draw.text((width - MARGIN, mid), SPEED_LABEL, fill=TEXT_COLOUR, font=font, anchor="rm")
    return np.asarray(canvas)


def title_card(size: tuple[int, int], text: str, font_path: Path) -> np.ndarray:
    width, height = size
    canvas = Image.new("RGB", size, BAND_COLOUR)
    font = fit_font(font_path, CARD_FONT_SIZE, text, width - 2 * MARGIN)
    ImageDraw.Draw(canvas).text((width / 2, height / 2), text, fill=TEXT_COLOUR, font=font, anchor="mm")
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path, help="JSON records written by run.py (the .mp4 beside each is read)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", type=Path, help="CJK-capable font file")
    args = parser.parse_args()

    font_path = find_font(args.font)

    frames: list[np.ndarray] = []
    fps = None
    for record_path in args.records:
        instruction = json.loads(record_path.read_text())["instruction"]
        reader = imageio.get_reader(record_path.with_suffix(".mp4"))
        fps = fps or reader.get_meta_data()["fps"]
        clip = [with_band(f, instruction, font_path) for f in reader]
        reader.close()
        height, width = clip[0].shape[:2]
        assert height % 2 == 0 and width % 2 == 0, (height, width)
        frames.extend([title_card((width, height), instruction, font_path)] * round(CARD_SECONDS * fps))
        frames.extend(clip)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=fps, macro_block_size=1)
    print(f"{args.out}: {len(frames)} frames, {len(frames) / fps:.1f} s, {frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == "__main__":
    main()
