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
CARD_NOTE_FONT_SIZE = 26
CARD_LINE_GAP = 48
INTRO_TITLE = "言語指示で SO-101 に pick-and-place させる"
INTRO_NOTE = f"MuJoCo シミュレーション / VLM: Gemini Robotics-ER / 再生速度 {SPEED_LABEL}"
TITLE_SECONDS = 3
DIAGRAM_SECONDS = 7
DIAGRAM_STAGES = (
    ("指示文 ＋ 俯瞰カメラの画像（top camera）",),
    ("VLM（Gemini Robotics-ER）が、", "掴む点と置く点を画像上で返す"),
    ("カメラの位置・向き・画角から、", "机の上の座標に直す"),
    ("把持: スクリプト（逆運動学）か、", "手先カメラ（wrist camera）の学習ポリシー"),
    ("運んで置く",),
)
DIAGRAM_BOX_COLOUR = (48, 48, 48)
DIAGRAM_BOX_OUTLINE = (200, 200, 200)
DIAGRAM_BOX_RADIUS = 10
DIAGRAM_BOX_PAD = 12
DIAGRAM_FONT_SIZE = 24
DIAGRAM_ARROW_GAP = 24
DIAGRAM_ARROW_WIDTH = 4
DIAGRAM_ARROW_HEAD = 10
LEARNED_NOTE = "把持: 手先カメラの学習ポリシー"
LEARNED_BAND_PREFIX = "[学習] "
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


def title_card(size: tuple[int, int], text: str, font_path: Path, note: str = "") -> np.ndarray:
    width, height = size
    canvas = Image.new("RGB", size, BAND_COLOUR)
    draw = ImageDraw.Draw(canvas)
    font = fit_font(font_path, CARD_FONT_SIZE, text, width - 2 * MARGIN)
    draw.text((width / 2, height / 2), text, fill=TEXT_COLOUR, font=font, anchor="mm")
    if note:
        note_font = fit_font(font_path, CARD_NOTE_FONT_SIZE, note, width - 2 * MARGIN)
        draw.text((width / 2, height / 2 + CARD_LINE_GAP), note, fill=TEXT_COLOUR, font=note_font, anchor="mm")
    return np.asarray(canvas)


def mechanism_card(size: tuple[int, int], font_path: Path) -> np.ndarray:
    width, height = size
    canvas = Image.new("RGB", size, BAND_COLOUR)
    draw = ImageDraw.Draw(canvas)
    count = len(DIAGRAM_STAGES)
    box_height = (height - 2 * MARGIN - (count - 1) * DIAGRAM_ARROW_GAP) / count
    inner = width - 2 * MARGIN - 2 * DIAGRAM_BOX_PAD
    for i, lines in enumerate(DIAGRAM_STAGES):
        top = MARGIN + i * (box_height + DIAGRAM_ARROW_GAP)
        draw.rounded_rectangle((MARGIN, top, width - MARGIN, top + box_height), DIAGRAM_BOX_RADIUS, fill=DIAGRAM_BOX_COLOUR, outline=DIAGRAM_BOX_OUTLINE)
        longest = max(lines, key=len)
        font = fit_font(font_path, DIAGRAM_FONT_SIZE, longest, inner)
        line_height = font.size * 1.3
        y = top + box_height / 2 - (len(lines) - 1) * line_height / 2
        for line in lines:
            draw.text((width / 2, y), line, fill=TEXT_COLOUR, font=font, anchor="mm")
            y += line_height
        if i < count - 1:
            x, y0 = width / 2, top + box_height
            y1 = y0 + DIAGRAM_ARROW_GAP
            draw.line((x, y0, x, y1 - DIAGRAM_ARROW_HEAD), fill=TEXT_COLOUR, width=DIAGRAM_ARROW_WIDTH)
            draw.polygon([(x - DIAGRAM_ARROW_HEAD, y1 - DIAGRAM_ARROW_HEAD), (x + DIAGRAM_ARROW_HEAD, y1 - DIAGRAM_ARROW_HEAD), (x, y1)], fill=TEXT_COLOUR)
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path, help="JSON records written by run.py (the .mp4 beside each is read)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--intro", action="store_true", help="prepend a title card and a pipeline diagram")
    parser.add_argument("--font", type=Path, help="CJK-capable font file")
    args = parser.parse_args()

    font_path = find_font(args.font)

    frames: list[np.ndarray] = []
    fps = None
    for record_path in args.records:
        record = json.loads(record_path.read_text())
        instruction = record["instruction"]
        learned = "policy" in record
        caption = (LEARNED_BAND_PREFIX if learned else "") + instruction
        reader = imageio.get_reader(record_path.with_suffix(".mp4"))
        fps = fps or reader.get_meta_data()["fps"]
        clip = [with_band(f, caption, font_path) for f in reader]
        reader.close()
        height, width = clip[0].shape[:2]
        assert height % 2 == 0 and width % 2 == 0, (height, width)
        if args.intro and not frames:
            frames.extend([title_card((width, height), INTRO_TITLE, font_path, INTRO_NOTE)] * round(TITLE_SECONDS * fps))
            frames.extend([mechanism_card((width, height), font_path)] * round(DIAGRAM_SECONDS * fps))
        frames.extend([title_card((width, height), instruction, font_path, LEARNED_NOTE if learned else "")] * round(CARD_SECONDS * fps))
        frames.extend(clip)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=fps, macro_block_size=1)
    print(f"{args.out}: {len(frames)} frames, {len(frames) / fps:.1f} s, {frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == "__main__":
    main()
