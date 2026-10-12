"""Ask a VLM for the pick and place points of an instruction, and parse its answer."""

import json
import re
import time

MODEL_ID = "gemini-robotics-er-2-preview"
THINKING_LEVEL = "low"
NORMALISED_RANGE = 1000
RETRY_ATTEMPTS = 4
RETRY_WAIT_S = 20.0
UNAVAILABLE_MARKERS = ("503", "service_unavailable")

PROMPT_TEMPLATE = (
    "Instruction: {instruction}.\n"
    "Point to the object to pick and to the place to put it. "
    'The answer should follow the json format: [{{"point": <point>, "label": "pick"}}, '
    '{{"point": <point>, "label": "place"}}]. '
    "The points are in [y, x] format normalized to 0-1000."
)

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")


def parse_points(text: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(pick_yx, place_yx)`` from a model answer, each normalised ``[y, x]`` in 0-1000."""
    try:
        items = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError as e:
        raise ValueError(f"answer is not JSON: {e}") from e
    if not isinstance(items, list):
        raise ValueError("answer is not a JSON list")
    points: dict[str, tuple[float, float]] = {}
    for item in items:
        label = item.get("label") if isinstance(item, dict) else None
        if label not in ("pick", "place"):
            continue
        if label in points:
            raise ValueError(f"duplicate label {label!r}")
        point = item.get("point")
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point)
            or not all(0 <= v <= NORMALISED_RANGE for v in point)
        ):
            raise ValueError(f"bad point for {label!r}: {point!r}")
        points[label] = (float(point[0]), float(point[1]))
    if len(points) != 2:
        raise ValueError(f"need one 'pick' and one 'place' label, got {sorted(points)}")
    return points["pick"], points["place"]


def ask_vlm(image_path: str, instruction: str) -> str:
    """Raw ``output_text`` of the model for the image and instruction. Retries on 503 only."""
    from google import genai  # imported lazily so replay needs neither the SDK nor a key

    client = genai.Client()
    uploaded = client.files.upload(file=image_path)
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.interactions.create(
                model=MODEL_ID,
                input=[
                    {"type": "image", "uri": uploaded.uri, "mime_type": "image/png"},
                    {"type": "text", "text": PROMPT_TEMPLATE.format(instruction=instruction)},
                ],
                generation_config={"thinking_level": THINKING_LEVEL},
            )
            return response.output_text
        except Exception as e:
            if attempt == RETRY_ATTEMPTS - 1 or not any(m in str(e) for m in UNAVAILABLE_MARKERS):
                raise
            time.sleep(RETRY_WAIT_S)
    raise AssertionError("unreachable")
