from pathlib import Path

import pytest

from pai_final.planner import parse_points

FIXTURE = Path(__file__).parent / "fixtures" / "red_to_cyan_response.txt"


def test_parse_points_real_response():
    assert parse_points(FIXTURE.read_text()) == ((564.0, 626.0), (378.0, 500.0))


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '[{"point": [1, 2], "label": "pick"}]',
        '[{"point": [1, 2], "label": "pick"}, {"point": [3], "label": "place"}]',
        '[{"point": [1, 2], "label": "pick"}, {"point": [3, 1001], "label": "place"}]',
        '[{"point": [1, 2], "label": "pick"}, {"point": [3, 4], "label": "pick"}, {"point": [5, 6], "label": "place"}]',
    ],
)
def test_parse_points_rejects(text):
    with pytest.raises(ValueError):
        parse_points(text)
