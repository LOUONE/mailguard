from __future__ import annotations

import numpy as np
import pytest

from mailguard.models.calibration import reliability_table
from mailguard.models.plots import reliability_svg


def test_reliability_svg_is_deterministic_and_themed() -> None:
    rows = reliability_table(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.95, 0.3]))
    curves = {"logreg": rows, "lightgbm": rows[:2]}

    svg = reliability_svg(curves)

    assert svg == reliability_svg(curves)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "prefers-color-scheme: dark" in svg and "<title>" in svg
    assert svg.count('class="s1 dot"') == len(rows) and svg.count('class="s2 dot"') == 2
    with pytest.raises(ValueError, match="at most"):
        reliability_svg({"a": rows, "b": rows, "c": rows})
