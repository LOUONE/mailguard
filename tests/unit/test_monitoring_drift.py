"""PSI and the drift decision, on distributions whose answer is known."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mailguard.monitoring.drift import PSI_DRIFT, bin_edges, compare, psi

RNG = np.random.default_rng(0)
NORMAL = RNG.normal(0, 1, 5000)


def test_same_distribution_is_stable() -> None:
    assert psi(NORMAL, RNG.normal(0, 1, 5000)) < 0.02


@pytest.mark.parametrize("shift", [0.3, 0.7, 1.5])
def test_psi_of_a_shifted_normal_is_about_the_shift_squared(shift: float) -> None:
    # PSI is a symmetrised Kullback-Leibler divergence: for two unit normals d apart, d².
    value = psi(NORMAL, RNG.normal(shift, 1, 5000))

    assert value == pytest.approx(shift**2, rel=0.25)
    assert (value > PSI_DRIFT) == (shift > 0.5)


def test_counts_are_binned_by_value() -> None:
    links = np.array([0] * 70 + [1] * 20 + [2] * 10, dtype=float)

    assert bin_edges(links).tolist() == [0.5, 1.5]
    assert psi(links, links) == pytest.approx(0.0)
    more_links = np.array([0] * 20 + [1] * 30 + [2] * 50, dtype=float)
    assert psi(links, more_links) > PSI_DRIFT


def test_values_never_seen_in_the_reference_do_not_break_it() -> None:
    assert np.isfinite(psi(np.zeros(100), np.full(100, 7.0)))


@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=200))
def test_psi_is_finite_and_non_negative(values: list[float]) -> None:
    value = psi(NORMAL, values)
    assert np.isfinite(value) and value >= -1e-12


def test_traffic_drifts_when_half_the_features_or_the_score_do() -> None:
    reference = {"a": NORMAL, "b": NORMAL, "score": NORMAL}
    stable = {"a": RNG.normal(0, 1, 2000), "b": RNG.normal(0, 1, 2000), "score": NORMAL}
    one_input = {**stable, "a": RNG.normal(2, 1, 2000)}
    score_only = {**stable, "score": RNG.normal(2, 1, 2000)}

    assert not compare(reference, stable).drifted
    report = compare(reference, one_input)
    assert report.share_drifted == 0.5 and report.drifted and not report.score_drifted
    assert compare(reference, score_only).drifted


def test_summary_is_sorted_and_numbers_only() -> None:
    report = compare({"a": NORMAL, "b": NORMAL}, {"a": NORMAL, "b": NORMAL + 3})

    summary = report.summary()

    assert [f["feature"] for f in summary["features"]] == ["b", "a"]
    assert summary["features"][0]["drifted"] is True
    assert summary["reference_rows"] == 5000 and summary["current_rows"] == 5000


def test_columns_must_overlap_and_hold_data() -> None:
    with pytest.raises(ValueError, match="no column in common"):
        compare({"a": NORMAL}, {"b": NORMAL})
    with pytest.raises(ValueError, match="no data"):
        compare({"a": NORMAL}, {"a": []})
