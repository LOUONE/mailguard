"""Variants compared over split seeds, on a tiny corpus; mailing-list detection; the probe."""

from __future__ import annotations

import numpy as np

from mailguard.data.synthetic import personal_probe
from mailguard.models.experiments import (
    METRICS,
    Variant,
    render,
    run_seed,
    sample_weights,
    seeds,
    summarise,
)
from mailguard.models.slices import from_mailing_list
from mailguard.parsing import parse_email
from tests.models import TINY
from tests.unit.test_models_evaluate import tiny_frame


def test_list_mail_is_recognised_by_its_headers() -> None:
    listed = parse_email(b"List-Id: <dev.lists.example.org>\nSubject: x\n\nbody")
    bulk = parse_email(b"Precedence: bulk\nSubject: x\n\nbody")
    personal = parse_email(personal_probe(1)[0])

    assert from_mailing_list(listed) and from_mailing_list(bulk)
    assert not from_mailing_list(personal)


def test_probe_is_personal_synthetic_mail() -> None:
    probe = [parse_email(raw) for raw in personal_probe(50, seed=3)]

    assert len({e.subject for e in probe}) > 5
    assert all((e.from_address or "").endswith("@example.org") for e in probe)
    assert not any(from_mailing_list(e) for e in probe)
    assert personal_probe(5, seed=3) == personal_probe(5, seed=3)


def test_personal_ham_is_weighted_up_to_match_list_ham() -> None:
    labels = np.array(["ham"] * 8 + ["phishing"] * 4)
    is_list = np.array([True] * 6 + [False] * 2 + [False] * 4)

    weights = sample_weights(labels, is_list)

    assert weights[6:8].tolist() == [3.0, 3.0]  # 6 list ham / 2 other ham
    assert weights[:6].sum() == weights[6:8].sum()
    assert (weights[8:] == 1).all()


def test_variants_over_seeds_are_paired_with_the_baseline() -> None:
    frame = tiny_frame()
    frame["stratum"] = frame["label"] + ":" + frame["period"]  # what re-splitting stratifies on
    emails = [parse_email(raw) for raw in frame["raw"]]
    is_list = np.zeros(len(frame), dtype=bool)
    is_list[::4] = True
    probe = [parse_email(raw) for raw in personal_probe(20)]
    variants = [Variant("served", TINY), Variant("reweighted", TINY, reweight_personal_ham=True)]

    results = [r for s in seeds(2) for r in run_seed(frame, emails, is_list, probe, s, variants)]
    summary = summarise(results, "served")

    assert len(results) == 8 and summary["seeds"] == seeds(2)  # 2 seeds x 2 models x 2 rules
    reweighted = summary["variants"]["reweighted"]["recall"]
    assert reweighted["seeds_lower"] + reweighted["seeds_higher"] <= 2
    assert "difference_mean" not in summary["variants"]["served"]["recall"]
    markdown = render(summary, "Tiny")
    assert all("| reweighted |" in markdown for _ in METRICS)
    assert markdown.count("## ") == len(METRICS)


def test_the_personal_threshold_is_set_on_ham_outside_lists() -> None:
    frame = tiny_frame()
    frame["stratum"] = frame["label"] + ":" + frame["period"]
    emails = [parse_email(raw) for raw in frame["raw"]]
    is_list = np.zeros(len(frame), dtype=bool)
    is_list[::4] = True
    probe = [parse_email(raw) for raw in personal_probe(5)]

    results = run_seed(frame, emails, is_list, probe, seeds(1)[0], [Variant("served", TINY)])

    assert [r["variant"] for r in results] == ["served", "served, personal threshold"]
    assert all(0 <= r["threshold"] <= 1 for r in results)
