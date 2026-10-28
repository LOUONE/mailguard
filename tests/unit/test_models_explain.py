from __future__ import annotations

import numpy as np
import pytest

from mailguard.models.explain import (
    contributions,
    explain,
    global_importance,
    group_importance,
    group_of,
)
from mailguard.models.pipelines import MODEL_NAMES
from mailguard.models.train import split_data, train_baseline
from mailguard.parsing import parse_email
from tests.unit.test_models_train import TINY, tiny_frame

SYNTHETIC = parse_email(
    b"From: PayPal <x@evil.example>\nSubject: verify\nContent-Type: text/html\n\n"
    b'<a href="http://198.51.100.9/login">www.paypal.com</a> Verify your account password.\n'
)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_contributions_add_up_to_the_margin(name: str) -> None:
    frame = tiny_frame()
    pipeline = train_baseline(name, frame, TINY)  # type: ignore[arg-type]
    emails = split_data(frame, "val").emails

    c = contributions(pipeline, emails)

    total = c.bias + np.asarray(c.values.sum(axis=1)).ravel()
    features = pipeline[:-1].transform(emails)
    if name == "lightgbm":
        margin = pipeline[-1].booster_.predict(features, raw_score=True)
    else:
        margin = pipeline[-1].decision_function(features)
    assert np.allclose(total, margin, atol=1e-6)
    assert c.values.shape == (len(emails), len(c.names))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_importance_and_local_explanation(name: str) -> None:
    frame = tiny_frame()
    pipeline = train_baseline(name, frame, TINY)  # type: ignore[arg-type]
    c = contributions(pipeline, split_data(frame, "val").emails)

    ranking = global_importance(c)
    shares = group_importance(c)
    reasons = explain(pipeline, SYNTHETIC, top=3)

    values = [v for _, v in ranking]
    assert values == sorted(values, reverse=True) and all(v > 0 for v in values)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert set(shares) <= {"headers", "attachments", "urls", "lexicon", "text"}
    assert 0 < len(reasons) <= 3
    for r in reasons:
        assert r["group"] == group_of(r["feature"])
        assert (r["direction"] == "phishing") == (r["contribution"] > 0)


def test_group_of() -> None:
    assert group_of("urls__url_ip_hosts") == "urls"
    assert group_of("text__word:your account") == "text"
