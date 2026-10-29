"""The single-pass scorer of the API gives exactly what the two-pass reference gives."""

from __future__ import annotations

import numpy as np
import pytest

from mailguard.models.explain import contributions, explain
from mailguard.models.pipelines import logreg_pipeline
from mailguard.models.train import calibrate, operating_thresholds
from mailguard.serving.model import ServedModel
from tests.models import TINY, messages, tiny_model


@pytest.fixture(scope="module", params=["lightgbm", "logreg"])
def served(request: pytest.FixtureRequest) -> ServedModel:
    if request.param == "lightgbm":
        model, thresholds = tiny_model()
    else:
        train, val = messages(40), messages(20, offset=100)
        pipeline = logreg_pipeline(TINY).fit(train.emails, train.target)
        model, _ = calibrate(pipeline, val)
        thresholds = operating_thresholds(model, val)
    return ServedModel(f"mailguard-{request.param}", "1", model, thresholds)


def test_score_matches_the_calibrated_pipeline(served: ServedModel) -> None:
    emails = messages(30, offset=500).emails

    fast = np.array([served.predict(e, base_rate=0.01).score for e in emails])

    reference = served.model.predict_proba(emails)[:, 1]
    np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-12)


def test_reasons_match_explain(served: ServedModel) -> None:
    for email in messages(10, offset=700).emails:
        assert served.predict(email, 0.01).reasons == explain(served.model.pipeline, email)


def test_contributions_add_up_to_the_raw_probability(served: ServedModel) -> None:
    emails = messages(10, offset=900).emails

    c = contributions(served.model.pipeline, emails)

    # Exact for TreeSHAP; the linear model computes its own margin in float32.
    tolerance = 1e-12 if "lightgbm" in served.name else 1e-6
    np.testing.assert_allclose(c.probability, served.model.raw_scores(emails), atol=tolerance)


def test_feature_names_are_computed_once(served: ServedModel) -> None:
    assert len(served.names) == served.model.pipeline[:-1].transform(messages(1).emails).shape[1]
