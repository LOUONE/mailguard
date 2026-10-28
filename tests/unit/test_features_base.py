from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion

from mailguard.features.base import EmailFeatures
from mailguard.parsing import Email


class Lengths(EmailFeatures):
    names: ClassVar[tuple[str, ...]] = ("subject_chars", "n_headers")

    def extract(self, email: Email) -> Mapping[str, float]:
        return {"subject_chars": len(email.subject), "n_headers": len(email.headers)}


class Broken(EmailFeatures):
    names: ClassVar[tuple[str, ...]] = ("a",)

    def extract(self, email: Email) -> Mapping[str, float]:
        return {"b": 1.0}


EMAILS = [Email(subject="hello"), Email()]


def test_rows_follow_declared_names() -> None:
    out = Lengths().fit_transform(EMAILS)

    assert out.tolist() == [[5.0, 0.0], [0.0, 0.0]]


def test_empty_input_keeps_the_column_count() -> None:
    assert Lengths().transform([]).shape == (0, 2)


def test_wrong_keys_fail_loudly() -> None:
    with pytest.raises(ValueError, match="Broken returned"):
        Broken().transform(EMAILS)


def test_base_class_must_be_specialised() -> None:
    with pytest.raises(NotImplementedError):
        EmailFeatures().transform(EMAILS)


def test_works_inside_scikit_learn_compositions() -> None:
    union = clone(FeatureUnion([("a", Lengths()), ("b", Lengths())]))

    out = union.fit_transform(EMAILS)

    assert out.shape == (2, 4)
    assert list(union.get_feature_names_out()) == [
        "a__subject_chars",
        "a__n_headers",
        "b__subject_chars",
        "b__n_headers",
    ]
    assert np.array_equal(out[:, :2], out[:, 2:])


def test_exclude_drops_columns_and_names() -> None:
    features = Lengths(exclude=("subject_chars",))

    assert features.fit_transform(EMAILS).tolist() == [[0.0], [0.0]]
    assert list(features.get_feature_names_out()) == ["n_headers"]
    assert clone(features).exclude == ("subject_chars",)


def test_exclude_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="no feature"):
        Lengths(exclude=("subject_char",)).transform(EMAILS)


def test_extractor_saved_before_exclude_keeps_every_feature() -> None:
    features = Lengths()
    del features.exclude  # what a model saved by v0.1.0 looks like once loaded

    assert features.transform(EMAILS).shape == (2, 2)


def test_declares_itself_stateless() -> None:
    assert Lengths().__sklearn_tags__().requires_fit is False
