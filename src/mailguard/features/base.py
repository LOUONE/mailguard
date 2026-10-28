"""Shared plumbing: stateless extractors that turn each `Email` into a fixed-length vector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from mailguard.parsing import Email


class EmailFeatures(TransformerMixin, BaseEstimator):  # type: ignore[misc]  # sklearn is untyped
    """scikit-learn transformer over a sequence of `Email`, one row per message.

    Subclasses list their feature `names` and implement `extract`. There is nothing to learn, so
    `fit` is a no-op: these extractors cannot leak anything from the data they are fitted on.

    `exclude` leaves some of the features out, for ablations: `UrlFeatures(exclude=("url_count",))`.
    """

    names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, exclude: tuple[str, ...] = ()) -> None:
        self.exclude = exclude

    @property
    def kept(self) -> tuple[str, ...]:
        # Models saved before `exclude` existed have no such attribute: they keep every feature.
        exclude = getattr(self, "exclude", ())
        unknown = set(exclude) - set(self.names)
        if unknown:
            raise ValueError(f"{type(self).__name__} has no feature {sorted(unknown)}")
        return tuple(name for name in self.names if name not in exclude)

    def extract(self, email: Email) -> Mapping[str, float]:
        raise NotImplementedError

    def fit(self, X: Sequence[Email], y: object = None) -> Self:
        return self

    def transform(self, X: Sequence[Email]) -> NDArray[np.float64]:
        kept = self.kept
        rows = [self.extract(email) for email in X]
        for row in rows:
            if row.keys() != set(self.names):
                raise ValueError(f"{type(self).__name__} returned {sorted(row)}")
        out = np.array([[row[name] for name in kept] for row in rows], dtype=np.float64)
        return out.reshape(len(rows), len(kept))

    def get_feature_names_out(self, input_features: object = None) -> NDArray[np.object_]:
        return np.asarray(self.kept, dtype=object)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        tags.input_tags.two_d_array = False  # input is a list of Email, not a matrix
        return tags
