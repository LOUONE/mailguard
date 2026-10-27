"""Invariants of the real dataset in data/processed/. Skipped until `make dataset` has run."""

from __future__ import annotations

import pandas as pd
import pytest

from mailguard.config import get_settings
from mailguard.data.dataset import (
    FRACTIONS,
    SPLITS,
    assignment_hash,
    load_dataset,
    load_manifest,
)

PROCESSED = get_settings().data_dir / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "dataset.parquet").is_file(), reason="run `make dataset` first"
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return load_dataset(PROCESSED).drop(columns="raw")


def test_no_cluster_crosses_a_split(dataset: pd.DataFrame) -> None:
    for column in ("split", "temporal_split"):
        assert dataset.groupby("cluster")[column].nunique().max() == 1


def test_label_shares_within_one_point_of_overall(dataset: pd.DataFrame) -> None:
    overall = dataset["label"].value_counts(normalize=True)
    for split in SPLITS:
        share = dataset.loc[dataset["split"] == split, "label"].value_counts(normalize=True)
        assert (share - overall).abs().max() < 0.01, split


def test_split_sizes_match_fractions(dataset: pd.DataFrame) -> None:
    sizes = dataset["split"].value_counts(normalize=True)
    for split in SPLITS:
        assert abs(sizes[split] - FRACTIONS[split]) < 0.01


def test_manifest_matches_the_table(dataset: pd.DataFrame) -> None:
    manifest = load_manifest(PROCESSED)

    assert manifest["assignment_sha256"] == assignment_hash(dataset)
    assert manifest["rows"] == len(dataset)


def test_temporal_test_set_is_recent_phishing_and_held_out_negatives(
    dataset: pd.DataFrame,
) -> None:
    test = dataset[dataset["temporal_split"] == "test"]
    phishing = test[test["label"] == "phishing"]

    assert phishing["year"].min() > 2019
    assert (test.loc[test["label"] != "phishing", "split"] == "test").all()
