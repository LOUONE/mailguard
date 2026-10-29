from __future__ import annotations

import json
import platform
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mailguard.data.dataset import (
    FRACTIONS,
    SPLITS,
    add_splits,
    assign_groups,
    assignment_hash,
    build_frame,
    build_manifest,
    cluster_id,
    file_year,
    load_dataset,
    load_manifest,
    main,
    manifest_hash,
    period,
    stratum,
    target,
    write_dataset,
)
from mailguard.data.sources import Source
from tests.fakes import source, write_mbox, write_tar

# --- per-message metadata ------------------------------------------------------------------


def test_target_follows_adr_0001() -> None:
    assert (target("phishing"), target("ham"), target("spam")) == (1, 0, 0)


@pytest.mark.parametrize(
    ("filename", "year", "expected_period"),
    [
        ("phishing-2021", 2021, "2015-2025"),
        ("phishing2.mbox", None, "2004-2007"),
        ("20030228_spam.tar.bz2", None, "2002-2003"),
    ],
)
def test_period_comes_from_the_file_not_the_date_header(
    filename: str, year: int | None, expected_period: str
) -> None:
    src = source(filename, "spam" if filename.endswith(".bz2") else "phishing")

    assert file_year(src) == year
    assert period(src) == expected_period


def test_stratum_splits_phishing_by_period_only() -> None:
    assert stratum("phishing", "2004-2007") == "phishing:2004-2007"
    assert stratum("ham", "2002-2003") == "ham"


def test_empty_bodies_get_their_own_cluster() -> None:
    assert cluster_id("a/b/1", "f" * 64) == "f" * 64
    assert cluster_id("a/b/1", None) != cluster_id("a/b/2", None)


# --- group assignment ----------------------------------------------------------------------

_group_sizes = st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=300)


@given(_group_sizes, _group_sizes, st.integers())
def test_every_group_lands_in_exactly_one_split(
    ham: list[int], phishing: list[int], seed: int
) -> None:
    groups = {f"h{i}": ("ham", n) for i, n in enumerate(ham)}
    groups |= {f"p{i}": ("phishing", n) for i, n in enumerate(phishing)}

    assignment = assign_groups(groups, seed=seed)

    assert assignment.keys() == groups.keys()
    assert set(assignment.values()) <= set(SPLITS)


@given(_group_sizes, st.integers())
def test_split_sizes_stay_within_two_groups_of_their_target(sizes: list[int], seed: int) -> None:
    # Deficits sum to zero and a split overshoots by at most one group, so no split can be
    # further than two of the largest groups away from its target.
    groups = {f"g{i}": ("ham", n) for i, n in enumerate(sizes)}
    assignment = assign_groups(groups, seed=seed)
    total, largest = sum(sizes), max(sizes)

    for split in SPLITS:
        filled = sum(n for g, (_, n) in groups.items() if assignment[g] == split)
        assert abs(filled - FRACTIONS[split] * total) <= 2 * largest


def test_assignment_is_deterministic_and_depends_on_the_seed() -> None:
    groups = {f"g{i}": ("ham", 1) for i in range(200)}

    assert assign_groups(groups, seed=1) == assign_groups(dict(reversed(groups.items())), seed=1)
    assert assign_groups(groups, seed=1) != assign_groups(groups, seed=2)


def test_large_balanced_strata_stay_within_one_point() -> None:
    sizes = [1] * 900 + [2] * 60 + [5] * 10 + [71]
    groups = {f"g{i}": ("phishing", n) for i, n in enumerate(sizes)}
    assignment = assign_groups(groups)
    total = sum(sizes)

    for split in SPLITS:
        share = sum(n for g, (_, n) in groups.items() if assignment[g] == split) / total
        assert abs(share - FRACTIONS[split]) < 0.01


def test_large_groups_are_spread_across_splits() -> None:
    # Regression: choosing the largest absolute shortfall sent every large group to train.
    sizes = [20] * 30 + [1] * 1400
    groups = {f"g{i}": ("phishing", n) for i, n in enumerate(sizes)}
    assignment = assign_groups(groups)
    overall = 600 / 2000  # share of messages that sit in a 20-message group

    for split in SPLITS:
        in_split = [n for g, (_, n) in groups.items() if assignment[g] == split]
        assert abs(sum(n for n in in_split if n > 1) / sum(in_split) - overall) < 0.05, split


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        assign_groups({"g": ("ham", 1)}, {"train": 0.8, "val": 0.1, "test": 0.05})


# --- splits on a frame ---------------------------------------------------------------------


def frame(rows: list[tuple[str, str, str, int | None]]) -> pd.DataFrame:
    """Rows of (id, cluster, label, year) with the columns `add_splits` needs."""
    df = pd.DataFrame(rows, columns=["id", "cluster", "label", "year"])
    df["year"] = df["year"].astype("Int64")
    df["stratum"] = df["label"]
    return df


def test_temporal_check_keeps_campaigns_on_one_side() -> None:
    rows: list[tuple[str, str, str, int | None]] = [
        ("old", "c-old", "phishing", None),  # 2004-2007 file
        ("y2018", "c-cross", "phishing", 2018),
        ("y2021-copy", "c-cross", "phishing", 2021),  # same campaign as y2018
        ("y2022", "c-new", "phishing", 2022),
    ]
    rows += [(f"ham{i}", f"c-ham{i}", "ham", None) for i in range(20)]

    out = add_splits(frame(rows)).set_index("id")

    assert out.loc["y2021-copy", "temporal_split"] == out.loc["y2018", "temporal_split"] != "test"
    assert out.loc["old", "temporal_split"] != "test"
    assert out.loc["y2022", "temporal_split"] == "test"
    ham = out[out["label"] == "ham"]
    assert (ham["temporal_split"] == ham["split"]).all()


def test_no_cluster_sits_in_two_splits() -> None:
    rows: list[tuple[str, str, str, int | None]] = [
        (f"m{i}", f"c{i % 37}", "phishing" if i % 3 else "ham", 2015 + i % 11) for i in range(400)
    ]

    out = add_splits(frame(rows))

    for column in ("split", "temporal_split"):
        assert (out.groupby("cluster")[column].nunique() == 1).all()


# --- build, persist, reload ----------------------------------------------------------------

HAM = b"From: a@example.org\nSubject: lunch\n\nLunch at {n}, Bob?\n"
PHISH = b"From: x@example.net\nSubject: locked\n\nYour account {n} is locked, verify now.\n"
OTHER = b"From: y@example.net\nSubject: hi\n\nA different campaign number {n}\n"


def write_corpus(raw_dir: Path) -> list[Source]:
    ham = source("ham.tar.bz2", "ham")
    old, new = source("phishing0.mbox", "phishing"), source("phishing-2022", "phishing")
    write_tar(raw_dir, ham, {f"{i:05}": HAM.replace(b"{n}", b"x" * i) for i in range(30)})
    # Numbers are masked by the fingerprint: the 10 copies of PHISH are one cluster.
    write_mbox(raw_dir, old, [PHISH.replace(b"{n}", str(i).encode()) for i in range(10)])
    write_mbox(raw_dir, new, [OTHER.replace(b"{n}", b"y" * i) for i in range(10)] + [b""])
    return [ham, old, new]


def test_build_frame_has_targets_clusters_and_splits(tmp_path: Path) -> None:
    sources = write_corpus(tmp_path)

    df = build_frame(tmp_path, sources)

    assert len(df) == 51
    assert df["id"].is_monotonic_increasing
    assert df.groupby("label")["target"].unique().to_dict() == {"ham": [0], "phishing": [1]}
    assert df.loc[df["file"] == "phishing0.mbox", "cluster"].nunique() == 1
    assert df.loc[df["file"] == "phishing-2022", "year"].eq(2022).all()
    assert df.loc[df["body_chars"] == 0, "cluster"].str.startswith("singleton:").all()
    assert set(df["split"]) == set(SPLITS)
    assert df.loc[df["file"] == "phishing-2022", "temporal_split"].eq("test").all()
    assert all(isinstance(raw, bytes) for raw in df["raw"])
    assert "normalised_fp" not in df.columns


def test_build_frame_refuses_an_empty_corpus(tmp_path: Path) -> None:
    empty = source("phishing-2020", "phishing")
    write_mbox(tmp_path, empty, [])

    with pytest.raises(ValueError, match="no message"):
        build_frame(tmp_path, [empty])


def test_round_trip_and_manifest(tmp_path: Path) -> None:
    sources = write_corpus(tmp_path / "raw")
    df = build_frame(tmp_path / "raw", sources)
    manifest = build_manifest(df, sources)

    write_dataset(df, manifest, tmp_path / "processed")
    reloaded = load_dataset(tmp_path / "processed")

    assert assignment_hash(reloaded) == manifest["assignment_sha256"]
    assert reloaded["raw"].tolist() == df["raw"].tolist()
    assert load_manifest(tmp_path / "processed") == json.loads(json.dumps(manifest))
    assert sum(sum(c.values()) for c in manifest["split"].values()) == manifest["rows"] == 51
    assert manifest["clusters"]["largest"] == 10
    assert not list((tmp_path / "processed").glob("*.part"))


def test_two_builds_give_the_same_assignment(tmp_path: Path) -> None:
    sources = write_corpus(tmp_path)

    first, second = build_frame(tmp_path, sources), build_frame(tmp_path, sources)

    assert assignment_hash(first) == assignment_hash(second)
    assert assignment_hash(first) != assignment_hash(build_frame(tmp_path, sources, seed=7))


def test_manifest_holds_no_message_content(tmp_path: Path) -> None:
    sources = write_corpus(tmp_path)
    text = json.dumps(build_manifest(build_frame(tmp_path, sources), sources))

    for secret in ("lunch", "locked", "example.org", "campaign"):
        assert secret not in text


def test_load_dataset_explains_how_to_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make dataset"):
        load_dataset(tmp_path)


def test_main_builds_into_out_dir(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sources = write_corpus(tmp_path / "raw")
    args = ["--raw-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "out")]

    with caplog.at_level("INFO"):
        assert main(args, sources=sources) == 0

    assert len(manifest_hash(tmp_path / "out")) == 64
    assert "51 rows written" in caplog.text


def test_main_fails_cleanly_without_raw_data(tmp_path: Path) -> None:
    args = ["--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]

    assert main(args, sources=[source("phishing-2020", "phishing")]) == 1
    assert not (tmp_path / "out").exists()


def test_manifest_bytes_do_not_depend_on_the_os(tmp_path: Path) -> None:
    sources = write_corpus(tmp_path / "raw")
    df = build_frame(tmp_path / "raw", sources)

    write_dataset(df, build_manifest(df, sources), tmp_path / "out")

    assert b"\r\n" not in (tmp_path / "out" / "manifest.json").read_bytes()


def test_manifest_records_the_python_that_built_it(tmp_path: Path) -> None:
    sources = write_corpus(tmp_path)

    manifest = build_manifest(build_frame(tmp_path, sources), sources)

    assert manifest["python"] == platform.python_version()


def test_the_running_python_is_the_pinned_one() -> None:
    """The stdlib email parser, hence the split, changes between patch releases (docs/data.md)."""
    pinned = Path(".python-version").read_text(encoding="utf-8").strip()

    assert platform.python_version() == pinned, "run `uv sync` to use the pinned interpreter"
