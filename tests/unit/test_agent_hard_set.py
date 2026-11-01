from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mailguard.agent.hard_set import (
    CATEGORIES,
    Annotation,
    agreement,
    annotate,
    defang,
    load_hard_set,
    read_annotations,
    render,
    select_candidates,
    write_annotations,
)
from mailguard.parsing import Attachment, Email

THRESHOLD = 0.3
BAND = (0.05, 0.95)


def rows(*specs: tuple[str, str, int]) -> pd.DataFrame:
    return pd.DataFrame(specs, columns=["id", "cluster", "target"])


def note(
    message_id: str = "a", label: str = "phishing", category: str = "credential"
) -> Annotation:
    return Annotation.model_validate(
        {
            "id": message_id,
            "cluster": f"c-{message_id}",
            "selected_as": "grey",
            "label": label,
            "category": category,
        }
    )


# --- selection -------------------------------------------------------------------------------


def test_grey_band_and_errors_are_selected_and_sure_right_answers_are_not() -> None:
    frame = rows(
        ("sure-phish", "c1", 1),
        ("sure-ham", "c2", 0),
        ("grey-phish", "c3", 1),
        ("missed", "c4", 1),
        ("false-alarm", "c5", 0),
        ("grey-error", "c6", 0),
    )
    scores = np.array([0.99, 0.01, 0.6, 0.02, 0.97, 0.5])
    picked = select_candidates(frame, scores, THRESHOLD, BAND)
    assert dict(zip(picked["id"], picked["selected_as"], strict=True)) == {
        "grey-phish": "grey",
        "missed": "error",
        "false-alarm": "error",
        "grey-error": "both",
    }


def test_one_message_per_cluster_whatever_the_row_order() -> None:
    frame = rows(("b", "c1", 1), ("a", "c1", 1), ("c", "c2", 1))
    scores = np.array([0.5, 0.5, 0.5])
    first = select_candidates(frame, scores, THRESHOLD, BAND)
    flipped = select_candidates(
        frame.iloc[::-1].reset_index(drop=True), scores[::-1], THRESHOLD, BAND
    )
    assert sorted(first["id"]) == sorted(flipped["id"]) == ["a", "c"]


def test_order_is_shuffled_and_seeded() -> None:
    frame = rows(*((f"m{i:02d}", f"c{i}", 1) for i in range(30)))
    scores = np.full(30, 0.5)
    once = select_candidates(frame, scores, THRESHOLD, BAND)["id"].tolist()
    again = select_candidates(frame, scores, THRESHOLD, BAND)["id"].tolist()
    other = select_candidates(frame, scores, THRESHOLD, BAND, seed=1)["id"].tolist()
    assert once == again != sorted(once)
    assert other != once


# --- annotations and files -------------------------------------------------------------------


def test_category_must_belong_to_the_label() -> None:
    with pytest.raises(ValidationError, match="not one of"):
        note(label="ham", category="credential")
    with pytest.raises(ValidationError):
        note(label="maybe", category="credential")


@pytest.mark.parametrize(
    ("label", "category", "target"),
    [
        ("phishing", "advance_fee", 1),
        ("spam", "marketing", 0),
        ("ham", "automated", 0),
        ("unclear", "unreadable", None),
    ],
)
def test_target_follows_adr_0001(label: str, category: str, target: int | None) -> None:
    assert note(label=label, category=category).target == target


def test_every_label_has_categories() -> None:
    assert all(CATEGORIES[label] for label in ("phishing", "spam", "ham", "unclear"))


def test_annotations_round_trip_through_csv(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "a.csv"
    notes = [note("x"), note("y", "spam", "marketing")]
    write_annotations(path, notes)
    assert read_annotations(path) == notes
    assert path.read_bytes().count(b"\r") == 0
    assert (
        path.read_text(encoding="utf-8").splitlines()[0] == "id,cluster,selected_as,label,category"
    )


def test_no_file_means_nothing_annotated(tmp_path: Path) -> None:
    assert read_annotations(tmp_path / "missing.csv") == []


def test_hard_set_loads_against_fresh_data(tmp_path: Path) -> None:
    path = tmp_path / "hard_set.csv"
    write_annotations(path, [note("x"), note("y", "unclear", "ambiguous")])
    frame = pd.DataFrame(
        {"id": ["y", "x", "z"], "cluster": ["c-y", "c-x", "c-z"], "raw": [b"Y", b"X", b"Z"]}
    )
    joined = load_hard_set(path, frame)
    assert joined["id"].tolist() == ["x", "y"]
    assert joined["raw"].tolist() == [b"X", b"Y"]
    assert joined["hand_target"].tolist()[0] == 1
    assert pd.isna(joined["hand_target"].tolist()[1])


def test_hard_set_refuses_missing_ids_and_empty_files(tmp_path: Path) -> None:
    path = tmp_path / "hard_set.csv"
    write_annotations(path, [note("gone")])
    frame = pd.DataFrame({"id": ["x"], "cluster": ["c"], "raw": [b"X"]})
    with pytest.raises(LookupError, match="1 ids not in the dataset"):
        load_hard_set(path, frame)
    write_annotations(path, [])
    with pytest.raises(ValueError, match="no annotation"):
        load_hard_set(path, frame)


def test_agreement_counts_corpus_against_hand_labels() -> None:
    notes = [note("a"), note("b", "spam", "marketing"), note("c")]
    table = agreement(notes, {"a": "phishing", "b": "phishing", "c": "phishing"})
    assert table.loc["phishing", "phishing"] == 2
    assert table.loc["phishing", "spam"] == 1


# --- what the annotator sees -----------------------------------------------------------------


def test_defang_breaks_links_but_not_prose() -> None:
    text = "Go to https://secure.paypa1.example/login or http://198.51.100.7, version 3.5, e.g."
    assert defang(text) == (
        "Go to hxxps://secure[.]paypa1[.]example/login or hxxp://198.51.100.7, version 3.5, e.g."
    )


EMAIL = Email(
    subject="Verify your account",
    from_name="Support",
    from_address="help@bank.example",
    to=("a@b.example", "c@d.example"),
    text_body="Click https://bank.example.org/x now. " + "padding " * 400,
    urls=tuple(f"https://u{i}.example/p" for i in range(20)),
    attachments=(
        Attachment(
            filename="invoice.pdf.exe", content_type="application/pdf", size=10, sha256="0" * 64
        ),
    ),
)


def test_render_shows_envelope_links_and_attachments_defanged() -> None:
    shown = render(EMAIL)
    assert "help@bank[.]example" in shown
    assert "2 recipient(s)" in shown
    assert "invoice[.]pdf[.]exe" in shown
    assert "hxxps://u0[.]example/p" in shown
    assert "5 more" in shown
    assert "https://" not in shown
    assert "type m" in shown
    assert "type m" not in render(EMAIL, full=True)
    assert len(render(EMAIL, full=True)) > len(shown)


def test_render_handles_a_bare_message() -> None:
    shown = render(Email())
    assert "<?>" in shown and "Reply-To:    -" in shown


class Session:
    """Scripted keyboard and screen for `annotate`."""

    def __init__(self, keys: list[str]) -> None:
        self.keys: Iterator[str] = iter(keys)
        self.screen: list[str] = []
        self.saved: list[list[Annotation]] = []

    def ask(self, prompt: str) -> str:
        self.screen.append(prompt)
        return next(self.keys)

    def show(self, text: str) -> None:
        self.screen.append(text)

    def save(self, notes: list[Annotation]) -> None:
        self.saved.append(notes)


CANDIDATES = pd.DataFrame(
    {
        "id": ["nazario/phishing-2019/1", "spamassassin/ham/2", "spamassassin/spam/3"],
        "cluster": ["c1", "c2", "c3"],
        "selected_as": ["grey", "error", "both"],
    }
)


def run(session: Session, done: list[Annotation] | None = None) -> list[Annotation]:
    return annotate(
        CANDIDATES, done or [], lambda _: EMAIL, session.ask, session.show, session.save
    )


def test_annotate_labels_each_message_and_saves_after_each() -> None:
    session = Session(["P", "9", "1", "m", "x", "h", "2", "s", "1"])
    notes = run(session)
    assert [(n.label, n.category) for n in notes] == [
        ("phishing", "credential"),
        ("ham", "list_or_newsletter"),
        ("spam", "marketing"),
    ]
    assert notes[1].selected_as == "error"
    assert [len(s) for s in session.saved] == [1, 2, 3]


def test_annotate_is_blind_to_ids_and_corpus() -> None:
    session = Session(["p", "1", "q"])
    run(session)
    screen = "\n".join(session.screen)
    assert "nazario" not in screen and "spamassassin" not in screen
    assert "1 / 3" in screen


def test_annotate_resumes_skips_and_quits() -> None:
    first = Session(["k", "u", "1", "q"])
    notes = run(first)
    assert [n.id for n in notes] == ["spamassassin/ham/2"]
    later = Session(["h", "3", "q"])
    notes = run(later, notes)
    assert [n.id for n in notes] == ["spamassassin/ham/2", "nazario/phishing-2019/1"]
    assert "2 / 3" in later.screen[0]
