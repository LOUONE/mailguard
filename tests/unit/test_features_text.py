from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mailguard.features.text import (
    CREDENTIAL_CUES,
    LEXICON_FEATURES,
    MAX_TEXT_CHARS,
    URGENCY_CUES,
    LexiconFeatures,
    TextFeatures,
    body_text,
    cue_count,
    exclamation_ratio,
    lexicon_features,
    recipient_words,
    scrubbed_text,
    uppercase_ratio,
)
from mailguard.parsing import Email

# Synthetic texts only.
TRAIN = [
    Email(subject="Lunch", text_body="Lunch at noon tomorrow? Bring the slides for the meeting."),
    Email(subject="Slides", text_body="Here are the slides from the meeting, see you at noon."),
    Email(subject="Meeting", text_body="The meeting moved to tomorrow, same room, bring slides."),
    Email(
        subject="URGENT: verify your account",
        text_body="Your account is suspended. Verify your password within 24 hours!",
    ),
    Email(
        subject="Account locked",
        text_body="Unusual activity: your account is locked. Confirm your password now.",
    ),
    Email(
        subject="Verify",
        text_body="Verify your account password today or your account will be suspended.",
    ),
]


# --- scrubbing -----------------------------------------------------------------------------


def test_scrubbing_removes_who_and_when_but_keeps_what() -> None:
    email = Email(
        to=("jane.doe@example.org",),
        text_body=(
            "Dear Jane, mail jose@monkey.org or visit https://evil.example/login?id=4821 "
            "before 2025. SpamAssassin and monkey.org say hi, DOE."
        ),
    )

    text = scrubbed_text(email)

    for gone in ("jane", "doe", "jose", "monkey", "org", "spamassassin", "evil", "4821", "2025"):
        assert gone not in text.casefold()
    assert "emailaddress" in text and "urladdress" in text
    assert "Dear" in text and "visit" in text and "before 0" in text


def test_recipient_words() -> None:
    email = Email(to=("jane.doe@example.org", "j_x-42@example.org", "ab@example.org"))

    assert recipient_words(email) == {"jane", "doe"}
    assert recipient_words(Email()) == set()


def test_scrubbing_without_recipients() -> None:
    assert scrubbed_text(Email(text_body="Call 555 1234")) == "Call 0 0"


def test_long_bodies_are_cut() -> None:
    assert len(body_text(Email(text_body="a" * (MAX_TEXT_CHARS + 10)))) == MAX_TEXT_CHARS


# --- TF-IDF --------------------------------------------------------------------------------


def small() -> TextFeatures:
    return TextFeatures(word_max_features=30, char_max_features=40, min_df=1)


def test_transform_refuses_to_run_before_fit() -> None:
    with pytest.raises(NotFittedError):
        small().transform(TRAIN)
    with pytest.raises(NotFittedError):
        _ = small().vocabulary_size


def test_vocabulary_is_bounded_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="mailguard.features.text"):
        features = small().fit(TRAIN)

    assert features.vocabulary_size == 70
    assert "word vocabulary 30" in caplog.text and "char vocabulary 40" in caplog.text


def test_vocabulary_comes_from_the_fitted_rows_only() -> None:
    features = TextFeatures(min_df=1).fit(TRAIN[:3])
    unseen = Email(text_body="zebra zebra zebra")

    names = set(features.get_feature_names_out())

    assert "word:slides" in names
    assert not any("zebra" in n for n in names)
    n_words = len(features.vectorizers_["word"].vocabulary_)
    assert features.transform([unseen])[:, :n_words].nnz == 0  # no known word in "zebra"


def test_output_is_sparse_float32_with_named_columns() -> None:
    features = small().fit(TRAIN)
    out = features.transform(TRAIN)

    assert sp.issparse(out) and out.dtype == np.float32
    assert out.shape == (len(TRAIN), features.vocabulary_size)
    names = features.get_feature_names_out()
    assert names[0].startswith("word:") and names[-1].startswith("char:")


def test_fitting_twice_is_deterministic() -> None:
    a, b = clone(small()).fit(TRAIN), clone(small()).fit(TRAIN)

    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    assert (a.transform(TRAIN) != b.transform(TRAIN)).nnz == 0


# --- lexicon -------------------------------------------------------------------------------


def test_each_cue_counts_once() -> None:
    assert cue_count("urgent URGENT urgently", URGENCY_CUES) == 1
    assert cue_count("Verify your password and confirm your account", CREDENTIAL_CUES) == 4
    assert cue_count("Lunch at noon?", URGENCY_CUES + CREDENTIAL_CUES) == 0


def test_ratios() -> None:
    assert uppercase_ratio("ABcd 12") == 0.5
    assert uppercase_ratio("123") == 0.0
    assert exclamation_ratio("hi!! ") == 0.5
    assert exclamation_ratio("   ") == 0.0


def test_lexicon_on_body_and_subject() -> None:
    f = lexicon_features(TRAIN[3])

    assert f.keys() == set(LEXICON_FEATURES)
    assert f["body_urgency_cues"] == 2  # suspended, within 24 hours ("urgent" is in the subject)
    assert f["subject_urgency_cues"] == 1
    assert f["subject_credential_cues"] == 2
    assert f["body_exclamation_ratio"] > 0
    assert f["body_empty"] == 0


def test_empty_body_is_flagged() -> None:
    assert lexicon_features(Email(html_body="<img src='x.png'>"))["body_empty"] == 1


def test_lexicon_transformer() -> None:
    out = LexiconFeatures().fit_transform(TRAIN)

    assert out.shape == (len(TRAIN), len(LEXICON_FEATURES))
    names = list(LEXICON_FEATURES)
    credential = out[:, names.index("body_credential_cues")]
    assert credential[:3].sum() == 0 and (credential[3:] > 0).all()
