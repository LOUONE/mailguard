"""Slices of the evaluation set: readable word counts and share of English text."""

from __future__ import annotations

from mailguard.models.slices import english_share, readable_words, slice_masks
from mailguard.parsing import Email

ENGLISH = "Please confirm the details of your account with us and we will be in touch. " * 3
GERMAN = "Bitte bestätigen Sie Ihre Kontodaten innerhalb von drei Tagen über unser Portal. " * 3


def test_readable_words_skip_placeholders_and_digits() -> None:
    email = Email(text_body="Pay 300 EUR to bob@example.org at https://example.org/x now")

    assert readable_words(email) == ["pay", "eur", "to", "at", "now"]


def test_english_share() -> None:
    assert english_share(readable_words(Email(text_body=ENGLISH))) > 0.3
    assert english_share(readable_words(Email(text_body=GERMAN))) < 0.1
    assert english_share([]) == 0.0


def test_slice_masks() -> None:
    emails = [Email(text_body="Click here"), Email(text_body=ENGLISH), Email(text_body=GERMAN)]

    masks = {name: mask.tolist() for name, mask in slice_masks(emails).items()}

    assert masks == {
        "under 20 words": [True, False, False],
        "20 to 99 words": [False, True, True],
        "100 words or more": [False, False, False],
        "20+ words, mostly English": [False, True, False],
        "20+ words, little English": [False, False, True],
        "from a mailing list": [False, False, False],
        "not from a mailing list": [True, True, True],
    }
