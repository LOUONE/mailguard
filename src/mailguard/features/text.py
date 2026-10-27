"""Text features on what the reader sees: `Email.body_text` and the subject.

The body is read through `Email.body_text`, so plain-text and HTML mail give the same kind of
input (94.1% of phishing is HTML against 4.5% of ham, an era marker). There is no "has HTML" flag.

Two kinds of features:

- `TextFeatures`: TF-IDF on word 1-2 grams and character 3-5 grams. It learns a vocabulary, so it
  is the one extractor with state, and it is only fitted inside the training pipeline
  (`mailguard.models`), on the training split.
- `LexiconFeatures`: stateless counts of urgency and credential cues, uppercase and exclamation
  ratios, for the body and for the subject.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from mailguard.features.base import EmailFeatures
from mailguard.parsing import Email

logger = logging.getLogger(__name__)

# Longer bodies are cut: 0.38% of the corpus, and one 1M-character message. This bounds the cost
# of a single request in production.
MAX_TEXT_CHARS: Final = 20_000


def body_text(email: Email) -> str:
    return email.body_text[:MAX_TEXT_CHARS]


# Words that name the corpora or their recipient rather than anything the sender chose. The
# phishing corpus is one person's inbox: "Dear jose" appears in 26% of it and in 0.4% of ham.
ARTEFACT_WORDS: Final = ("jose", "nazario", "monkey", "spamassassin", "taint")

_ADDRESS_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")
# The word and any domain labels after it: "monkey.org" must not leave "org" behind.
_ARTEFACT_RE = re.compile(
    r"\b(?:" + "|".join(ARTEFACT_WORDS) + r")(?:\.[a-z]{2,})*\b", re.IGNORECASE
)
_LOCAL_PART_SPLIT_RE = re.compile(r"[._+-]+")


def recipient_words(email: Email) -> set[str]:
    """Words of the recipients' local parts (`jane.doe@x` -> jane, doe), three letters or more."""
    words: set[str] = set()
    for address in email.to:
        local = address.rpartition("@")[0]
        words.update(w for w in _LOCAL_PART_SPLIT_RE.split(local.casefold()) if len(w) >= 3)
    return {w for w in words if w.isalpha()}


def scrubbed_text(email: Email) -> str:
    """Body text without what identifies the corpus, the recipient or the date.

    Addresses and URLs become placeholders (URL features cover them), digits become 0 (years are
    the strongest era marker), and the recipient's own name and the corpus names are removed. The
    TF-IDF vocabulary is learnt on this text, so it cannot memorise who received the message.
    """
    text = body_text(email)
    text = _ADDRESS_RE.sub(" emailaddress ", text)
    text = _URL_RE.sub(" urladdress ", text)
    text = _DIGITS_RE.sub("0", text)
    text = _ARTEFACT_RE.sub(" ", text)
    names = recipient_words(email)
    if names:
        pattern = r"\b(?:" + "|".join(map(re.escape, sorted(names))) + r")\b"
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


# --- TF-IDF --------------------------------------------------------------------------------


class TextFeatures(TransformerMixin, BaseEstimator):  # type: ignore[misc]  # sklearn is untyped
    """Word and character TF-IDF on the scrubbed body text, with a bounded vocabulary."""

    def __init__(
        self,
        word_max_features: int = 50_000,
        char_max_features: int = 100_000,
        min_df: int = 3,
    ) -> None:
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.min_df = min_df

    def _vectorizers(self) -> dict[str, TfidfVectorizer]:
        common: dict[str, Any] = {
            "min_df": self.min_df,
            "sublinear_tf": True,  # 1 + log(tf): ten "verify" are not ten times one
            "strip_accents": "unicode",
            "dtype": np.float32,
        }
        return {
            "word": TfidfVectorizer(
                ngram_range=(1, 2), max_features=self.word_max_features, **common
            ),
            "char": TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=self.char_max_features,
                **common,
            ),
        }

    def fit(self, X: Sequence[Email], y: object = None) -> Self:
        texts = [scrubbed_text(email) for email in X]
        self.vectorizers_ = self._vectorizers()
        for name, vectorizer in self.vectorizers_.items():
            vectorizer.fit(texts)
            logger.info("text features: %s vocabulary %d", name, len(vectorizer.vocabulary_))
        return self

    def transform(self, X: Sequence[Email]) -> sp.csr_matrix:
        check_is_fitted(self, "vectorizers_")
        texts = [scrubbed_text(email) for email in X]
        blocks: list[sp.csr_matrix] = [v.transform(texts) for v in self.vectorizers_.values()]
        out: sp.csr_matrix = sp.hstack(blocks, format="csr", dtype=np.float32)
        return out

    @property
    def vocabulary_size(self) -> int:
        check_is_fitted(self, "vectorizers_")
        return sum(len(v.vocabulary_) for v in self.vectorizers_.values())

    def get_feature_names_out(self, input_features: object = None) -> NDArray[np.object_]:
        check_is_fitted(self, "vectorizers_")
        names = [
            f"{name}:{term}"
            for name, v in self.vectorizers_.items()
            for term in v.get_feature_names_out()
        ]
        return np.asarray(names, dtype=object)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.string = True
        return tags

    # Models are saved with skops, which rebuilds only trusted types instead of running pickle
    # code. skops stores a dict entry by entry: a 150k-term vocabulary took 150 s and 92 MB. The
    # fitted state is therefore saved as arrays (terms in column order, IDF weights), which skops
    # stores natively, and the vectorisers are rebuilt from them on load.

    def __getstate__(self) -> dict[str, Any]:
        # A copy: object.__getstate__ returns the instance's own __dict__, and saving a model must
        # not unfit it.
        state: dict[str, Any] = dict(super().__getstate__())
        vectorizers = state.pop("vectorizers_", None)
        if vectorizers is not None:
            state["fitted_"] = {
                name: {
                    "terms": np.asarray(v.get_feature_names_out(), dtype=np.str_),
                    "idf": np.asarray(v.idf_, dtype=np.float64),
                }
                for name, v in vectorizers.items()
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        fitted = state.pop("fitted_", None)
        super().__setstate__(state)
        if fitted is not None:
            self.vectorizers_ = self._vectorizers()
            for name, vectorizer in self.vectorizers_.items():
                terms = fitted[name]["terms"].tolist()
                vectorizer.vocabulary_ = {term: i for i, term in enumerate(terms)}
                vectorizer.idf_ = fitted[name]["idf"]


# --- lexicon -------------------------------------------------------------------------------


def _cues(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# Each cue counts once per message, however often it repeats: the feature is "how many distinct
# pressure tactics", which does not grow with the length of the message.
URGENCY_CUES: Final = _cues(
    r"\burgent",
    r"\bimmediate(ly)?\b",
    r"\bsuspen(d|ded|sion)\b",
    r"\bexpir(e|es|ed|ation|ing)\b",
    r"\bwithin (24|48|72) ?(hours|hrs|h)\b",
    r"\baction (is )?required\b",
    r"\bfinal (notice|warning|reminder)\b",
    r"\b(locked|blocked|lock(ed)? out)\b",
    r"\bunusual (activity|sign-?in|login)\b",
    r"\bunauthori[sz]ed\b",
    r"\b(de|re)activat",
    r"\brestrict(ed|ion)?\b",
)
CREDENTIAL_CUES: Final = _cues(
    r"\bpass(word|code)\b",
    r"\buser ?(name|id)\b",
    r"\blog ?in\b|\blog-in\b",
    r"\bsign ?in\b",
    r"\bverif(y|ication|ied)\b",
    r"\bconfirm(ation)?\b",
    r"\baccount\b",
    r"\b(ssn|social security)\b",
    r"\bpin\b",
    r"\b(credit|debit) card\b|\bcard number\b|\bcvv\b",
    r"\bbilling\b",
    r"\bupdate (your )?(account|information|details|payment)\b",
)

LEXICON_FEATURES: Final = (
    *(
        f"{part}_{name}"
        for part in ("body", "subject")
        for name in ("urgency_cues", "credential_cues", "uppercase_ratio", "exclamation_ratio")
    ),
    "body_empty",
)


def cue_count(text: str, cues: Sequence[re.Pattern[str]]) -> int:
    return sum(1 for cue in cues if cue.search(text))


def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def exclamation_ratio(text: str) -> float:
    """Share of `!` among non-space characters."""
    visible = len(text) - sum(c.isspace() for c in text)
    return text.count("!") / visible if visible else 0.0


def _text_features(prefix: str, text: str) -> dict[str, float]:
    return {
        f"{prefix}_urgency_cues": float(cue_count(text, URGENCY_CUES)),
        f"{prefix}_credential_cues": float(cue_count(text, CREDENTIAL_CUES)),
        f"{prefix}_uppercase_ratio": uppercase_ratio(text),
        f"{prefix}_exclamation_ratio": exclamation_ratio(text),
    }


def lexicon_features(email: Email) -> dict[str, float]:
    body = body_text(email)
    return {
        **_text_features("body", body),
        **_text_features("subject", email.subject),
        # 3.2% of phishing has no text at all (image or attachment only): the text model is blind.
        "body_empty": float(not body.strip()),
    }


class LexiconFeatures(EmailFeatures):
    """Urgency and credential cues, uppercase and exclamation ratios, for body and subject."""

    names: ClassVar[tuple[str, ...]] = LEXICON_FEATURES

    def extract(self, email: Email) -> Mapping[str, float]:
        return lexicon_features(email)
