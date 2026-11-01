"""The hard evaluation set: test emails the classifier is unsure of or wrong about, read by hand.

The agent only costs something on the cases it is sent, so it is judged on those. Selection
(`select_candidates`) takes the test split, keeps every message whose calibrated score falls in a
wide band around the threshold or that the classifier gets wrong at the 1% operating point, and
keeps one message per duplicate cluster. The band is wider than any grey zone the agent will be
given (#49), so each grey zone tried later is a subset of the set.

Annotation is blind: messages come in a seeded random order, without their corpus label or score,
and each gets a label and one rationale category from `CATEGORIES` (guideline in
`docs/hard_set.md`). Only ids, clusters, labels and categories are committed
(`evals/hard_set.csv`); the text is rebuilt from the downloaded corpora by `load_hard_set`.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from mailguard.parsing import Email

HandLabel = Literal["phishing", "spam", "ham", "unclear"]
SelectedAs = Literal["grey", "error", "both"]

CATEGORIES: Final[Mapping[HandLabel, tuple[str, ...]]] = {
    "phishing": (
        "credential",  # fake login, account verification or suspension
        "payment",  # invoice, billing update, gift card, wire transfer
        "personal_data",  # identity documents, tax refund forms, survey for data
        "callback",  # call a number: fake support, fake order to cancel
        "malware",  # attachment or download to open
        "advance_fee",  # lottery, inheritance, 419: money first, reward never
        "other_phishing",
    ),
    "spam": ("marketing", "other_spam"),
    "ham": ("personal", "list_or_newsletter", "automated", "other_ham"),
    "unclear": ("unreadable", "ambiguous"),
}
LABEL_KEYS: Final[Mapping[str, HandLabel]] = {
    "p": "phishing",
    "s": "spam",
    "h": "ham",
    "u": "unclear",
}
FIELDS: Final = ("id", "cluster", "selected_as", "label", "category")
SHUFFLE_SEED: Final = 20261101
# Enough to judge most messages; the annotator can ask for the rest.
EXCERPT_CHARS: Final = 2500
MAX_URLS: Final = 15


class Annotation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    cluster: str
    selected_as: SelectedAs
    label: HandLabel
    category: str

    @model_validator(mode="after")
    def _category_fits_label(self) -> Annotation:
        if self.category not in CATEGORIES[self.label]:
            raise ValueError(f"category {self.category!r} is not one of {self.label}'s")
        return self

    @property
    def target(self) -> int | None:
        """1 phishing, 0 ham or spam (ADR 0001), None when the annotator could not tell."""
        return None if self.label == "unclear" else int(self.label == "phishing")


# --- selection -------------------------------------------------------------------------------


def select_candidates(
    rows: pd.DataFrame,
    scores: NDArray[np.float64],
    threshold: float,
    band: tuple[float, float],
    seed: int = SHUFFLE_SEED,
) -> pd.DataFrame:
    """One message per cluster among the grey band and the errors, in a seeded random order.

    `rows` needs `id`, `cluster` and `target`, in the order of `scores`. A cluster is kept when
    any of its messages qualifies; its representative is the qualifying message with the
    smallest id, so the choice does not depend on row order.
    """
    low, high = band
    grey = (scores >= low) & (scores <= high)
    error = (scores > threshold) != (rows["target"].to_numpy() == 1)
    kept = rows.loc[grey | error, ["id", "cluster"]].assign(
        selected_as=np.where(grey & error, "both", np.where(grey, "grey", "error"))[grey | error]
    )
    representatives = kept.sort_values("id").drop_duplicates("cluster", keep="first")
    order = np.random.default_rng(seed).permutation(len(representatives))
    shuffled: pd.DataFrame = representatives.iloc[order].reset_index(drop=True)
    return shuffled


# --- files -----------------------------------------------------------------------------------


def write_annotations(path: Path, annotations: Iterable[Annotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(a.model_dump() for a in annotations)


def read_annotations(path: Path) -> list[Annotation]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return [Annotation.model_validate(row) for row in csv.DictReader(source)]


def load_hard_set(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """The committed set joined to a freshly built dataset: hand labels next to raw messages.

    Fails if an id is missing, which means the corpora or the dataset build changed.
    """
    annotations = read_annotations(path)
    if not annotations:
        raise ValueError(f"{path} holds no annotation")
    hand = pd.DataFrame([a.model_dump() | {"hand_target": a.target} for a in annotations])
    joined = hand.merge(frame.drop(columns=["cluster"]), on="id", how="left", validate="one_to_one")
    missing = joined.loc[joined["raw"].isna(), "id"].tolist()
    if missing:
        raise LookupError(f"{len(missing)} ids not in the dataset, first: {missing[:3]}")
    return joined


# --- annotation ------------------------------------------------------------------------------

_SCHEME_RE = re.compile(r"\bhttp(s?)://", re.IGNORECASE)
# A host name: labels and dots ending in a letter-only TLD. "3.5" and "e.g." are left alone.
_HOST_RE = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b", re.IGNORECASE)


def defang(text: str) -> str:
    """Links a terminal will not make clickable: `hxxp://`, `example[.]com`."""
    text = _SCHEME_RE.sub(lambda m: f"hxxp{m.group(1)}://", text)
    return _HOST_RE.sub(lambda m: m.group(0).replace(".", "[.]"), text)


def render(email: Email, full: bool = False) -> str:
    """What the annotator reads: envelope, links, attachments and body, all defanged."""
    lines = [
        f"From:        {email.from_name or ''} <{email.from_address or '?'}>",
        f"Reply-To:    {email.reply_to or '-'}",
        f"Return-Path: {email.return_path or '-'}",
        f"Subject:     {email.subject}",
        f"To:          {len(email.to)} recipient(s)",
    ]
    if email.attachments:
        lines.append("Attachments:")
        lines += [
            f"  - {a.filename or '(no name)'}  {a.content_type}"
            f"  sniffed={a.detected_type or '?'}  {a.size} B"
            for a in email.attachments
        ]
    if email.urls:
        lines.append(f"Links ({len(email.urls)}):")
        lines += [f"  - {url[:150]}" for url in email.urls[:MAX_URLS]]
        if len(email.urls) > MAX_URLS:
            lines.append(f"  ... {len(email.urls) - MAX_URLS} more")
    body = " ".join(email.body_text.split())
    lines += ["", body if full else body[:EXCERPT_CHARS]]
    if not full and len(body) > EXCERPT_CHARS:
        lines.append(f"[... {len(body) - EXCERPT_CHARS} more characters: type m]")
    return defang("\n".join(lines))


def menu(label: HandLabel) -> str:
    return "  ".join(f"{i}={c}" for i, c in enumerate(CATEGORIES[label], start=1))


def annotate(
    candidates: pd.DataFrame,
    done: list[Annotation],
    email_of: Callable[[str], Email],
    ask: Callable[[str], str],
    show: Callable[[str], None],
    save: Callable[[list[Annotation]], None],
) -> list[Annotation]:
    """Ask a label and a category for each candidate not yet annotated, saving after each one.

    Keys: p/s/h/u for the label, then the category number; m shows the whole body, k skips to
    the next message, q stops. Progress is saved after every answer, so stopping loses nothing.
    """
    annotated = {a.id for a in done}
    todo = candidates[~candidates["id"].isin(annotated)]
    for n, row in enumerate(todo.itertuples(index=False), start=len(done) + 1):
        email = email_of(str(row.id))
        # No id on screen: it names the corpus file, hence the corpus label.
        show(f"\n===== {n} / {len(candidates)}\n{render(email)}")
        label: HandLabel | None = None
        while label is None:
            key = ask("label [p]hishing [s]pam [h]am [u]nclear, [m]ore, s[k]ip, [q]uit: ")
            key = key.strip().lower()
            if key == "q":
                return done
            if key == "k":
                break
            if key == "m":
                show(render(email, full=True))
                continue
            label = LABEL_KEYS.get(key)
        if label is None:
            continue
        category: str | None = None
        while category is None:
            choice = ask(f"category {menu(label)}: ").strip()
            options = CATEGORIES[label]
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                category = options[int(choice) - 1]
        done = [
            *done,
            Annotation(
                id=str(row.id),
                cluster=str(row.cluster),
                selected_as=row.selected_as,
                label=label,
                category=category,
            ),
        ]
        save(done)
    return done


def agreement(annotations: Iterable[Annotation], corpus_labels: Mapping[str, str]) -> pd.DataFrame:
    """Hand label against corpus label, as counts: where the corpus is wrong, and how often."""
    pairs = pd.DataFrame(
        [(corpus_labels[a.id], a.label) for a in annotations], columns=["corpus", "hand"]
    )
    return pd.crosstab(pairs["corpus"], pairs["hand"])
