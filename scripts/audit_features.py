"""Era-leakage audit of each feature group on training phishing. See docs/features.md.

Prints a Markdown table of counts and AUCs only, safe to paste into the docs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from mailguard.config import get_settings
from mailguard.data.dataset import load_dataset
from mailguard.features.audit import ALERT_AUC, era_audit
from mailguard.features.base import EmailFeatures
from mailguard.features.headers import AttachmentFeatures, HeaderFeatures
from mailguard.features.urls import UrlFeatures
from mailguard.parsing import parse_email

GROUPS: dict[str, EmailFeatures] = {
    "headers": HeaderFeatures(),
    "attachments": AttachmentFeatures(),
    "urls": UrlFeatures(),
}


def main(argv: Sequence[str] | None = None) -> int:
    frame = load_dataset(get_settings().data_dir / "processed")
    phishing = frame[(frame["split"] == "train") & (frame["label"] == "phishing")]
    emails = [parse_email(raw) for raw in phishing["raw"]]
    recent = (phishing["period"] == "2015-2025").to_numpy()
    clusters = phishing["cluster"].tolist()
    print(f"{len(emails)} training phishing messages, {recent.mean():.1%} from 2015-2025\n")

    print("| Group | Era AUC | Top features (univariate AUC) |")
    print("|---|---|---|")
    for name, extractor in GROUPS.items():
        X = extractor.transform(emails)
        result = era_audit(X, recent, clusters, extractor.names)
        top = sorted(result.feature_auc.items(), key=lambda kv: -kv[1])[:3]
        flag = " ⚠" if result.alert else ""
        cells = ", ".join(f"`{f}` {auc:.2f}" for f, auc in top)
        print(f"| {name} | {result.group_auc:.3f}{flag} | {cells} |")

    everything = np.hstack([e.transform(emails) for e in GROUPS.values()])
    names = [n for e in GROUPS.values() for n in e.names]
    combined = era_audit(everything, recent, clusters, names)
    print(f"| all | {combined.group_auc:.3f}{' ⚠' if combined.alert else ''} | |")
    print(f"\nAlert threshold: {ALERT_AUC}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
