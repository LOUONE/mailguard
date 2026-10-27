"""Curated reference lists used by the features, each with its source and snapshot date.

They are data, not code: refreshing one is a reviewed change like a manifest checksum.
"""

from __future__ import annotations

from typing import Final

# URL shorteners: hide the real destination from the reader and from link checks.
# Common public services, checked by hand on 2026-10-27.
SHORTENERS: Final = frozenset(
    {
        "bit.ly", "bit.do", "buff.ly", "cutt.ly", "goo.gl", "is.gd", "lnkd.in", "ow.ly",
        "qrco.de", "rb.gy", "rebrand.ly", "s.id", "shorturl.at", "t.co", "t.ly", "tiny.cc",
        "tinyurl.com", "v.gd", "x.co",
    }
)  # fmt: skip

# The 20 TLDs with the highest phishing score (phishing domains per 10,000 registered) in
# Interisle's Phishing Landscape 2025, May 2024 to April 2025, via the Cybercrime Information
# Center: https://www.cybercrimeinfocenter.org/phishing-activity-in-tlds-may-april-2025
# None of them existed in 2002, so this feature can only fire on recent mail (see docs/features.md).
SUSPICIOUS_TLDS: Final = frozenset(
    {
        "xin", "bond", "help", "win", "cfd", "finance", "support", "top", "world", "fyi",
        "lol", "icu", "cyou", "mom", "zone", "monster", "pics", "sbs", "best", "buzz",
    }
)  # fmt: skip

# Words that phishing URLs put in hosts and paths to look like a login flow.
URL_KEYWORDS: Final = (
    "account", "banking", "confirm", "login", "password", "secure", "signin", "update",
    "verify", "webscr",
)  # fmt: skip
