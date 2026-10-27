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

# Brands most impersonated in phishing display names ("PayPal Security" <x@evil.example>), as
# lowercase tokens compared with the sender's registrable domain. Hand-picked from the brands that
# APWG phishing trend reports and vendor brand-phishing rankings name year after year.
BRANDS: Final = frozenset(
    {
        "adobe", "amazon", "americanexpress", "apple", "bankofamerica", "barclays", "chase",
        "citibank", "dhl", "docusign", "dropbox", "ebay", "facebook", "fedex", "google", "hsbc",
        "instagram", "irs", "linkedin", "microsoft", "netflix", "office365", "outlook", "paypal",
        "santander", "usps", "wellsfargo", "whatsapp", "yahoo",
    }
)  # fmt: skip

# Attachment extensions that run code or render a page when opened.
RISKY_EXTENSIONS: Final = frozenset(
    {
        "bat", "cmd", "com", "cpl", "exe", "hta", "htm", "html", "img", "iso", "jar", "js", "jse",
        "lnk", "msi", "pif", "ps1", "scr", "shtml", "vbe", "vbs", "wsf", "docm", "xlsm", "pptm",
    }
)  # fmt: skip
ARCHIVE_EXTENSIONS: Final = frozenset({"7z", "gz", "rar", "tgz", "zip"})

# Domains that the corpus maintainers substituted into headers: they identify the corpus, not the
# sender (docs/data.md). Addresses under them are treated as unknown.
ARTEFACT_DOMAINS: Final = ("spamassassin.taint.org", "taint.org", "monkey.org")
