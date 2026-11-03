"""Investigation tools that need no network: headers, links, attachments, the classifier's view.

They reuse the parsing and feature code of Phase 1, but read what the model was not allowed to:
`Authentication-Results` (useless on the corpora, where only recent phishing carries it, and
exactly what a receiving server adds on live mail) and the full structure of each link.
"""

from __future__ import annotations

import re
from typing import Final, Literal
from urllib.parse import urlsplit

from pydantic import Field

from mailguard.agent.tools import (
    Arguments,
    Case,
    NoArguments,
    Output,
    Tool,
    clip,
)
from mailguard.features.auth import authentication_results
from mailguard.features.headers import (
    brand_mismatch,
    differs,
    domain_of,
    double_extension,
    type_mismatch,
)
from mailguard.features.lists import (
    ARCHIVE_EXTENSIONS,
    BRANDS,
    RISKY_EXTENSIONS,
    SHORTENERS,
    SUSPICIOUS_TLDS,
    URL_KEYWORDS,
)
from mailguard.features.psl import public_suffix
from mailguard.features.urls import anchor_mismatch, parse_url
from mailguard.parsing.filetypes import extensions
from mailguard.parsing.html import read_html

MAX_ATTACHMENTS: Final = 20
MAX_REASONS: Final = 8

# --- check_auth_headers ----------------------------------------------------------------------


class AuthCheck(Output):
    header_found: bool
    authserv_id: str | None
    trusted: bool = Field(
        description="True only if the header was added by the configured receiving server."
    )
    spf: str | None
    dkim: str | None
    dmarc: str | None
    from_domain: str | None
    reply_to_domain: str | None
    return_path_domain: str | None
    reply_to_differs: bool
    return_path_differs: bool
    display_name: str | None
    display_name_names_other_brand: bool


async def check_auth_headers(case: Case, _: NoArguments) -> AuthCheck:
    email = case.email
    results = authentication_results(email, case.trusted_authserv_id)
    from_domain = domain_of(email.from_address)
    reply_to, return_path = domain_of(email.reply_to), domain_of(email.return_path)
    return AuthCheck(
        header_found=results.authserv_id is not None,
        authserv_id=clip(results.authserv_id, 100),
        trusted=case.trusted_authserv_id is not None and results.authserv_id is not None,
        spf=results.spf,
        dkim=results.dkim,
        dmarc=results.dmarc,
        from_domain=from_domain,
        reply_to_domain=reply_to,
        return_path_domain=return_path,
        reply_to_differs=differs(reply_to, from_domain),
        return_path_differs=differs(return_path, from_domain),
        display_name=clip(email.from_name, 100),
        display_name_names_other_brand=brand_mismatch(email.from_name, from_domain),
    )


CHECK_AUTH_HEADERS: Final = Tool(
    name="check_auth_headers",
    description=(
        "SPF, DKIM and DMARC results from the Authentication-Results header, and whether the "
        "From, Reply-To and Return-Path domains agree. Untrusted unless `trusted` is true."
    ),
    arguments=NoArguments,
    run=check_auth_headers,
)

# --- analyze_url -----------------------------------------------------------------------------

# Characters that stand in for others in lookalike domains: digits for letters, and the
# Cyrillic and Greek letters that render like Latin ones (a small part of Unicode's
# confusables; Phase 4 brings the full UTS #39 skeletons).
_CONFUSABLE: Final = str.maketrans(
    {
        "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    }
    # Cyrillic a e o p c y x i j s d, Latin script g, Greek o a v p, by code point: as
    # characters they are unreadable in a diff, which is the whole problem.
    | {
        chr(code): latin
        for code, latin in (
            (0x0430, "a"), (0x0435, "e"), (0x043E, "o"), (0x0440, "p"), (0x0441, "c"),
            (0x0443, "y"), (0x0445, "x"), (0x0456, "i"), (0x0458, "j"), (0x0455, "s"),
            (0x0501, "d"), (0x0261, "g"), (0x03BF, "o"), (0x03B1, "a"), (0x03BD, "v"),
            (0x03C1, "p"),
        )
    }
)  # fmt: skip
_MULTI_CONFUSABLE: Final = (("rn", "m"), ("vv", "w"), ("cl", "d"))
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Short brands ("irs", "dhl", "chase") sit inside ordinary words ("first", "purchase"): they
# only match as whole tokens. Longer ones also match inside a label ("securepaypal").
MIN_SUBSTRING_BRAND: Final = 6

Squatting = Literal["brand_domain", "typosquat", "combosquat", "homoglyph"]


class BrandSignal(Output):
    brand: str
    kind: Squatting | Literal["brand_outside_domain"]


class UrlAnalysis(Output):
    url: str
    in_email: bool = Field(description="The URL appears in the email's links or text.")
    parsed: bool
    host: str | None = None
    registrable_domain: str | None = None
    unicode_host: str | None = Field(None, description="The host as a reader sees it.")
    is_ip: bool = False
    punycode: bool = False
    subdomain_depth: int = 0
    has_userinfo: bool = Field(False, description="`user@host` form: the part before @ is decor.")
    non_standard_port: bool = False
    shortener: bool = False
    suspicious_tld: bool = False
    keywords: tuple[str, ...] = ()
    brands: tuple[BrandSignal, ...] = ()
    shown_as: tuple[str, ...] = Field((), description="Link texts of anchors pointing here.")
    shown_domain_differs: bool = Field(False, description="A link text names another domain.")


class UrlArguments(Arguments):
    url: str = Field(min_length=1, max_length=2048, description="A URL from the email.")


def osa_distance(a: str, b: str) -> int:
    """Edit distance counting adjacent swaps as one edit (optimal string alignment)."""
    previous2: list[int] = []
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                current[j] = min(current[j], previous2[j - 2] + 1)
        previous2, previous = previous, current
    return previous[-1]


def deconfuse(text: str) -> str:
    text = text.casefold().translate(_CONFUSABLE)
    for many, one in _MULTI_CONFUSABLE:
        text = text.replace(many, one)
    return text


def _decode_host(host: str) -> str:
    labels = []
    for label in host.split("."):
        try:
            labels.append(
                label.encode("ascii").decode("idna") if label.startswith("xn--") else label
            )
        except UnicodeError:
            labels.append(label)
    return ".".join(labels)


def typo_budget(brand: str) -> int:
    """Edits a lookalike may differ by. None for short brands: `dhs` is not `dhl`."""
    return 0 if len(brand) <= 4 else 1 if len(brand) <= 7 else 2


def brand_signals(registrable: str, rest: str) -> list[BrandSignal]:
    """How a URL relates to known brands: their own domain, a lookalike, or a decoy.

    `registrable` is the registrable domain (punycode or not); `rest` the subdomains, path and
    query, where a brand name proves nothing about who owns the link.
    """
    suffix = public_suffix(registrable) or ""
    label = _decode_host(registrable.removesuffix("." + suffix)).casefold()
    tokens = set(_TOKEN_RE.findall(label))
    skeleton = deconfuse(label)
    found: dict[str, Squatting] = {}
    for brand in sorted(BRANDS):
        if label == brand:
            found[brand] = "brand_domain"
        elif skeleton == brand:
            found[brand] = "typosquat" if label.isascii() else "homoglyph"
        elif 0 < osa_distance(label, brand) <= typo_budget(brand):
            found[brand] = "typosquat"
        elif brand in tokens or (len(brand) >= MIN_SUBSTRING_BRAND and brand in skeleton):
            found[brand] = "combosquat"
    signals = [BrandSignal(brand=b, kind=k) for b, k in found.items()]
    # Raw tokens too: deconfusing turns the digits of "office365" into letters.
    rest = rest.casefold()
    elsewhere = (set(_TOKEN_RE.findall(rest)) | set(_TOKEN_RE.findall(deconfuse(rest)))) & BRANDS
    signals += [
        BrandSignal(brand=b, kind="brand_outside_domain") for b in sorted(elsewhere - set(found))
    ]
    return signals


async def analyze_url(case: Case, arguments: UrlArguments) -> UrlAnalysis:
    url = arguments.url.strip()
    email = case.email
    anchors = read_html(email.html_body).anchors if email.html_body else ()
    in_email = url in email.urls or url in email.body_text
    parsed = parse_url(url)
    if parsed is None:
        return UrlAnalysis(url=clip(url, 300) or "", in_email=in_email, parsed=False)
    lowered = url.casefold()
    unicode_host = _decode_host(parsed.host)
    parts = urlsplit(url if "://" in url else "http://" + url)
    rest = f"{parsed.host.removesuffix(parsed.domain)} {parts.path} {parts.query}"
    pointing_here = [a for a in anchors if a.href.strip() == url]
    shown = [a.text.strip() for a in pointing_here if a.text.strip()]
    return UrlAnalysis(
        url=clip(url, 300) or "",
        in_email=in_email,
        parsed=True,
        host=clip(parsed.host, 253),
        registrable_domain=parsed.domain,
        unicode_host=clip(unicode_host, 253) if unicode_host != parsed.host else None,
        is_ip=parsed.is_ip,
        punycode="xn--" in parsed.host,
        subdomain_depth=parsed.subdomain_depth,
        has_userinfo=parsed.has_userinfo,
        non_standard_port=not parsed.port_is_standard,
        shortener=parsed.domain in SHORTENERS,
        suspicious_tld=parsed.domain.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS,
        keywords=tuple(k for k in URL_KEYWORDS if k in lowered),
        brands=() if parsed.is_ip else tuple(brand_signals(parsed.domain, rest)),
        shown_as=tuple(clip(t, 100) or "" for t in shown[:5]),
        shown_domain_differs=any(map(anchor_mismatch, pointing_here)),
    )


ANALYZE_URL: Final = Tool(
    name="analyze_url",
    description=(
        "Structure of one URL from the email, without visiting it: host, registrable domain, "
        "IP or punycode host, shortener, suspicious TLD, login keywords, lookalike brand "
        "domains (typosquat, combosquat, homoglyph), and link texts that name another domain."
    ),
    arguments=UrlArguments,
    run=analyze_url,
)

# --- inspect_attachments ---------------------------------------------------------------------


class AttachmentReport(Output):
    filename: str | None
    content_type: str
    detected_type: str | None = Field(description="Family sniffed from the first bytes.")
    size: int
    extensions: list[str]
    risky_extension: bool = Field(description="Runs code or renders a page when opened.")
    archive: bool
    double_extension: bool = Field(description="`invoice.pdf.exe`: a decoy extension first.")
    type_mismatch: bool = Field(description="Name or declared type disagrees with the bytes.")


class Attachments(Output):
    count: int
    attachments: list[AttachmentReport]


async def inspect_attachments(case: Case, _: NoArguments) -> Attachments:
    reports = []
    for attachment in case.email.attachments[:MAX_ATTACHMENTS]:
        exts = extensions(attachment.filename)
        last = exts[-1] if exts else ""
        reports.append(
            AttachmentReport(
                filename=clip(attachment.filename, 150),
                content_type=clip(attachment.content_type, 100) or "",
                detected_type=attachment.detected_type,
                size=attachment.size,
                extensions=exts[-4:],
                risky_extension=last in RISKY_EXTENSIONS,
                archive=last in ARCHIVE_EXTENSIONS,
                double_extension=double_extension(attachment.filename),
                type_mismatch=type_mismatch(attachment),
            )
        )
    return Attachments(count=len(case.email.attachments), attachments=reports)


INSPECT_ATTACHMENTS: Final = Tool(
    name="inspect_attachments",
    description=(
        "Metadata of every attachment, never its content: name, declared and sniffed type, "
        "size, risky or double extensions, and name/content mismatches."
    ),
    arguments=NoArguments,
    run=inspect_attachments,
)

# --- get_classifier_explanation --------------------------------------------------------------


class ReasonOut(Output):
    feature: str
    contribution: float = Field(description="Log-odds; positive pushes towards phishing.")


class Explanation(Output):
    score: float = Field(description="Calibrated phishing score on the training data's prior.")
    probability: float = Field(description="Phishing probability at the deployment base rate.")
    label: Literal["phishing", "legitimate"]
    threshold: float
    reasons: list[ReasonOut]


async def get_classifier_explanation(case: Case, _: NoArguments) -> Explanation:
    opinion = case.opinion
    return Explanation(
        score=round(opinion.score, 4),
        probability=round(opinion.probability, 4),
        label="phishing" if opinion.is_phishing else "legitimate",
        threshold=round(opinion.threshold, 4),
        reasons=[
            ReasonOut(feature=r.feature, contribution=round(r.contribution, 3))
            for r in opinion.reasons[:MAX_REASONS]
        ],
    )


GET_CLASSIFIER_EXPLANATION: Final = Tool(
    name="get_classifier_explanation",
    description=(
        "What the LightGBM classifier concluded and the features that moved its score most "
        "(TreeSHAP). Words appear as `text__word:...`; URL, header and attachment features "
        "by name."
    ),
    arguments=NoArguments,
    run=get_classifier_explanation,
)

OFFLINE_TOOLS: Final = (
    CHECK_AUTH_HEADERS,
    ANALYZE_URL,
    INSPECT_ATTACHMENTS,
    GET_CLASSIFIER_EXPLANATION,
)
