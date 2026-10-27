"""SPF, DKIM and DMARC verdicts from `Authentication-Results` headers (RFC 8601).

Not a model feature: on these corpora the header exists only in recent phishing (0% of ham, 61% of
2015-2025 phishing), so a model would learn the era, not the verdict. The Phase 3 agent uses it on
live mail instead.

Any sender can write an `Authentication-Results` header. Only one added by the receiving
infrastructure, recognised by its `authserv-id` (the first token), can be trusted; an MTA is
expected to strip forged ones with its own id before adding its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from mailguard.parsing import Email

Method = Literal["spf", "dkim", "dmarc"]
METHODS: Final[tuple[Method, ...]] = ("spf", "dkim", "dmarc")

_RESULT_RE = re.compile(r"(?<![\w.-])(spf|dkim|dmarc)\s*=\s*([a-z]+)", re.IGNORECASE)
_COMMENT_RE = re.compile(r"\([^()]*\)")


@dataclass(frozen=True, slots=True)
class AuthResults:
    authserv_id: str | None = None
    spf: str | None = None
    dkim: str | None = None
    dmarc: str | None = None

    def get(self, method: Method) -> str | None:
        value: str | None = getattr(self, method)
        return value


def parse_authentication_results(value: str) -> AuthResults:
    """Verdicts of one header value. The first result of each method wins; comments are ignored.

    `mx.example.org; spf=pass smtp.mailfrom=a.example; dkim=fail (bad sig) header.d=a.example`
    """
    text = " ".join(value.split())
    head, _, rest = text.partition(";")
    authserv_id = head.strip().split(" ", 1)[0].casefold() or None
    found: dict[str, str] = {}
    for method, result in _RESULT_RE.findall(_COMMENT_RE.sub(" ", rest)):
        found.setdefault(method.casefold(), result.casefold())
    return AuthResults(authserv_id, found.get("spf"), found.get("dkim"), found.get("dmarc"))


def authentication_results(email: Email, trusted_authserv_id: str | None = None) -> AuthResults:
    """The verdicts to rely on for `email`.

    With `trusted_authserv_id`, only a header from that server counts. Without it, the topmost
    header is used, which is the one added last, closest to the recipient. Empty if none applies.
    """
    for value in email.header_values("Authentication-Results"):
        results = parse_authentication_results(value)
        if trusted_authserv_id is None or results.authserv_id == trusted_authserv_id.casefold():
            return results
    return AuthResults()
