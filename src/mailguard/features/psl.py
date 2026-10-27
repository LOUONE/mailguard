"""Registrable domains from a bundled Public Suffix List snapshot. No network access, ever.

The list (https://publicsuffix.org, MPL 2.0) says which suffixes are open to registration:
`co.uk`, `github.io`, `*.ck`. The registrable domain is that suffix plus one label, so
`login.paypal.co.uk` and `www.paypal.co.uk` both belong to `paypal.co.uk`. The snapshot is
refreshed by hand, like the corpus manifest (see docs/features.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources

SNAPSHOT = "public_suffix_list.dat"


@dataclass(frozen=True, slots=True)
class PublicSuffixList:
    rules: frozenset[str]
    wildcards: frozenset[str]  # "ck" for the rule "*.ck"
    exceptions: frozenset[str]  # "www.ck" for the rule "!www.ck"

    @classmethod
    def parse(cls, text: str) -> PublicSuffixList:
        rules: set[str] = set()
        wildcards: set[str] = set()
        exceptions: set[str] = set()
        for line in text.splitlines():
            rule = line.strip().split(maxsplit=1)[0] if line.strip() else ""
            if not rule or rule.startswith("//"):
                continue
            rule = _to_ascii(rule.casefold())
            if rule.startswith("!"):
                exceptions.add(rule[1:])
            elif rule.startswith("*."):
                wildcards.add(rule[2:])
            else:
                rules.add(rule)
        return cls(frozenset(rules), frozenset(wildcards), frozenset(exceptions))

    def suffix_length(self, labels: list[str]) -> int:
        """Number of trailing labels that form the public suffix (at least 1)."""
        for i in range(len(labels)):
            candidate = ".".join(labels[i:])
            if candidate in self.exceptions:
                return len(labels) - i - 1
            if candidate in self.rules:
                return len(labels) - i
            if i + 1 < len(labels) and ".".join(labels[i + 1 :]) in self.wildcards:
                return len(labels) - i
        return 1  # the default rule "*": an unknown TLD is a public suffix

    def registrable_domain(self, host: str) -> str | None:
        """`a.b.example.co.uk` -> `example.co.uk`. None for a bare suffix or an empty host."""
        labels = _to_ascii(host.casefold()).rstrip(".").split(".")
        if not labels or any(not label for label in labels):
            return None
        n = self.suffix_length(labels)
        return ".".join(labels[-(n + 1) :]) if len(labels) > n else None


def _to_ascii(name: str) -> str:
    """IDNA-encode non-ASCII labels (`bücher.de` -> `xn--bcher-kva.de`), leaving bad ones as-is."""
    if name.isascii():
        return name
    labels = []
    for label in name.split("."):
        try:
            labels.append(label.encode("idna").decode("ascii") if label else label)
        except UnicodeError:
            labels.append(label)
    return ".".join(labels)


@cache
def default_list() -> PublicSuffixList:
    text = resources.files("mailguard.features").joinpath("data", SNAPSHOT).read_text("utf-8")
    return PublicSuffixList.parse(text)


def registrable_domain(host: str) -> str | None:
    """Registrable domain of `host` according to the bundled snapshot."""
    return default_list().registrable_domain(host)


def public_suffix(host: str) -> str | None:
    """The public suffix of `host` (`co.uk` for `a.example.co.uk`). None for an empty host."""
    labels = _to_ascii(host.casefold()).rstrip(".").split(".")
    if any(not label for label in labels):
        return None
    return ".".join(labels[-default_list().suffix_length(labels) :])
