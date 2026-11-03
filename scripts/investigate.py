"""Run the investigation agent on raw emails, locally: classifier, tools and LLM.

    uv run python scripts/investigate.py docs/demo/phishing.eml
    uv run python scripts/investigate.py --no-network some.eml     # offline tools only

Needs the production model (`make promote`) and the LLM of the settings (Ollama by default,
`qwen3:8b`). Network tools use RDAP and the feeds of `make reputation` when present. Prints the
classifier's opinion and the investigation as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from mailguard.agent.backends import make_client
from mailguard.agent.investigator import Investigator
from mailguard.agent.tools import Case, Opinion, Reason, Tool, Toolbox
from mailguard.agent.tools.network import (
    MANIFEST,
    RdapClient,
    ReputationIndex,
    default_bootstrap,
    network_tools,
)
from mailguard.agent.tools.offline import OFFLINE_TOOLS
from mailguard.config import Settings, get_settings
from mailguard.parsing import parse_email
from mailguard.serving.model import ServedModel, load_production


def opinion_of(model: ServedModel, raw: bytes, settings: Settings) -> tuple[Case, dict[str, Any]]:
    email = parse_email(raw)
    prediction = model.predict(email, settings.base_rate)
    opinion = Opinion(
        score=prediction.score,
        probability=prediction.probability,
        is_phishing=prediction.is_phishing,
        threshold=model.threshold,
        reasons=tuple(Reason(r["feature"], r["contribution"]) for r in prediction.reasons),
    )
    summary = {"score": round(prediction.score, 4), "is_phishing": prediction.is_phishing}
    return Case(email, opinion, settings.trusted_authserv_id), summary


async def run(paths: Sequence[Path], network: bool) -> None:
    settings = get_settings()
    model = load_production(settings)
    client = make_client(settings)
    tools: list[Tool[Any, Any]] = list(OFFLINE_TOOLS)
    rdap: RdapClient | None = None
    if network:
        rdap = RdapClient(default_bootstrap(), settings.rdap_timeout_s)
        has_feeds = (settings.reputation_dir / MANIFEST).is_file()
        index = ReputationIndex.load(settings.reputation_dir) if has_feeds else None
        tools += network_tools(rdap, index, settings.reputation_max_age_days)
    investigator = Investigator(client, Toolbox.of(tools))
    try:
        for path in paths:
            case, classifier = opinion_of(model, path.read_bytes(), settings)
            investigation = await investigator.investigate(case)
            report = {
                "email": str(path),
                "classifier": classifier,
                "investigation": investigation.model_dump(mode="json"),
            }
            print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        await client.aclose()
        if rdap is not None:
            await rdap.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="investigate")
    parser.add_argument("emails", nargs="+", type=Path)
    parser.add_argument("--no-network", action="store_true", help="offline tools only")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # stdout carries the JSON report only, so it can be piped; log lines go to stderr.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    asyncio.run(run(args.emails, network=not args.no_network))
    return 0


if __name__ == "__main__":
    sys.exit(main())
