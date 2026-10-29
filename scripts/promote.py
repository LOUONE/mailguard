"""Give a registered model version the `production` alias, the one the API serves.

    uv run python scripts/promote.py                      # latest mailguard-lightgbm
    uv run python scripts/promote.py --version 3

Manual for now; the retraining flow (#39) promotes only on a significant gain.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mailguard.config import get_settings
from mailguard.models.registry import PRODUCTION, Version, client, latest_version, promote


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="promote")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--version", default="latest")
    parser.add_argument("--alias", default=PRODUCTION)
    args = parser.parse_args(argv)

    try:
        if args.version == "latest":
            version = latest_version(args.model, settings)
        else:
            found = client(settings).get_model_version(args.model, args.version)
            version = Version(args.model, str(found.version), dict(found.tags))
        promote(version, settings, args.alias)
    except (LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{version.name} v{version.version} is now @{args.alias}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
