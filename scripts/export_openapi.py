"""Write the API's OpenAPI document to docs/api/openapi.json (committed, checked by a test)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mailguard.config import Settings
from mailguard.serving.app import create_app
from mailguard.serving.model import ServedModel

OUT = Path("docs/api/openapi.json")


def no_model(settings: Settings) -> ServedModel:
    raise LookupError("the OpenAPI document does not need a model")


def main() -> int:
    document = create_app(Settings(), loader=no_model).openapi()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
