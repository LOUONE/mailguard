"""Evaluate the registered baselines on the test split. Writes reports/evaluation.{json,md}."""

import sys

from mailguard.models.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
