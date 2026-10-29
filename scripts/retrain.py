"""Retrain and promote only on a significant gain. See src/mailguard/pipeline/retrain.py."""

import sys

from mailguard.pipeline.retrain import main

if __name__ == "__main__":
    sys.exit(main())
