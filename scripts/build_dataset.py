"""Build data/processed/ (dataset + manifest) from data/raw/. See docs/data.md."""

import sys

from mailguard.data.dataset import main

if __name__ == "__main__":
    sys.exit(main())
