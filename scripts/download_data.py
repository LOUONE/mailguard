"""Download the training corpora into data/raw/. See docs/data.md."""

import sys

from mailguard.data.download import main

if __name__ == "__main__":
    sys.exit(main())
