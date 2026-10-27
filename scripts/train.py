"""Train the Phase 1 baselines and log them to MLflow. See docs/features.md and configs/."""

import sys

from mailguard.models.train import main

if __name__ == "__main__":
    sys.exit(main())
