"""Compare recorded traffic with the validation data. See src/mailguard/monitoring/drift_job.py."""

import sys

from mailguard.monitoring.drift_job import main

if __name__ == "__main__":
    sys.exit(main())
