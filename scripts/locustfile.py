"""Load test of the scoring API (docs/performance.md).

    make serve                                   # in one terminal
    uv run --group perf locust -f scripts/locustfile.py --headless -u 8 -r 8 -t 60s \
        --host http://127.0.0.1:8000 --csv reports/perf/run

Two kinds of payload:

- `MAILGUARD_LOADTEST_CORPUS=1`: real messages of the test split, read from data/processed on
  this machine. Realistic sizes (median 5 KB, p99 100 KB, HTML), never written anywhere.
- otherwise: the made-up messages of `demo_traffic.py`, a few hundred bytes each. A floor for
  latency, useful on a machine without the corpora.

Each simulated user sends its next request as soon as the previous one returns (a closed loop):
throughput then measures what the service can absorb, and latency what it costs.
"""

from __future__ import annotations

import base64
import os
import random

from locust import HttpUser, constant, events, task

from demo_traffic import legitimate, phishing  # same folder, on locust's path

PAYLOADS: list[dict[str, object]] = []


@events.init.add_listener
def _load_payloads(**_: object) -> None:
    if os.environ.get("MAILGUARD_LOADTEST_CORPUS") == "1":
        from mailguard.config import get_settings
        from mailguard.data.dataset import load_dataset

        frame = load_dataset(get_settings().data_dir / "processed")
        raws = frame.loc[frame["split"] == "test", "raw"].tolist()
        PAYLOADS.extend(
            {"raw": base64.b64encode(raw).decode(), "encoding": "base64"} for raw in raws
        )
    else:
        rng = random.Random(0)
        PAYLOADS.extend(
            {"raw": phishing(rng) if i % 5 == 0 else legitimate(rng)} for i in range(500)
        )


class Mailer(HttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        self.rng = random.Random(id(self))

    @task
    def predict(self) -> None:
        self.client.post("/v1/predict", json=self.rng.choice(PAYLOADS), name="/v1/predict")
