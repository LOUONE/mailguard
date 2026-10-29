"""The drift job: recorded traffic against the validation data, as JSON and an HTML page.

    uv run python scripts/drift_report.py                 # once
    uv run python scripts/drift_report.py --every 300     # the compose service

Reference: the served model's structured features and scores on the validation split, data the
model was not trained on but that is drawn like its training data. Computing it scores ~2,000
emails, so it is cached per model version. Current: the last `--window` recorded emails scored
by that same version (a new version starts a new comparison).

Outputs, in `drift_dir`: `summary.json`, numbers only, which the API exposes as metrics, and
`report.html`, a table for a person. Neither holds message content.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from mailguard.config import Settings, get_settings
from mailguard.data.dataset import load_dataset
from mailguard.models.registry import aliased_version
from mailguard.monitoring.drift import PSI_DRIFT, SCORE, DriftReport, compare
from mailguard.monitoring.recorder import SUMMARY_FILE, read_records
from mailguard.parsing import parse_email
from mailguard.serving.model import ServedModel, load_production

logger = logging.getLogger(__name__)

WINDOW: Final = 2000
MIN_ROWS: Final = 200  # below this, PSI on 10 bins is mostly noise
REPORT_FILE: Final = "report.html"

Columns = dict[str, list[float]]


def reference(model: ServedModel, settings: Settings, cache_dir: Path) -> Columns:
    """Structured features and score of the validation split, cached per model version."""
    cache = cache_dir / f"reference-{model.name}-v{model.version}.json"
    if cache.is_file():
        cached: Columns = json.loads(cache.read_text(encoding="utf-8"))
        return cached
    frame = load_dataset(settings.data_dir / "processed")
    raws = frame.loc[frame["split"] == "val", "raw"].tolist()
    logger.info("scoring %d validation emails for the reference", len(raws))
    columns: Columns = {}
    for raw in raws:
        prediction = model.predict(parse_email(raw), settings.base_rate)
        for name, value in {**prediction.features, SCORE: prediction.score}.items():
            columns.setdefault(name, []).append(value)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(columns), encoding="utf-8", newline="\n")
    return columns


def current(records: Sequence[dict[str, Any]], version: str, window: int) -> Columns:
    """Column-wise values of the last `window` records scored by `version`."""
    same = [r for r in records if r.get("model_version") == version][-window:]
    columns: Columns = {}
    for record in same:
        for name, value in {**record["features"], SCORE: record["score"]}.items():
            columns.setdefault(name, []).append(float(value))
    return columns


def render_html(report: DriftReport, model: ServedModel, generated: str) -> str:
    verdict = "Drift detected" if report.drifted else "No drift"
    rows = "\n".join(
        f"<tr class={'drift' if f.drifted else 'ok'}><td>{html.escape(f.feature)}</td>"
        f"<td>{f.psi:.3f}</td><td>{f.ks_statistic:.3f}</td><td>{f.reference_mean:.3g}</td>"
        f"<td>{f.current_mean:.3g}</td><td>{'yes' if f.drifted else ''}</td></tr>"
        for f in sorted(report.features, key=lambda f: -f.psi)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>mailguard drift</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --warn:#fdecea; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#111; --fg:#f2f2f7; --muted:#a1a1a6; --line:#2c2c2e; --warn:#3a1d1b; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif;
  margin:0 auto; max-width:900px; padding:24px 16px; }}
p {{ color:var(--muted); }} table {{ border-collapse:collapse; width:100%; }}
th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }} tr.drift {{ background:var(--warn); }}
</style></head><body>
<h1>{verdict}</h1>
<p>{html.escape(model.name)} v{html.escape(model.version)}, generated {generated}.
{report.current_rows} recent emails against {report.reference_rows} validation emails.
{100 * report.share_drifted:.0f}% of input features drift (PSI above {PSI_DRIFT}); the score
{"drifts" if report.score_drifted else "does not drift"}.</p>
<table><thead><tr><th>Feature</th><th>PSI</th><th>KS</th><th>Mean, validation</th>
<th>Mean, recent</th><th>Drift</th></tr></thead><tbody>
{rows}
</tbody></table></body></html>
"""


def run_once(settings: Settings, model: ServedModel, window: int) -> dict[str, Any] | None:
    if settings.feature_log_dir is None or settings.drift_dir is None:
        raise ValueError("set MAILGUARD_FEATURE_LOG_DIR and MAILGUARD_DRIFT_DIR")
    recent = current(read_records(settings.feature_log_dir), model.version, window)
    rows = len(recent.get(SCORE, []))
    if rows < MIN_ROWS:
        logger.info("%d recorded emails for v%s, waiting for %d", rows, model.version, MIN_ROWS)
        return None
    report = compare(reference(model, settings, settings.drift_dir), recent)
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    summary = {"generated": generated, "model_version": model.version, **report.summary()}
    settings.drift_dir.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name and renamed, so the API never reads half a file.
    tmp = settings.drift_dir / (SUMMARY_FILE + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    tmp.replace(settings.drift_dir / SUMMARY_FILE)
    (settings.drift_dir / REPORT_FILE).write_text(
        render_html(report, model, generated), encoding="utf-8", newline="\n"
    )
    logger.info(
        "drift: %s, %.0f%% of features, score %s",
        report.drifted,
        100 * report.share_drifted,
        "drifts" if report.score_drifted else "stable",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drift_report")
    parser.add_argument("--window", type=int, default=WINDOW, help="most recent emails compared")
    parser.add_argument("--every", type=float, default=0, help="repeat every N seconds (0: once)")
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)-7s %(name)s %(message)s")
    try:
        model = load_production(settings)
        while True:
            run_once(settings, model, args.window)
            if not args.every:
                return 0
            time.sleep(args.every)
            # A promotion since the last run: compare the new version from now on.
            if aliased_version(model.name, settings, settings.model_alias).version != model.version:
                model = load_production(settings)
    except (LookupError, ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
