# Drift simulation: a model meets campaigns it has never seen

The corpora contain the drift a production system faces: phishing from years a model has not
seen. `make simulate-drift` (`scripts/simulate_drift.py`) holds out every 2024-2025 phishing
message and replays the story end to end, through the real API code, drift job and retraining
flow, in its own MLflow. Results, numbers only: [`reports/drift_simulation.json`](../reports/drift_simulation.json).

## Setup

- **Held out**: 884 phishing messages from 2024-2025, plus 11 more in the same duplicate
  clusters, 895 in all.
- **The old model**: LightGBM, same configuration as production, trained without them and put in
  production in the simulation's registry.
- **Normal traffic**: the 1,988 test messages of the old world (ham, spam, older phishing).
- **The new campaigns**: the 895 held-out messages shuffled with the same 907 ham and spam,
  1,802 in all.
- **Drift reference**: the old model's features and scores on its validation split.

## Results

| | Normal traffic | New campaigns arrive |
|---|---|---|
| Drift detected | **no** | **yes** |
| Score distribution drifts | no | yes |
| Share of input features above PSI 0.25 | 0% (largest PSI 0.07) | 13.5% (5 of 37) |

The features that moved, validation mean then 2024-2025 traffic mean:

| Feature | PSI | Validation | New campaigns |
|---|---|---|---|
| `url_keyword_hits` (login words in URLs) | 0.34 | 0.45 | 0.12 |
| `url_ip_hosts` (links to a bare IP) | 0.31 | 0.23 | 0.05 |
| `body_credential_cues` | 0.31 | 1.12 | 0.51 |
| `url_anchor_mismatches` | 0.27 | 0.42 | 0.09 |
| `hdr_message_id_domain_mismatch` | 0.26 | 0.48 | 0.24 |

**The drift has a meaning: phishing changed.** Recent campaigns use far fewer IP-hosted links,
login words in URLs and credential vocabulary, the textbook signs older phishing carried. The old
model still caught 98.1% of them, on the strength of the text model, but the signals it relies on
are fading, which is exactly what a drift monitor is for: a warning before recall drops.

## Retraining

The retraining flow trained a candidate on the full corpus (so on most of the 2024-2025
campaigns) and compared it with the old model on the 2,113 test messages, none of which the old
model had seen:

| Check (ADR 0008) | Result |
|---|---|
| Recall gain, cluster-bootstrap interval above zero | **pass**: +0.33 points [+0.08, +0.69] |
| McNemar, more messages only the candidate gets right, p < 0.05 | **fail**: 8 against 3, p = 0.23 |
| Ham FPR difference within +0.5 points | **fail**: upper bound +0.79 |

**Production was kept.** That is the rule doing its job, and it shows its limits:

- The gain is real but concentrated: 125 of the held-out messages are in the test split, among
  1,206 phishing, so a better catch on the new campaigns barely moves overall recall.
- The bootstrap and McNemar disagree. The bootstrap sees a consistent small gain; McNemar sees 11
  discordant messages, too few to call. The rule requires both, by design.
- The FPR guard fails on uncertainty more than on harm: one or two more flagged ham out of 622
  give an upper bound above half a point.

What would change the outcome is the evaluation set, not the rule: weighting recent traffic, or
a dedicated check on the drifted slice. Both are open questions for the next iteration, recorded
here rather than tuned until the candidate passes.

## Reproducing

```bash
make dataset            # if not done
make simulate-drift     # about 15 minutes; writes reports/drift_simulation.json
```

Everything runs under `data/simulation/` (gitignored) and leaves the real registry untouched.
