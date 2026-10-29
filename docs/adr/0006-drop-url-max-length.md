# 0006. Drop `url_max_length` from the served model

- Status: Accepted
- Date: 2026-10-29
- Issue: #71
- Amends: [0005](0005-model-choice.md) (its consequence "`url_max_length` stays in the model")

## Context

The leakage check of Phase 1 found that the strongest LightGBM feature, `url_max_length`, is
probably an era marker: the ham corpus is from 2002, when URLs were short (median 48 characters
on the training split), while phishing of every era has long ones (about 70). Modern legitimate
mail is full of long tracking URLs, and the ham false positives of v0.1.0 were exactly that
([error analysis](../error_analysis.md#last-leakage-check)).

ADR 0005 kept the feature until modern ham could test it. The ablation added to the evaluation
report in #69 tested what can be tested without modern ham: the same configuration retrained
without the feature, thresholds set on validation, paired on the same bootstrap replicates
([`reports/evaluation.md`](../../reports/evaluation.md)):

| | With `url_max_length` (v0.1.0) | Without |
|---|---|---|
| Recall at 1% ham FPR, test | 98.8% [97.8, 99.5] | 98.8% [97.9, 99.6] |
| Ham flagged on test | 5 of 622 | 6 of 622 |
| Spam flagged on test | 3.9% [1.5, 6.6] | 6.0% [3.3, 9.2] |
| PR-AUC | 0.9993 | 0.9992 |
| 2020-2025 phishing, trained up to 2019 | 95.3% [94.1, 96.3] | **97.6% [96.8, 98.4]** |

Paired, without minus with: recall +0.1 points [0.0, 0.3], ham FPR +0.2 [0.0, 0.5], spam FPR
+2.1 [0.7, 3.8], PR-AUC -0.0001 [-0.0003, 0.0004].

## Options

### A. Keep it (status quo of ADR 0005)

Nothing to retrain, and it slightly reduces the spam the model flags. But the main split cannot
tell an era marker from a phishing signal, and the one measurement that can, the temporal check,
says the feature costs 2.3 points of recall on campaigns the model has never seen. Production is
made of exactly those campaigns.

### B. Drop it

Same recall and PR-AUC on test; better generalisation to new campaigns, which moves LightGBM
above the text-only challenger on the temporal check (96.5%). The cost is real but of the cheaper
kind (ADR 0001): about 2 more points of spam flagged as phishing, and one more test ham.

### C. Replace it with a robust version (length relative to the domain, or capped)

It would keep whatever real signal URL length carries. But there is no data to validate such a
transform against modern ham either, and it would be a new feature tuned on the same 2002 corpus.

## Decision

**Option B.** `configs/baselines.toml` sets `lightgbm.exclude_features = ["url_max_length"]`; the
feature code is unchanged, so the variant with it stays one config line away and the evaluation
report keeps measuring both (the ablation section now retrains the variant *with* the feature).

## Consequences

- The served LightGBM is retrained. Its calibrated operating thresholds are recorded on the
  model version, so the API never recomputes them.
- An effect the ablation of #69 did not show, because it only reported the 1% point: at the
  strict threshold (no validation ham flagged), recall rises from 96.4% to 98.3% [97.3, 99.2],
  for 5 test ham flagged instead of 4. Against the text-only challenger the gain there grows
  from 1.8 to 3.8 points [2.6, 5.3].
- Calibration is slightly worse: ECE 0.9% [0.5, 1.7] instead of 0.7% [0.5, 1.6], inside each
  other's intervals.
- Spam flagged as phishing rises from about 4% to about 6% on test. In production a spam filter
  usually sits in front of the model; if it does not, this is the first cost an operator sees.
- Part of the temporal gain comes from a looser threshold on that split (ham FPR 0.6% to 1.1%,
  three more messages): the direction of the effect is measured, its size is uncertain.
- The drift report (#38) still watches URL lengths: if modern ham shows a feature still encodes
  the era, the same test (ablation plus temporal check) applies.
- Revisit with modern ham, or if Phase 4 shows that URL length is a useful evasion signal.

## Update, 2026-10-29: the temporal evidence did not hold

Pinning the Python version (v0.1.2) changed the split of 850 spam messages
([data](../data.md#reproducibility)). On the reproducible split the same ablation gives:

| | With `url_max_length` | Without (served) |
|---|---|---|
| Recall at 1% ham FPR, test | 98.5% [97.5, 99.3] | 99.0% [98.1, 99.6] |
| Spam flagged | 3.5% | 5.3% |
| 2020-2025 phishing, trained up to 2019 | 96.7% [95.8, 97.6] | 96.1% [95.2, 97.0] |

Paired, without minus with: recall +0.5 points [0.2, 0.9], ham FPR +0.3 [0.0, 0.8], spam FPR
+1.8 [0.4, 3.4].

The temporal gain that was the main measured argument above (+2.3 points) is now a 0.6-point
loss, inside the intervals. Its threshold rests on the ham of one validation split, and moving
some spam between splits was enough to change it: a single split cannot measure an effect of this
size, which the cluster bootstrap (resampling the test set, not the split) does not show.

The decision stands, on narrower grounds: the feature encodes the era of the ham corpus by
construction, the main-split recall is significantly better without it, and modern ham is full of
the long tracking URLs it penalises. Measuring effects of a few points will need the comparison
repeated over several split seeds; that is now part of the method for the next ablation (#73).
