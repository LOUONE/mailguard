# 0005. Ship LightGBM, keep the text-only model as a challenger, no transformer yet

- Status: Accepted
- Date: 2026-10-28
- Issue: #30

## Context

Phase 1 trained two baselines on the same split, calibrated both on validation and evaluated
them once on test ([`reports/evaluation.md`](../../reports/evaluation.md); every number below
comes from it, with 95% cluster-bootstrap intervals):

| | Logistic regression, text only | LightGBM, all feature groups |
|---|---|---|
| Recall at 1% ham FPR | 97.9% [96.7, 98.9] | 98.8% [97.8, 99.5] |
| Recall at the 0.1% threshold | 94.5% [92.8, 96.0] | 96.4% [95.0, 97.6] |
| PR-AUC | 0.9978 [0.9960, 0.9991] | 0.9993 [0.9985, 0.9998] |
| ECE after calibration | 1.0% [0.6, 1.9] | 0.7% [0.5, 1.6] |
| 2020-2025 phishing, trained up to 2019 | **96.5% [95.6, 97.4]** | 95.3% [94.1, 96.3] |

Paired differences, LightGBM minus logistic regression: +0.8 points of recall at 1% FPR
[0.2, 1.7] (McNemar: 9 messages only the regression gets right, 20 only LightGBM, p = 0.061);
+1.8 points at the 0.1% threshold [0.4, 3.3] (p = 0.00064); PR-AUC +0.0015 [0.0006, 0.0027].

Phase 2 needs one model behind the API, a probability the agent can route on, and reasons it can
return.

## Options

### A. Logistic regression on text

Simple, fast, and the most robust to new campaigns in the temporal check: it loses 1.2 points on
2020-2025 phishing against 3.5 for LightGBM. But it is blind to everything outside the body text
(sender consistency, link structure, attachments) and can only explain itself with words.

### B. LightGBM on all feature groups

Better at every operating point on the main split, better calibrated, and its explanations name
structural reasons (`url_anchor_mismatches`, `hdr_display_name_brand_mismatch`) that the API and
the Phase 3 agent can use. Its weakness is the one the leakage audit predicted: its URL features
partly encode the era, so it loses more on campaigns from years it never saw, and its strongest
feature (`url_max_length`) is a probable era confound.

### C. A fine-tuned transformer (DistilBERT and similar)

Likely the best text model on paper. But with 14,084 messages, 2002-only ham and 2004-2025
phishing, it would learn writing style by era even more readily than TF-IDF, and nothing in the
data could show it. It also needs a GPU or a much larger CPU budget per message, for a service
that should answer in milliseconds and send the hard cases to an LLM anyway.

## Decision

**Option B, with A as a challenger.** LightGBM (`mailguard-lightgbm`, calibrated) is the model the
API serves. The logistic regression stays registered and evaluated at every retraining; the Phase 2
promotion gate (#39) compares both on the same split and on the temporal check, so the text-only
model takes over if it ever wins.

No transformer in Phase 1 or 2. The grey zone, where a bigger model would help, goes to the LLM
agent in Phase 3, which reads the message rather than fitting the corpus.

## Consequences

- The difference at 1% FPR is small and McNemar alone does not call it significant (p = 0.061);
  the cluster-bootstrap interval of the difference excludes zero. The choice rests on that interval,
  the 0.1% point, calibration and explanations together, not on one test.
- The temporal weakness is accepted, monitored and mitigated: drift reports (#38), retraining on
  recent phishing (#39) and a test with a held-out campaign (#40).
- `url_max_length` stays in the model and is the first limitation in the model card. A version
  without it is the first experiment once a licensed corpus of modern ham exists.
- Revisit if modern ham becomes available, if the challenger wins a promotion gate, or if
  Phase 4 shows the structural features are easier to evade than text.
