# The mailing-list artefact (#73)

What the model learnt from a ham corpus that is mostly mailing lists, what was tried against it,
and what it says about the next step. Numbers: [`reports/experiments/mailing_lists.md`](../reports/experiments/mailing_lists.md),
from `scripts/experiment_mailing_lists.py` (five split seeds, three models, two threshold rules).

## The finding

- **75% of training ham carries mailing-list headers** (`List-Id`, `List-Unsubscribe`,
  `Precedence: list`...), against 18% of spam and 3% of phishing. The model never reads those
  headers (five allowlisted headers only), but it reads what comes with them: a `Return-Path`
  that differs from `From` (the list's bounce address), the word `list`, quoted replies, footers.
- **A message with no signal starts on the phishing side.** Averaged over 300 made-up personal
  emails, the TreeSHAP base value is +1.66 log-odds, and the strongest pushes towards phishing are
  absences: no word `list` (+0.40), a consistent return path (+0.37).
- **The 1% threshold is set by list mail.** Three quarters of validation ham is list mail that
  scores near zero, so the 1% FPR threshold lands low (0.29 for the served model), and a short
  personal email without strong legitimate signals crosses it.

Over five seeds, the served configuration flags 0.2% of list ham, 3.5% of other ham, and 41% of
the made-up personal emails (0% to 67% depending on the seed).

## What was tried

| Variant (mean over 5 seeds, paired with `served`) | Recall | Other-ham FPR | Personal emails flagged | Verdict |
|---|---|---|---|---|
| Drop `hdr_return_path_mismatch` | +0.27 pts (2 seeds up, 2 down) | -0.25 pts (2 down, 1 up) | -0.4 pts | noise |
| Weight non-list ham up to match list ham | +0.73 pts (4 up, 1 down) | +0.12 pts (1 / 1) | +2.3 pts | noise on FPR |
| Set the 1% threshold on non-list validation ham | -0.46 pts (3 down) | -0.92 pts (3 down) | -13.7 pts (3 down) | a real trade-off |

- **Neither the feature nor the weighting is the cause on its own.** Removing the feature moves
  nothing consistently; reweighting does not teach the model what short personal mail looks like,
  because the corpus barely contains any.
- **The threshold rule does move the needle**, in the direction expected, on every seed where the
  two rules differ: fewer personal emails and less non-list ham flagged, at the cost of about
  half a point of recall. It is a choice of which ham the 1% target protects, not a free gain.

## What it changed

- **The model stays as it is.** No variant improved the problem without costing recall, and the
  data to fix it (modern, personal legitimate mail) is not in a licensed public corpus.
- **The evaluation report gains two slices**: ham from mailing lists and other ham, so the gap is
  measured at every evaluation instead of found by accident.
- **A second finding, about thresholds.** On one seed the 1% threshold jumped to 0.995 and recall
  fell to 88.6%: nine copies of one legitimate email landed in validation and scored high, and
  the threshold, the seventh highest ham score, fell inside them. Splits and intervals treat
  duplicates by cluster; the threshold did not. Tracked in #85.
- **For production**, short personal mail is the most likely false alarm. The Phase 3 agent, which
  reads the message instead of fitting the corpus, is where grey-zone scores should go.
