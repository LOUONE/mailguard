# Architecture decision records

An ADR records one structural decision: the context that forced it, the options that were on the
table, what was chosen and what it costs. They are short on purpose. The code says *what* the
system does; ADRs say *why it does not do something else*.

## Index

| # | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-binary-target.md) | Binary target: phishing against ham and spam | Accepted | 2026-10-27 |
| [0002](0002-parser-design.md) | Parser design: lenient, never raises, metadata-only attachments | Accepted (retroactive) | 2026-10-27 |
| [0003](0003-model-serialisation.md) | Save models with skops and an explicit list of trusted types | Accepted | 2026-10-27 |
| [0004](0004-grouped-splits.md) | Split by duplicate cluster before any feature, with a separate temporal check | Accepted | 2026-10-27 |
| [0005](0005-model-choice.md) | Ship LightGBM, keep the text-only model as a challenger, no transformer yet | Accepted, amended by 0006 | 2026-10-28 |
| [0006](0006-drop-url-max-length.md) | Drop `url_max_length` from the served model | Accepted, updated | 2026-10-29 |
| [0007](0007-drift-without-evidently.md) | Detect drift with PSI and KS in our own code, not Evidently | Accepted | 2026-10-29 |
| [0008](0008-significance-gated-promotion.md) | Promote a retrained model only on a significant gain, from a plain-Python flow | Accepted | 2026-10-29 |

## Writing one

1. Copy [`template.md`](template.md) to `NNNN-short-title.md`, with the next free number.
2. Keep it under a page. Write the options as they looked *before* the decision, including the
   ones that lost, with their real advantages.
3. Put numbers in the context whenever they exist (EDA, benchmarks). A decision that cannot point
   to evidence says so.
4. Open it in the same PR as the code it governs, or before that PR if the code depends on it.
5. ADRs are not edited once accepted. A new decision gets a new ADR that marks the old one
   `Superseded by NNNN`. When the evidence changes but the decision does not, a dated `Update`
   section is appended, and the original text stays as it was written.

Statuses: `Proposed`, `Accepted`, `Superseded by NNNN`, `Rejected`. A retroactive ADR documents a
decision already in the code, so that it can be reviewed like the others.
