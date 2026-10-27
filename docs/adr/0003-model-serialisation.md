# 0003. Save models with skops and an explicit list of trusted types

- Status: Accepted
- Date: 2026-10-27
- Issue: #26

## Context

A trained pipeline is written to the MLflow registry by `make train` and, from Phase 2, loaded by
the API at startup. Whatever format is used, loading a model is reading a file that someone could
have replaced.

With `pickle` (and `joblib`, `cloudpickle`), loading a file **runs code chosen by whoever wrote
it**. Anyone who can write to the registry, or to the volume the API reads it from, gets code
execution inside the service that sees every email. MLflow 3 changed its default for scikit-learn
models to [skops](https://skops.readthedocs.io), which rebuilds objects only from types the caller
explicitly trusts and never runs code from the file.

The first `make train` with skops took 7 minutes and failed: skops refused our own transformers
(untrusted types), and it stores a Python dict entry by entry, so the 150k-term TF-IDF vocabulary
took 149 s and 92 MB per model, against 0.13 s and 3.6 MB with pickle.

## Options

### A. Pickle (cloudpickle)

Fast and small. Safe only as long as nobody but us can ever write to the registry, which is an
assumption about infrastructure, not a property of the code.

### B. ONNX

Portable and code-free, but our pipeline is custom Python from `list[Email]` (parsing-dependent
extractors, scrubbing), which ONNX cannot express without rewriting the feature code.

### C. skops with an explicit trust list

Loading cannot execute code. The cost is a list of trusted types to maintain and the dict
serialisation problem.

## Decision

**Option C.** `mlflow.sklearn.log_model(serialization_format="skops",
skops_trusted_types=TRUSTED_TYPES)`, with `TRUSTED_TYPES` in
[`models/pipelines.py`](../../src/mailguard/models/pipelines.py): our five extractors, LightGBM's
`LGBMClassifier` and `Booster`, and `OrderedDict`.

`TextFeatures` saves its fitted state as numpy arrays (terms in column order, IDF weights), which
skops stores natively, and rebuilds its vectorisers on load: **0.1 s and 13 MB**, identical output.

## Consequences

- A tampered model file can at worst fail to load or predict badly; it cannot run code.
- Adding a new custom class to a pipeline means adding it to `TRUSTED_TYPES`. A test saves both
  baselines and fails if skops reports a type outside the list, so this cannot be forgotten.
- `TextFeatures.__getstate__` must copy the state before changing it: `object.__getstate__`
  returns the instance's own `__dict__`, and an early version unfitted the model it was saving. A
  test checks the original still predicts after a save.
- Models stay larger than with pickle (13 MB against 3.6 MB per TF-IDF), which is acceptable.
- Revisit if skops support lags behind a scikit-learn or LightGBM upgrade.
