# Tests

Pytest suite for the Wearable Flag Pipeline. The layout mirrors the package
structure so each workstream's tests sit alongside the package they cover:

```
tests/
  conftest.py            # puts the repo root on sys.path
  test_scaffolding.py    # cross-package smoke test (Task 1.1)
  data/                  # data/ package: physionet_load, clean, windows, schema
  flag/                  # flag/ package: baseline, score, tiers, context, explain
  eval/                  # eval/ package: labels, metrics, calibrate, run_eval
  app/                   # app/ package: demo
  export/                # export/ package: fhir, care_update (stretch)
```

## Running

```bash
pip install -e .[test]   # installs runtime deps + pytest
pytest                   # discovers tests/ per pyproject.toml
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/) at a
minimum of 100 iterations each and are tagged
`# Feature: wearable-flag-pipeline, Property N: ...`.
