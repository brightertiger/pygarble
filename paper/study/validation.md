# Validation record

Evaluation performed on an Apple M2, 8 GiB RAM, macOS ARM64, Python 3.12.2.
The frozen protocol/harness commit is `22f30d8`; exact evaluation source
hashes are in `results/code-manifest.json`. No detector implementation or
configuration was changed after observing outcomes.

## Executed checks

- Thirteen standard-library unit tests passed in both the working interpreter
  and a fresh virtual environment. Tests cover smoothing, document boundaries,
  tied-score calibration, empty/invalid inputs, applicability, disjoint family
  roles, nonoverlapping blocks, short documents and uncertainty calculations.
- A complete rerun in a fresh virtual environment with no installed research
  dependencies produced byte-identical predictions, calibration settings,
  summaries, record/document manifests, overlap results and the report.
- `verify_results.py` recomputes summaries from saved predictions, verifies
  every file in the original code manifest, compares public `predict` decisions
  with the harness and checks the reproduction artifacts. Machine-readable
  counts are recorded in `validation.json`.
- Black, isort and flake8 checks apply to this folder's Python source files.
  Ignored cache files, downloaded upstream code and virtual environments are
  excluded; they are not modified to satisfy this project's formatting.
- The empirical manuscript is rendered with Pandoc/citeproc and compiled with
  Tectonic. The 7-page PDF has been inspected for readable text, tables and
  figures. TeX reports a 0.124-point table alignment overflow; no content is
  visibly clipped. The author must inspect the final post-review build too.
- The review source archive compiled in an isolated directory with identical
  extracted page text; all text blocks fit within the page boundaries.
- The JOSS draft's citations parse in a generic Pandoc HTML preview. Official
  JOSS Inara/Docker rendering has not run; Docker is unavailable locally.

The repository's existing PR CI covers library tests, optional backends,
quality, docs and package building. It does not discover `paper/study` tests;
those are explicitly run locally as documented here. PR checks must be
checked on the latest pushed commit before any merge.

## Commands

Run from the repository root:

```bash
python -m unittest paper.study.test_study -v
python -m black --check paper/study/*.py
python -m isort --check-only paper/study/*.py
python -m flake8 paper/study/*.py
python -m venv paper/study/.cache/repro-venv
paper/study/.cache/repro-venv/bin/python -m paper.study.run \
  --output paper/study/reproduction --skip-timing
paper/study/.cache/repro-venv/bin/python -m paper.study.verify_results
```

Plotting uses optional Matplotlib/NumPy dependencies in
`requirements-figures.txt`; the evaluation itself does not need them:

```bash
python -m paper.study.figures
python -m paper.study.build_paper
```

The paper build additionally requires Pandoc and Tectonic. Timing varies by
machine and is deliberately excluded from byte-identical comparisons. A
fresh virtual environment on the same computer is an environment-isolation
check, not an independent human reproduction or cross-platform validation.

## Scope and spending

Source labels were reused without a new human annotation/audit. Automated
verification checks computations, not whether the inherited labels suit a
future deployment. No paid data, model inference, annotation service, cloud
compute or publication service was procured. Additional external-service
spend for this execution: US$0, excluding existing subscriptions/hardware.

Study results are for author review. No manuscript has been submitted, no
software release was published, and no acceptance or Scholar indexing claim
is made.
