# Validation record

Measured on an Apple M2, 8 GiB RAM, macOS ARM64, Python 3.12.2.
The full-corpus scoring protocol and core were frozen in `717ceee` before
new predictions. Exact source hashes are in `full-results/code-manifest.json`.
The earlier capped experiment remains frozen at `22f30d8` with its original
code and results. No package configuration or scoring threshold was changed.

## Complete corpus: executed checks

- All 109 labelled documents and 31,964,664 normalized characters were covered
  by 79,969 lossless, consecutive chunks. Source/model asset hashes matched.
  No inputs were truncated; the largest model input had 348 tokens.
- Every input identity, offset, length and hash was checked against the
  reconstructed source. All 399,845 saved pygarble decisions and 159,938 HF
  policy decisions were checked against their saved scores/probabilities.
  Document summaries and aggregate statistics were recomputed.
- A new offline process replayed 325 fixed first/middle/last inputs across
  all documents. Every policy decision matched. The maximum HF probability
  difference was 1.103e-6, below the predeclared 1e-5 tolerance. All 1,625
  replayed pygarble public `predict` decisions and score/status tuples matched.
  This is a subset replay, not a second complete neural inference run.
- No exact duplicate chunks or conflicting exact-hash inherited labels were
  found. All 438 preliminary primary input hashes occur in the full corpus;
  the extension is not an independent replication.
- Twenty-one standard-library study tests passed in both the working
  interpreter and a clean virtual environment without research dependencies.
  Tests include tied-score calibration, grouping, source boundaries, tails,
  majority ties, neural label mapping and equal document weighting.
- All 18 study Python files passed Black, isort and flake8 checks. Downloaded
  assets and ignored virtual environments are excluded from these checks.
- CPU timing and RSS ran in fresh, separate workers after exhaustive scoring
  ended. Both systems used the same 76 inputs, one thread and batch size one;
  one warm pass preceded five timed passes. Timings are observations from
  this machine, not deployment guarantees or energy measurements.
- The generated ten-page empirical PDF compiled without TeX warnings.
  Its tables/figures were visually inspected, and every extracted text block
  fits inside its page. An isolated extraction of `review-source.tar.gz`
  compiled without warnings and produced identical extracted page text.
  Numerical prose guards and generated tables check the manuscript against
  saved measurements. `full-results/paper-validation.json` records the build.
- The JOSS software draft's citations parsed in a generic Pandoc HTML preview.
  Official JOSS Inara/Docker rendering has not run; Docker is unavailable.

Machine-readable checks are in `full-results/validation.json`. The complete
scoring pass took about 48.7 minutes on this CPU, excluding preparation and
subsequent validation/timing. No cross-machine or independent human
reproduction of the transformer experiment is claimed.

## Retained preliminary reproduction

The original standard-library experiment reproduced seven deterministic
artifacts byte for byte in a fresh virtual environment: predictions,
calibration settings, summaries, records, document manifests, overlap results
and its report. `verify_results.py` verified 130 original source hashes,
11,470 prediction rows through summary recomputation and 6,560 public API
decisions. The original `validation.json` retains those counts.

## Commands

Run from the repository root; prepare the source/model assets as described
in [README.md](README.md) before full artifact verification:

```bash
python -m unittest paper.study.test_study paper.study.test_full_corpus -v
python -m paper.study.full_verify --replay
python -m paper.study.full_diagnostics
python -m paper.study.full_runtime
python -m paper.study.full_figures
python -m paper.study.build_paper
python -m black --check paper/study/*.py
python -m isort --check-only paper/study/*.py
python -m flake8 paper/study/*.py
```

The isolated PDF check used PyMuPDF to compare extracted text and page bounds;
PDF compilation uses Pandoc and Tectonic. The model dependencies and measured
versions are recorded separately from the dependency-free package. Existing
PR CI covers library tests, optional backends, quality, docs and packaging;
it does not discover these study-specific tests, which were run explicitly.
Check the PR's latest commit before any merge.

## Scope and spending

Source labels were inherited without a new human annotation/audit. Automated
verification checks computations, not the scientific adequacy of those labels.
Author review of methods, claims and AI-assisted text remains outstanding.
No paid data, inference endpoint, annotation service, cloud compute or
publication service was procured. Additional external-service spend: US$0,
excluding existing subscriptions and hardware. No paper has been submitted,
package release published, merge performed or indexing outcome asserted.
