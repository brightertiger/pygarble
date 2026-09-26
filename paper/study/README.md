# Published-corpus study

Everything needed for this evaluation lives here. The library itself is
unchanged. This work evaluates inherited published labels without a new human
audit, as requested by the author. It is not submitted or peer reviewed.

## Reproduce

From the repository root, using Python 3.8 or later (measured on 3.12):

```bash
python -m unittest paper.study.test_study -v
python -m paper.study.run
```

The runner uses only Python's standard library and the local pygarble source.
The first run downloads about 14 MB from the pinned GitHub revision and checks
SHA-256 hashes. No account, model weights, paid inference or API key is needed.
Subsequent runs use the verified cache. Run from this repository's root, not
from this subdirectory. Allow a few minutes on an ordinary CPU.

```bash
python -m paper.study.run --output paper/study/reproduction --skip-timing
```

Compare deterministic `predictions.csv.gz`, `summary.json`, `calibration.json`,
`documents.json`, `records.json` and `overlap.json` with `results/`. Runtime and
environment metadata intentionally vary by run. Inspect the code manifest to
identify the exact scripts and library resources used.

## Review draft and figures

Start with [the 7-page PDF](manuscript.pdf) and [submission handoff](submission.md).
To rebuild tables from the saved results and compile the paper:

```bash
# Optional plotting dependencies, preferably in a separate environment:
python -m pip install -r paper/study/requirements-figures.txt
python -m paper.study.figures
# Requires Pandoc and Tectonic:
python -m paper.study.build_paper
```

Editing `manuscript.template.md` preserves generated numerical tables.
The compiled paper is a review draft with outstanding author declarations.
The short JOSS software draft is `../paper.md`.

## Files

- `protocol.md`: frozen methods, data rules, calibration and limitations.
- `sources.json`, `LICENSES.md`: pinned source provenance and separate rights.
- `data.py`: verified retrieval, selection, record IDs and overlap checks.
- `detectors.py`: package adapters and independently trained character models.
- `metrics.py`: source-level summaries and descriptive uncertainty.
- `run.py`, `runtime.py`: evaluation and isolated CPU/RSS measurement.
- `test_study.py`: meaningful checks of grouping, thresholds and statistics.
- `verify_results.py`: saved-output, frozen-code and public-API verification.
- `figures.py`, `build_paper.py`: plots and manuscript generation.
- `validation.md`, `validation.json`: exact executed checks and their limits.
- `results/`: full metrics and predictions; start with `results/report.md`.
- `deviations.md`: corrections or departures from the frozen protocol.

Raw source text stays in ignored `.cache/`. Results contain hashes and offsets
into whitespace-normalized text, not redistributed comparison prose. The two
Bible translations share a family. Every gibberish prefix retains its source
ID; repeated length views and calibration folds are not independent samples.

## Scope

The primary evaluation includes 38 released gibberish documents and up to
100 400-character blocks from each of four English comparison documents.
All meaningful/gibberish classes come from the original collection. We do
not certify labels or treat historical spelling as modern clean English.
The sample is small, curated and confounded by source, so headline precision
or a general deployment accuracy claim would be misleading.

The calibration experiment uses separate source families for model fitting,
threshold selection and false-positive evaluation. A 1% rate on validation
blocks is not a guaranteed 1% rate elsewhere. Character models are adaptations
of the documented approach, not a claimed reproduction of upstream scores.
A local evaluation is research evidence for author review; it does not by
itself establish that JOSS's scope or substantial-contribution criteria are met.
