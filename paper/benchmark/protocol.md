# Benchmark protocol (frozen 2026-10-11)

This protocol governs the benchmark reported in the preprint
(`paper/preprint/main.tex`). It was first committed before any B test row was
scored. On 2026-10-11 the maintainer narrowed the benchmark to default
decisions on all of A and all of B; the operating points, split and
calibration of the first version were removed (see `deviations.md`). This
file describes what is now done. pygarble defaults, thresholds and data tables are frozen at main
`e3135ed` (0.11.0); nothing under `pygarble/` changes on this branch.

## Datasets

**A. Survey junk, positives only.** Kaggle
`johnwdata/gibberish-text-classification`, file `Gibberish.xls`, licence
"Unknown". Despite the extension the file is a Windows-1252 CSV with header
`Response,Label` and 3,767 rows, all label 1. Each response is stripped;
empty responses are dropped; exact duplicates are dropped keeping the first
occurrence. 3,513 rows remain. Row id is `a:<0-based data row>`. Endpoint:
capture rate. A has no negatives and no thresholds are fitted on it.

**B. Random strings against tweets, English only.** Hugging Face
`Sowmya15/gibberish_multilingual`, Apache-2.0. The train, validation and
test files are pooled; rows whose `language` is "english" (case-insensitive)
and whose text is non-empty after stripping are kept, with the text as
given. 4,000 rows have label 1 (random string) and 5,971 label 0 (tweet).
Labels are used as given. Row id is `<split>:<0-based data row>`.
Duplicates are not removed (215 rows repeat an earlier text). 26.4 percent of tweets contain a
mojibake signature (`U+00F0 0x9F`, `U+00E2 0x80` or `U+00C3`); this is
reported as a note and nothing is re-labelled.

Sources, licences, sha256 hashes, counts and every row id are in
`manifest.json`. No text from either dataset is committed.

## Decision and metrics

Every method is scored at its own default decision: the binary output it
gives with its default settings. No threshold is fitted on either dataset.
Each method also yields a continuous score (higher means more junk), which
is saved with the predictions but not used in any reported metric.

- A: capture rate (flagged / rows).
- B: precision, recall and false-positive rate (FPR), over all 9,971
  English rows. Precision is reported as null when nothing is flagged.
- 95 percent intervals: 1,000 row bootstrap resamples with replacement,
  percentile method. A and B are resampled independently with one
  `numpy.random.default_rng(20261011)`, A first; the same resamples are
  used for all methods. Resamples where a metric is undefined are skipped
  for that metric.

No other breakdowns (by length, script or otherwise) are reported.

## Methods

Five methods are compared (reduced from the first version on 2026-10-11;
see `deviations.md`).

pygarble 0.11.0 from this repository, every detector built with
`threads=1`; score from `predict_proba`, decision from `predict`:

- `english`: `EnsembleDetector(profile="english")`, the package default
  (Markov chain, log-likelihood ratio, word anomaly, mojibake, keyboard
  adjacency, control characters; `any` voting; threshold 0.5).
- `english_fusion`: `EnsembleDetector(profile="english_fusion")`.
- Tuned weights (post hoc in its use of labels, evaluated on held-out
  rows): `tune.py` learns a weight per strategy and a threshold under
  `weighted` voting from a stratified 20 percent of A and of B, for five
  seeds. The sample is drawn by unit: rows whose stripped text is
  identical (the manifest's `duplicates` groups; 86 groups covering 215
  repeated rows in B, none in A) form one unit, so every copy of a text
  falls on the same side of the split. Per stratum (A, B tweets, B random
  strings) the units are shuffled and the prefix whose row count is
  closest to 20 percent of the stratum is taken. It evaluates on the other 80 percent next to `english` and
  `english_fusion` on the same rows. Results are in `results/tuning/`; see
  `deviations.md`. The package is unchanged.

Baselines (CPU, pinned in `paper/scripts/benchmark/requirements.txt`,
installed in a Python 3.12 venv outside the repository):

- **gibberish-detector 0.1.1** (Aaron Loo's port of rrenaud's Markov
  detector). The wheel ships no model, so a model is trained with the
  package's own `trainer.train` (default ASCII-letter charset) on the
  repository's `examples/big.txt` at commit `ecac969` (sha256
  `e77a1d4a...71d39d7`); training is count-based and needs no seed. Score:
  `calculate_probability_of_being_gibberish`; decision: score above the CLI
  default limit 4.0.
- **lingua-language-detector 2.2.0**, policy "not English is junk".
  Detector built from all languages with default settings. Decision: the
  detected language is not English (no detection counts as not English).
  Score: one minus the English confidence.

## Footprint

For every method, on all B texts: the default decision call over all
texts (pygarble `predict` on the list, baselines in a loop), one warm-up
run and five timed runs in one process on one thread; the median is
reported as milliseconds per 1,000 texts. Import time: median over five
fresh interpreters of the time to import the method's modules. Install
size: bytes on disk of the installed files of the distribution and its
non-stdlib runtime dependencies, excluding `__pycache__`; for pygarble the
files a wheel installs. Offline: whether scoring needs network access
after installation.

## Pilot disclosure

On 2026-10-11, before this protocol, all of A and all English B rows were
scored once with pygarble defaults at threshold 0.5: `english_fusion`
capture 0.895 on A; precision 0.906, recall 0.991 and FPR 0.068 on B.
Nothing in the package changed afterwards.

## Scoring

`run.py` scores every method once on all of A and all of B. Per-row
scores and decisions (row id, score, decision; no text) are saved under
`results/predictions/A/` and `results/predictions/B/`; `verify.py`
recomputes `results/metrics.json` from them and the manifest. Scores are
deterministic: for the four methods kept from the first run, the A
predictions are byte for byte those of that run, and the B predictions
equal the union of its B dev and B test predictions.
Losses are reported as they come.
