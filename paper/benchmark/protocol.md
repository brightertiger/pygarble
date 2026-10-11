# Benchmark protocol (frozen 2026-10-11)

This protocol governs the preprint benchmark described in
`paper/PREPRINT-PLAN.md` (v4). It was committed before any B test row was
scored. pygarble defaults, thresholds and data tables are frozen at main
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
Duplicates are not removed (215 rows repeat an earlier text; 132 test rows
share their text with a dev row). 26.4 percent of tweets contain a
mojibake signature (`U+00F0 0x9F`, `U+00E2 0x80` or `U+00C3`); this is
reported as a note and nothing is re-labelled.

Sources, licences, sha256 hashes, counts and every row id are in
`manifest.json`. No text from either dataset is committed.

## Split

B is split 30 percent development, 70 percent test, stratified by label:
for label 0 then label 1, the ids in file order are shuffled with one
`random.Random(20261011)` and the first `round(0.3 n)` go to development.
B dev has 2,991 rows (1,200 positive); B test has 6,980 rows (2,800
positive). A is scored whole.

## Operating points

Every method yields a continuous score (higher means more junk) and a
native binary decision. Three operating points are evaluated:

- `native`: the method's own decision at its default settings.
- `fpr1` and `fpr0.1`: on the 1,791 B dev tweets, let `k = floor(target *
  n)` (17 and 1). The cut `c` is the (k+1)-th highest dev tweet score, and
  a text is flagged when its score is strictly greater than `c`. This is the
  most permissive cut whose dev FPR is at most the target.

The cuts are fitted on B dev only and applied unchanged to A and to B test.

## Metrics

- A: capture rate (flagged / rows).
- B test: precision, recall and FPR at each operating point; AUROC from the
  continuous score (Mann-Whitney, ties count one half); precision-recall
  curve points from the continuous score, thinned to points where recall
  moves by at least 0.005. Precision is reported as null when nothing is
  flagged.
- 95 percent intervals: 1,000 row bootstrap resamples with replacement,
  percentile method. A and B test are resampled independently with
  `numpy.random.default_rng(20261011)`; the same resamples are used for all
  methods. Cuts are not refitted inside the bootstrap. Resamples where a
  metric is undefined are skipped for that metric.

No other breakdowns (by length, script or otherwise) are reported.

## Checkpoint

After B dev cuts are fitted, `english_fusion` capture on A at `fpr1` is
computed. If it is below 0.70 the run stops before scoring B test and the
result is reported for discussion. Otherwise B test is scored once.

## Methods

pygarble 0.11.0 from this repository, every detector built with
`threads=1`; score from `predict_proba`, decision from `predict`.

- Profiles: `EnsembleDetector(profile=p)` for `english`, `english_extended`,
  `english_fusion`, `legacy`, `corruption`, `spoofing`, `llm_output`.
- Voting modes on the `english` profile: `majority`, `any`, `all`,
  `average`, `weighted` (equal weights of 1.0, as no weights are shipped),
  `fisher`. `any` is the profile default and duplicates `english`.
- Every strategy alone via `GarbleDetector(strategy)` (appendix).

Baselines (CPU, pinned in `paper/scripts/benchmark/requirements.txt`,
installed in a Python 3.12 venv outside the repository):

- **nostril**, Hucka's Nonsense String Evaluator, GitHub `casics/nostril`
  at `fbc0c91`. (PyPI `nostril` 0.1.1 is an unrelated nose wrapper.)
  Decision: `nostril.nonsense(text)`; texts it refuses as too short (fewer
  than 6 letters after its sanitising) are not flagged. Score: nostril's
  own tf-idf string score with its module defaults on the sanitised text;
  0 when too short or when its simple-real rule accepts; 1,000,000 when its
  simple-nonsense rule rejects.
- **gibberish-detector 0.1.1** (Aaron Loo's port of rrenaud's Markov
  detector). The wheel ships no model, so a model is trained with the
  package's own `trainer.train` (default ASCII-letter charset) on the
  repository's `examples/big.txt` at commit `ecac969` (sha256
  `e77a1d4a...71d39d7`); training is count-based and needs no seed. Score:
  `calculate_probability_of_being_gibberish`; decision: score above the CLI
  default limit 4.0.
- **gibberish-chat-detector 0.1.2**. Decision: the package's `gibberish`
  output with its default rules. Score: how many of the six default rules
  fire, evaluated with the package's rule string on the package's own
  features. Both gibberish packages install a top-level module named
  `gibberish_detector`; their submodules do not collide and are imported
  directly.
- **lingua-language-detector 2.2.0**, policy "not English is junk".
  Detector built from all languages with default settings. Decision: the
  detected language is not English (no detection counts as not English).
  Score: one minus the English confidence.
- **Gopher rules** and **C4 rules**, implemented in
  `paper/scripts/benchmark/rules.py` from the RedPajama-V2 quality-signal
  definitions (`togethercomputer/RedPajama-Data` at `6d2cee9`) with the
  thresholds of Rae et al. (2021) and Raffel et al. (2020). Score: number
  of rules failed; decision: any rule failed. The rule lists are in
  `rules.py`; Gopher's duplicate-line and duplicate-paragraph rules have no
  RedPajama-V2 signal and are omitted. The C4 bad-word rule uses the
  LDNOOBW English list at commit `5faf2ba`. C4's line rules are applied at
  document level: a text fails when any of its lines fails.

## Footprint

For every method, on the B test texts: the native decision call over all
texts (pygarble `predict` on the list, baselines in a loop), one warm-up
run and five timed runs in one process on one thread; the median is
reported as milliseconds per 1,000 texts. Import time: median over five
fresh interpreters of the time to import the method's modules. Install
size: bytes on disk of the installed files of the distribution and its
non-stdlib runtime dependencies, excluding `__pycache__`; for pygarble the
files a wheel installs. Offline: whether scoring needs network access
after installation.

## Calibration

For `english` and `english_fusion`: 500 tweets and 500 random strings are
drawn from B dev with `random.Random(20261011).sample` over manifest order.
`calibrate(detector, garbled, clean, objective="max_fpr",
max_false_positive_rate=0.01)` picks a threshold; the profile is rebuilt
with that threshold and its decisions give the realised FPR and recall on
B test and the capture rate on A.

## Pilot disclosure

On 2026-10-11, before this protocol, all of A and all English B rows were
scored once with pygarble defaults at threshold 0.5: `english_fusion`
capture 0.895 on A; precision 0.906, recall 0.991 and FPR 0.068 on B.
Nothing in the package changed afterwards.

## Single test run

`run.py` scores A and B dev, fits cuts, applies the checkpoint, then
scores B test once. Per-row scores and decisions (row id, score, decision;
no text) are saved under `results/predictions/`; `verify.py` recomputes
`results/metrics.json` from them and the manifest. Losses are reported
as they come.
