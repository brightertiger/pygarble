# Preprint plan (v5, 2026-10-11)

Deliverables: benchmark code, frozen results and analysis, LaTeX source and
PDF of the preprint. Scope: one preprint showing what pygarble captures on two public gibberish
datasets. Nothing else. pygarble defaults are frozen at main e3135ed.

## Data

**A. Survey junk, positives only.** Kaggle `johnwdata/gibberish-text-classification`,
`Gibberish.xls` (CSV, sha256 091d5d38...), 3,513 unique junk rows after strip
and dedupe. Endpoint: capture rate (recall) per method. No negatives from this source.
Licence "Unknown": store row indices and hashes, never the text.

**B. Random strings against tweets, English only.** Hugging Face
`Sowmya15/gibberish_multilingual` (Apache-2.0), all splits pooled, rows with
language English: 4,000 random strings (label 1), 5,971 tweets (label 0).
Endpoint: precision, recall, FPR, AUROC, precision-recall curve. Labels used
as given; the mojibake-tweet share is reported as a note, not re-labelled.

## Protocol

- B is split 30 percent development, 70 percent test by seed 20261011,
  stratified by label. A is scored whole; it has no thresholds to fit.
- The paper reports every method at its own default decision only. The
  fixed-FPR operating points and the calibration run computed in PR 18 stay
  in the frozen result files but are not headline results; the limitations
  section states in one sentence, with numbers, that at a matched 1 percent
  false-positive rate the Markov gibberish-detector captures more survey
  junk than english_fusion (v5 change, made after seeing those results).
- Pilot on 2026-10-11 (defaults, threshold 0.5: fusion recall 0.895 on A;
  precision 0.906 and recall 0.991 on B) is disclosed; nothing changed after.
- Test scoring runs once from the frozen runner; predictions saved;
  `verify.py` recomputes every number from them.
- Bootstrap 95 percent intervals over rows, 1,000 resamples.

## Methods

pygarble: profiles `english`, `english_extended`, `english_fusion`, `legacy`,
`corruption`, `spoofing`, `llm_output`; six voting modes on `english`; each
strategy alone (appendix). Baselines, CPU, pinned: nostril, gibberish-detector
(Markov), gibberish-chat-detector, lingua language ID ("not English is junk"),
Gopher and C4 rules. No neural baselines.

Footprint for every method: ms per 1,000 texts single-threaded, median of 5;
import time; install size; offline.

## Figures and tables

Fig 1 capture rate on A, all methods, default decision. Fig 2 precision
and recall on B, default decision. Fig 3 speed against capture rate. Table 1
results table (A capture; B precision, recall, FPR; default decision).
Table 2 footprint. All numbers read from result files.

## Work packages

| WP | Scope | Model |
| --- | --- | --- |
| 0 | Manifest with hashes; loaders from `/Users/ujjwal/Downloads/solo/pygarble-data/`; dedupe; B split; `protocol.md` frozen | Opus |
| 1 | Method wrappers, runner, metrics with bootstrap, B dev operating points, single test run on A and B; footprint; calibration | Opus |
| 2 | Figures, tables, `verify.py`, preprint in LaTeX built with tectonic to PDF; Fable edits introduction, limitations, guidance | Opus |

One PR and one Opus review each. Checkpoint after WP1 dev: if fusion capture
on A at the 1 percent point is below 0.7, stop and discuss.

## Honesty

Losses reported with the same prominence as wins; label noise
in A left in; pilot disclosed; no default or threshold changed after scoring.

## Preprint outline

Title: "How much junk do rules catch? pygarble on real survey responses and
social-media text". Sections: introduction; pygarble in one page; datasets
and protocol; results on A; results on B; cost; where it fails; guidance;
limitations (including the matched-FPR sentence), reproducibility, AI
assistance disclosure.
