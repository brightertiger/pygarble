# Preprint plan (v7, 2026-10-11)

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

- Every method at its own default decision only. B is scored whole, no
  split. The matched-FPR operating points, split and calibration computed in
  PR 18 were removed at the maintainer's direction (v5/v6) and remain only in
  git history; the limitations section says so in one sentence.
- Pilot on 2026-10-11 (defaults, threshold 0.5: fusion recall 0.895 on A;
  precision 0.906 and recall 0.991 on B) is disclosed; nothing changed after.
- Test scoring runs once from the frozen runner; predictions saved;
  `verify.py` recomputes every number from them.
- Bootstrap 95 percent intervals over rows, 1,000 resamples.

## Methods

Three methods, both datasets, each at its own default decision:
pygarble (two rows: the default `english` profile and `english_fusion`),
gibberish-detector 0.1.1 (rrenaud Markov port, trained on its big.txt,
pinned), lingua-language-detector 2.2.0 with "not English is junk". Plus one
post hoc row, "domain-specific tuning": the `english` profile with the
MOJIBAKE strategy removed (same five other members, voting any, threshold
0.5), constructed after seeing that 26.4 percent of tweets carry mojibake;
reported in its own subsection as post hoc. Nothing else. Footprint for the same rows: ms per 1,000 texts single-threaded,
median of 5; install size.

## Figures and tables

Table 1: four rows (pygarble english, pygarble english_fusion,
gibberish-detector, lingua) plus the separated tuned row; columns A capture, B precision, B recall, B FPR,
with 95 percent bootstrap intervals. Table 2: footprint for the same rows.
One optional two-panel figure (A capture; B precision and recall). All
numbers read from result files.

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
