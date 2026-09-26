# Published-corpus evaluation protocol

Frozen before detector predictions on 27 September 2026. This protocol
supersedes the broader pilot and annotation proposals in `../publication-plan.md`.
The author requested the published GitHub dataset, no new human audit, local
execution and review before submission. No new labels or synthetic examples
are introduced. This is an exploratory external evaluation, not a preregistered
trial or evidence that pygarble understands meaning.

## Question and dataset

How do local English-oriented heuristics respond to the released human-written
gibberish and English comparison texts of Gaskell and Bowern (2022), and how
does calibrating on one meaningful source transfer to another?

Pin `danielgaskell/voynich` at
`d076a7d081f35098fa405928239595afd2e75927`. Use all 38 released gibberish `.txt`
files, including both DA-prefixed files; do not infer participant identities
or cohort membership from filenames. The original study reports 42 volunteers
and additional author samples; this archive is not assumed to be complete.
Use the four meaningful files explicitly named English. Exclude other
languages and Voynichese without reclassifying them as gibberish.

Inherit the source's meaningful/gibberish classifications without new human
annotation or audit. The controls include historical spelling and two Bible
versions. This is a source-label evaluation, not verified modern clean prose.
The data were collected for a different question and classes differ in origin.
Do not infer population precision, downstream corpus utility, multilingual
coverage, PII/secret/profanity performance or a Voynich decipherment result.

Normalize whitespace with `" ".join(text.split())`, without changing case,
accents or punctuation. Primary input length is 400 Unicode characters:
one prefix per gibberish document, and up to 100 evenly spaced, nonoverlapping
blocks per meaningful document. Record offsets in normalized text and hashes;
do not publish the comparison texts themselves. Length sensitivity uses 100
and 800 characters, omitting documents too short for a complete block and
reporting counts. Also report whole-gibberish-document detection separately.
These overlapping views are not independent replications.

## Methods fixed before evaluation

- Always keep, a trivial lower bound on removals.
- pygarble `english`, `english_extended` and `legacy` profiles, defaults
  (`any` voting, threshold 0.5, one thread).
- pygarble `word_lookup` and `entropy_based` strategies at default 0.5,
  representing inexpensive single-signal references. They share package
  resources and are not independent packages.
- A Python 3 adaptation of rrenaud/Gibberish-Detector's smoothed character
  bigram likelihood, trained locally on the specified source family. Keep
  its 27-character alphabet and add-10 smoothing. Use negative mean log
  likelihood as the increasing anomaly score. Do not claim reproduction of
  its upstream training corpus, good/bad examples or threshold.
- A character trigram conditional-likelihood baseline with the same alphabet,
  add-1 smoothing, and training/calibration assignments.

Default methods are evaluated without tuning on all selected views. Whole
positive documents are a secondary endpoint. The extended/legacy profiles
provide fixed strategy-group comparisons; no profile is chosen after results.

Calibration is a separate, primary-length-only experiment. There are three
meaningful source families: `bible` (both translations kept together),
`secreta` and `wiki`. Fixed rotations `(test, train, validation)` are:
`(bible, wiki, secreta)`, `(secreta, bible, wiki)`, `(wiki, secreta, bible)`.
Fit character models on the first 100,000 normalized characters per training
document, with no transitions across documents. No positive examples are
used for fitting or threshold selection. The pygarble models are unchanged.

For each method, select the lowest candidate threshold that flags at most
1% of meaningful validation blocks. Candidates are zero and the next
representable float above each applicable validation score. Scores use `>=`;
ties are kept together. A threshold above a method's entire score range means
an external keep-all policy, not a valid new pygarble constructor setting.
Record calibration count, observed FPR, threshold and whether it exceeds 1
for bounded pygarble scores. Test on the held-out meaningful family and the
same 38 gibberish prefixes. Never pool those repeatedly evaluated positives
as if they were 114 distinct documents. No hyperparameter search is performed.

Do not fit a supervised text classifier: there are only three meaningful
source families, too few for an informative independent train/validation/test
comparison with a broad generalization claim. This scope reduction is made
before outcomes, not because of a baseline's performance. The bigram and
trigram models provide independently implemented lightweight comparisons.

## Metrics and dependence

Store every prediction with record ID, source, length, score, applicability,
status, threshold and decision. Inapplicable positives count as misses for
automatic removal; separately report coverage. Report positive recall and
false-positive counts/rates for each meaningful document and macro-average
over documents; class prevalence is artificial, so omit overall accuracy,
precision and F1 from headline results. Report all methods and all folds.

For positive recall, show a 95% Wilson interval conditional on treating released
documents as independent; missing participant metadata and shared collection
conditions weaken that assumption. Use 2,000 seeded paired document bootstrap
replicates for recall differences against default English (seed 20260926).
These are descriptive intervals, not multiplicity-adjusted hypothesis tests.
Do not give passage-binomial CIs for FPR: hundreds of excerpts still come from
only four documents. Negative source counts remain visible in every report.

Check exact duplicates and five-word-shingle overlap between source families
and with old development corpora. Publish the automated overlap results;
overlap checks are not human label review and cannot establish independence
from all bundled dictionaries or historical training resources. If severe
cross-family duplication is found, pause claims about those folds and record
the issue rather than silently reassign examples after scoring.

## Runtime and reproducibility

Use the same 400-character inputs for timing: all positive prefixes plus an
equal-size deterministic sample of meaningful blocks. Time the full score
and applicability path used by this harness, not pygarble's short-circuit
`predict` optimization. Run each method in a fresh subprocess, recording import,
construction/training, first-call and five warm passes separately. Report
median/p95 per-call time, warm throughput and whole-process peak RSS, which
includes Python, imports and resident data. Measure RSS in a separate worker
from timed calls. Use one Python thread; no remote inference or paid compute.
Report one calibrated character model per order (the wiki-held-out fold);
do not present that fit as representative of every training domain.

Commit source manifests, protocol, tests and analysis code before predictions.
Keep raw archives/text in ignored `.cache`; publish only provenance, record
hashes/offsets, metrics, predictions, figures and drafts. Re-run in a fresh
stdlib-only virtual environment and compare deterministic outputs. Runtime
numbers are machine-dependent and are excluded from byte-identical checks.
Log any corrections or protocol deviations in `deviations.md`.

All work remains under `paper/study/`, aside from updating existing paper
documents to link the completed study. No runtime library change, paid service,
paper submission or merge is part of this execution.
