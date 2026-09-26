---
title: 'pygarble on published human-generated gibberish: local heuristics and calibration transfer'
author:
  - 'Ujjwal Singh Rao\thanks{Independent Researcher, India}'
institute: 'Independent Researcher, India'
date: 27 September 2026
bibliography: references.bib
fontsize: 10pt
geometry: margin=0.8in
colorlinks: true
abstract: |
  Inexpensive text filters can identify unusual character or word patterns,
  but their decisions depend on the input domain and operating threshold.
  We evaluate pygarble profiles and lightweight character models using
  published meaningful/gibberish classifications without new annotation.
  The primary analysis uses one 400-character prefix from each of 38 released
  human-generated gibberish documents and 100 blocks from each of four English
  comparison documents. Default English and extended profiles detect 16 and
  19 positive documents, respectively. A single dictionary strategy detects
  all 38 with no false flags in the 400 selected control blocks. However,
  source-held-out calibration reveals substantial transfer failures: a 1%
  validation false-positive budget can yield much higher rates on another
  meaningful source. These results describe a small curated collection, not
  general accuracy or semantic understanding. We provide pinned retrieval,
  predictions, descriptive uncertainty, CPU measurements and a reproducible
  evaluation harness to support further scrutiny.
---

**Draft for author review. Not submitted or peer reviewed. Author review,
funding/conflict declarations and the final AI disclosure remain pending.**

# Introduction

Gibberish screening is often an early step in text processing: a program
flags suspicious material before a more expensive analysis or a decision to
retain it. This can be useful when inputs contain keyboard noise, corrupted
characters or unfamiliar strings. The decision is nevertheless task-specific.
Historical spelling, names and specialist vocabulary can be meaningful while
looking unusual to a modern English dictionary. Conversely, invented text can
imitate the statistical properties of language without carrying its meaning.

pygarble supplies deterministic strategies, configurable ensembles and
inspectable scores for local screening [@pygarble]. Its base installation
has no runtime dependencies. A separate module handles selected PII, secret
and profanity patterns; those capabilities are outside this evaluation.
The present study asks a narrower question: how do existing local strategies
respond to a published collection of human-generated gibberish and English
comparison text, and how stable are thresholds across the available sources?

The contribution is an external, reproducible assessment of configurations
and a small calibration-transfer study. It is not a new detection algorithm
or an estimate of performance on arbitrary user input. The default ensemble
is not assumed to outperform its individual strategies. We retain unfavorable
results and distinguish the collection's labels from a new determination of
whether particular text is meaningful.

# Data and evaluation design

## Published classifications

Gaskell and Bowern collected handwritten invented-language samples for a study
of statistical resemblance to the Voynich manuscript [@gaskell2022]. Their
reported collection involved 42 volunteers and additional author samples.
We use the released GitHub archive at revision
`d076a7d081f35098fa405928239595afd2e75927`, which contains 38 gibberish transcript
files. We include all 38 and do not infer their individual cohort membership.
The study evaluates the available release, not the original study's complete
collection or its Voynich conclusions.

We select the four comparison files explicitly identified as English: the
King James Version New Testament (KJV), a modern NET Bible text, a historical
English Secreta Alberti transcription, and a Wikipedia-derived Voynich article.
Their supplied meaningful/gibberish classifications are inherited without
new annotation or human label audit. This does not establish that every
comparison passage is clean modern prose. Non-English comparison files and
Voynichese are excluded without classifying them as gibberish.

Source rights differ. The gibberish data carry the authors' modified MIT
notice and citation requirement; meaningful comparison texts retain their
original terms. Our repository publishes retrieval instructions, hashes,
offsets and measurements rather than redistributing the comparison texts.
No new participants were recruited and no participant identities were inferred.

## Inputs and dependence

Whitespace is collapsed to single spaces while case, accents and punctuation
are retained. The primary input is exactly 400 Unicode characters. We take
one prefix per gibberish document and up to 100 evenly spaced, nonoverlapping
blocks per comparison document. All four controls provide 100 primary blocks.
This controls input length without producing multiple nominally independent
positive examples from each transcript. A prefix can end within a word;
the same fixed-length slicing rule applies to both classes.

Predeclared sensitivity views use 100 and 800 characters, with documents too
short for a complete block omitted. The 800-character view has 36 positive
documents and 362 control blocks. Whole positive documents are an additional
endpoint. These overlapping views are dependent and are not independent
replications. Source/class confounding remains: gibberish and comparison
prose were produced under different conditions.

An automated check found no exact duplicate primary blocks and no primary
block with at least half its five-word shingles covered by the repository's
existing JSON regression material. Across separate control families, maximum
five-word-shingle containment was below 0.1%. The Bible translations share
content and are kept in one family despite their different wording. These
checks cannot establish independence from every dictionary or historical
resource used in detector development.

# Methods

## Fixed configurations

We evaluate pygarble's `english`, `english_extended` and `legacy` profiles
with their default threshold of 0.5, `any` voting and one thread. The English
profile combines its legacy language signals with encoding/control-character
and keyboard signals; the extended profile adds pattern, local-anomaly and
repetition checks. The legacy profile provides a fixed strategy-group
comparison. Single `word_lookup` and `entropy_based` strategies use their
default settings. An always-keep policy establishes the trivial no-removal
reference. These package strategies share implementation/resources and are
not presented as independent software systems.

For calibration experiments, two additional baselines fit conditional
character models locally. The bigram baseline adapts Renaud's approach
[@renaud]: retain lower-case ASCII letters and spaces, add ten pseudocounts
per character transition, and average log transition likelihoods. We reverse
the sign so that larger scores indicate less resemblance to training prose.
The trigram extension uses two-character contexts and add-one smoothing.
Neither uses pretrained weights or an LLM. Training data and threshold
selection differ from the upstream program; these are documented adaptations,
not claims to reproduce its original accuracy.

## Source-held-out calibration

The controls form three families: Bible (both translations), Secreta and
Wiki. The fixed rotations assign distinct families to testing, training and
validation:

| Test family | Character-model training | Threshold validation |
| --- | --- | --- |
| Bible | Wiki | Secreta |
| Secreta | Bible | Wiki |
| Wiki | Secreta | Bible |

Character models use the first 100,000 normalized characters from each
training document, with no transitions across documents. Package resources
remain unchanged. No gibberish transcript is used to fit any model or select
a threshold. On meaningful validation blocks, select the lowest candidate
threshold producing at most 1% false positives. Candidates are zero and the
next representable floating-point number above each distinct applicable
score; decisions use `>=`, preserving ties. This policy maximizes flags
subject to the empirical constraint without optimizing on positive test data.

The held-out meaningful family and the same 38 positive prefixes are then
scored. Those positives recur across folds; they are not pooled as 114
independent observations. The available three control families are too few
for a broad supervised-classifier comparison, so a trained linear classifier
from the initial planning document was excluded before scoring. No model or
threshold was selected after inspecting final outcomes.

## Metrics and execution

We report positive-document recall, false-positive counts and rates within
each control document, and macro-average control-document FPR where useful.
The collection's artificial class prevalence makes overall accuracy,
precision and F1 unsuitable as headline deployment measures. Inapplicable
positive results count as misses; coverage is also recorded.

Recall intervals use the 95% Wilson construction conditional on independent
documents. A paired document bootstrap with 2,000 replicates and seed 20260926
provides descriptive recall differences against default English. Unknown
participant dependence limits these intervals. We do not calculate
passage-binomial FPR confidence intervals: hundreds of blocks still come
from four documents. No multiplicity-adjusted significance claim is made.

The protocol and harness were committed as `22f30d8` before detector
predictions. Scripts use the standard library and the unchanged local package.
A fresh virtual environment reproduced the deterministic prediction and
summary artifacts byte for byte. Scoring requires no external inference;
source downloads occur during preparation only.

# Results

## Default decisions

{{DEFAULT_TABLE}}

The dictionary strategy detected all released positive prefixes while the
default English ensemble missed 22 of 38. This is evidence that the ensemble
is not uniformly stronger than a single lexical signal on this collection.
The dictionary recall interval is approximately 90.8–100% under the stated
independence assumption. Its zero observed false positives across four
control documents does not establish a zero population FPR.

The extended profile's recall increase over English is 7.9 percentage points;
the paired bootstrap interval is 0.0–15.8 points. Coverage was 100% for these
views. The entropy strategy made no positive decisions at its default cutoff.
That result describes this implementation and threshold, not a claim that
entropy cannot contribute to another method.

![Default 400-character results. Recall intervals are conditional on document independence. Control false-positive rates are descriptive within-source measurements.](figures/default-results.pdf){width=100%}

## Length sensitivity

{{SENSITIVITY_TABLE}}

The extended profile detects more complete transcripts than 400-character
prefixes, consistent with different opportunities to trigger its strategies.
Because the whole-document endpoint has no matched control length, it cannot
establish improved overall discrimination. Length-view denominators and
control rates are available in the machine-readable results.

## Calibration transfer

{{CALIBRATION_TABLE}}

The 1% validation constraint did not generally transfer. For example,
calibrating word lookup on Wiki before testing Secreta produced 99 false
flags in 100 Secreta blocks, compared with none at the original default
threshold. Both character models detected all 38 positives in every fold,
but their Wiki false-positive rates were 42% and 44%. Their training family
was Secreta and their validation family was Bible in that fold.

These differences show why low false-positive behavior on one source is
insufficient to promise the same behavior on another. They do not establish
which threshold is optimal in a future application. Outcomes can reflect
both source domain and the fixed assignment of training and validation
families. Exhaustive rotations or alternative assignments would be additional
exploratory analyses and are not implied by this study.

![Test false-positive rates after calibration on a different family. Every validation rate meets the empirical 1% budget; test rates can be much higher.](figures/calibration-transfer.pdf){width=80%}

\clearpage

## CPU and memory measurements

Measurements used an Apple M2 with 8 GiB RAM, macOS and Python 3.12.2. Each
method ran in a fresh process on 76 fixed 400-character inputs: 38 positive
prefixes and 38 deterministically selected controls. After an unmeasured warm
pass, five timed passes measured the complete score-and-applicability path.
This differs from the library's potentially faster short-circuit `predict`
path. Package import, construction/training and first-call measurements are
stored separately. Peak RSS was measured in separate worker processes and
includes Python, imports and resident corpus data.

{{RUNTIME_TABLE}}

Character-model timings use the Secreta-trained fit. Measurements come from
one machine and a fixed input order; the trivial always-keep cost approaches
timer overhead. They are not production latency guarantees or cloud-cost
estimates. No paid data, inference or cloud-compute service was procured.

# Discussion and limitations

The most direct finding is modest: lexical unfamiliarity separates this
published invented-language collection from the selected English controls
more strongly than several existing default profiles. That is compatible
with a collection built from intentionally invented words. It does not show
that the dictionary method recognizes semantic nonsense made entirely of
ordinary words, and it does not justify replacing a production default on
this evidence alone.

The study also provides a concrete counterexample to interpreting an
empirical calibration budget as a general guarantee. A threshold learned on
modern technical text can reject historical meaningful spelling. Even with
all positives detected, a method can be unsuitable for automatic removal
because it flags substantial meaningful material. Reusing the supplied
classifications made the study inexpensive but leaves label suitability
unverified; no new human audit was performed.

Four control documents and 38 released positive documents cannot represent
the diversity of real inputs. Limited source diversity, prefix selection,
missing participant metadata, untested languages and possible overlap with
pre-existing lexical resources all constrain inference. This is a secondary
analysis of an existing collection, not independent confirmation of its
labels. The absence of a domain-matched deployment corpus also prevents
claims of downstream research-quality improvement.

For software users, the practical implication is to expose configuration and
source-specific errors rather than equating a composite detector with a
universal answer. The reproducible harness makes those choices inspectable.
A larger independently sourced evaluation and actual adopter feedback would
be needed for stronger claims about broad research utility. No journal
acceptance or indexing outcome follows from the present results.

# Availability and author review

Code, pinned source retrieval, data notices, predictions, figures and
reproduction instructions are in `paper/study/` of
[the pygarble repository](https://github.com/brightertiger/pygarble).
The raw comparison corpus is not republished. Exact script/library hashes
are recorded in `results/code-manifest.json`.

The author reports using OpenAI Codex and Anthropic Claude in the wider
project. Codex (GPT-6 in this session) assisted with study design, research
code, automated execution, analysis and manuscript drafting. No LLM was used
for detector inference or creation of evaluation labels. Claude's precise
scope and model versions remain to be confirmed. Human review of this draft
and responsibility for its scientific decisions have not yet been asserted.

Funding and conflicts of interest: author declarations pending. This draft
must be reviewed and these declarations completed before any submission.
A prospective JOSS article would describe the software contribution and
reference this evaluation, rather than duplicate a results-focused article.

# References
