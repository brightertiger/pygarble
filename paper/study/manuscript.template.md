---
title: 'Low-cost gibberish screening: pygarble and DistilBERT on a complete published corpus'
author:
  - 'Ujjwal Singh Rao\thanks{Independent Researcher, India}'
institute: 'Independent Researcher, India'
date: 27 September 2026
bibliography: references.bib
fontsize: 10pt
geometry: margin=0.8in
colorlinks: true
abstract: |
  Local text screening requires a trade-off between detecting suspicious input,
  retaining meaningful text and inference cost. We compare five fixed pygarble
  configurations with a published DistilBERT gibberish classifier on the complete
  labelled release of a human-generated gibberish research corpus: 38 invented
  texts and 71 meaningful documents. Every normalized character is evaluated
  through 79,969 consecutive chunks, with English controls separated from
  other-language diagnostics. Under a fixed majority-chunk document rule,
  word lookup and two transformer label policies detect all 38 positive
  documents; pygarble's English and extended profiles detect 13 and 20.
  Word lookup flags none of the 5,200 English control chunks. The transformer's
  document-macro English false-positive rate is {{HF_MACRO_FPR}} when all
  non-clean classes trigger a flag, and {{HF_STRICT_FPR}} when only noise and
  word salad do. The transformer's median single-thread CPU scoring latency is
  {{WORD_HF_RATIO}} times word lookup's on the fixed timing workload. These findings
  characterize invented-word detection in this collection, not general
  semantic understanding or deployment accuracy. We release pinned code,
  prediction-level artifacts, source-aware summaries and reproducibility checks.
---

**Draft for author review. Not submitted or peer reviewed. Author review,
funding/conflict declarations and the final AI disclosure remain pending.**

# Introduction

A first-stage text filter may flag malformed or suspicious input before a
more expensive processing step. Such a filter must balance detection against
the cost of rejecting meaningful material. Character statistics and lexical
coverage can be inexpensive, but historical spelling, names and specialist
vocabulary can resemble noise to an English detector. A pretrained neural
classifier offers another operating point, with additional inference and
installation costs. Neither approach should be assumed to recognize meaning
in every domain.

pygarble provides deterministic strategies, configurable ensembles and
inspectable scores for local screening [@pygarble]. Its base installation
has no runtime dependencies. Its separate PII, secrets and profanity module
is outside this study. We compare its existing gibberish configurations with
a publicly available DistilBERT classifier [@jindal; @sanh2019], using a
published linguistic research corpus [@gaskell2022]. Our questions are:

1. How do the fixed configurations compare on all released invented texts
   and all text from the English controls?
2. How sensitive is the comparison to the transformer's binary label mapping
   and the rule for aggregating chunk decisions?
3. What false-flag behavior appears outside English, and what CPU and memory
   costs accompany each implementation?

The contribution is a reproducible empirical comparison, including negative
results and inference costs. It is not a new classifier architecture. In
particular, we test rather than assume that an ensemble improves on its
individual strategies. The corpus's limited source diversity prevents broad
accuracy claims, even when every released character is processed.

# Corpus and coverage

## Published source labels

Gaskell and Bowern collected handwritten invented-language samples while
studying statistical resemblance to the Voynich manuscript [@gaskell2022].
Their reported collection involved 42 volunteers and additional author
samples. We evaluate all 38 transcript files in their released archive at
GitHub revision `d076a7d081f35098fa405928239595afd2e75927`. We do not infer
individual cohort membership or claim that the archive contains every sample
from the original study.

We also include all 71 documents in the accompanying meaningful-text archive.
These span 33 source language/variant labels, including historical and
constructed languages. Four are explicitly English: the KJV New Testament,
a modern NET Bible text, a historical Secreta Alberti transcription, and a
Wikipedia-derived Voynich article. Those four provide the primary negative
controls. The remaining 67 meaningful documents are a separate scope diagnostic;
we retain their negative labels and do not redefine foreign-language text as
gibberish. Voynichese has no established binary label and is not included.
Thus, *complete corpus* refers to the entire released labelled collection.

All classifications come from the source collection. No new human annotation
or label audit was performed. This includes inheriting a document's label
for short fragments, an assumption whose consequences we examine by excluding
tails. The gibberish material carries the authors' modified MIT notice and
citation requirement; comparison documents retain their separate source
terms. We publish hashes, normalized offsets and predictions, without
redistributing comparison prose. No new participants were recruited.

## Exhaustive chunking and dependence

We collapse whitespace to single spaces, retaining other characters, then
partition each normalized document into consecutive, nonoverlapping chunks
of 400 Unicode characters. The final short chunk is retained. This produces
31,964,664 normalized characters and 79,969 chunks across 109 documents:

| Source group | Documents | Chunks | Role |
| --- | ---: | ---: | --- |
| Human-generated gibberish | 38 | 173 | Positive class |
| English meaningful text | 4 | 5,200 | Primary negative controls |
| Other meaningful text | 67 | 74,596 | Separate language diagnostic |

No normalized character is dropped, sampled away or silently truncated.
A chunk can end within a word; the same rule applies to every method and class.
There are {{TAILS}} short final chunks. Exact-hash checking finds {{DUPLICATES}}
duplicate chunk occurrences beyond their first occurrence; all remain in the
coverage analysis. Document IDs and source metadata remain attached to every
prediction. Correlated passages and translations are not independent examples.

An earlier exploratory study used one 400-character prefix per positive and
100 selected blocks per English document. The full run contains {{OVERLAP}}
of those 438 primary input hashes. It expands coverage but is not an independent
replication. The preliminary outcomes were known when this extension was
designed. Its protocol and scoring code were committed as `717ceee` before
the new predictions; this is a versioned prospective analysis plan for the
extension, not a preregistered or previously untouched benchmark.

# Methods

## pygarble configurations

We evaluate the unchanged `english`, `english_extended` and `legacy` profiles,
and the individual `word_lookup` and `entropy_based` strategies. Each uses its
default threshold of 0.5 and one thread. Ensemble defaults use `any` voting.
The legacy profile combines Markov-chain, log-likelihood-ratio and word-anomaly
strategies. English adds encoding/control-character and keyboard signals;
extended also adds pattern, local-anomaly and repetition checks. The separately
evaluated `word_lookup` strategy is not a member of these profiles. It uses
an embedded English word set, folds diacritics, tokenizes Latin letters and
half-weights unknown title-cased words. Strategy applicability is preserved: an insufficient-evidence
result does not trigger a flag and counts as a miss if the inherited label is
positive. Abstention counts are retained separately from accuracy statistics.

These configurations share code and resources; they are not five independent
software systems. No package resource, threshold or default was altered after
observing this corpus. An always-keep rule would have zero recall and zero false
flags and serves as the trivial reference without an additional inference run.

## Hugging Face benchmark and label mapping

The external benchmark is Madhur Jindal's published AutoNLP gibberish detector
[@jindal], a 66,956,548-parameter DistilBERT sequence classifier [@sanh2019].
We pin revision `76672dd7d357` (full commit in the asset manifest) and record hashes
for all model and tokenizer assets. Its safetensors weights occupy 267,838,720
bytes. The model card declares an MIT license. Its named training dataset
was not accessible through the unauthenticated public API, so fine-tuning
overlap with the evaluation material is unknown. Neither the model card's
own validation accuracy nor its marketing claims are used as our results.

The supplied labels are *clean*, *mild gibberish*, *noise* and *word salad*.
We fix two binary policies before scoring: **HF non-clean** flags any winning
class except clean; **HF noise/salad** flags only noise or word salad. Both
policies use the same probabilities and model, and both are reported regardless
of outcome. The winner is the highest-probability class; thresholding the sum
of non-clean probabilities at 0.5 would be a different decision rule.
There is no fine-tuning or threshold selection on this corpus.

Inference is local and offline after verified asset downloads. The full run
uses PyTorch 2.9.0, Transformers 4.53.3, full-precision CPU inference, evaluation
mode, no gradients, scaled-dot-product attention, four intra-op threads, one
inter-op thread and batches of eight. Tokenizer parallelism is disabled.
Inputs are dynamically padded without truncation; the largest observed input
has {{MAX_TOKENS}} tokens, within the 512-token limit. Both systems receive
identical text. The transformer is a research benchmark dependency only;
pygarble's base package remains model-free.

## Endpoints and statistical units

A document is flagged if **at least half of its chunks are flagged**, with
ties flagged. This is an explicit study aggregation rule, not a native
long-document interface of either detector. Primary results are positive-document
recall and false flags among the four English documents. To expose behavior
hidden by majority aggregation, we also report each document's flagged-chunk
fraction and average those fractions with equal document weight. We call
these the positive mean flagged fraction and negative mean chunk FPR.
All micro counts are available, but 79,969 chunks are not treated as 79,969
independent samples.

Sensitivity analyses report any-chunk document decisions and mean chunk rates
excluding final short chunks. Any-chunk flags tend to increase with document
length; comparing their counts without that context would be misleading.
Other-language meaningful texts receive their own document and language
summaries, without pooling them into the English headline.

The 95% Wilson intervals for positive-document recall are conditional on
document independence, which unknown participant metadata may violate. A
2,000-replicate paired document bootstrap, seeded 20260926, describes recall
differences from HF non-clean. We do not give passage-binomial confidence
intervals for FPR: 5,200 English chunks still come from four documents, two
of which are Bible translations. No multiple-testing significance claim is
made. Artificial class balance also makes headline accuracy, precision and
F1 poor substitutes for deployment evaluation.

\clearpage

# Results

## Detection and English false flags

Table 1 reports document decisions and equally weighted within-document chunk
rates. The last two columns refer only to the four English meaningful sources.

{{PRIMARY_TABLE}}

**Table 1.** TP/FP refer to majority-chunk document decisions. Pos./neg. means
average flagged-chunk fractions across positive/English negative documents.
Recall intervals describe 38 documents under the stated independence assumption.

Word lookup and both HF policies flag all 38 positive documents. Word lookup
and HF non-clean also flag all 173 positive chunks; HF noise/salad flags 172.
Perfect observed document recall has a conditional Wilson interval of roughly
90.8--100%, and does not establish equivalence on a wider population. Default
English and extended profiles detect 13 and 20 positive documents, respectively;
these default configurations detect fewer positive documents than the
separate word-lookup strategy. This is a comparison of configurations, not
an ablation of a shared ensemble component. Entropy produces no flags at its
default operating point.

The English controls show a substantial policy effect in the transformer.
HF non-clean flags 1,494 of 5,200 chunks and a majority of the Secreta document.
HF noise/salad flags six chunks and no control-document majority. Its macro
FPR is {{HF_STRICT_FPR}}, compared with {{HF_MACRO_FPR}} for HF non-clean.
The difference is entirely due to the treatment of the model's winning
*mild gibberish* class, not retraining or different inputs. Word lookup has
zero observed false flags in these four controls; this is not a zero-FPR
guarantee outside the collection.

{{CONTROL_TABLE}}

**Table 2.** False-flag counts / complete chunk counts for each English source.
A majority-only table would obscure the considerable within-document errors
of HF non-clean on KJV, NET and Wiki, even though those documents remain
below the majority threshold.

![Complete English comparison. Left: majority-chunk recall over 38 positive documents, with conditional Wilson intervals. Right: complete within-source English false-flag rates.](figures/full-comparison.pdf){width=100%}

## Aggregation and short-fragment sensitivity

{{TAIL_TABLE}}

**Table 3.** Any-chunk document decisions, tail-excluded majority positive
detections (Full TP), and document-macro flagged fractions after excluding
short tails. Full-width rates still describe correlated chunks.
The alternative document rule and tail analysis are sensitivity checks, not
additional independent experiments or grounds to select an optimal rule.

The primary comparison favors a simple lexical signal on invented-word text.
This should not be read as evidence about grammatical but semantically incoherent
sentences: the corpus was not designed to balance those types of error.
The two transformer policies have identical document recall here, despite
one differing positive-chunk decision and markedly different control behavior.

## Other-language scope diagnostic

{{LANGUAGE_TABLE}}

**Table 4.** False flags on the 67 meaningful documents outside the explicitly
English controls. These are retained source negatives, not extra gibberish
examples. Rates average documents equally and remain outside the primary
English comparison.

![False-flag fractions by supplied language/variant label, averaged over documents. Parentheses show document counts; cells round to whole percentages. Meaningful source labels are retained; English is included for orientation.](figures/full-languages.pdf){width=90%}

Word lookup falsely flags 49 of the 67 other-language documents under the
majority rule; HF non-clean and HF noise/salad flag 62 and 52. No saved
pygarble evaluation reports an API-level insufficient-evidence status.

These results test scope, not multilingual competence. The word-lookup
strategy returns a zero score when it finds no eligible Latin-letter words;
zero flags therefore need not establish linguistic coverage. Conversely,
unfamiliar vocabulary or script can trigger flags in other configurations.
The diagnostic separately records API-level insufficient-evidence counts;
that API status is not itself a guarantee of language coverage. Related Bible content, parallel translations and the small
number of documents per language limit cross-language comparisons. Neither
English detector should be presented as a language-independent arbiter of
meaningfulness on this evidence.

\clearpage

## CPU and memory cost

The timing workload uses the original fixed 76 inputs of 400 characters:
38 positive prefixes and 38 deterministically selected English control blocks.
It is a latency probe, not a random sample of the full multilingual workload.
Measurements run on an Apple M2 with 8 GiB RAM, macOS ARM64 and Python 3.12.2,
with each method in a fresh process. We use one thread and batch size one
for the timing comparison, including tokenizer, model and probability work
for HF, and the full score/applicability path for pygarble. After one warm
pass, five measured passes produce 380 per-input observations per method.
These differ from the four-thread, batched exhaustive run and from pygarble's
potentially faster short-circuit `predict` interface.

{{RUNTIME_TABLE}}

**Table 5.** Warm scoring latency and separate-process peak RSS. Both HF policies
share one model run. RSS includes Python, imports, model state and resident
corpus data; it is not the isolated model's tensor memory. Import, construction,
first-call and throughput measurements are separately archived.

On this workload, HF median latency is {{WORD_HF_RATIO}} times word lookup's
and {{ENGLISH_HF_RATIO}} times the English profile's. This describes one CPU,
fixed input order and implementation, not a production service-level promise.
GPU, quantization, compilation and alternative transformer runtimes were not
tested. Model download time is excluded. No energy measurement or cloud-cost
extrapolation is made. No paid data, inference endpoint or cloud compute was
procured; execution used existing hardware.

# Discussion

Full coverage strengthens the earlier comparison by removing its cap on
meaningful text and scoring every positive fragment. It also adds a materially
different model family and makes its label-policy trade-off visible. The
strongest supported claim remains specific: lexical unfamiliarity separates
the released invented texts from these four English comparison documents at
low measured CPU cost. It does not establish a universal advantage over
transformer classifiers. The stricter transformer policy is also effective
on the primary task; the broader policy appears mismatched to retaining
all inherited meaningful text, particularly the historical Secreta source.

A first-stage filter should therefore expose its intended language, rejection
policy and fallback behavior. Cheap rejection may be useful when conspicuous
lexical noise is the target, but unfamiliar meaningful input can be costly to
lose. The full results make those trade-offs inspectable without adding a
neural dependency to the library. We have not measured a deployed cascade,
real user prevalence, downstream research-quality improvements or independent
adoption, and do not infer those benefits from this benchmark.

The preliminary source-held-out calibration experiment remains available with
its original protocol and outputs. It fit adapted character bigram/trigram
models following Renaud's approach [@renaud] and calibrated thresholds to a
1% empirical validation FPR. Its failures illustrate transfer risk: calibrating
word lookup on Wiki yielded 99 false flags among 100 Secreta blocks, and the
character models produced 42% and 44% Wiki FPR in their held-out fold. These
are earlier capped results, not measurements from a new full-corpus calibration
run. They motivate caution about carrying a threshold across domains.

# Limitations and reproducibility

The most consequential limitation is source diversity. Exhaustive chunking
adds text, not independent participants or control domains. The release
contains only 38 positive documents and four English controls; participant
relationships are not fully known. Meaningful and invented texts were produced
under different conditions, and their labels were inherited without a new
human audit. Fixed character boundaries and inherited tail labels introduce
further measurement assumptions. The evaluation does not comprehensively
cover keyboard noise, ordinary-word semantic nonsense, names, code or modern
user submissions.

HF fine-tuning overlap is unknown because its named training dataset could
not be inspected. Pretraining and existing package lexical resources also
prevent claiming a guaranteed contamination-free test. The earlier and full
experiments overlap. The two HF policies and five related package configurations
are not seven independent algorithms; small conditional intervals or a zero
paired difference cannot establish equivalence. All conclusions are conditional
on these inputs and operating rules.

Artifacts include pinned downloads, asset checksums, the frozen scoring-code
manifest, per-chunk probabilities and decisions, source-aware summaries and
CPU measurements. Raw comparison prose is kept in an ignored local cache.
Automated validation checks complete membership and character coverage,
recomputes summaries, checks every saved decision mapping and replays a fixed
first/middle/last subset from every document in a fresh offline process.
This verifies computations, not inherited label validity. The preliminary
standard-library study also has a separate byte-identical reproduction.
The full transformer run is not claimed to have been independently reproduced
across machines or by a human researcher.

# Availability and declarations

All study code, provenance, results and manuscript sources are in
`paper/study/` of [the pygarble repository](https://github.com/brightertiger/pygarble).
`full-results/` holds this evaluation; `results/` retains the preliminary study.
The full protocol is `full_corpus_protocol.md`. Exact script/library hashes
are recorded in each result directory's code manifest. The optional HF
requirements are isolated from the base package.

The author reports using OpenAI Codex and Anthropic Claude in the wider project.
Codex assisted with study design, research code, automated execution, analysis
and manuscript drafting. No generative LLM produced evaluation labels or
served as a judge. The DistilBERT classifier is an explicitly evaluated neural
baseline. Claude's precise scope and recoverable model versions remain to be
confirmed. Human review of this draft and responsibility for its scientific
decisions have not yet been asserted.

Funding and conflicts of interest: author declarations pending. Author review
and completed declarations are required before submission. A prospective
JOSS article would describe the software contribution and reference this
study, rather than duplicate the empirical manuscript. No journal acceptance
or search-indexing outcome follows from these measurements.

\clearpage

# Appendix: paired recall differences

{{RECALL_DIFFERENCE_TABLE}}

Paired document-bootstrap intervals compare the same 38 majority decisions.
A zero interval when two methods agree on every document is a property of
this empirical resampling distribution, not proof of equal generalization.

# References
