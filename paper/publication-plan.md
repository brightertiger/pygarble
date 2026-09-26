# Publication and discovery plan

Status: proposed protocol, 26 September 2026. No experiments in this plan
have been run, no paper has been submitted, and no acceptance or indexing
outcome is promised. This document plans the work; it does not authorize
paid services or publication of an unfinished manuscript.

## Objective and route

Produce a defensible software paper supported by a reproducible comparison
and a concrete research application, release a complete preprint on arXiv,
and pursue journal publication and Google Scholar discovery.

Primary route: a complete arXiv preprint, followed by a fee-free JOSS
submission if actual research-use evidence, scope and its other requirements
are met. Journal of Open Research Software (JORS) is an alternative only with
a confirmed waiver that keeps total spending within US$100. Its listed
software-metapaper fee of GBP824 exceeds this budget. Choose one journal
before adapting the final manuscript; do not submit the same paper to multiple
journals simultaneously. A second empirical paper is optional and requires
a distinct contribution. A preprint is not peer-reviewed journal acceptance.

Confirmed author: Ujjwal Singh Rao, Independent Researcher, India. The author
has a verified arXiv account and reports using Codex and Claude. An ORCID,
funding statement, detailed AI-use inventory and human-review confirmation
remain outstanding. Verify category-specific submission eligibility later.

Confirmed total budget ceiling: US$100. Target incremental cash spending is
US$0 using existing hardware, free datasets, local CPU experiments, arXiv
and a fee-free journal. Existing equipment, internet and author time are
assumed available; this is not a claim of zero total economic cost. Reserve
the budget for a concrete need, not automatic spending. No paid inference,
cloud compute, model downloads or publication fees are planned.

The author prefers reusing public datasets from Kaggle and similar sources.
Human-review availability remains unconfirmed. Reuse suitable existing labels
and propose a small audit rather than assume the author can label a corpus
from scratch. Source candidates and unresolved checks are recorded in
[dataset-candidates.md](dataset-candidates.md).

## Research question and scope

**When can inexpensive local detectors remove corrupted English text while
preserving legitimate unusual text, and how does their behavior change
across domains?**

The first study focuses on gibberish/corruption detection in text used for
corpus-quality assessment. PII, secrets and profanity are described as
separate package capabilities, not benchmarked under a shared accuracy score.
The Trident citation motivates an optional later filename study; it is not
evidence of adoption or measured malware-detection improvement.

Define classes before collecting labels:

- `legitimate`: interpretable text in the task's intended domain, including
  technical terms, unfamiliar names and ordinary spelling errors.
- `corrupted`: text made unusable for the stated task by character disorder,
  keyboard mashing, encoding damage or degenerate repetition. Store the subtype
  and severity; a corruption operation alone does not establish this label.
- `uncertain`: insufficient context or genuine ambiguity. Preserve these
  cases and report them separately rather than force a favorable binary label.
- `out_of_scope`: valid non-English text, code or identifiers outside the
  primary English-prose task. Audit false flags separately; do not call these
  positive gibberish examples merely because they are unfamiliar English.

Do not interpret fluent but nonsensical prose as a task this detector has
demonstrated it can solve. Confirm the annotation rubric with the author.

## Milestones and exit criteria

| Milestone | Work and deliverable | Exit criterion |
| --- | --- | --- |
| 1. Scope and sources | Research question, related-work matrix, dataset source/licensing inventory, annotation guide | Author reviews task definition; at least one useful real-text source can be obtained and reproduced within its terms |
| 2. Pilot | Approximately 600–1,000 examples; baseline adapters; initial label and runtime audit | Labels are meaningful, sources are traceable, runner works, and the proposed comparison answers an identifiable question |
| 3. Freeze protocol | Revise this plan after the pilot; pin versions, splits, metrics, model-selection budget and statistical precision target | A dated commit records the protocol before final test results are examined |
| 4. Full evaluation | Reproducible measurements, raw predictions, confidence intervals, subgroup results, plots and error analysis | All planned methods and failure cases are reported; deviations are documented; held-out data remain separate from tuning |
| 5. Reference application | Corpus-quality audit using the frozen methods, with a human audit of retained/rejected material | Demonstrates an actual analysis and its limits; no claim of downstream improvement without measuring it |
| 6. Manuscript and reproduction | Complete manuscript, bibliography, figures, environment instructions, release/archive preparation | Author verifies claims; a fresh-environment run reproduces results; an independent person's rerun is strongly preferred |
| 7. arXiv and journal | Compilable source bundle, submission metadata, chosen license/category, venue-specific files | Author reviews the final rendered document and submits; journal fees/waiver and declarations are resolved |
| 8. Discovery | Public article landing page, paper links and citation metadata; Scholar checks | Verify actual article availability and Scholar search presence, recording unresolved indexing issues accurately |

The pilot is a decision point. If the task is poorly defined or data access
is inadequate, revise the scope before scaling. pygarble does not have to win:
negative results, limitations and trade-offs remain reportable. Do not change
the question or suppress a baseline to manufacture a favorable comparison.

## Data and annotation

Select sources during milestone 1, recording source URLs, acquisition dates,
versions, attribution, permitted redistribution and hashes. Prefer public
corpora with clear terms; do not assume publicly viewable text is freely
redistributable. When redistribution is restricted, provide permitted
retrieval instructions and identifiers instead of copying the corpus.

Include ordinary and technical prose, genuinely observed corruption where
available, and controlled seeded corruptions of legitimate text. Report
natural and synthetic results separately. If only synthetic positives are
available, scope conclusions explicitly to those corruptions and discuss
whether the resulting contribution is sufficient for the target journal.
No LLM-generated labels are ground truth. Detector verdicts must not be used
as reference labels for evaluating those same detectors.

Keep, at minimum, each record's ID, text or retrieval pointer, source/document
ID, domain, length, source license reference, parent example, transformation
and seed when applicable, label, label rationale, annotator and review status.
Store ambiguous cases and exclusion reasons. Do not include private user
content or active credentials in a publication corpus.

Human reviewers label without seeing model predictions. Ideally two people
independently label an overlapping subset and resolve disagreements with a
record of the original decisions. If only the author is available, disclose
single-annotator limitations; do not describe AI checks as independent human
review. Use pilot disagreements to refine the guide before the final labels.

Reuse original labels only after mapping their definitions to this task.
Preserve original labels, mapped labels and mapping rationale separately.
For the pilot, propose a stratified audit of 100 examples across sources and
classes, plus ambiguous mappings; this is a workload proposal, not completed
annotation or an author commitment. Expand or revise the audit if it exposes
systematic label problems. A small audit does not certify every label.
Wikipedia membership alone is not a verified clean label, and the fact that
a generator altered text does not establish that it became unusable.

Existing `regression/benchmark_data.json`, `english_challenge.json`, golden
files and scan vectors are development/regression material. Do not relabel
their existing holdouts as unseen evidence after extensive development use.
Audit new examples for overlap with old datasets and known model resources.
Record unavoidable or uncertain overlap with bundled dictionaries and
character statistics rather than claiming complete training independence.

## Splits and freezing

- Plan source/document-grouped training, validation and test partitions;
  provisional proportions are 60/20/20, subject to pilot feasibility.
- Keep every original text and all its corruptions or near-duplicates in
  the same partition. Check exact and near-duplicate leakage before fitting.
- Use training data for learned baselines, validation for configuration and
  thresholds, and test data only for final evaluation.
- Include a held-out source/domain analysis with no target-domain threshold
  tuning. Avoid confounding one source with only positives or only negatives.
- The pilot remains development data. Final evaluation uses newly selected
  groups and a frozen manifest. Record seeds, hashes and exclusion decisions.
- Choose final sample sizes from the desired uncertainty and source diversity
  after the pilot, not from a desired headline score. Sparse subgroup results
  receive explicit uncertainty and no strong ranking claims.
- If final errors lead to code fixes or tuning, report the original outcome
  and use a new untouched evaluation set for confirmatory claims.

## Comparison protocol

Proposed methods, to be finalized after checking maintained implementations,
licenses, versions and supported Python environments:

| Method | Purpose |
| --- | --- |
| Always keep / simple length and character rules | Establish trivial and inexpensive reference points |
| Character entropy and dictionary-coverage heuristics | Test whether a simple local signal explains the benefit |
| Character-transition gibberish detector | Compare with a directly related approach, such as rrenaud/Gibberish-Detector |
| Character n-gram linear classifier | Include a lightweight trained baseline using only permitted training data |
| Frozen pygarble profiles | Compare default behavior and validation-calibrated behavior separately |
| Predeclared pygarble ablations | Identify which strategy groups affect errors and cost |

Use the same examples and comparable preprocessing. Document unavoidable
differences in vocabulary, pretrained resources and training requirements.
Keep development/calibration effort comparable and record search budgets.
Upstream defaults and retrained variants, if both used, get separate labels.
Run research dependencies in a separate environment; the package's base
installation remains dependency-free. No remote LLM inference is needed.

Primary operating point: select a threshold on validation data to maximize
recall with an empirical false-positive budget of 1%. Freeze it and report
both recall and the *actual* test false-positive rate. If validation cannot
meet the constraint, record that; do not silently report a fallback as success.
The test rate may exceed 1%; label that outcome honestly. A test-set ROC curve
may be descriptive but must not determine the deployment threshold.

Secondary outputs: confusion counts, precision/recall, PR curves, performance
by corruption subtype/domain/length, out-of-scope false flags, and abstention
coverage. State prevalence when reporting precision or F1. Predictions of
insufficient evidence remain distinguishable from confident clean decisions.
Count unflagged corrupt examples as misses for the automatic-removal task,
and separately report the workload of routing abstentions to human review.

Use paired, group-aware bootstrap intervals for method differences where
data share source documents or generated parents. Report the resampling unit,
seed and counts; do not treat derived variants as independent observations.
Use simple binomial intervals only when the independence assumptions fit.
Limit confirmatory comparisons; mark additional investigations exploratory.

## Cost and reference application

Measure on one documented CPU environment with pinned dependency versions.
Separate import/construction, cold execution and warmed inference. Specify
thread counts, batch/document size, timing repetitions and UTF-8 byte counts.
Report latency distributions and throughput at defined input lengths.
Measure peak process memory separately from timed runs; Python allocation
tracking is not a substitute for whole-process memory. Record training costs
for learned baselines separately. Do not extrapolate local timings into cloud
cost savings or production latency guarantees.

For the reference application, run the frozen screen over a permitted text
collection, retain reasons and offsets, and audit samples from both retained
and rejected strata. Use sampling weights if estimating full-corpus rates.
Report usable text lost, corrupted text retained and human-review workload.
Controlled injected corruption is a separate stress test. This must become
an actual completed analysis before it is described as research use.

## Planned repository artifacts

These locations are a proposed layout, not files already implemented:

```text
research/gibberish/
  README.md                 # one-command reproduction and scope
  protocol.md               # frozen study choices and deviations
  data/manifest.json        # provenance, hashes, split membership
  annotations/              # permitted labels, rubric, adjudication
  configs/                  # pinned detectors, thresholds and seeds
  baselines/                # independent method adapters
  run.py                    # predictions and run metadata
  analyze.py                # metrics and uncertainty from saved predictions
  results/                  # raw metrics/predictions where redistribution permits
paper/
  paper.md                  # manuscript source, adapted to chosen venue
  paper.bib                 # verified references
  figures/                  # generated from saved results
```

Before implementing, inspect existing interfaces and callers. Reuse suitable
metric logic only after checking its statistical assumptions. The current
`regression/evaluate.py` does not by itself provide this study's independent
data, baseline training, source-grouped uncertainty or publication protocol.
Do not replace engineering regression tests with the research harness.

## Division of work and submission handoff

The assistant can inventory sources and related work, implement the harness,
run permitted local experiments, investigate errors, produce plots and draft
the paper, check references, build submission files and prepare discovery
metadata. Progress should be reported by completed milestone and evidence,
with important unresolved decisions made visible.

The human author defines and reviews the scientific claims, supplies or
reviews labels, confirms data permissions and disclosures, approves authorship,
funding and license choices, and reads the final manuscript. A second human
reviewer/reproducer is desirable; no such collaborator is assumed to exist.
Exact Claude/model versions should be recovered where available, not invented.

arXiv expects author self-submission. Prepare the complete reviewed bundle
first, then guide the author through category/license selection, compilation
preview and final submission. Account verification does not remove moderation
or all category-specific requirements. TeX-generated papers need their source
bundle, not just an exported PDF. No submission is made during planning.

Journal submission follows the chosen venue's template and declarations.
Complete the materials before seeking final publication approval. In a JOSS
review, author/editor/reviewer conversations must be human-written except for
translation assistance under its current policy; assistant technical work
on the repository remains separate. Other venues' policies must be checked
when selected. This is a policy-specific handoff, not a claim that automated
submission or editorial correspondence has been authorized.

## Scholar discovery after release

Publish a full paper with consistent author/title metadata and an accessible
abstract/PDF. Prefer the arXiv/journal record and link it from a publications
page and the repository. If hosting an author copy, provide citation metadata
and a searchable PDF using accurate preprint or publication dates. Add the
journal DOI/reference to the preprint after acceptance where appropriate.

Search Scholar by exact title after public release; distinguish a profile
entry from a globally indexed article. Keep unresolved indexing status visible.
The existing Search Console verification and docs sitemap help ordinary web
discovery, but are not Scholar acceptance or indexing confirmation.

## Sources checked for this plan

- [JORS scope and review](https://openresearchsoftware.metajnl.com/about)
- [JORS submission requirements and fees](https://openresearchsoftware.metajnl.com/about/submissions)
- [JOSS requirements and co-publication](https://joss.readthedocs.io/en/latest/submitting.html)
- [arXiv submission overview](https://info.arxiv.org/help/submit/index.html)
- [Google Scholar inclusion](https://scholar.google.com/intl/en/scholar/inclusion.html)
- [Google Scholar profiles](https://scholar.google.com/intl/en/scholar/citations.html)
