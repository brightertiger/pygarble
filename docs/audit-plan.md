**Repository audit and deterministic detection plan — 6 September 2026**

Audited baseline: `a3db3a2`, version 0.8.0. This document proposes future
implementation; it does not change detector behavior. Findings come from source
inspection, the existing tests, rerunning all benchmark strategies, targeted
reproductions, and the primary sources linked below.

**Recommendation**

Keep the dependency-free character-model approach. First fix evaluation labels,
language applicability, shared preprocessing, and API contracts. Then measure the
incremental value of existing specialists before introducing a small conditional
trigram model. A compact set of complementary signals should be the product's
center; 26 public strategies need not mean 26 independent votes.

Working scope: detect malformed text and implausible character sequences, with
English language scoring and language-independent corruption checks. Valid code,
identifiers, other languages, and semantic nonsense need explicit policies.
Whether a valid identifier belongs in a prose field is a separate decision.

**Basic tenets worth preserving**

- Zero runtime dependencies, network calls, model downloads, or fitting.
- Deterministic inference from versioned rules and embedded tables. Offline
  estimation of counts and thresholds is compatible with deterministic inference.
- Precision first: names, uncommon vocabulary, spelling mistakes, multilingual
  content, and technical text deserve conservative handling.
- Small, inspectable features: dictionary membership, character likelihood,
  phonotactics, keyboard paths, repetition, and encoding artifacts.
- Simple scalar and batch APIs, with strategies usable independently.
- Explicit limits: plausible invented words and grammatical semantic nonsense
  cannot reliably be rejected using character plausibility alone.

The current architecture has a useful strategy interface and separate files.
`core.py` combines registration, input handling, threading, thresholding, and
ensemble aggregation. `BaseStrategy` combines interface mechanics with English
dictionary policy, normalization, and a universal long-token rule. Embedded data
contains 49,330 words, 729 bigram scores, and 2,000 common trigrams. The default
ensemble uses Markov, likelihood ratio, and word anomaly with union voting.

**Validation and baseline**

Local environment: CPython 3.12.2 on macOS. These are local results, not a claim
that every supported Python version was tested during this audit.

| Check | Observed result |
|---|---|
| `python3 -m pytest -q` | 515 passed in 0.42 seconds |
| Benchmark via `load_test_cases()` and `run_benchmark()` | All 26 strategies and default ensemble evaluated |
| Default ensemble | TP 654, FP 5, TN 875, FN 110 |
| Default precision / recall / F1 | 99.24% / 85.60% / 91.92% |
| Markov precision / recall | 99.23% / 84.29% |
| Word lookup precision / recall | 54.95% / 91.49% |
| Black 24.8.0, `--check pygarble tests` | Failed: 35 files would be reformatted |
| mypy 1.11.2, `pygarble` | Failed: 37 errors in 21 files |
| flake8 7.0.0, `pygarble tests --count --statistics` | Failed: 5,076 findings, including 5,069 line-length findings; generated tables dominate |

The benchmark has 1,644 rows but 1,628 distinct strings. Source counts are 420
internal, 600 external dictionary, 599 externally generated, and 25 externally
curated. No identical string had conflicting labels. The default's recall on
the internal subset is only 95/165 = 57.58%; the generated subset contributes
559 true positives. The headline result therefore hides major category gaps.

To reproduce the benchmark without rewriting tracked result files:

```python
from regression.benchmark import load_test_cases, run_benchmark

cases = load_test_cases("regression/benchmark_data.json")
results = run_benchmark(cases)
print({k: v for k, v in results["ensemble"].items() if k != "predictions"})
```

**Prioritized findings**

| Priority | Finding and evidence | Proposed improvement |
|---|---|---|
| P0 | Benchmark labels mix corruption with valid structured content. `SELECT * FROM users WHERE id = 1;`, `3.14159265359`, and ordinary URLs are labeled garbled. `wishy` is labeled garbage because it came from the random generator. See `regression/benchmark_data.json`. | Define categories and profile-specific expectations; manually adjudicate ambiguous generated samples. Preserve the original benchmark as a legacy comparison. |
| P0 | Unsupported scripts become high-confidence gibberish. The default flags `नमस्ते दुनिया`, `你好世界`, `Привет мир`, and `مرحبا بالعالم`, each scoring approximately 1.0. `BaseStrategy._novel_words()` retains Unicode alphabetic characters, but Markov's table covers ASCII. | Restrict English scoring to supported spans; report unsupported language or insufficient evidence. A script gate cannot identify non-English text written in Latin script, so document that residual limitation. |
| P0 | Ensemble batch validation is inconsistent. `predict([None])`, `predict([0])`, and `predict([False])` return `[False]`; `predict([123])` raises `AttributeError`. `predict_proba` raises `TypeError` on these inputs. See `core.py:255–321`. | Validate the entire batch before work begins, in both APIs and both execution modes. |
| P1 | The base class forces every strategy to return 1.0 for a non-URL token over 1,000 characters. Even standalone mojibake and script checks flag `"你" * 1001`. See `strategies/base.py:24–59`. | Move length policy out of the base class. Resource limits should produce an explicit error or abstention; classify repetition with its own evidence. |
| P1 | Nonfinite ensemble weights are accepted. A one-member weighted ensemble with `[float("nan")]` or `[float("inf")]` produces a NaN score and a false prediction on `qxzjkwp`. | Validate finite numeric weights and strategy parameters; reject invalid worker counts and unknown strategy values with stable errors. |
| P1 | `WordLookupStrategy.unknown_threshold` is stored and validated but never used. Values 0.1 and 0.9 give the same result on `hello qxzjkwp`. The newer strategies also accept largely unchecked kwargs. | Introduce typed strategy configuration and explicit threshold semantics. Deprecate or implement ineffective settings; reject misspelled options after a compatibility period. |
| P1 | Default coverage excludes existing corruption specialists: `CafÃ© au lait`, `Hello � world`, and `hello\x00world` pass. `qwerty` also passes. | Evaluate specialist inclusion by profile and add a bounded control-character detector. Avoid advertising all specialist coverage for the default. |
| P1 | Normalization and tokenization differ: whitespace splitting with alphabetic concatenation versus ASCII regex extraction; digits/acronyms/URLs are skipped by some strategies but scored by others. | Build shared token features with original spans and explicit token kinds. Retain raw Unicode for corruption detection and a separate folded view for English scoring. |
| P1 | Repetition counts each distinct word using `words.count`, giving O(words × unique words). A helper timing grew from ~8 ms at 1,000 unique tokens to ~136 ms at 4,000. | Use `Counter` for linear counting and bounded cycle lengths for phrase repetition. |
| P1 | The benchmark has no enforced untouched holdout. Defaults explicitly refer to tuning on this same dataset. Its report calls external data unbiased and averages F1 across sources that can contain only one class. | Split by source and generation family before tuning; report class counts, false-positive rate, category recall, abstention coverage, and uncertainty. Pool appropriate confusion counts instead of averaging incompatible F1 scores. |
| P2 | Markov, LLR, word anomaly, and entropy reuse the same bigram table. Against a uniform null, mean LLR is just mean log likelihood plus `log(27)` for the same extracted bigrams. | Share a scoring primitive and group correlated features. Current strategies differ in token selection and aggregation, but LLR is not independent linguistic evidence. |
| P2 | `applicable()` defaults to true even when some scorers cannot judge. LLR extracts bigrams in applicability, scoring, and again in `_average_llr`; word anomaly retokenizes too. | Return applicability and score from one evaluation, with counts and reason codes. Prepare features once per input and compute optional features lazily. |
| P2 | Scores are hand-mapped values, not demonstrated calibrated probabilities. Majority `predict()` uses vote counts while its `predict_proba()` averages scores, so thresholding the latter need not reproduce the former. | Add an explicit `score`/analysis contract; retain `predict_proba` for compatibility with clear semantics. Test each voting mode against its documented decision rule. |
| P2 | Every strategy is eagerly imported; importing data loads the ~563 KB word-list source even for specialists. Thread pools add overhead and the single-detector timeout can substitute a clean result while still waiting for shutdown. | Measure import time/memory before changing representation, load only required resources, consolidate batch handling, and report processing failures explicitly. |
| P2 | Data generation downloads an unpinned source and takes the top 50,000 alphabetic entries. It does not encode the 670-entry cleanup described in the changelog. | Persist source hashes, curated exclusions/additions, parameters, stable ordering, table versions, and rebuild checks. Verify a rebuild reproduces the committed 49,330-word artifact. |
| P2 | CI has commented-out lint/type jobs; pre-commit only checks YAML, large files, and merge conflicts. `test.yml` unnecessarily installs NumPy. Sphinx docs advertise 24 strategies and stale metrics, including 92.7% word-lookup precision. | Enable checks after separating generated-data style policy, remove stale development setup, and generate strategy/default documentation from one registry. |

The source-data manifest should distinguish code licensing from data provenance.
The upstream [Norvig page](https://www.norvig.com/ngrams/) explicitly grants MIT
terms to its code and identifies the data's origin; that page alone does not
substantiate this repository's blanket statement that all derived data is MIT.
Record verified data terms rather than copying that assertion into new artifacts.

**Existing strategies: measure before expanding**

Adding each strategy individually to the default with union voting gave the
following additional detections on the current labels. These gains overlap;
they cannot be summed and do not predict performance on a fresh dataset.

| Candidate | Additional true positives | Additional false positives |
|---|---:|---:|
| Pattern matching | 20 | 0 |
| Mojibake | 12 | 0 |
| Keyboard adjacency | 9 | 0 |
| Consonant sequence | 3 | 0 |
| N-gram frequency | 13 | 13 |
| Word lookup | 61 | 570 |

First evaluate pattern matching, mojibake, and keyboard adjacency on corrected
labels and new data. Keep word lookup as supporting evidence, especially for
names and technical vocabulary. Zipf produced no positives on this benchmark;
that does not prove it is useless for long documents. Evaluate specialists on
their applicable slices before deprecating them.

**Research-backed candidates and extensions**

These are implementation hypotheses, not measured improvements. None requires
a runtime ML framework or an external service.

| Order | Technique | Implementation and cost | Main qualification |
|---|---|---|---|
| 1 | Conditional character trigrams with bigram backoff | Estimate `P(c_i | c_(i-2), c_(i-1))` offline with smoothing and boundaries. Freeze interpolated/backoff scores in an array. For 27 symbols, 27³ = 19,683 entries: ~77 KiB as float32, excluding metadata and packaging overhead. O(n) inference. | Existing common-trigram membership is not a conditional trigram model. Tune by token length and protect rare valid words with evidence, not universal dictionary exemptions. |
| 1 | Local corruption scoring | Extend word anomaly with worst-token score, suspicious-character fraction, and fixed-size rolling windows using prefix sums. Return offsets. O(n) for a fixed set of window sizes. | A maximum score alone overflags long documents; require sufficient bad-span length and length-aware thresholds. |
| 1 | Keyboard paths | Extend existing adjacency scoring with physical coordinates, direction changes, coverage, shifted keys, and digits; package layouts as data. O(n × layouts), with a small configured layout count. | Ordinary words can be keyboard paths. Require long coverage plus weak linguistic evidence; test `typewriter`, `power`, names, and identifiers. |
| 1 | Encoding and control artifacts | Extend mojibake with contextual signatures and a bounded number of strict Latin-1/Windows-1252 → UTF-8 round-trip hypotheses. Add NUL/control density and anomalous combining-mark runs. O(n) for a fixed candidate set. | Successful decoding alone is insufficient: require recognizable corruption evidence and reduced badness. Preserve tabs/newlines and legitimate combining marks, emoji joiners, and Indic/Arabic text. |
| 2 | Generalized periodicity | Extend repetition to token cycles of 1–8 words and bounded character periods. Use linear frequency counts and coverage thresholds. | Headers, tables, laughter, poetry, and emphasis are hard negatives. Avoid unconstrained backtracking or all-substring searches. |
| 2 | Character-class transition model | A tiny fixed model over lower/upper/digit/punctuation classes can complement alphabetic scoring on inputs such as `1a2b3c4d5e6f7g8h`. O(n), constant table size. | Identifier/profile recognition must come first. Product codes, UUIDs, versions, and measurements are valid structured tokens. |
| 3 | Word-order anomaly for long prose | Extend existing collocation/function-word features using a small frozen frequent-word/class bigram table with backoff. | Experimental, prose-only, minimum-length gated. Lists, headlines, and grammatical semantic nonsense defeat simple assumptions; do not promise general nonsense detection. |

The [original Markov gibberish detector](https://github.com/rrenaud/Gibberish-Detector)
demonstrates the compact 27-symbol bigram approach already used here. Moving to
conditional trigrams is our proposed extension, with acceptance dependent on
held-out results.

[Baldwin and Lui's language-identification study](https://aclanthology.org/N10-1027.pdf)
evaluates character/byte n-gram methods and shows that shorter documents and
broader language sets make classification harder. This supports length-stratified
evaluation and conservative applicability; it is not direct evidence that a new
pygarble gibberish strategy will improve recall.

[Dropbox's zxcvbn matching implementation](https://github.com/dropbox/zxcvbn/blob/master/src/matching.coffee)
provides concrete examples of spatial keyboard matching, repetitions, sequences,
and multiple layouts. Reuse those ideas selectively; password predictability and
gibberish are different targets.

[ftfy's documented badness heuristics](https://ftfy.readthedocs.io/en/latest/heuristic.html)
use contextual character sequences to identify likely mojibake and describe
limitations of partial decoding. A small detection-only implementation fits this
repository; adopting the whole repair package is unnecessary for the core goal.

[Unicode UTS #39](https://www.unicode.org/reports/tr39/)
defines script/confusable mechanisms using Script_Extensions and compatible
writing systems. Replace character-name prefix guesses with versioned property
data if stronger Unicode support is implemented. Keep spoofing as a separate
reason/profile: mixed scripts are not automatically gibberish. Do not claim full
UTS #39 conformance for a hand-picked subset.

Avoid prioritizing more global entropy/vowel thresholds, compression on short
strings, dictionary rejection alone, or another uniform-null bigram score.
They add little independent evidence or have obvious short-text/rare-word limits.

**Target structure and API**

Keep the architecture small, with a stable compatibility facade in `core.py`:

```text
pygarble/
  core.py                 public imports and compatibility
  detector.py             input contract and single-detector orchestration
  ensemble.py             profiles and aggregation
  analysis.py             immutable result, reasons, applicability, spans
  preprocessing.py        raw/folded views, token kinds, shared features
  registry.py             strategy metadata, factories, documented parameters
  scoring.py              shared character likelihood and bounded mappings
  strategies/             existing detectors, migrated incrementally
  data/                   versioned tables, provenance, optional resources
```

Names are illustrative. Split files only as responsibilities move; avoid a
framework rewrite. Shared features should be local to a request, with lazy
evaluation and no unbounded cache of user text. Frozen result/config types can
use standard-library dataclasses. Preserve original offsets across normalization.

Proposed `analyze(text)` returns the decision, heuristic score, applicable
signals, reasons, suspicious spans, profile/model version, and an explicit
insufficient-evidence status. The bool adapter returns false for abstention and
documents that this is not certification of meaningful text. Preserve existing
`predict`/`predict_proba` entry points during migration.

Profiles should select both policy and strategy configuration: English prose,
short free text, structured mixed content, encoding corruption, and identifier
spoofing. Start with only profiles backed by distinct tests. Add a caller-owned
allowlist for domain vocabulary, with documented scope; it must not suppress raw
encoding/control evidence. Per-strategy configuration replaces shared ambiguous
kwargs, including collisions such as `min_word_length`.

**Implementation sequence and acceptance gates**

1. **Evaluation contract and corpus, before changing defaults.** Add typed labels
   for clean prose, lexical gibberish, corruption, structured content, unsupported
   language, spoofing, and ambiguous/semantic cases. Review current labels and
   duplicates. Create source/family-separated training, development, and frozen
   holdout splits. Include names, rare words, typos, code, URLs, multilingual text,
   short strings, keyboard walks, mixed digits, and localized corruption. A
   generator seed is provenance, not a label guarantee. Gate: audited labels,
   reproducible split manifests, per-category confusion counts and coverage.
2. **Correctness fixes in a separate PR.** Fix batch validation, finite parameters,
   unused settings, long-token policy, and unsupported-script handling. Add
   targeted regressions for the reproductions above. Gate: 515 legacy tests plus
   justified updated policy expectations; scalar/batch/thread parity, finite
   bounded scores, and explicit abstention semantics.
3. **Shared features and configuration.** Extract preprocessing, one-pass
   evaluation, shared likelihood, typed config, and result explanations. Retain
   public enums and compatibility imports. Gate: no unexplained prediction
   changes on the frozen legacy corpus; applicability and span tests cover
   accents, apostrophes, hyphens, case, token kinds, and mixed scripts.
4. **Lightweight execution.** Replace quadratic repetition counting, remove
   duplicate extraction, measure lazy imports, and unify batch execution/error
   handling. On this machine, median default batch time over five runs was
   ~37 ms serial versus ~45 ms with four threads for 1,644 samples. This is a
   local microbenchmark, not a throughput promise. Gate: length-scaling checks,
   import/peak-memory measurements, p50/p95 timing, scalar/batch equivalence,
   and no clean substitutions on processing failure.
5. **Specialist profiles and localized detection.** Evaluate existing pattern,
   keyboard, mojibake, and repetition signals first; then implement bounded
   control detection and local aggregation. Measure each addition and the
   combined set, accounting for overlapping errors. Gate: incremental recall at
   matched false-positive rate on development data, then final holdout reporting.
6. **Conditional trigram experiment.** Build reproducible smoothed tables,
   compare against bigram-only and existing common-trigram baselines, and sweep
   length-specific thresholds on development data. Gate: useful incremental
   detection within a provisional ≤100 KiB raw table budget and no material
   latency regression on the reference environment. Ship only if the extra
   coverage justifies the cost; preserve the old profile for comparison.
7. **Release quality and documentation.** Restore lint/type checks with explicit
   generated-data exclusions, align package/dev configuration, remove unused
   NumPy installation, test the installed wheel, and refresh Sphinx/README
   metrics from versioned outputs. Gate: CI passes on the declared Python matrix,
   artifact rebuilds match checksums, and score/probability limitations are clear.

Run comparisons at fixed false-positive budgets such as 0.1%, 0.5%, and 1%,
where sample size permits meaningful estimates. Publish uncertainty and
per-profile results; zero observed false positives is not a universal guarantee.
The existing 880 negative rows cannot substantiate very small error-rate claims.
Use several thousand diverse clean holdout examples before making a strong
precision claim, and match deployment class prevalence when reporting precision.
If holdout findings lead to retuning, create a new holdout for the next claim.

Determinism checks should cover repeated processes and hash seeds, batch ordering,
threads, and supported Python versions. Record Unicode database versions; runtime
normalization/property data can change between Python releases. If exact
cross-version decisions are required, pin the relevant data/normalization
contract and test threshold-boundary cases.

Each implementation step should use its own feature/fix PR with the relevant
validation evidence. This audit PR is ready for review; implementation and merging
are separate subsequent actions.
