> Archived planning document from the 0.9.0 development cycle. Numbers and blockers here are historical.

**English detection implementation — version 0.9.0**

The user clarified the audit's scope: Hindi and other non-English text scoring
as gibberish is expected. English scoring retains that behavior and now reports
`outside_english_alphabet` where applicable. It does not attempt multilingual
abstention or guarantee identification of non-English Latin-script text.

The implementation keeps zero runtime dependencies and preserves public imports
from `pygarble` and `pygarble.core`. It separates registration, single-detector
orchestration, ensembles, validation, results, preprocessing, and shared scoring.
Models and dictionaries load lazily. Shared features stay within each request;
no global user-text cache was added. Existing specialized strategies remain
independently usable and are migrated through a small compatibility interface.

**Completed changes**

- Default English profile: Markov, LLR, word anomaly, mojibake, keyboard
  adjacency, and control characters. The former three-member set is available as
  `legacy`, using current preprocessing and correctness fixes.
- Extended English profile: adds pattern matching, local anomaly, and repetition.
  It deliberately trades more false positives for coverage. Independent
  `corruption` and `spoofing` profiles do not inherit English-language judgments.
- Immutable explanations include applicability, score, reasons, model version,
  and original-string offsets. English models use shared tokenization and
  likelihood primitives; raw corruption checks retain the unnormalized input.
- Scalar/batch/thread validation is consistent. Nonfinite weights and invalid
  numeric settings are rejected. Timeouts/errors propagate. Threaded submission
  is bounded. `max_input_length` is an explicit resource limit; the universal
  implicit 1,000-character decision is gone. Its opt-in compatibility option
  remains available.
- Per-member `strategy_kwargs` isolates settings, including strategy thresholds
  from detector thresholds. Previously ignored word-lookup thresholds now affect
  scoring. Unknown legacy options warn before a future removal.
- Domain allowlists apply to shared English character scorers without suppressing
  raw corruption. Structured tokens have a common exclusion policy for those
  scorers. Other specialists still inspect their own relevant evidence.
- Keyboard adjacency includes digit-row neighbors, QWERTY/AZERTY/QWERTZ, and a
  minimum path-coverage guard. Repetition uses linear counting and bounded phrase
  cycles. Local anomalies use severe-token evidence and bounded token windows.
- Source checksums, the 670 word exclusions, and artifact hashes are versioned.
  Rebuilding reproduces all 49,330 words, 729 bigrams, and 2,000 common trigrams.
  Generated resource files remain exempt from source-formatting rules.
- Quality CI runs formatting, import ordering, lint, typing, reproducible data,
  generated strategy documentation, and wheel installation without dependencies.
  PR documentation builds are separated from protected Pages deployment; Pages
  environment restrictions are preserved. No release/deployment was performed.

**Evaluation decisions**

Original labels remain in `benchmark_data.json`. Reviewed English policy
corrections live in `label_overrides.json`; deduplication occurs in the new
runner. Clear keyboard mashing remains positive even when the historical category
was named `password_like`. Short ambiguous single letters are not forced positive.
Meaningful non-English examples remain targets for the English profile.

The new challenge set contains 132 authored cases, split by family into 68
for development and 64 for holdout before evaluating candidates. It is a small
engineering challenge set, not an independently sampled production dataset.
Parameters/default membership were selected using development/historical data;
no parameter was adjusted to remove the final holdout misses. Some vocabulary and
patterns naturally overlap language-model training; source/family separation
cannot establish complete independence from the old dictionary corpus.

| Evaluation | Detector | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|
| Original 1,644 rows | v0.8 default, audit baseline | 654 | 5 | 875 | 110 |
| Original 1,644 rows | v0.9 default English | 669 | 3 | 877 | 95 |
| Reviewed 1,628 texts | Current three-member set | 628 | 4 | 913 | 83 |
| Reviewed 1,628 texts | Default English | 652 | 4 | 913 | 59 |
| Reviewed 1,628 texts | Extended English | 659 | 7 | 910 | 52 |
| Development challenge | Default English | 27 | 2 | 39 | 0 |
| Holdout challenge | Current three-member set | 17 | 0 | 36 | 11 |
| Holdout challenge | Default English | 25 | 0 | 36 | 3 |
| Holdout challenge | Extended English | 27 | 0 | 36 | 1 |

The original-label default comparison corresponds to precision 99.55% and
recall 87.57%, versus 99.24% and 85.60% at the audit baseline. Those historical
labels conflate gibberish with some structured content, so these figures are
comparison metrics, not a general guarantee. Reviewed-label figures must not
be compared directly with original-label figures.

Development still flags rare words such as `syzygy` and `chutzpah`. The default
holdout misses are `1qaz2wsx3edc` and two repeated multiword cycles. The extended
profile catches the cycles but still misses that discontinuous keyboard walk.
The holdout has only 36 negative examples: its zero observed false positives
still gives a Wilson 95% upper bound of about 9.64% on false-positive rate.
Larger independent evaluation remains necessary for deployment-level claims.

Full reproducible aggregates, coverage, and uncertainty are in
`regression/english_results.json`. Use `--details` to obtain categories and errors.
The source-comparison report now pools confusion counts instead of averaging F1
across sources that sometimes contain only one class.

**Conditional trigram experiment**

The experiment builds 19,683 little-endian float32 entries (78,732 bytes) from
the pinned source, using add-one smoothing and 80% conditional-trigram / 20%
bigram interpolation. It evaluated thresholds from -3.0 to -5.0 on development
and reviewed historical examples.

At -4.5, the first tested threshold with zero standalone false positives on the
development challenge, the model added no true positives over the new default
on either development or reviewed historical data. Less conservative thresholds
change that tradeoff; the result does not imply trigram models are inherently
ineffective. The candidate was rejected from the runtime package because it did
not justify extra data and execution at the selected precision constraint.
The experimental generator and `regression/trigram_results.json` are retained.

```bash
python regression/trigram_experiment.py \
  --source /path/to/count_1w.txt \
  --output /tmp/pygarble-trigram.bin \
  --report /tmp/trigram-results.json
```

**Performance and remaining research**

On CPython 3.12.2 / arm64, median warm batch times across five runs for 1,644
strings were ~34 ms for the three-member set, ~50 ms for the six-member default,
and ~73 ms for extended English. The audit measured ~37 ms for the former
default; this is a local comparison across runs, not a portable latency promise.
Four threads were slower than serial execution for these short samples.

Repetition counting for 1,000 / 2,000 / 4,000 distinct tokens took approximately
0.19 / 0.36 / 0.71 ms, versus the audit's ~8 / 31 / 136 ms. A fresh process using
only the control specialist did not load the dictionary. Import-memory results
include interpreter/module infrastructure and tracemalloc overhead, not merely
the size of the detector. Reproduce with `python regression/performance.py`.

Character-class models, general word-order anomaly detection, richer encoding
round-trip inference, and full Unicode property tables remain research
candidates. They were not added merely to increase strategy count. No general
semantic-nonsense or multilingual-understanding claim is made. Cross-version
Unicode differences and very short strings remain limitations.
