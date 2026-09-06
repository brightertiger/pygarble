# pygarble

**Deterministic, lightweight gibberish detection for English text.**

A zero-dependency Python library using fixed character models, English word patterns, keyboard paths, and encoding checks. Inference requires no network, training, or model downloads.

**English-specific scope:** language strategies judge whether text resembles English. Hindi and other non-English text may receive high gibberish scores even when meaningful in their own language; this is expected behavior. This library is not a multilingual validator or a language identifier. Non-English text written with Latin letters may still pass. A passing result does not prove that a sentence makes semantic sense.

This branch contains the upcoming 0.9.0 changes. Install the checkout with `pip install -e .` to use them before release.

## Quick start

```bash
pip install pygarble
```

```python
from pygarble import EnsembleDetector

detector = EnsembleDetector()
detector.predict("Hello world")       # False
detector.predict("asdfghjkl")         # True
detector.predict("qwerty")            # True
detector.predict("नमस्ते दुनिया")      # True: expected for English checks
detector.predict("CafÃ© au lait")     # True: encoding corruption
detector.predict("hello\x00world")    # True: control character

detector.predict(["Hello world", "qxzjkwpv"])  # [False, True]
detector.score("qxzjkwpv")            # Heuristic score in [0, 1]
```

`score()` and `predict_proba()` return the same **heuristic score, not a calibrated probability**. Names, uncommon English words, spelling mistakes, and technical identifiers can still be misclassified. Use domain vocabulary and choose a profile appropriate to the field.

## Profiles

| Profile | Behavior |
|---|---|
| `english` (default) | Union of Markov, likelihood ratio, word anomaly, mojibake, keyboard adjacency, and control-character checks |
| `english_extended` | Also uses pattern matching, localized anomaly detection, and repetition; more aggressive and more prone to false positives |
| `legacy` | The former three-member strategy set: Markov, likelihood ratio, word anomaly; uses current preprocessing and correctness fixes |
| `corruption` | Mojibake and control artifacts, independent of English plausibility |
| `spoofing` | Unicode script/confusable heuristic; not a complete phishing detector or full Unicode UTS #39 implementation |

```python
from pygarble import EnsembleDetector, GarbleDetector, Strategy

# English scoring may flag Hindi; a corruption-only profile checks encoding.
EnsembleDetector(profile="corruption").predict("नमस्ते दुनिया")  # False

# Find severe garbage embedded in a longer paragraph.
detector = EnsembleDetector(profile="english_extended")

# Use any strategy independently.
keyboard = GarbleDetector(
    Strategy.KEYBOARD_ADJACENCY, keyboard_layout="azerty"
)
keyboard.predict("azerty")  # True

# Caller-owned vocabulary for the shared English character models.
detector = EnsembleDetector(allowlist=["syzygy", "myproductname"])
detector.predict("syzygy")  # False
```

The allowlist applies to Markov, likelihood ratio, word anomaly, and local anomaly scoring. It does not override keyboard, pattern, or raw encoding/control evidence. URLs, paths, digit-containing tokens, and camel-case identifiers are excluded from shared English word scoring; other strategies may still flag them. Structured data and passwords are not universally equivalent to gibberish.

## Explanations

```python
from dataclasses import asdict
from pygarble import GarbleDetector, Strategy

text = "Please review qxzjkwpvm before delivery."
result = GarbleDetector(Strategy.LOCAL_ANOMALY).analyze(text)

result.garbled       # Decision
result.score         # Heuristic score
result.status        # "clean", "garbled", or "insufficient_evidence"
result.signals       # Per-strategy score, applicability, reason, and spans
result.model_version # Version of the inference contract

for span in result.spans:
    print(text[span.start:span.end], span.reason)

record = asdict(result)  # Can be serialized with json.dumps()
```

Offsets refer to the **original Python string**, even when accents or ligatures are folded for English scoring. `analyze()` evaluates every selected strategy; `predict()` can stop early for `any` and `all`. Empty or wholly inapplicable input returns `False` with `insufficient_evidence`; that does not certify meaningful English.

## Configuration and voting

```python
from pygarble import EnsembleDetector, Strategy

detector = EnsembleDetector(
    strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_ANOMALY],
    voting="weighted",
    weights=[0.7, 0.3],
    threshold=0.5,
    strategy_kwargs={
        Strategy.MARKOV_CHAIN: {"min_length": 4},
        Strategy.WORD_ANOMALY: {"min_word_length": 6},
    },
    max_input_length=100_000,
)
```

- Choose `profile` or `strategies`, not both. Profiles default to `any`; custom strategy lists default to `majority`.
- `any` uses the maximum applicable score; `all` uses the minimum. `average` and `weighted` aggregate applicable scores. Zero-weight members do not participate in weighted decisions.
- `majority` requires strictly more than half the applicable members to cross `threshold`. Its reported score is their arithmetic mean, so thresholding that mean need not reproduce the majority decision.
- Abstaining members do not dilute votes. Weights must be finite, nonnegative, and not all zero.
- `GarbleDetector(..., strategy_kwargs={...})` can configure a strategy's own `threshold` independently of the detector's decision threshold. Legacy `**kwargs` remain supported; unknown settings now emit a deprecation warning.
- `WORD_LOOKUP.unknown_threshold` now controls the unknown-word fraction mapped to the decision boundary; its default 0.5 preserves the previous score mapping.

Both detector classes provide `predict`, `score`, `predict_proba`, and `analyze` for a string or list of strings. A batch is fully type-validated before processing. Invalid batch entries raise `TypeError`; invalid numeric configuration raises `ValueError`.

`threads` is an optional positive integer. Serial execution is recommended for short strings; threads are not a guaranteed speedup. `max_input_length` raises an error when a scalar or batch member exceeds the limit. `timeout_per_text` bounds waits for threaded results, **not total execution time**: Python worker threads cannot be killed, and executor shutdown may still wait. Timeouts and worker failures propagate rather than returning a clean classification.

Length alone no longer forces every strategy to score 1.0. The explicit legacy `max_string_length` option retains that policy when requested; use `max_input_length` for resource limits.

## Strategies

The [generated strategy reference](docs/strategies.rst) lists all 28 strategies and their accepted settings.

New in 0.9.0:

- `CONTROL_CHARACTERS`: NUL and unexpected controls, replacement characters, lone surrogates, and excessive combining-mark runs. Normal tabs, newlines, accents, and emoji joiners are preserved.
- `LOCAL_ANOMALY`: severe unknown tokens and bounded token windows, with original-text offsets. Useful when whole-document averages dilute localized corruption.
- Keyboard adjacency now supports QWERTY, AZERTY, and QWERTZ, includes digit-row neighbors, and requires substantial path coverage.
- Repetition counting is linear in token count, with additional phrase cycles up to eight words.

A conditional-trigram experiment is retained in `regression/trigram_experiment.py`. Its 78,732-byte candidate table did not add coverage beyond the specialist combination at the conservative development threshold, so it is **not shipped or loaded at runtime**. Global character entropy and character likelihood cannot reliably detect grammatical semantic nonsense.

## Evaluation

Run the legacy comparison and the reviewed English evaluation separately:

```bash
python regression/benchmark.py
python regression/evaluate.py --split all --output /tmp/english-results.json
# Add --details for per-category metrics and individual errors.
```

The original 1,644-row benchmark is retained for historical comparisons. It includes duplicates and labels some valid structured content as garbled. `label_overrides.json` documents the English policy corrections; the reviewed corpus deduplicates to 1,628 texts. The 132-case authored challenge set is separated by family into 68 development and 64 holdout cases, with a checksum to expose accidental edits.

These are engineering datasets, **not representative production precision estimates**. Non-English Latin-script text is deliberately labeled against the English target, even though character models may accept it. A random generator can produce a real word, and zero observed false positives does not guarantee perfect precision. The evaluator reports confusion counts, applicability coverage, per-category results with `--details`, and a 95% interval for false-positive rate; metrics undefined for a single-class slice are `null`.

See [the recorded evaluation](regression/english_results.json) and [implementation notes](docs/implementation.md) for measured results, limitations, and the trigram decision.

## Development and reproducibility

Requires Python 3.8+. Runtime dependencies: none.

```bash
pip install -e ".[dev]"
pytest -q
black --check pygarble tests scripts regression
isort --check-only pygarble tests scripts regression
flake8 pygarble tests scripts regression
mypy pygarble

# Downloads the pinned source and verifies its checksum and generated files.
python scripts/generate_data.py --check
# An offline rebuild can use a previously downloaded source file.
python scripts/generate_data.py --source /path/to/count_1w.txt --check
```

The data manifest records source and artifact checksums. Curated word exclusions are versioned in `scripts/data_curation.json`. Runtime resources load lazily; using a control-character specialist does not load the English dictionary. Shared features are local to each request, without an unbounded cache of user text.

Inference is deterministic for fixed package, settings, and Python/Unicode data versions. Unicode normalization/property tables may differ across Python releases. CI checks supported Python versions, hashes, formatting, typing, documentation builds, and installation of the wheel without runtime dependencies.

## License

Library code: MIT. Data source attribution and provenance are recorded separately in `scripts/data_curation.json`.
