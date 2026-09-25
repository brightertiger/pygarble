# pygarble

Deterministic, lightweight gibberish detection for **English text**, for Python 3.8 and later. pygarble combines fixed character models, English word patterns, keyboard paths, and encoding checks. It has **zero runtime dependencies** and requires no training, network access, or model downloads at inference time.

[Documentation](https://brightertiger.github.io/pygarble/) · [Source](https://github.com/brightertiger/pygarble) · [Issues](https://github.com/brightertiger/pygarble/issues)

## Installation

Install from PyPI:

```bash
python -m pip install pygarble
```

Check your installed version with `python -m pip show pygarble`.

## Quick start

```python
from pygarble import EnsembleDetector

detector = EnsembleDetector()

assert detector.predict("Hello world") is False
assert detector.predict("asdfghjkl") is True
assert detector.predict("qwerty") is True
assert detector.predict("नमस्ते दुनिया") is True

texts = ["Hello world", "qxzjkwpv"]
assert detector.predict(texts) == [False, True]
scores = detector.score(texts)  # List of floats in [0, 1]
```

`True` means the selected checks flag the text. `False` means they did not flag it; it does not prove that the text is meaningful English. `score()` and `predict_proba()` return the same heuristic score, **not a calibrated probability**. These methods accept a Python string or list of strings and preserve batch order. No `fit()` step is needed.

## English-specific behavior

Meaningful Hindi and other non-English text may score as gibberish. **This is expected for English-specific checks.** pygarble is not a multilingual validator, language identifier, or semantic nonsense detector. Non-English text written in Latin letters can pass, and names, rare English words, and technical terms can be flagged.

For encoding and control-character checks without English plausibility scoring:

```python
from pygarble import EnsembleDetector

corruption = EnsembleDetector(profile="corruption")
assert corruption.predict("नमस्ते दुनिया") is False
assert corruption.predict("CafÃ© au lait") is True
assert corruption.predict("hello\x00world") is True
```

## Choose a profile

| Profile | Checks and intended use |
| --- | --- |
| `english` (default) | Markov, likelihood ratio, word anomaly, mojibake, keyboard adjacency, and control characters; general English screening |
| `english_extended` | Adds pattern matching, localized anomalies, and repetition; more aggressive screening with more potential false positives |
| `legacy` | Former three-member set: Markov, likelihood ratio, and word anomaly, using current preprocessing and fixes |
| `corruption` | Mojibake and control artifacts, independent of English plausibility |
| `spoofing` | Unicode script/confusable heuristic; not a complete phishing detector |

Named profiles default to `any` voting: an applicable member reaching the decision threshold flags the text. Use the [API guide](https://brightertiger.github.io/pygarble/api.html) for custom voting rules and per-member settings.

## Use an individual strategy

All 28 strategies are available through `GarbleDetector` and the `Strategy` enum. Two strategies are new in 0.9.0:

```python
from pygarble import GarbleDetector, Strategy

controls = GarbleDetector(Strategy.CONTROL_CHARACTERS)
assert controls.predict("hello\x00world") is True
assert controls.predict("hello\nworld") is False

local = GarbleDetector(Strategy.LOCAL_ANOMALY)
text = "Please review qxzjkwpvm before delivery."
assert local.predict(text) is True
```

`CONTROL_CHARACTERS` detects unexpected controls, replacement characters, lone surrogates, and excessive combining-mark runs. It is included in the default English and corruption profiles. `LOCAL_ANOMALY` finds severe unknown tokens and bounded token windows inside otherwise readable English; it is included in `english_extended`.

Keyboard adjacency also supports QWERTY, AZERTY, and QWERTZ. Repetition detection includes repeated phrases. See the [strategy guide](https://brightertiger.github.io/pygarble/strategy-guide.html) for examples and the [strategy reference](https://brightertiger.github.io/pygarble/strategies.html) for accepted settings.

## Inspect explanations

```python
import json
from dataclasses import asdict
from pygarble import GarbleDetector, Strategy

text = "Please review qxzjkwpvm before delivery."
result = GarbleDetector(Strategy.LOCAL_ANOMALY).analyze(text)

assert result.garbled is True
assert result.status == "garbled"
for span in result.spans:
    print(text[span.start:span.end], span.reason)

payload = json.dumps(asdict(result))
```

Analysis records contain the decision, score, status, per-strategy signals, profile, and model version. Span offsets index the original Python string, with an exclusive end. Not every strategy produces spans. `analyze()` evaluates all selected members; `predict()` may stop early for `any` and `all`.

An empty input or an ensemble with no applicable members returns `False` and status `insufficient_evidence`. Applications should validate required fields separately.

## Configure domain vocabulary and limits

```python
from pygarble import EnsembleDetector

detector = EnsembleDetector(
    allowlist=["syzygy", "myproductname"],
    max_input_length=100_000,
)
assert detector.predict("syzygy") is False
```

Allowlists apply to Markov, likelihood ratio, word anomaly, and local anomaly scoring. They do not suppress keyboard, pattern, or raw corruption evidence. Shared English scoring excludes structured tokens such as URLs, paths, digit-containing identifiers, and camel-case identifiers; other strategies can still flag them.

Oversized input raises `ValueError`. Invalid batch entries raise `TypeError` before processing begins. Optional threads can process batches, but are not a guaranteed speedup. `timeout_per_text` only bounds waits for threaded results; it is not a hard execution deadline. See the [API reference](https://brightertiger.github.io/pygarble/api.html) for configuration and error behavior.

## Upgrading and evaluation

The 0.9.0 default adds specialist checks and changes preprocessing. The `legacy` profile restores the former strategy selection, not exact earlier scores. Review the [upgrade guide](https://brightertiger.github.io/pygarble/migration.html) before changing versions.

The repository contains reproducible benchmark tooling and a small authored challenge set. These engineering datasets are not production accuracy estimates; measure false positives and missed detections on your own English inputs before choosing thresholds. Inference is deterministic for a fixed package, configuration, and Python/Unicode data version.

- [Changelog](https://github.com/brightertiger/pygarble/blob/main/CHANGELOG.md)
- [Evaluation and implementation report](https://github.com/brightertiger/pygarble/blob/main/docs/dev/2026-07-implementation.md)
- [Recorded evaluation results](https://github.com/brightertiger/pygarble/blob/main/regression/english_results.json)
- [Data provenance and curation](https://github.com/brightertiger/pygarble/blob/main/scripts/data_curation.json)

## Contributing

```bash
git clone https://github.com/brightertiger/pygarble.git
cd pygarble
python -m pip install -e ".[dev]"
python -m pytest -q
```

See the [contributing guide](https://brightertiger.github.io/pygarble/contributing.html) for quality checks, documentation builds, and data regeneration. Please include your Python version, package version, selected profile, and a minimal input when reporting a problem.

## License

Library code is MIT licensed. Data provenance and attribution are recorded separately in the curation manifest linked above.
