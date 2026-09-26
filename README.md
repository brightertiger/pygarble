# pygarble

**Detect gibberish, keyboard mashing, mojibake and degenerate model output in English text. Pure Python, zero dependencies, deterministic, explainable.**

[![PyPI](https://img.shields.io/pypi/v/pygarble.svg)](https://pypi.org/project/pygarble/)
[![Python](https://img.shields.io/pypi/pyversions/pygarble.svg)](https://pypi.org/project/pygarble/)
[![Tests](https://github.com/brightertiger/pygarble/actions/workflows/test.yml/badge.svg)](https://github.com/brightertiger/pygarble/actions/workflows/test.yml)
[![License](https://img.shields.io/pypi/l/pygarble.svg)](LICENSE)

[Documentation](https://brightertiger.github.io/pygarble/) · [Changelog](https://github.com/brightertiger/pygarble/blob/main/CHANGELOG.md) · [Issues](https://github.com/brightertiger/pygarble/issues)

## Why pygarble

- **Zero dependencies, no model downloads.** `pip install pygarble` and go; nothing touches the network at inference time.
- **Deterministic and explainable.** The same input always gives the same score, and `analyze()` tells you which spans triggered which check.
- **28 strategies behind 6 profiles.** Character models, English word patterns, keyboard paths, repetition, encoding damage and Unicode spoofing, combined by profile for your use case.
- **A CLI for pipelines.** `pygarble check` reads stdin or files, emits text, TSV or JSONL, and exits non-zero when something is garbled.
- **Calibrate to your data.** `pygarble.calibrate()` sweeps thresholds over your own labeled samples and recommends one.

## Ten-second start

```bash
python -m pip install pygarble
printf 'hello world\nasdfghjkl\n' | pygarble check
```

```python
from pygarble import EnsembleDetector

detector = EnsembleDetector()
assert detector.predict("Hello world") is False
assert detector.predict("asdfghjkl") is True
assert detector.predict(["Hello world", "qxzjkwpv"]) == [False, True]
```

`True` means the selected checks flagged the text. `score()` returns the same heuristic value as a float in `[0, 1]`; it is not a calibrated probability.

## Use cases

| Need | Start with |
| --- | --- |
| Reject junk in form fields (names, messages, usernames) | `EnsembleDetector()` and an `allowlist` of your product words |
| Guard LLM responses against loops and encoding damage | `EnsembleDetector(profile="llm_output")` |
| Clean scraped or OCR'd corpora | `pygarble check --format tsv` in your pipeline |
| Drop noise lines from logs | `pygarble check --strategy control_characters` |

## Choose a profile

| Profile | Checks and intended use |
| --- | --- |
| `english` (default) | Markov, likelihood ratio, word anomaly, mojibake, keyboard adjacency, and control characters; general English screening |
| `english_extended` | Adds pattern matching, localized anomalies, and repetition; more aggressive screening with more potential false positives |
| `legacy` | Former three-member set: Markov, likelihood ratio, and word anomaly, using current preprocessing and fixes |
| `corruption` | Mojibake and control artifacts, independent of English plausibility |
| `spoofing` | Unicode script/confusable heuristic; not a complete phishing detector |
| `llm_output` | Repetition, control characters, mojibake, local anomaly; deterministic pre-check for degenerate model output, quiet on code and technical prose |

Named profiles default to `any` voting: an applicable member reaching the decision threshold flags the text. Use the [API guide](https://brightertiger.github.io/pygarble/api.html) for custom voting rules and per-member settings.

## Command line

```console
$ printf 'hello world\nasdfghjkl\n' | pygarble check
clean	hello world
garbled	asdfghjkl
$ pygarble score -t "please review qxzjkwpvm"
0.9974	please review qxzjkwpvm
$ pygarble check --profile llm_output -t "the the the the the the the the"
garbled	the the the the the the the the
```

`check`, `score` and `analyze` read one text per line from files or stdin, and `--field NAME` reads JSON lines. `check` exits with 1 when any line is garbled and 2 on a usage or input error. See the [CLI guide](https://brightertiger.github.io/pygarble/cli.html) for every option.

## Calibrate the threshold

```python
from pygarble import EnsembleDetector, calibrate

garbled = ["qxzjkwpv bnmqwer", "asdfghjkl"]
clean = ["hello world", "please send the invoice"]
report = calibrate(EnsembleDetector(), garbled, clean)
detector = EnsembleDetector(threshold=report.recommended.threshold)
assert detector.predict(garbled) == [True, True]
assert detector.predict(clean) == [False, False]
```

`calibrate()` scores both samples once, reports precision, recall, F1 and false-positive rate at every observed score, and recommends a threshold by F1 or by a maximum false-positive rate. `pygarble calibrate --garbled bad.txt --clean good.txt` does the same from files. See the [calibration guide](https://brightertiger.github.io/pygarble/calibration.html).

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

`CONTROL_CHARACTERS` detects unexpected controls, replacement characters, lone surrogates, and excessive combining-mark runs. It is included in the default English and corruption profiles. `LOCAL_ANOMALY` finds severe unknown tokens and bounded token windows inside otherwise readable English; it is included in `english_extended` and `llm_output`.

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

Since 0.9.0 allowlists apply to every strategy: allowlisted words are excluded from English scoring and blanked out of the text that keyboard, pattern, and other raw-text checks scan. `SYMBOL_RATIO` reads the raw text because allowlisted letters can only lower its score. Control characters and encoding damage outside allowlisted words are still reported. Shared English scoring excludes structured tokens such as URLs, paths, digit-containing identifiers, and camel-case identifiers; other strategies can still flag them.

Oversized input raises `ValueError`. Invalid batch entries raise `TypeError` before processing begins. Optional threads can process batches, but are not a guaranteed speedup. `timeout_per_text` only bounds waits for threaded results; it is not a hard execution deadline. See the [API reference](https://brightertiger.github.io/pygarble/api.html) for configuration and error behavior.

## English-specific behavior

Meaningful Hindi and other non-English text may score as gibberish. **This is expected for English-specific checks.** A `False` result means the checks did not flag the text; it does not prove the text is meaningful English. pygarble is not a multilingual validator, language identifier, or semantic nonsense detector. Non-English text written in Latin letters can pass, and names, rare English words, and technical terms can be flagged.

The default profile flags Hindi. The `corruption` profile checks encoding and control artifacts without English plausibility scoring:

```python
from pygarble import EnsembleDetector

assert EnsembleDetector().predict("नमस्ते दुनिया") is True

corruption = EnsembleDetector(profile="corruption")
assert corruption.predict("नमस्ते दुनिया") is False
assert corruption.predict("CafÃ© au lait") is True
assert corruption.predict("hello\x00world") is True
```

## Upgrading and evaluation

0.10.0 is additive over 0.9.0. The 0.9.0 default adds specialist checks and changes preprocessing. The `legacy` profile restores the former strategy selection, not exact earlier scores. Review the [upgrade guide](https://brightertiger.github.io/pygarble/migration.html) before changing versions.

The repository contains reproducible benchmark tooling and a small authored challenge set. These engineering datasets are not production accuracy estimates; measure false positives and missed detections on your own English inputs before choosing thresholds. Inference is deterministic for a fixed package, configuration, and Python/Unicode data version.

- [Changelog](https://github.com/brightertiger/pygarble/blob/main/CHANGELOG.md)
- [Evaluation and implementation report](https://github.com/brightertiger/pygarble/blob/main/docs/dev/2026-07-implementation.md)
- [Recorded evaluation results](https://github.com/brightertiger/pygarble/blob/main/regression/english_results.json)
- [Golden corpus](https://github.com/brightertiger/pygarble/blob/main/regression/golden.jsonl) of frozen detector outputs for every profile, checked in CI
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
