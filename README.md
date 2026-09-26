# pygarble

**A local first line of defence for text: secrets, PII, profanity and gibberish, with explainable findings and redaction. No LLM calls or model downloads; the base install has no runtime dependencies.**

[![PyPI](https://img.shields.io/pypi/v/pygarble.svg)](https://pypi.org/project/pygarble/)
[![Python](https://img.shields.io/pypi/pyversions/pygarble.svg)](https://pypi.org/project/pygarble/)
[![Tests](https://github.com/brightertiger/pygarble/actions/workflows/test.yml/badge.svg)](https://github.com/brightertiger/pygarble/actions/workflows/test.yml)
[![License](https://img.shields.io/pypi/l/pygarble.svg)](https://github.com/brightertiger/pygarble/blob/main/LICENSE)

[Documentation](https://brightertiger.github.io/pygarble/) · [Changelog](https://github.com/brightertiger/pygarble/blob/main/CHANGELOG.md) · [Issues](https://github.com/brightertiger/pygarble/issues)

## Why pygarble

- **Local processing.** Native checks use bundled rules, dictionaries and character statistics. Optional integrations run locally and are enabled explicitly.
- **Deterministic and explainable.** Every finding has a kind, span, confidence and reason. `Finding` and `ScanReport` objects never carry the matched text, so logging a report from Python cannot leak a secret. `pygarble scan` rows include the input line; use `pygarble redact` when output goes to logs.
- **Separate APIs.** Screen secrets, PII and profanity together, or check gibberish independently. The original combined scanner still supports all four categories.
- **Redaction in three modes.** Placeholders such as `[EMAIL]`, length-preserving masks, or partial masks that keep the last four digits of a card or phone number.
- **A document CLI.** `pygarble-screen scan` emits findings-only JSON per document; `pygarble-screen redact` preserves document structure, including multiline private keys. The original line-oriented CLI remains available.

## Install and choose an API

```bash
python -m pip install pygarble
```

This README describes the current source, including **unreleased** module
separation and optional backends. For those APIs, install a checkout of the
reviewed branch or commit with `python -m pip install -e .`; see the
[installation guide](https://brightertiger.github.io/pygarble/installation.html) and [migration notes](https://brightertiger.github.io/pygarble/migration.html).

| Use case | Import | Default behavior |
| --- | --- | --- |
| Secrets, PII and profanity | `from pygarble.screening import Scanner` | Three native rule categories; optional backends are opt-in |
| Gibberish detection | `from pygarble.gibberish import EnsembleDetector` | English gibberish profile |
| Existing combined applications | `from pygarble import Scanner` | All four categories, unchanged |

## Ten-second start

Use `pygarble.screening` for **secrets, PII and profanity**. Keep gibberish
detection separate with `pygarble.gibberish.GarbleDetector` /
`pygarble.gibberish.EnsembleDetector`. The existing
top-level `Scanner` retains its four-category defaults for compatibility.

```python
from pygarble.screening import Scanner

scanner = Scanner(max_input_length=100_000)
assert scanner.redact("mail jane@example.com").text == "mail [EMAIL]"
assert not scanner.scan("qxzjkwpv bnmqwer zzxqv").flagged
```

The dedicated document CLI keeps source text out of scan reports and handles
multiline private keys during redaction:

```bash
python -m pygarble.screening scan document.txt
python -m pygarble.screening redact document.txt
```

Optional local backends are explicitly enabled, and do not change the base
installation. In a source checkout, `pip install -e '.[screening]'` adds `phonenumberslite`,
`python-stdnum` and `detect-secrets`. Gitleaks requires a separately installed
executable. None requires LLM calls or model downloads.

After installing the optional dependencies and Gitleaks:

```text
from pygarble.screening import Scanner

scanner = Scanner(
    backends=["phonenumbers", "stdnum", "detect-secrets", "gitleaks"],
    backend_options={"phonenumbers": {"region": "GB"}},
    max_input_length=100_000,
)
```

See the [standalone screening guide](https://brightertiger.github.io/pygarble/standalone-screening.html)
for individual extras, backend options, custom detectors and coverage limits.
Reuse a scanner across documents; Gitleaks starts a subprocess per document.

## Existing combined API and line CLI

The original API remains supported. Its line-oriented CLI echoes input text;
use the document CLI above for findings-only logs and multiline redaction.

```bash
python -m pip install pygarble
printf 'mail jane@example.com\nhello\n' | pygarble scan
printf 'key AKIAIOSFODNN7EXAMPLE\n' | pygarble redact
```

```console
$ printf 'mail jane@example.com\nhello\n' | pygarble scan
flagged	email	mail jane@example.com
clean		hello
$ printf 'key AKIAIOSFODNN7EXAMPLE\n' | pygarble redact
key [AWS_ACCESS_KEY_ID]
```

```python
from pygarble import redact, scan

report = scan("mail jane@example.com, key AKIAIOSFODNN7EXAMPLE, damn")
assert report.kinds() == ("aws_access_key_id", "email", "profanity")
assert redact("mail jane@example.com").text == "mail [EMAIL]"
```

`report.flagged` is true when any finding reaches `min_confidence` (default 0.5) or the gibberish ensemble flags the text. Each `Finding` has `category`, `kind`, `start`, `end`, `confidence` and `reason`; `report.to_dict()` is JSON-ready.

## What it catches

| Category | Kinds | How |
| --- | --- | --- |
| `secrets` | AWS, GitHub, GitLab, Slack, Stripe, Google, OpenAI, Anthropic, Hugging Face, npm, PyPI and SendGrid keys; JWTs; private key blocks; credentials in URLs; bearer tokens; generic secrets | Unique vendor prefixes; JWTs at 1.0 (0.8 when the header does not decode); PEM/PGP private key blocks; credentials in URLs; `Bearer` prefixes; keyword plus entropy for generic secrets |
| `pii` | email, phone, credit card, IBAN, IPv4/IPv6; US SSN; UK National Insurance and NHS numbers; Indian Aadhaar and PAN | Structure plus checksums (Luhn, mod-97, mod-11, Verhoeff) and locale packs `us`, `uk`, `in` |
| `profanity` | profanity (strong and mild tiers) | Attributed English word list with leetspeak, elongation, masking, spacing and phrase handling; token-level, so Scunthorpe stays clean |
| `gibberish` | garbled | The existing English gibberish ensemble, whole text |

Confidence is 1.0 for checksum-verified or vendor-prefixed matches, down to 0.6 for keyword-plus-entropy secrets and ambiguous masking, and 0.5 for standalone high-entropy strings, which are opt-in (`secrets_without_context=True`); see the [screening guide](https://brightertiger.github.io/pygarble/screening.html).

**Coverage limits:** native rules do not detect names, postal addresses,
free-text dates of birth, hate speech beyond the word list, secrets without a
recognisable shape, or non-English profanity. A clean result means the selected
checks found nothing; choose any further review according to your application's
requirements.

## Redact

```python
from pygarble.screening import Scanner

scanner = Scanner(categories=["secrets", "pii"])
text = "card 4111 1111 1111 1111, mail jane@example.com"
assert scanner.redact(text).text == "card [CREDIT_CARD], mail [EMAIL]"
assert scanner.redact(text, mode="partial").text == (
    "card ***************1111, mail ****************"
)
```

Overlapping findings merge into one region. `placeholder` templates accept `{KIND}`, `{kind}` and `{category}`; `mask` preserves length; `partial` keeps the last four characters when a region ends with a `credit_card`, `phone`, `iban`, `ssn_us`, `nhs_number` or `aadhaar` finding. Gibberish is never redacted.

## Throughput

Measured on an Apple M2 (macOS arm64, Python 3.12.2) with `python regression/throughput.py --size-mb 2`, on a synthetic corpus of five ASCII English paragraphs with 5% planted findings. MB is 10^6 UTF-8 bytes of scanned text; your numbers will differ.

| Categories | Short lines (~110 bytes), MB/s | 4 KB documents (`--chunk-bytes 4096`), MB/s |
| --- | --- | --- |
| `secrets` | 23.0 | 14.7 |
| `pii` | 13.3 | 7.7 |
| `profanity` | 10.7 | 10.7 |
| `secrets`, `pii`, `profanity` | 4.8 (about 4.5 with a varied vocabulary) | 3.5 |
| `gibberish` | 0.8 | 1.1 |
| all four | 0.7 | 0.8 |

These measurements use the native rules through the combined scanner and do
not include optional backends. Use `pygarble.screening.Scanner` for the three
rule categories without the gibberish ensemble. A single 1 MB line through
the three rule categories took 0.11 s (repeated sentence) to 0.37 s (random
dictionary words) in this benchmark. Reuse scanner instances; Gitleaks adds a
subprocess invocation for each document.

## Gibberish detection

The `gibberish` category is pygarble's original English gibberish detector, still available on its own. It flags keyboard mashing, mojibake, control artifacts and degenerate model output.

```python
from pygarble.gibberish import EnsembleDetector

detector = EnsembleDetector()
assert detector.predict("Hello world") is False
assert detector.predict("asdfghjkl") is True
assert detector.predict(["Hello world", "qxzjkwpv"]) == [False, True]
```

`True` means the selected checks flagged the text. `score()` returns the same heuristic value as a float in `[0, 1]`; it is not a calibrated probability.

### Profiles and use cases

| Profile | Checks and intended use |
| --- | --- |
| `english` (default) | Markov, likelihood ratio, word anomaly, mojibake, keyboard adjacency, and control characters; general English screening, junk in form fields |
| `english_extended` | Adds pattern matching, localized anomalies, and repetition; more aggressive, more potential false positives |
| `legacy` | Former three-member set: Markov, likelihood ratio, and word anomaly |
| `corruption` | Mojibake and control artifacts, independent of English plausibility; scraped or OCR'd corpora |
| `spoofing` | Unicode script/confusable heuristic; not a complete phishing detector |
| `llm_output` | Repetition, control characters, mojibake, local anomaly; degenerate model output, quiet on code and technical prose |

Named profiles default to `any` voting. `pygarble check`, `score` and `analyze` read one text per line (`--field NAME` for JSON lines); `check` exits 1 when any line is garbled. See the [CLI guide](https://brightertiger.github.io/pygarble/cli.html) and [API guide](https://brightertiger.github.io/pygarble/api.html).

```console
$ printf 'hello world\nasdfghjkl\n' | pygarble check
clean	hello world
garbled	asdfghjkl
$ pygarble check --profile llm_output -t "the the the the the the the the"
garbled	the the the the the the the the
```

### Calibrate, explain, configure

```python
from pygarble.gibberish import EnsembleDetector, calibrate

garbled = ["qxzjkwpv bnmqwer", "asdfghjkl"]
clean = ["hello world", "please send the invoice"]
report = calibrate(EnsembleDetector(), garbled, clean)
detector = EnsembleDetector(threshold=report.recommended.threshold)
assert detector.predict(garbled) == [True, True]
assert detector.predict(clean) == [False, False]
```

`calibrate()` reports precision, recall, F1 and false-positive rate at every observed score and recommends a threshold; `pygarble calibrate --garbled bad.txt --clean good.txt` does the same from files. See the [calibration guide](https://brightertiger.github.io/pygarble/calibration.html).

```python
import json
from dataclasses import asdict
from pygarble.gibberish import GarbleDetector, Strategy

controls = GarbleDetector(Strategy.CONTROL_CHARACTERS)
assert controls.predict("hello\x00world") is True
assert controls.predict("hello\nworld") is False

text = "Please review qxzjkwpvm before delivery."
result = GarbleDetector(Strategy.LOCAL_ANOMALY).analyze(text)
assert result.garbled is True
assert result.status == "garbled"
for span in result.spans:
    print(text[span.start:span.end], span.reason)
payload = json.dumps(asdict(result))
```

All 28 strategies are available through `GarbleDetector` and the `Strategy` enum; see the [strategy guide](https://brightertiger.github.io/pygarble/strategy-guide.html). `analyze()` records the decision, score, status, per-strategy signals and spans (Python string offsets, exclusive end). Empty or wholly inapplicable input returns `False` with status `insufficient_evidence`.

```python
from pygarble.gibberish import EnsembleDetector

detector = EnsembleDetector(
    allowlist=["syzygy", "myproductname"],
    max_input_length=100_000,
)
assert detector.predict("syzygy") is False
```

Allowlists apply to every strategy. Oversized input raises `ValueError`; invalid batch entries raise `TypeError` before processing begins.

### English-specific behavior

Meaningful Hindi and other non-English text may score as gibberish. **This is expected for English-specific checks.** A `False` result does not prove the text is meaningful English; pygarble is not a language identifier or semantic nonsense detector. The `corruption` profile checks encoding and control artifacts without English plausibility scoring:

```python
from pygarble.gibberish import EnsembleDetector

assert EnsembleDetector().predict("नमस्ते दुनिया") is True

corruption = EnsembleDetector(profile="corruption")
assert corruption.predict("नमस्ते दुनिया") is False
assert corruption.predict("CafÃ© au lait") is True
assert corruption.predict("hello\x00world") is True
```

### Upgrading and evaluation

The unreleased module reorganization preserves existing imports, defaults and
scores. Version 0.11.0 added the combined scanner; it did not introduce the new
module layout. Review the [upgrade guide](https://brightertiger.github.io/pygarble/migration.html) before changing
versions. The benchmark and challenge sets are engineering regression data,
not production accuracy estimates; measure on your own inputs before choosing
thresholds.

- [Changelog](https://github.com/brightertiger/pygarble/blob/main/CHANGELOG.md)
- [Evaluation and implementation report](https://github.com/brightertiger/pygarble/blob/main/docs/dev/2026-07-implementation.md)
- [Recorded evaluation results](https://github.com/brightertiger/pygarble/blob/main/regression/english_results.json)
- [Golden corpus](https://github.com/brightertiger/pygarble/blob/main/regression/golden.jsonl) of frozen detector outputs for every profile, and a [golden scan corpus](https://github.com/brightertiger/pygarble/blob/main/regression/golden_scan.jsonl) for the scanner, both checked in CI
- [Data provenance and curation](https://github.com/brightertiger/pygarble/blob/main/scripts/data_curation.json)

## Repository layout and compatibility

```text
pygarble/
  screening/
    pii/           # detector, patterns, checksums
    profanity/     # detector, word lists, normalization
    secrets/       # detector, patterns, entropy
    backends/      # phonenumbers, stdnum, detect-secrets, Gitleaks
    scanner.py     # standalone three-category scanner
  gibberish/
    strategies/    # individual heuristic strategies
    detector.py    # single-strategy API
    ensemble.py    # profiles and voting
    calibration.py # threshold selection
  data/            # shared tables and portable JSON copies
  scanner.py       # compatible four-category scanner
```

Old paths such as `pygarble.core`, `pygarble.strategies`, `pygarble.pii` and
`pygarble.profanity` remain compatibility pointers. Old and new imports share
classes, enums, rule tables and caches. See the [architecture guide](https://brightertiger.github.io/pygarble/architecture.html)
for the full layout, import mappings and lazy-loading behavior.

## Contributing

```bash
git clone https://github.com/brightertiger/pygarble.git
cd pygarble
python -m pip install -e ".[dev]"
python -m pytest -q
```

See the [contributing guide](https://brightertiger.github.io/pygarble/contributing.html) for quality checks, documentation builds, and data regeneration. Please include your Python version, package version, selected profile, and a minimal input when reporting a problem.

## Open source and publishing

The repository is public, and releases are distributed on
[PyPI](https://pypi.org/project/pygarble/). Contributions and synthetic bug
reports are welcome; see [CONTRIBUTING.md](https://github.com/brightertiger/pygarble/blob/main/CONTRIBUTING.md)
and [SECURITY.md](https://github.com/brightertiger/pygarble/blob/main/SECURITY.md).
The [maintainer guide](https://brightertiger.github.io/pygarble/publishing.html)
covers releases, Google Search Console and documentation discovery.

## License

Library code is MIT licensed. The profanity word list is seeded from the LDNOOBW English list (Shutterstock), CC-BY-4.0, filtered and extended by the pygarble maintainers. Data provenance for the gibberish tables is recorded in the curation manifest linked above.
