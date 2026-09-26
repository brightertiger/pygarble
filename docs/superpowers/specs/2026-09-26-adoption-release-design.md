# pygarble 0.10.0 "Adoption Release" Design

Date: 2026-09-26. Author: Claude (delegated by the maintainer: "you decide, based on your research, then implement it end to end").

## Goal

Make pygarble the default answer to "detect gibberish in Python" and lay the groundwork for a JavaScript port, without changing the package's positioning (pure Python, zero dependencies, deterministic, English) or breaking the 0.9 API.

## Why these five things

Research (2026-09-26, live npm/PyPI stats and competitor issue trackers) found:

- The Python download leader, `gibberish-detector` (domanchi), gets ~16x pygarble's traffic despite being unmaintained since 2022. Its two differentiators are a **CLI** and a train-your-own-corpus story. pygarble has no CLI.
- The most common open issues on every competitor (rrenaud #15, nostril #18, domanchi #3) are **"how do I pick a threshold"** and **"give me the raw score"**. pygarble exposes raw scores but no calibration help.
- The Guardrails AI hub's gibberish validator depends on a HuggingFace model. A **deterministic, CPU-only LLM-output guard** is an open niche pygarble already has the strategies for.
- A **TypeScript port** is the recommended route into JS (pure JS beat WASM 5:1 for tiktoken). Its prerequisites are language-neutral data tables and a golden corpus so two implementations can be held to identical output.
- Adoption is decided on the README in the first ten seconds. The current README opens with a paragraph of caveats.

Deferred to later releases, in order: TypeScript port (large), name/username field presets (needs a names benchmark), competitor benchmark table (needs third-party installs), custom-corpus retraining, multilingual.

## Constraints

- Backward compatible with 0.9.0: no public name removed or renamed; every profile, kwarg and default unchanged. New behaviour is additive.
- Pure Python, `dependencies = []`, Python >= 3.8, black line length 79, mypy clean, determinism preserved.
- Wheel stays lean: the new JSON data tables ship in the repo and sdist, not the wheel.
- Every README and docs snippet must execute (existing snippet runner).

## Components

### 1. Command-line interface

Module `pygarble/cli.py` exposing `main(argv: Optional[List[str]] = None) -> int`. Entry points: `[project.scripts] pygarble = "pygarble.cli:main"` and `pygarble/__main__.py` so `python -m pygarble` works.

Subcommands:

| Subcommand | Output per input line | Exit code |
|---|---|---|
| `check` | text: `garbled\t<text>` or `clean\t<text>` or `insufficient\t<text>` | 0 if no input was garbled, 1 if any was, 2 on usage/IO error |
| `score` | text: `<score>\t<text>` with 4 decimals | 0, or 2 on error |
| `analyze` | JSON object per line (default format `jsonl`) | 0, or 2 on error |
| `calibrate` | see component 3 | 0, or 2 on error |

Shared options for `check`, `score`, `analyze`:

- Inputs: positional `inputs` (file paths, or `-` for stdin; default stdin when none given), each line is one text, trailing newline stripped, blank lines evaluated (they yield `insufficient`). `-t/--text TEXT` (repeatable) evaluates literal texts instead of reading inputs.
- `--profile NAME` (default `english`) or `--strategy NAME` (a `Strategy` value; mutually exclusive with `--profile`).
- `--threshold FLOAT` (default 0.5).
- `--allowlist FILE`: one word per line, `#` comments and blank lines ignored.
- `--format {text,tsv,jsonl}`: `text` as above; `tsv` = `garbled(0/1)\tscore\tstatus\ttext`; `jsonl` = `{"text": ..., "garbled": ..., "score": ..., "status": ..., "profile": ..., "spans": [{"start","end","reason"}], "signals": [{"strategy","score","applicable","reason"}]}`. Default `text` for `check`/`score`, `jsonl` for `analyze`.
- `--field NAME`: treat each input line as a JSON object and evaluate `obj[NAME]`; output JSONL echoes the original object plus a `pygarble` key holding the analysis. Lines that fail to parse or lack the field go to stderr and count as errors (exit 2 at the end, other lines still processed).
- `--version` prints `pygarble <version>`.

Errors: unknown profile/strategy, unreadable file, bad JSON → message on stderr, exit 2. Text is never truncated. Output is UTF-8. The CLI imports `pygarble` lazily inside `main` so `--help` costs nothing.

### 2. `llm_output` profile

`PROFILES["llm_output"] = (Strategy.REPETITION, Strategy.CONTROL_CHARACTERS, Strategy.MOJIBAKE, Strategy.LOCAL_ANOMALY)`, voting `any` like every named profile.

Purpose: catch degenerate model output (repetition loops, tail loops, mojibake from bad decoding, control characters, dense token salad) while staying quiet on technical prose, code, identifiers and product names, which the Markov/word-anomaly members would flag. Documented as "a cheap deterministic pre-check, not a hallucination detector".

Acceptance examples: "the the the the the the the the" and "and so on and so on and so on and so on and so on" → garbled; a 40-word ordinary assistant paragraph → clean; a Python snippet `def parse(row): return row.split(",")` → clean; a response containing `�` or `Ã©` mojibake → garbled.

### 3. Calibration helper

Module `pygarble/calibration.py`:

```python
@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float

@dataclass(frozen=True)
class CalibrationReport:
    recommended: ThresholdPoint
    objective: str            # "f1" or "max_fpr"
    max_false_positive_rate: Optional[float]
    garbled: int              # sample counts
    clean: int
    points: Tuple[ThresholdPoint, ...]   # sorted by threshold ascending

def calibrate(detector, garbled, clean, *, objective="f1", max_false_positive_rate=None, thresholds=None) -> CalibrationReport
```

- `detector` is anything with a `.score(list) -> list` method (GarbleDetector or EnsembleDetector). Scores are computed once per text.
- Candidate thresholds: `thresholds` if given, else the sorted set of all observed scores plus 0.0 and 1.0.
- A text counts as flagged at threshold t iff `score >= t`.
- `objective="f1"` picks the highest F1; ties resolve to the highest threshold (fewest false positives). `objective="max_fpr"` requires `max_false_positive_rate` and picks the threshold with the highest recall among those whose FPR ≤ limit, ties to the highest threshold; if none qualifies, threshold 1.0 is returned.
- Empty `garbled` or `clean` → `ValueError`. Non-string entries → `TypeError` (via the detector's own validation).
- Precision with zero flagged is defined as 1.0; recall with zero positives cannot happen (validated).
- `report.recommended.threshold` can be passed straight to a detector constructor. Documented caveat: under `voting="majority"` the ensemble decision counts member votes, so the calibrated threshold is applied per member; the report still measures the aggregate score.

Exported from `pygarble`: `calibrate`, `CalibrationReport`, `ThresholdPoint`.

CLI: `pygarble calibrate --garbled FILE --clean FILE [--profile|--strategy] [--objective f1|max_fpr] [--max-fpr FLOAT] [--format text|jsonl]`. Text output: a table of points (threshold, precision, recall, F1, FPR) and a final line `recommended threshold: <t> (f1=<...>, fpr=<...>)`.

### 4. Language-neutral data tables and golden corpus

**JSON tables.** `scripts/generate_data.py` writes, next to the `.py` tables, `words.json` (sorted list of strings), `bigrams.json` (`{"default_log_prob": -10.0, "log_probs": {"ab": -3.1, ...}}` with keys sorted), `trigrams.json` (sorted list). The manifest `files` map records SHA-256 for `*.py` and `*.json` (excluding `manifest.json`). `--check` verifies the JSON files exactly like the `.py` files. JSON is written with `ensure_ascii=True, sort_keys=True, indent=None, separators=(",", ":")` plus a trailing newline, so it is byte-reproducible. The JSON files are committed. `MANIFEST.in` already includes `*.json` under `pygarble` (sdist); `[tool.setuptools.package-data]` is NOT extended, so wheels do not carry them. A test asserts `set(json words) == ENGLISH_WORDS`, bigram dict equality including the default, and trigram set equality. `test_model_manifest_matches_packaged_data` keeps passing because it iterates the manifest.

**Golden corpus.** `regression/golden.py` with `--write` and `--check`. Rows (JSONL, one per text × profile) with keys `text`, `profile`, `garbled`, `score` (rounded to 12 decimals), `status`, `spans` (list of `[start, end, reason]`). Sources: all 132 challenge cases and a fixed list of 24 edge inputs (empty, whitespace, single char, emoji, CJK, Cyrillic, Arabic, mojibake, U+FFFD, control char, 5k-char repeat, URL, path, hex, base64, UUID, contractions with curly apostrophe, Title-Case names, ALL-CAPS acronyms, numbers, mixed script spoof, combining marks, "test test test", a 40-word paragraph). Profiles: `english`, `english_extended`, `legacy`, `corruption`, `spoofing`, `llm_output`. Output file `regression/golden.jsonl`, plus `regression/golden.sha256`. `--check` regenerates in memory and diffs; a non-zero exit lists the first ten differing rows. A test `tests/test_golden.py` runs the same comparison (skipped when `regression/` is absent). CI: the quality job runs `python regression/golden.py --check`. The docs state that any port must reproduce `golden.jsonl` exactly, with span offsets as Unicode code points.

### 5. README and docs overhaul

README order: title + one-sentence pitch; badges (PyPI version, Python versions, license, tests workflow); "Why pygarble" (five bullets: zero dependencies, deterministic and explainable spans, 28 strategies behind 6 profiles, CLI, calibrate to your data); ten-second start (pip, three-line Python, one CLI line); "Use cases" (form input validation, LLM output guard, data cleaning, log noise); then the existing sections condensed: profiles table (now including `llm_output`), CLI, calibration, explanations, vocabulary and limits, English-only behaviour, evaluation, contributing, license. Every code block executable. Badge URLs use shields.io for PyPI and the repo's `test.yml` workflow badge.

Docs: new `docs/cli.rst` and `docs/calibration.rst` (added to the toctree after `quickstart`); `docs/api.rst` gains `calibrate`/`CalibrationReport`/`ThresholdPoint` and the `llm_output` profile row; `docs/strategy-guide.rst` gains a short "LLM output" section; `docs/contributing.rst` mentions the JSON tables, golden corpus and their `--check` commands.

### 6. Release metadata

`__version__ = "0.10.0"`. CHANGELOG gains `## [0.10.0] - 2026-09-26` with Added (CLI, `llm_output` profile, calibration, JSON tables + golden corpus, README) and a note that 0.9.0 was never published to PyPI and its entry is folded into this release's user-facing notes. `docs/migration.rst` gains a one-paragraph 0.10.0 section ("no breaking changes; new entry point `pygarble`"). The branch is pushed and a PR opened; tagging and publishing are left to the maintainer.

## Testing

- CLI: subprocess-free tests call `main(argv)` with `capsys`, covering every subcommand, format, exit code, `--field`, `--allowlist`, stdin via `monkeypatch.setattr("sys.stdin", io.StringIO(...))`, error paths, and `python -m pygarble --version` via `subprocess` once.
- Profile: the acceptance examples above, plus the existing `test_every_profile_constructs_and_votes_any` now covers six profiles automatically.
- Calibration: synthetic scores through a stub detector (a class with `.score`) for exact F1/FPR math, plus one real `EnsembleDetector` run; tie-breaking; `max_fpr` fallback to 1.0; validation errors.
- Data: JSON/py equality; manifest hashes; `generate_data.py --check` in CI.
- Golden: reproduction test; `--check` in CI.
- Docs: snippet runner over README and all rst; `sphinx -W`.
- Full gate as in the 0.9.0 plan.

## Out of scope (recorded for the next spec)

TypeScript port; name/username/freetext presets and a names benchmark; competitor benchmark table; custom-corpus retraining; Guardrails AI adapter package; multilingual packs.
