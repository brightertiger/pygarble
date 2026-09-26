# pygarble 0.10.0 Adoption Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 0.10.0 with a CLI, an `llm_output` profile, a threshold calibration helper, language-neutral JSON data tables plus a golden corpus, and a README that sells the package.

**Architecture:** All new code is additive. `pygarble/cli.py` is a thin argparse layer over the existing detectors. `pygarble/calibration.py` is pure functions over `detector.score()`. The data generator gains JSON output alongside the `.py` tables; a new `regression/golden.py` freezes detector output for ports and CI. Docs and README are rewritten last, after the APIs exist.

**Tech Stack:** Python 3.8+, stdlib only (argparse, json, dataclasses), pytest, black 79, isort, flake8, mypy, Sphinx.

**Spec:** `docs/superpowers/specs/2026-09-26-adoption-release-design.md`

## Global Constraints

- Backward compatible with 0.9.0: no public name removed or renamed; every existing profile, kwarg and default unchanged.
- `dependencies = []`; Python >= 3.8 syntax only (no `list[str]`, no `X | None`, no walrus, no `match`).
- black line length 79; `isort`, `flake8`, `mypy pygarble` clean; suite clean under `python -m pytest -q -W error::FutureWarning`.
- Ad-hoc `python -c` snippets need `PYTHONPATH=.` (an old pygarble in site-packages shadows the repo).
- Wheel must not gain the JSON tables: do not touch `[tool.setuptools.package-data]`.
- Commit per task with the given message; end each message with the `Co-Authored-By:` trailer the harness specifies.
- CHANGELOG lines go under `## [0.10.0]` → `### Added` (Task 6 creates the heading; Tasks 1-5 append their lines under a temporary `## [Unreleased]` → `### Added`, which Task 6 renames).

## Review Focus

1. CLI on a file containing a blank line and a line with only spaces must print `insufficient` and not crash — Task 1.
2. `--field` with a line that is valid JSON but whose field is not a string must go to stderr and yield exit 2 while other lines are still processed — Task 1.
3. `calibrate()` when every score is identical (e.g. all 0.0) must return a valid report, not divide by zero — Task 3.
4. `llm_output` on a code snippet with repeated identifiers (`x = x + x + x + x`) must stay clean — Task 2.
5. Golden `--check` must fail loudly (non-zero, first differing rows printed) when a score changes, and the test must skip cleanly from an sdist without `regression/` — Task 5.

---

### Task 1: Command-line interface

**Files:**
- Create: `pygarble/cli.py`, `pygarble/__main__.py`, `tests/test_cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]`), `CHANGELOG.md`

**Interfaces:**
- Produces: `pygarble.cli.main(argv: Optional[List[str]] = None) -> int`; `pygarble.cli.build_parser() -> argparse.ArgumentParser`; `pygarble.cli.load_allowlist(path: str) -> List[str]`; `pygarble.cli.make_detector(args) -> Union[GarbleDetector, EnsembleDetector]` (Task 3 adds the `calibrate` subcommand to this parser).

- [ ] **Step 1: Write the failing tests**

```python
"""Command-line interface contracts (no subprocess except --version)."""

import io
import json
import subprocess
import sys

import pytest

from pygarble import __version__
from pygarble.cli import main


def run(capsys, argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def test_check_text_inputs(capsys):
    code, out, err = run(
        capsys, ["check", "-t", "hello world", "-t", "qxzjkwpv bnmqwer"]
    )
    assert code == 1
    assert out.splitlines() == [
        "clean\thello world",
        "garbled\tqxzjkwpv bnmqwer",
    ]
    assert err == ""


def test_check_all_clean_exits_zero(capsys):
    code, out, _ = run(capsys, ["check", "-t", "hello world"])
    assert code == 0
    assert out == "clean\thello world\n"


def test_check_reads_stdin_and_handles_blank_lines(capsys, monkeypatch):
    code, out, _ = run(
        capsys, ["check"], stdin="hello world\n\n   \nqxzjkwpv\n",
        monkeypatch=monkeypatch,
    )
    assert code == 1
    assert out.splitlines() == [
        "clean\thello world",
        "insufficient\t",
        "insufficient\t   ",
        "garbled\tqxzjkwpv",
    ]


def test_check_reads_files(capsys, tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("hello world\nasdfghjkl\n", encoding="utf-8")
    code, out, _ = run(capsys, ["check", str(path)])
    assert code == 1
    assert out.splitlines()[1] == "garbled\tasdfghjkl"


def test_score_format(capsys):
    code, out, _ = run(capsys, ["score", "-t", "hello world"])
    assert code == 0
    value, text = out.strip().split("\t")
    assert text == "hello world"
    assert 0.0 <= float(value) < 0.5


def test_tsv_format(capsys):
    _, out, _ = run(capsys, ["check", "--format", "tsv", "-t", "qxzjkwpv"])
    garbled, score, status, text = out.strip().split("\t")
    assert garbled == "1"
    assert float(score) >= 0.5
    assert status == "garbled"
    assert text == "qxzjkwpv"


def test_analyze_jsonl(capsys):
    code, out, _ = run(capsys, ["analyze", "-t", "please review qxzjkwpvm"])
    assert code == 0
    row = json.loads(out)
    assert row["text"] == "please review qxzjkwpvm"
    assert row["garbled"] is True
    assert row["profile"] == "english"
    assert {"start", "end", "reason"} <= set(row["spans"][0])
    assert {"strategy", "score", "applicable", "reason"} <= set(
        row["signals"][0]
    )


def test_profile_and_strategy_selection(capsys):
    _, out, _ = run(
        capsys, ["check", "--profile", "corruption", "-t", "CafÃ© au lait"]
    )
    assert out.startswith("garbled\t")
    _, out, _ = run(
        capsys,
        ["check", "--strategy", "control_characters", "-t", "hello\x00x"],
    )
    assert out.startswith("garbled\t")
    code, _, err = run(capsys, ["check", "--profile", "nope", "-t", "x"])
    assert code == 2 and "unknown profile" in err


def test_threshold_and_allowlist(capsys, tmp_path):
    words = tmp_path / "allow.txt"
    words.write_text("# domain words\nqxzjkwpv\n\n", encoding="utf-8")
    _, out, _ = run(
        capsys, ["check", "--allowlist", str(words), "-t", "hello qxzjkwpv"]
    )
    assert out.startswith("clean\t")
    _, out, _ = run(capsys, ["check", "--threshold", "0.99", "-t", "qxzjkwpv"])
    assert out.startswith("clean\t")


def test_field_mode(capsys, monkeypatch):
    lines = "\n".join(
        [
            json.dumps({"id": 1, "msg": "hello world"}),
            "not json",
            json.dumps({"id": 3}),
            json.dumps({"id": 4, "msg": 7}),
            json.dumps({"id": 5, "msg": "qxzjkwpv"}),
        ]
    )
    code, out, err = run(
        capsys, ["check", "--field", "msg"], stdin=lines,
        monkeypatch=monkeypatch,
    )
    rows = [json.loads(line) for line in out.splitlines()]
    assert [r["id"] for r in rows] == [1, 5]
    assert rows[0]["pygarble"]["garbled"] is False
    assert rows[1]["pygarble"]["garbled"] is True
    assert code == 2
    assert "line 2" in err and "line 3" in err and "line 4" in err


def test_missing_file_is_exit_2(capsys, tmp_path):
    code, _, err = run(capsys, ["check", str(tmp_path / "missing.txt")])
    assert code == 2 and "missing.txt" in err


def test_version_via_module():
    out = subprocess.check_output(
        [sys.executable, "-m", "pygarble", "--version"], text=True
    )
    assert out.strip() == f"pygarble {__version__}"


def test_help_does_not_import_data(capsys):
    program = (
        "import sys; from pygarble.cli import build_parser; "
        "build_parser(); "
        "assert 'pygarble.data.words' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_cli.py -q`
Expected: ImportError on `pygarble.cli`.

- [ ] **Step 3: Implement `pygarble/cli.py`**

```python
"""Command-line interface: check, score, analyze and calibrate texts."""

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

EXIT_OK = 0
EXIT_GARBLED = 1
EXIT_ERROR = 2


class CliError(Exception):
    """User-facing error; message goes to stderr with exit code 2."""


def load_allowlist(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as error:
        raise CliError(f"cannot read allowlist {path}: {error}")
    words = []
    for line in lines:
        word = line.split("#", 1)[0].strip()
        if word:
            words.append(word)
    return words


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="pygarble",
        description="Deterministic gibberish detection for English text.",
    )
    parser.add_argument(
        "--version", action="version", version=f"pygarble {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    def add_common(sub: argparse.ArgumentParser, default_format: str) -> None:
        sub.add_argument(
            "inputs",
            nargs="*",
            help="files to read, one text per line; '-' or none = stdin",
        )
        sub.add_argument(
            "-t",
            "--text",
            action="append",
            default=None,
            help="evaluate this text instead of reading inputs (repeatable)",
        )
        group = sub.add_mutually_exclusive_group()
        group.add_argument(
            "--profile", default=None, help="ensemble profile (default english)"
        )
        group.add_argument(
            "--strategy", default=None, help="single strategy name"
        )
        sub.add_argument("--threshold", type=float, default=0.5)
        sub.add_argument(
            "--allowlist", default=None, help="file of words never flagged"
        )
        sub.add_argument(
            "--format",
            choices=["text", "tsv", "jsonl"],
            default=default_format,
        )
        sub.add_argument(
            "--field",
            default=None,
            help="read JSON objects and evaluate this field; echo JSONL",
        )

    add_common(subparsers.add_parser("check", help="flag garbled lines"), "text")
    add_common(subparsers.add_parser("score", help="print scores"), "text")
    add_common(
        subparsers.add_parser("analyze", help="print full analyses"), "jsonl"
    )
    return parser


def make_detector(args: argparse.Namespace) -> Any:
    from . import EnsembleDetector, GarbleDetector, Strategy

    allowlist = load_allowlist(args.allowlist) if args.allowlist else None
    try:
        if args.strategy:
            return GarbleDetector(
                Strategy(args.strategy),
                threshold=args.threshold,
                allowlist=allowlist,
            )
        return EnsembleDetector(
            threshold=args.threshold,
            profile=args.profile or "english",
            allowlist=allowlist,
        )
    except ValueError as error:
        raise CliError(str(error))


def iter_lines(inputs: List[str]) -> Iterator[str]:
    sources = inputs or ["-"]
    for source in sources:
        if source == "-":
            for line in sys.stdin.read().splitlines():
                yield line
            continue
        try:
            with open(source, encoding="utf-8") as handle:
                for line in handle.read().splitlines():
                    yield line
        except OSError as error:
            raise CliError(f"cannot read {source}: {error}")


def analysis_row(text: str, analysis: Any) -> Dict[str, Any]:
    return {
        "text": text,
        "garbled": analysis.garbled,
        "score": analysis.score,
        "status": analysis.status,
        "profile": analysis.profile,
        "spans": [asdict(span) for span in analysis.spans],
        "signals": [
            {
                "strategy": s.strategy,
                "score": s.score,
                "applicable": s.applicable,
                "reason": s.reason,
            }
            for s in analysis.signals
        ],
    }


def format_row(command: str, fmt: str, text: str, analysis: Any) -> str:
    if fmt == "jsonl":
        return json.dumps(analysis_row(text, analysis), ensure_ascii=False)
    if fmt == "tsv":
        return "\t".join(
            [
                "1" if analysis.garbled else "0",
                f"{analysis.score:.4f}",
                analysis.status,
                text,
            ]
        )
    if command == "score":
        return f"{analysis.score:.4f}\t{text}"
    label = (
        "garbled"
        if analysis.garbled
        else ("insufficient" if analysis.status == "insufficient_evidence" else "clean")
    )
    return f"{label}\t{text}"


def run_texts(args: argparse.Namespace, out: Any, err: Any) -> int:
    detector = make_detector(args)
    any_garbled = False
    had_error = False
    if args.text is not None:
        pairs: Iterable[Tuple[int, Any]] = enumerate(args.text, 1)
    else:
        pairs = enumerate(iter_lines(args.inputs), 1)
    for number, line in pairs:
        if args.field is None:
            analysis = detector.analyze(line)
            any_garbled = any_garbled or analysis.garbled
            out.write(format_row(args.command, args.format, line, analysis) + "\n")
            continue
        try:
            obj = json.loads(line)
            value = obj[args.field]
            if not isinstance(obj, dict) or not isinstance(value, str):
                raise TypeError("field is not a string")
        except (ValueError, KeyError, TypeError) as error:
            err.write(f"line {number}: {error}\n")
            had_error = True
            continue
        analysis = detector.analyze(value)
        any_garbled = any_garbled or analysis.garbled
        obj["pygarble"] = analysis_row(value, analysis)
        del obj["pygarble"]["text"]
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if had_error:
        return EXIT_ERROR
    if args.command == "check" and any_garbled:
        return EXIT_GARBLED
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_texts(args, sys.stdout, sys.stderr)
    except CliError as error:
        sys.stderr.write(f"pygarble: {error}\n")
        return EXIT_ERROR
```

`pygarble/__main__.py`:

```python
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
```

`pyproject.toml`, after `[project.urls]`:

```toml
[project.scripts]
pygarble = "pygarble.cli:main"
```

- [ ] **Step 4: Run tests, lint, reinstall the entry point**

Run: `pip install -e . -q && python -m pytest tests/test_cli.py -q -W error::FutureWarning && pygarble --version && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble`
Expected: all pass; `pygarble 0.9.0` printed (version bump is Task 6). If mypy complains about `Any` returns, annotate `make_detector` return as `Any` (already) and `analysis: Any`.

Note: `test_profile_and_strategy_selection` asserts the message contains "unknown profile"; `EnsembleDetector` raises `ValueError("unknown profile: nope")`, which `CliError` forwards.

- [ ] **Step 5: CHANGELOG + commit**

Under a new `## [Unreleased]` → `### Added` at the top of CHANGELOG (above `## [0.9.0]`): `- Command-line interface: \`pygarble check|score|analyze\` reads files or stdin, supports --profile/--strategy/--threshold/--allowlist, text/TSV/JSONL output, JSON --field mode, and exit code 1 when any input is garbled.`

```bash
git add pygarble/cli.py pygarble/__main__.py pyproject.toml tests/test_cli.py CHANGELOG.md
git commit -m "feat: command-line interface (check, score, analyze)"
```

### Task 2: `llm_output` profile

**Files:**
- Modify: `pygarble/ensemble.py:16-35` (PROFILES), `CHANGELOG.md`
- Create: `tests/test_profile_llm_output.py`

- [ ] **Step 1: Failing tests**

```python
"""llm_output profile: degenerate model output, quiet on technical prose."""

import pytest

from pygarble import EnsembleDetector
from pygarble.ensemble import PROFILES

PARAGRAPH = (
    "Sure. To rotate the logs, set the handler to RotatingFileHandler with "
    "a maximum size of ten megabytes and keep five backups. Restart the "
    "service afterwards and confirm that the new file is being written. "
    "If nothing appears, check the directory permissions first."
)


def test_profile_members():
    assert [s.value for s in PROFILES["llm_output"]] == [
        "repetition",
        "control_characters",
        "mojibake",
        "local_anomaly",
    ]
    assert EnsembleDetector(profile="llm_output").voting == "any"


@pytest.mark.parametrize(
    "text",
    [
        "the the the the the the the the",
        "and so on and so on and so on and so on and so on",
        "I am happy to help! I am happy to help! I am happy to help! "
        "I am happy to help! I am happy to help!",
        "The cafÃ© was closed",
        "Result: ���",
        "hello\x00world",
        "Please review the qxzkvbwq qzxkvjwp xkqzvbwr output carefully",
    ],
)
def test_degenerate_output_is_flagged(text):
    assert EnsembleDetector(profile="llm_output").predict(text) is True


@pytest.mark.parametrize(
    "text",
    [
        PARAGRAPH,
        'def parse(row):\n    return row.split(",")',
        "x = x + x + x + x",
        "Use Kubernetes with Prometheus and Grafana on GKE.",
        "The API returned 200 OK with ETag W/\"abc123\".",
    ],
)
def test_technical_prose_is_clean(text):
    assert EnsembleDetector(profile="llm_output").predict(text) is False
```

- [ ] **Step 2: Run** → KeyError on the profile.

- [ ] **Step 3: Implement** — add to `PROFILES` in `pygarble/ensemble.py` after `"spoofing"`:

```python
    # Degenerate model output: loops, encoding damage, dense token salad.
    # Deliberately excludes the Markov/word-anomaly members so technical
    # prose, code and product names stay quiet.
    "llm_output": (
        Strategy.REPETITION,
        Strategy.CONTROL_CHARACTERS,
        Strategy.MOJIBAKE,
        Strategy.LOCAL_ANOMALY,
    ),
```

- [ ] **Step 4: Run** `python -m pytest tests/test_profile_llm_output.py tests/test_ensemble.py -q -W error::FutureWarning` → all pass (the existing profile-parametrized tests pick up the sixth profile). If a "flagged" case does not fire, first check Repetition's thresholds on the text (`GarbleDetector(Strategy.REPETITION).score(text)`); if the text genuinely needs a longer loop, lengthen the test text rather than changing the strategy. If a "clean" case fires, report which member fired; do not remove the case without a ruling.

- [ ] **Step 5: CHANGELOG + commit**

Append under `### Added`: `- \`llm_output\` profile (repetition, control characters, mojibake, local anomaly): a deterministic pre-check for degenerate model output that stays quiet on code and technical prose.`

```bash
git add pygarble/ensemble.py tests/test_profile_llm_output.py CHANGELOG.md
git commit -m "feat: llm_output profile for degenerate model output"
```

### Task 3: Calibration helper and `calibrate` subcommand

**Files:**
- Create: `pygarble/calibration.py`, `tests/test_calibration.py`
- Modify: `pygarble/__init__.py` (exports), `pygarble/cli.py` (subcommand), `tests/test_cli.py` (append), `CHANGELOG.md`

**Interfaces:**
- Produces: `pygarble.calibrate(detector, garbled, clean, *, objective="f1", max_false_positive_rate=None, thresholds=None) -> CalibrationReport`; `CalibrationReport(recommended, objective, max_false_positive_rate, garbled, clean, points)`; `ThresholdPoint(threshold, precision, recall, f1, false_positive_rate)`.

- [ ] **Step 1: Failing tests**

```python
"""Threshold calibration: exact math on stub scores, one real run."""

import pytest

from pygarble import (
    CalibrationReport,
    EnsembleDetector,
    ThresholdPoint,
    calibrate,
)


class Stub:
    def __init__(self, table):
        self.table = table

    def score(self, texts):
        return [self.table[t] for t in texts]


def test_f1_picks_best_threshold_ties_to_highest():
    stub = Stub({"g1": 0.9, "g2": 0.7, "c1": 0.6, "c2": 0.1})
    report = calibrate(stub, ["g1", "g2"], ["c1", "c2"])
    assert isinstance(report, CalibrationReport)
    assert report.objective == "f1"
    assert report.garbled == 2 and report.clean == 2
    assert report.recommended.threshold == pytest.approx(0.7)
    assert report.recommended.f1 == pytest.approx(1.0)
    assert report.recommended.false_positive_rate == 0.0
    assert [p.threshold for p in report.points] == pytest.approx(
        [0.0, 0.1, 0.6, 0.7, 0.9, 1.0]
    )


def test_point_math():
    stub = Stub({"g": 0.8, "c": 0.8})
    report = calibrate(stub, ["g"], ["c"], thresholds=[0.5, 0.9])
    low, high = report.points
    assert isinstance(low, ThresholdPoint)
    assert low.precision == pytest.approx(0.5)
    assert low.recall == 1.0 and low.false_positive_rate == 1.0
    assert high.precision == 1.0  # nothing flagged: precision defined as 1
    assert high.recall == 0.0 and high.f1 == 0.0


def test_max_fpr_objective_and_fallback():
    stub = Stub({"g1": 0.9, "g2": 0.4, "c1": 0.5, "c2": 0.1})
    report = calibrate(
        stub, ["g1", "g2"], ["c1", "c2"], objective="max_fpr",
        max_false_positive_rate=0.0,
    )
    assert report.recommended.threshold == pytest.approx(0.9)
    assert report.recommended.recall == pytest.approx(0.5)
    strict = calibrate(
        Stub({"g": 0.3, "c": 0.9}), ["g"], ["c"], objective="max_fpr",
        max_false_positive_rate=0.0,
    )
    assert strict.recommended.threshold == 1.0


def test_identical_scores_do_not_divide_by_zero():
    report = calibrate(Stub({"g": 0.0, "c": 0.0}), ["g"], ["c"])
    assert report.recommended.threshold in (0.0, 1.0)
    assert 0.0 <= report.recommended.f1 <= 1.0


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (dict(garbled=[], clean=["a"]), ValueError),
        (dict(garbled=["a"], clean=[]), ValueError),
        (dict(garbled=["a"], clean=["b"], objective="nope"), ValueError),
        (dict(garbled=["a"], clean=["b"], objective="max_fpr"), ValueError),
        (dict(garbled=[1], clean=["b"]), TypeError),
    ],
)
def test_validation(kwargs, error):
    with pytest.raises(error):
        calibrate(EnsembleDetector(), **kwargs)


def test_real_detector_round_trip():
    garbled = ["qxzjkwpv bnmqwer", "asdfghjkl", "xkrf plmq bvzt nwsd"]
    clean = ["hello world", "the quick brown fox", "please send the invoice"]
    report = calibrate(EnsembleDetector(), garbled, clean)
    assert report.recommended.recall == 1.0
    assert report.recommended.false_positive_rate == 0.0
    detector = EnsembleDetector(threshold=report.recommended.threshold)
    assert detector.predict(garbled) == [True, True, True]
    assert detector.predict(clean) == [False, False, False]
```

Append to `tests/test_cli.py`:

```python
def test_calibrate_subcommand(capsys, tmp_path):
    garbled = tmp_path / "g.txt"
    clean = tmp_path / "c.txt"
    garbled.write_text("qxzjkwpv bnmqwer\nasdfghjkl\n", encoding="utf-8")
    clean.write_text("hello world\nthe quick brown fox\n", encoding="utf-8")
    code, out, _ = run(
        capsys, ["calibrate", "--garbled", str(garbled), "--clean", str(clean)]
    )
    assert code == 0
    assert "recommended threshold:" in out.splitlines()[-1]
    code, out, _ = run(
        capsys,
        ["calibrate", "--garbled", str(garbled), "--clean", str(clean),
         "--format", "jsonl"],
    )
    row = json.loads(out)
    assert 0.0 <= row["recommended"]["threshold"] <= 1.0
    assert row["objective"] == "f1"
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement `pygarble/calibration.py`**

```python
"""Pick a decision threshold from labeled examples."""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .validation import unit_interval, validate_batch


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
    objective: str
    max_false_positive_rate: Optional[float]
    garbled: int
    clean: int
    points: Tuple[ThresholdPoint, ...]


def _point(
    threshold: float, garbled: Sequence[float], clean: Sequence[float]
) -> ThresholdPoint:
    tp = sum(1 for s in garbled if s >= threshold)
    fp = sum(1 for s in clean if s >= threshold)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / len(garbled)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return ThresholdPoint(threshold, precision, recall, f1, fp / len(clean))


def calibrate(
    detector: Any,
    garbled: Iterable[str],
    clean: Iterable[str],
    *,
    objective: str = "f1",
    max_false_positive_rate: Optional[float] = None,
    thresholds: Optional[Iterable[float]] = None,
) -> CalibrationReport:
    """Score both samples once and sweep candidate thresholds.

    objective="f1" picks the highest F1; objective="max_fpr" picks the
    highest recall whose false-positive rate stays within
    max_false_positive_rate. Ties resolve to the highest threshold. Under
    voting="majority" the ensemble decision counts member votes, so the
    recommended threshold is applied per member; the report still measures
    the aggregate score.
    """
    garbled_texts = list(garbled)
    clean_texts = list(clean)
    if not garbled_texts or not clean_texts:
        raise ValueError("garbled and clean must each contain at least one text")
    validate_batch(garbled_texts)
    validate_batch(clean_texts)
    if objective not in ("f1", "max_fpr"):
        raise ValueError("objective must be 'f1' or 'max_fpr'")
    limit: Optional[float] = None
    if objective == "max_fpr":
        if max_false_positive_rate is None:
            raise ValueError(
                "max_false_positive_rate is required for objective='max_fpr'"
            )
        limit = unit_interval("max_false_positive_rate", max_false_positive_rate)
    garbled_scores: List[float] = list(detector.score(garbled_texts))
    clean_scores: List[float] = list(detector.score(clean_texts))
    if thresholds is None:
        candidates = sorted(
            set(garbled_scores) | set(clean_scores) | {0.0, 1.0}
        )
    else:
        candidates = sorted({unit_interval("threshold", t) for t in thresholds})
    points = tuple(_point(t, garbled_scores, clean_scores) for t in candidates)
    if objective == "f1":
        best = max(points, key=lambda p: (p.f1, p.threshold))
    else:
        eligible = [p for p in points if p.false_positive_rate <= limit]
        if eligible:
            best = max(eligible, key=lambda p: (p.recall, p.threshold))
        else:
            best = _point(1.0, garbled_scores, clean_scores)
    return CalibrationReport(
        best, objective, limit, len(garbled_texts), len(clean_texts), points
    )
```

`pygarble/__init__.py`: add `from .calibration import CalibrationReport, ThresholdPoint, calibrate` and the three names to `__all__`.

`pygarble/cli.py`: in `build_parser`, add after the `analyze` subparser:

```python
    cal = subparsers.add_parser("calibrate", help="recommend a threshold")
    cal.add_argument("--garbled", required=True, help="file of garbled lines")
    cal.add_argument("--clean", required=True, help="file of clean lines")
    group = cal.add_mutually_exclusive_group()
    group.add_argument("--profile", default=None)
    group.add_argument("--strategy", default=None)
    cal.add_argument("--allowlist", default=None)
    cal.add_argument("--objective", choices=["f1", "max_fpr"], default="f1")
    cal.add_argument("--max-fpr", type=float, default=None)
    cal.add_argument("--format", choices=["text", "jsonl"], default="text")
    cal.set_defaults(threshold=0.5)
```

and a runner, dispatched from `main` when `args.command == "calibrate"`:

```python
def run_calibrate(args: argparse.Namespace, out: Any) -> int:
    from .calibration import calibrate

    detector = make_detector(args)
    garbled = [line for line in iter_lines([args.garbled]) if line.strip()]
    clean = [line for line in iter_lines([args.clean]) if line.strip()]
    try:
        report = calibrate(
            detector,
            garbled,
            clean,
            objective=args.objective,
            max_false_positive_rate=args.max_fpr,
        )
    except ValueError as error:
        raise CliError(str(error))
    if args.format == "jsonl":
        out.write(json.dumps(asdict(report)) + "\n")
        return EXIT_OK
    out.write("threshold\tprecision\trecall\tf1\tfpr\n")
    for p in report.points:
        out.write(
            f"{p.threshold:.4f}\t{p.precision:.3f}\t{p.recall:.3f}\t"
            f"{p.f1:.3f}\t{p.false_positive_rate:.3f}\n"
        )
    r = report.recommended
    out.write(
        f"recommended threshold: {r.threshold:.4f} "
        f"(f1={r.f1:.3f}, fpr={r.false_positive_rate:.3f})\n"
    )
    return EXIT_OK
```

In `main`: `if args.command == "calibrate": return run_calibrate(args, sys.stdout)` inside the `try`.

- [ ] **Step 4: Run** `python -m pytest tests/test_calibration.py tests/test_cli.py -q -W error::FutureWarning && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble` → all pass.

- [ ] **Step 5: CHANGELOG + commit**

Append under `### Added`: `- \`pygarble.calibrate(detector, garbled, clean)\` sweeps thresholds over labeled samples and recommends one by F1 or by a maximum false-positive rate; \`pygarble calibrate\` does the same from files.`

```bash
git add pygarble/calibration.py pygarble/__init__.py pygarble/cli.py tests/test_calibration.py tests/test_cli.py CHANGELOG.md
git commit -m "feat: threshold calibration helper and CLI subcommand"
```

### Task 4: Language-neutral JSON data tables

**Files:**
- Modify: `scripts/generate_data.py` (write JSON, hash `*.json`), `docs/contributing.rst` (one sentence), `CHANGELOG.md`
- Create: `pygarble/data/words.json`, `pygarble/data/bigrams.json`, `pygarble/data/trigrams.json` (generated), `tests/test_data_json.py`
- Regenerate: `pygarble/data/manifest.json`

- [ ] **Step 1: Failing test**

```python
"""JSON data tables must equal the Python tables byte-for-byte in content."""

import json
from pathlib import Path

import pytest

from pygarble.data import (
    BIGRAM_LOG_PROBS,
    COMMON_TRIGRAMS,
    DEFAULT_LOG_PROB,
    ENGLISH_WORDS,
)

DATA = Path(__file__).resolve().parent.parent / "pygarble" / "data"


def load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not present (wheel install)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_words_json_matches():
    words = load("words.json")
    assert words == sorted(words)
    assert set(words) == set(ENGLISH_WORDS)


def test_bigrams_json_matches():
    table = load("bigrams.json")
    assert table["default_log_prob"] == DEFAULT_LOG_PROB
    assert table["log_probs"] == dict(BIGRAM_LOG_PROBS)


def test_trigrams_json_matches():
    trigrams = load("trigrams.json")
    assert trigrams == sorted(trigrams)
    assert set(trigrams) == set(COMMON_TRIGRAMS)


def test_manifest_lists_json_files():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert {"words.json", "bigrams.json", "trigrams.json"} <= set(
        manifest["files"]
    )
```

- [ ] **Step 2: Run** → skips/fails because the files do not exist and the manifest lacks them. (The skip guard must not hide the failure here: run with `-rs` and confirm the manifest test fails.)

- [ ] **Step 3: Implement in `scripts/generate_data.py`**

Add a writer:

```python
def write_json(payload: object, filepath: Path) -> None:
    """Byte-reproducible JSON: sorted keys, compact, ASCII, trailing newline."""
    filepath.write_text(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
```

In `main`, after the three `write_*_file` calls:

```python
        write_json(sorted(words), directory / "words.json")
        write_json(
            {"default_log_prob": DEFAULT_LOG_PROB, "log_probs": bigrams},
            directory / "bigrams.json",
        )
        write_json(sorted(trigrams), directory / "trigrams.json")
```

and change the manifest `files` comprehension to hash both kinds:

```python
            "files": {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted(directory.iterdir())
                if file.suffix in (".py", ".json") and file.name != "manifest.json"
            },
```

Check that `bigrams` at that point is the plain `Dict[str, float]` the `.py` writer receives (read `write_bigrams_file` and `compute_bigram_probabilities`); if the `.py` file applies rounding when writing, apply the same rounding to the JSON payload so the equality test holds. Confirm `--check` compares every generated file (it iterates `directory.iterdir()`).

- [ ] **Step 4: Regenerate and verify**

Run: `python scripts/generate_data.py && python scripts/generate_data.py --check && python -m pytest tests/test_data_json.py tests/test_english_api.py -q -W error::FutureWarning && git status --short pygarble/data`
Expected: check passes; tests pass; `git status` shows only the three new `.json` files and `manifest.json` modified (the `.py` tables must be byte-identical: `git diff --stat pygarble/data/*.py` prints nothing).

Run: `python -m build --wheel --outdir /tmp/pygarble-wheel && unzip -l /tmp/pygarble-wheel/*.whl | grep -c "data/.*json"`
Expected: `1` (only `manifest.json`).

- [ ] **Step 5: Docs, CHANGELOG, commit**

`docs/contributing.rst`, in the data regeneration paragraph: `The generator also writes language-neutral JSON copies of the tables (\`\`words.json\`\`, \`\`bigrams.json\`\`, \`\`trigrams.json\`\`) for ports to other languages; they are hashed in \`\`manifest.json\`\` and verified by \`\`--check\`\`.`

CHANGELOG `### Added`: `- Language-neutral JSON copies of the word, bigram and trigram tables under \`pygarble/data/\` (repo and sdist only), hashed in the manifest, as the shared source for ports.`

```bash
git add scripts/generate_data.py pygarble/data/*.json tests/test_data_json.py docs/contributing.rst CHANGELOG.md
git commit -m "feat: emit language-neutral JSON data tables with manifest hashes"
```

### Task 5: Golden corpus

**Files:**
- Create: `regression/golden.py`, `regression/golden.jsonl`, `regression/golden.sha256`, `tests/test_golden.py`
- Modify: `.github/workflows/test.yml` (quality job), `docs/contributing.rst`, `CHANGELOG.md`

- [ ] **Step 1: Write `regression/golden.py`**

```python
"""Frozen detector outputs that every implementation must reproduce.

Rows are JSONL: text, profile, garbled, score (12 decimals), status and
spans as [start, end, reason] with code-point offsets. Regenerate with
--write only when a behaviour change is intended; --check is run in CI.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import EnsembleDetector  # noqa: E402
from pygarble.ensemble import PROFILES  # noqa: E402
from regression.evaluate import challenge  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "golden.jsonl"
CHECKSUM = ROOT / "golden.sha256"

EDGE_INPUTS = [
    "",
    "   ",
    "a",
    "\U0001F44D\U0001F44F\U0001F64F",
    "你好世界",
    "Привет мир",
    "مرحبا بالعالم",
    "CafÃ© crÃ¨me",
    "result ��",
    "hello\x00world",
    "ab" * 2500,
    "https://example.com/path?q=1",
    "/usr/local/bin/python3",
    "4f8a9b2c1d3e5f6a7b8c9d0e",
    "aGVsbG8gd29ybGQgdGhpcyBpcw==",
    "123e4567-e89b-12d3-a456-426614174000",
    "don’t stop believin’",
    "Alice Nguyen and Priya Ramaswamy",
    "NASA FBI NATO UNESCO",
    "Order 000123 shipped 2026-09-26 at 10:00",
    "pаypal login",
    "café latté",
    "test test test",
    (
        "Sure. To rotate the logs, set the handler to RotatingFileHandler "
        "with a maximum size of ten megabytes and keep five backups. "
        "Restart the service afterwards and confirm that the new file is "
        "being written. If nothing appears, check the permissions first."
    ),
]


def texts():
    seen = []
    for split in ("development", "holdout"):
        for case in challenge(split):
            seen.append(case["text"])
    seen.extend(EDGE_INPUTS)
    return seen


def rows():
    detectors = {name: EnsembleDetector(profile=name) for name in sorted(PROFILES)}
    for text in texts():
        for name, detector in detectors.items():
            analysis = detector.analyze(text)
            yield {
                "text": text,
                "profile": name,
                "garbled": analysis.garbled,
                "score": round(analysis.score, 12),
                "status": analysis.status,
                "spans": [[s.start, s.end, s.reason] for s in analysis.spans],
            }


def render():
    return "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        for row in rows()
    )


def check():
    expected = OUTPUT.read_text(encoding="utf-8")
    if hashlib.sha256(expected.encode("utf-8")).hexdigest() != (
        CHECKSUM.read_text().strip()
    ):
        print("golden.jsonl does not match golden.sha256", file=sys.stderr)
        return 1
    actual = render()
    if actual == expected:
        print(f"golden corpus reproduced ({actual.count(chr(10))} rows)")
        return 0
    shown = 0
    for old, new in zip(expected.splitlines(), actual.splitlines()):
        if old != new:
            print(f"- {old}\n+ {new}", file=sys.stderr)
            shown += 1
            if shown == 10:
                break
    print("golden corpus differs; run --write if intended", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        content = render()
        OUTPUT.write_text(content, encoding="utf-8")
        CHECKSUM.write_text(
            hashlib.sha256(content.encode("utf-8")).hexdigest() + "\n"
        )
        print(f"wrote {content.count(chr(10))} rows")
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `tests/test_golden.py`**

```python
"""The committed golden corpus must reproduce exactly."""

import pytest


def test_golden_corpus_reproduces():
    golden = pytest.importorskip(
        "regression.golden", reason="regression/ is not shipped in sdist"
    )
    assert golden.render() == golden.OUTPUT.read_text(encoding="utf-8")


def test_golden_rows_cover_every_profile_and_edge_input():
    golden = pytest.importorskip("regression.golden")
    from pygarble.ensemble import PROFILES

    rows = [
        __import__("json").loads(line)
        for line in golden.OUTPUT.read_text(encoding="utf-8").splitlines()
    ]
    assert {r["profile"] for r in rows} == set(PROFILES)
    texts = {r["text"] for r in rows}
    assert set(golden.EDGE_INPUTS) <= texts
    assert len(rows) == len(texts) * len(PROFILES)
```

- [ ] **Step 3: Generate, check, test**

Run: `python regression/golden.py --write && python regression/golden.py --check && python -m pytest tests/test_golden.py -q -W error::FutureWarning`
Expected: `wrote 936 rows` (156 texts × 6 profiles), `golden corpus reproduced (936 rows)`, 2 passed.

Prove the check fails loudly: temporarily edit one score in `golden.jsonl`, run `--check`, expect exit 1 with a `-`/`+` pair, then `git checkout regression/golden.jsonl` (the file is not yet tracked at this point, so instead re-run `--write`).

- [ ] **Step 4: CI, docs, CHANGELOG**

`.github/workflows/test.yml`, quality job, after the `generate_data.py --check` step:

```yaml
    - name: Golden corpus reproduces
      run: python regression/golden.py --check
```

`docs/contributing.rst`: a short "Golden corpus" paragraph: what it is, `--check` in CI, regenerate with `--write` only for intended behaviour changes, and "any port must reproduce it exactly; span offsets are Unicode code points".

CHANGELOG `### Added`: `- Golden corpus \`regression/golden.jsonl\` (challenge cases and edge inputs × every profile) with a CI check; ports in other languages are held to it.`

```bash
git add regression/golden.py regression/golden.jsonl regression/golden.sha256 tests/test_golden.py .github/workflows/test.yml docs/contributing.rst CHANGELOG.md
git commit -m "feat: golden corpus of frozen detector outputs with CI check"
```

### Task 6: Docs, README, version, CHANGELOG, final gate

**Files:**
- Modify: `README.md` (rewrite), `docs/index.rst` (toctree), `docs/api.rst`, `docs/strategy-guide.rst`, `docs/migration.rst`, `docs/quickstart.rst` (mention CLI), `pygarble/__init__.py` (version), `CHANGELOG.md`, `docs/strategies.rst` (regenerated if the generator includes profiles; else untouched)
- Create: `docs/cli.rst`, `docs/calibration.rst`

- [ ] **Step 1: Version and CHANGELOG**

`pygarble/__init__.py`: `__version__ = "0.10.0"`. CHANGELOG: rename `## [Unreleased]` to `## [0.10.0] - 2026-09-26`; add a `### Notes` line under it: `- 0.9.0 was prepared but never published to PyPI; users upgrading from 0.8.0 should read both entries.`

- [ ] **Step 2: `docs/cli.rst`** (new; every code block executable or a literal shell transcript)

```rst
Command-line interface
======================

``pygarble`` is installed as a console script; ``python -m pygarble`` is
equivalent.

Check lines from stdin or files
-------------------------------

.. code-block:: console

   $ printf 'hello world\nasdfghjkl\n' | pygarble check
   clean	hello world
   garbled	asdfghjkl
   $ echo $?
   1

Exit code 0 means nothing was flagged, 1 means at least one line was, 2
means a usage or input error. Blank lines print ``insufficient``.

Scores and full analyses
------------------------

.. code-block:: console

   $ pygarble score -t "please review qxzjkwpvm"
   0.9000	please review qxzjkwpvm
   $ pygarble analyze -t "please review qxzjkwpvm"
   {"text": "please review qxzjkwpvm", "garbled": true, ...}

Options shared by ``check``, ``score`` and ``analyze``
------------------------------------------------------

``--profile NAME`` or ``--strategy NAME``, ``--threshold FLOAT``,
``--allowlist FILE`` (one word per line, ``#`` comments), ``--format
text|tsv|jsonl`` and ``--field NAME`` for JSON-lines input, which echoes
each object with a ``pygarble`` key added.

.. code-block:: console

   $ printf '{"id":1,"msg":"hello"}\n' | pygarble check --field msg
   {"id": 1, "msg": "hello", "pygarble": {"garbled": false, ...}}

Calibrate a threshold
---------------------

.. code-block:: console

   $ pygarble calibrate --garbled bad.txt --clean good.txt
   threshold	precision	recall	f1	fpr
   ...
   recommended threshold: 0.6000 (f1=0.980, fpr=0.010)

See :doc:`calibration` for the Python API.
```

(The score `0.9000` in the transcript must match the real output; run it and paste the real value. Console blocks are not executed by the snippet runner.)

- [ ] **Step 3: `docs/calibration.rst`** (new)

```rst
Calibrating the threshold
=========================

Scores are heuristic, not probabilities, so the right decision threshold
depends on your data. ``calibrate`` scores labeled samples once and sweeps
every observed score as a candidate threshold.

.. code-block:: python

   from pygarble import EnsembleDetector, calibrate

   garbled = ["qxzjkwpv bnmqwer", "asdfghjkl"]
   clean = ["hello world", "please send the invoice"]
   report = calibrate(EnsembleDetector(), garbled, clean)
   detector = EnsembleDetector(threshold=report.recommended.threshold)
   assert detector.predict(garbled) == [True, True]
   assert detector.predict(clean) == [False, False]

``objective="f1"`` (default) maximises F1. ``objective="max_fpr"`` with
``max_false_positive_rate=0.01`` picks the highest recall whose
false-positive rate stays at or below one percent, or threshold ``1.0`` if
no threshold qualifies. Ties resolve to the higher threshold.

``report.points`` lists precision, recall, F1 and false-positive rate at
every candidate. Under ``voting="majority"`` the ensemble counts member
votes, so the threshold applies per member; the report still measures the
aggregate score.
```

- [ ] **Step 4: Other docs**

- `docs/index.rst` toctree: insert `cli` and `calibration` after `quickstart`.
- `docs/api.rst`: add `llm_output` to the profile table; add a "Calibration" subsection documenting `calibrate`, `CalibrationReport`, `ThresholdPoint` (signatures from Task 3).
- `docs/strategy-guide.rst`: add an "LLM output" section (3-5 sentences: what the profile catches, what it deliberately ignores, the guard pattern `if EnsembleDetector(profile="llm_output").predict(response): retry`).
- `docs/quickstart.rst`: one paragraph pointing at the CLI with one `console` example.
- `docs/migration.rst`: `0.10.0` section: no breaking changes; new console script `pygarble`; new profile; new `calibrate`; JSON tables are repo/sdist only.

- [ ] **Step 5: README rewrite**

Replace the top of `README.md` through the end of "Quick start" with:

````markdown
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
````

Then keep the existing sections in this order, edited where noted: "Choose a profile" (add the `llm_output` row: `Repetition, control characters, mojibake, local anomaly; deterministic pre-check for degenerate model output, quiet on code and technical prose`), a new short "Command line" section (three console lines and a link to the CLI docs), a new short "Calibrate the threshold" section (the calibration.rst Python block), "Use an individual strategy", "Inspect explanations", "Configure domain vocabulary and limits" (fix the sentence that says allowlists apply only to four strategies: since 0.9.0 they apply to every strategy), "English-specific behavior", "Upgrading and evaluation" (add the golden corpus link), "Contributing", "License".

- [ ] **Step 6: Full gate**

```bash
pip install -e . -q && black --check pygarble tests scripts regression && isort --check-only pygarble tests scripts regression && flake8 pygarble tests scripts regression && mypy pygarble && python -m pytest -q -W error::FutureWarning && python scripts/update_strategy_docs.py --check && python scripts/generate_data.py --check && python regression/golden.py --check && python -m sphinx -b html -W --keep-going -q docs /tmp/pygarble-docs && python -m build --outdir /tmp/pygarble-final && pygarble --version && echo ALL-GREEN
```
Expected: `pygarble 0.10.0` then `ALL-GREEN`.

Then run the doc-snippet runner from `regression/`? No such script exists; use this one-off (fixed version from the 0.9.0 cycle):

```bash
python - <<'EOF'
import re, subprocess, sys, pathlib, textwrap
blocks = []
for path in ["README.md", *map(str, pathlib.Path("docs").glob("*.rst"))]:
    text = pathlib.Path(path).read_text()
    blocks += [(path, b) for b in re.findall(r"```python\n(.*?)```", text, re.S)]
    blocks += [(path, textwrap.dedent(b)) for b in re.findall(r".. code-block:: python\n\n((?:[ \t]+.*\n|\n)+)", text)]
ok = 0
for path, code in blocks:
    r = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, env={"PYTHONPATH": "."})
    if r.returncode:
        print("FAILED in", path, "\n", code, r.stderr); sys.exit(1)
    ok += 1
print("snippets ok:", ok)
EOF
```
Expected: `snippets ok: N` with no failures.

- [ ] **Step 7: Commit**

```bash
git add README.md docs pygarble/__init__.py CHANGELOG.md
git commit -m "docs: README overhaul, CLI and calibration guides; release 0.10.0"
```

---

## Manual follow-ups for the maintainer (not automated)

- Tag `v0.10.0` and let `release.yml` publish.
- Add GitHub topics: `gibberish-detection`, `text-validation`, `spam-detection`, `llm-guardrails`, `data-cleaning`, `python`.
- Switch GitHub Pages source to "GitHub Actions" (carried over from the 0.9.0 plan).
