# pygarble Cleanup and Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every confirmed defect from the 2026-09-26 audit, remove dead and duplicated code, and leave the repo releasable as 0.9.0 with no public API removed.

**Architecture:** Fixes land in dependency order: release plumbing first (no behavior change), then the core API layer (`base.py`, `options.py`, `detector.py`, `ensemble.py`, `validation.py`, `preprocessing.py`), then the strategies in the shipped profiles, then opt-in strategies, then cleanup, then contract tests and regression re-baseline. Each task is independently testable and committed on its own. New tests go in subject-named files (`tests/test_ensemble.py`, `tests/test_allowlist.py`, `tests/test_strategy_<name>.py`, `tests/test_contracts.py`), never in the date-named legacy files.

**Tech Stack:** Python 3.8+, pure stdlib, pytest, black (line length 79), isort, flake8, mypy. Regression tooling under `regression/`.

**Spec:** `docs/superpowers/specs/2026-09-26-audit-findings.md` (finding IDs like A1, C3, D11 below refer to it).

## Global Constraints

- Backward compatibility: no public name removed or renamed. `PronouncabilityStrategy` keeps working. Every kwarg in `pygarble/options.py` `PARAMETERS` stays accepted. Default voting stays `"any"` for named profiles and `"majority"` for custom strategy lists.
- Python >= 3.8 syntax only: no `list[str]` annotations, no `X | None`, no walrus in hot code, no `match`, no `str.removeprefix`.
- Zero runtime dependencies. `dependencies = []` stays.
- Line length 79 (black). Run `black pygarble tests scripts regression && isort pygarble tests scripts regression` before every commit.
- Every behavior change gets a line under `## [0.9.0]` in `CHANGELOG.md`.
- Never run `python regression/benchmark.py` before Task 3 lands (it overwrites tracked results on any invocation).
- Run the suite from the repo root with `python -m pytest -q`. If snippets are run with `python -c`, prefix with `PYTHONPATH=.` because an old pygarble 0.1.6 is installed in site-packages on this machine.
- Commit message format: `<type>: <summary>` with types `fix`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`. End each commit message with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

Input classes the spec implies but that need explicit pinning; each has a test added to the owning task below.

1. Allowlisted word attached to punctuation (`"ping asdfgh, then deploy"`) must still be scrubbed — Task 5.
2. Text where every word is allowlisted must return `insufficient_evidence`, not `garbled` — Task 5.
3. `EnsembleDetector(profile=..., strategy_kwargs={Strategy.X: {"typo": 1}})` must warn exactly once and point at the caller's file — Task 6.
4. WordAnomaly on text made only of URLs / numeric tokens must be not-applicable rather than 0.0-clean — Task 9.
5. A user-supplied `patterns={"consonant_cluster": ...}` override must still be honored after the consonant pattern is scoped to novel words — Task 11.

---

## Phase 0 — Release plumbing (no behavior change)

### Task 1: Remove broken release automation and stale files

**Files:**
- Delete: `.github/workflows/publish.yml`, `.github/workflows/test-pypi.yml`, `.github/workflows/publish-current.yml`, `.bumpversion.cfg`, `.github/instructions.txt`
- Modify: `.github/workflows/test.yml:3-7` (triggers), every `actions/setup-python@v4` → `@v5`

**Interfaces:**
- Consumes: nothing
- Produces: `release.yml` is the only publish path

- [ ] **Step 1: Confirm the files to delete are the ones the spec names**

Run: `ls .github/workflows .bumpversion.cfg .github/instructions.txt && grep -n "bump2version\|current_version" .github/workflows/*.yml .bumpversion.cfg`
Expected: `publish.yml` and `test-pypi.yml` reference `bump2version`; `.bumpversion.cfg` says `current_version = 0.3.1`.

- [ ] **Step 2: Delete them**

```bash
git rm .github/workflows/publish.yml .github/workflows/test-pypi.yml .github/workflows/publish-current.yml .bumpversion.cfg .github/instructions.txt
```

- [ ] **Step 3: Fix test.yml triggers and pin setup-python**

In `.github/workflows/test.yml` replace lines 3-7 with:

```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
```

Then: `sed -i '' 's/actions\/setup-python@v4/actions\/setup-python@v5/g' .github/workflows/*.yml`

- [ ] **Step 4: Verify workflow YAML still parses**

Run: `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('ok')"`
Expected: `ok` (if PyYAML is missing: `pip install pyyaml` into the dev venv, it is dev-only).

- [ ] **Step 5: Commit**

```bash
git add -A .github .bumpversion.cfg
git commit -m "ci: remove broken bump2version publish workflows and stale instructions

release.yml (tag-gated, version-checked) is the single publish path.
publish.yml and test-pypi.yml failed on a stale .bumpversion.cfg (0.3.1)
and checked out the pre-bump SHA; publish-current.yml could publish
without a tag. test.yml no longer triggers on non-existent branches.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 2: Packaging metadata and single-sourced version

**Files:**
- Modify: `pyproject.toml` (build-system, authors, classifiers, dynamic version, dev extra)
- Modify: `docs/conf.py:16-20`
- Modify: `.github/workflows/release.yml:44-50` (version check)
- Modify: `requirements-dev.txt`, `Makefile` (`lint`, `format`)

**Interfaces:**
- Produces: `pygarble.__version__` is the only version string in the repo.

- [ ] **Step 1: Write a test that the version is single-sourced**

Create `tests/test_packaging.py`:

```python
"""Packaging invariants that CI must keep true."""

import re
from pathlib import Path

import pygarble

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_has_no_static_version():
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version\s*=\s*"', text, re.M) is None
    assert 'dynamic = ["version"]' in text
    assert 'version = {attr = "pygarble.__version__"}' in text


def test_docs_conf_uses_package_version():
    text = (ROOT / "docs" / "conf.py").read_text()
    assert "pygarble.__version__" in text
    assert "version = '0." not in text


def test_author_email_is_not_placeholder():
    text = (ROOT / "pyproject.toml").read_text()
    assert "example.com" not in text
    assert pygarble.__email__ in text
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_packaging.py -q`
Expected: 3 failed.

- [ ] **Step 3: Edit pyproject.toml**

Replace the `[build-system]` block, the `version`/`authors` lines, the classifiers list, and the `[project.optional-dependencies]` block so they read:

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pygarble"
dynamic = ["version"]
description = "Deterministic, zero-dependency gibberish detection for English text"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Ujjwal Singh Rao", email = "ujjwalsrao@gmail.com"},
]
```

Add `"Programming Language :: Python :: 3.13",` after the 3.12 classifier.

```toml
[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black==24.8.0",
    "flake8>=3.8",
    "isort>=5.0",
    "pre-commit>=2.0",
    "mypy==1.11.2",
    "build>=1.0",
    "twine>=4.0",
]
docs = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
]
```

Add after `[tool.setuptools.package-data]`:

```toml
[tool.setuptools.dynamic]
version = {attr = "pygarble.__version__"}
```

- [ ] **Step 4: Edit docs/conf.py lines 16-20**

```python
copyright = '2026, pygarble contributors'
author = 'pygarble contributors'

from pygarble import __version__  # noqa: E402

version = __version__
release = __version__
```

- [ ] **Step 5: Edit release.yml version check**

Replace the `Check tag matches package version` step's `run:` block with:

```yaml
        run: |
          tag="${GITHUB_REF_NAME#v}"
          version=$(PYTHONPATH=. python -c "import pygarble; print(pygarble.__version__)")
          if [ "$tag" != "$version" ]; then
            echo "Tag v$tag does not match pygarble.__version__ $version" >&2
            exit 1
          fi
```

- [ ] **Step 6: Replace requirements-dev.txt contents and fix Makefile**

`requirements-dev.txt`:

```
# Development dependencies are declared in pyproject.toml.
-e .[dev,docs]
```

In `Makefile` replace the `lint` and `format` targets:

```make
lint:
	black --check pygarble tests scripts regression
	isort --check-only pygarble tests scripts regression
	flake8 pygarble tests scripts regression
	mypy pygarble

format:
	isort pygarble tests scripts regression
	black pygarble tests scripts regression
```

- [ ] **Step 7: Verify**

Run: `pip install -e ".[dev,docs]" -q && python -m pytest tests/test_packaging.py -q && python -c "import pygarble; print(pygarble.__version__)" && python -m build --sdist --wheel --outdir /tmp/pygarble-build -q && ls /tmp/pygarble-build`
Expected: 3 passed; `0.9.0`; `pygarble-0.9.0-py3-none-any.whl` and `pygarble-0.9.0.tar.gz`.

Run: `python -m sphinx -b html -W --keep-going -q docs /tmp/pygarble-docs`
Expected: no output (clean build).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml docs/conf.py .github/workflows/release.yml requirements-dev.txt Makefile tests/test_packaging.py
git commit -m "build: single-source version, real author metadata, setuptools>=61

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 3: sdist-safe test and benchmark argparse

**Files:**
- Modify: `tests/test_english_api.py:319-323`
- Modify: `regression/benchmark.py` (bottom `if __name__ == "__main__":` block, ~line 495)

- [ ] **Step 1: Guard the regression import**

Replace the body of `test_frozen_challenge_has_no_family_leakage` with:

```python
def test_frozen_challenge_has_no_family_leakage():
    evaluate = pytest.importorskip(
        "regression.evaluate", reason="regression/ is not shipped in sdist"
    )
    assert evaluate.challenge("development")
    assert evaluate.challenge("holdout")
```

- [ ] **Step 2: Give benchmark.py an argument parser**

Read the current `__main__` block first (`sed -n 480,520p regression/benchmark.py`) to find the two hard-coded output paths. Then replace the block with:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("benchmark_results.json"),
        help="where to write JSON results (default: tracked file)",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path(__file__).with_name("benchmark_results.txt"),
        help="where to write the text report (default: tracked file)",
    )
    args = parser.parse_args()
    run_benchmark(args.output, args.text_output)


if __name__ == "__main__":
    main()
```

Thread the two paths into whatever function currently writes them (rename the existing top-level code into `run_benchmark(json_path: Path, text_path: Path)` if it is not already a function). Add `from pathlib import Path` at the top if missing.

- [ ] **Step 3: Verify --help does not touch tracked files**

Run: `python regression/benchmark.py --help && git status --short regression/`
Expected: usage text; `git status` prints nothing.

- [ ] **Step 4: Verify from an sdist**

```bash
rm -rf /tmp/pygarble-sdist && mkdir /tmp/pygarble-sdist && python -m build --sdist --outdir /tmp/pygarble-sdist -q && cd /tmp/pygarble-sdist && tar xzf pygarble-0.9.0.tar.gz && cd pygarble-0.9.0 && python -m pytest -q -p no:cacheprovider 2>&1 | tail -2; cd /Users/ujjwal/Downloads/solo/pygarble
```
Expected: `617 passed, 1 skipped` (or similar; zero failures).

- [ ] **Step 5: Commit**

```bash
git add tests/test_english_api.py regression/benchmark.py
git commit -m "build: skip regression-backed test outside the repo; benchmark takes --output

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 4: Docs: pre-release wording, planning docs, CHANGELOG

**Files:**
- Modify: `README.md:15`, `docs/index.rst:12`, `docs/api.rst:8`, `docs/installation.rst:15-25`, `docs/migration.rst:4`, `docs/quickstart.rst:4`, `docs/examples.rst:4`
- Move: `docs/audit-plan.md` → `docs/dev/2026-07-audit-plan.md`, `docs/implementation.md` → `docs/dev/2026-07-implementation.md`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Find every pre-release sentence**

Run: `grep -n "upcoming\|not yet\|not been released\|install from the repository" README.md docs/*.rst`
Expected: 8 hits (the ones listed in the spec F1).

- [ ] **Step 2: Rewrite each**

For each hit, delete the sentence that says the API is upcoming / unreleased, and where it told users to install from git, replace with `pip install pygarble`. Keep any surrounding sentence that documents the API. In `docs/installation.rst` the heading `Install the upcoming API` becomes `Install`. In `docs/migration.rst` line 4 becomes `Version 0.9.0 replaces the 0.8 strategy list with profiles and the analyze API.`

- [ ] **Step 3: Move planning docs**

```bash
mkdir -p docs/dev
git mv docs/audit-plan.md docs/dev/2026-07-audit-plan.md
git mv docs/implementation.md docs/dev/2026-07-implementation.md
```

Prepend to both moved files: `> Archived planning document from the 0.9.0 development cycle. Numbers and blockers here are historical.`

- [ ] **Step 4: Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to pygarble are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.9.0] - unreleased

### Added
- `EnsembleDetector` profiles (`english`, `english_extended`, `legacy`,
  `corruption`, `spoofing`), `analyze()` with spans, `allowlist`,
  `strategy_kwargs`, `max_input_length`, `timeout_per_text`.
- Strategies: `CONTROL_CHARACTERS`, `LOCAL_ANOMALY`, `WORD_ANOMALY`,
  `LOG_LIKELIHOOD_RATIO`, `KEYBOARD_ADJACENCY`.
- Correctly spelled `PronounceabilityStrategy` alias.
- `GarbleDetector` and `EnsembleDetector.strategy_kwargs` accept strategy
  names as strings.

### Changed
- Unknown strategy settings now raise a visible `FutureWarning` at the call
  site instead of a hidden `DeprecationWarning`.
- Ensemble-level `**kwargs` are forwarded only to member strategies that
  accept them.

### Fixed
- (entries appended by later tasks)

### Removed
- Five legacy strategies (see docs/migration.rst).

## [0.8.0] - 2026-07-04
- Previous release; see git history.
```

Add to `README.md` near the Documentation links: `- [Changelog](CHANGELOG.md)`.

- [ ] **Step 5: Verify docs still build and snippets still run**

Run: `python -m sphinx -b html -W --keep-going -q docs /tmp/pygarble-docs && grep -c "upcoming" README.md docs/*.rst`
Expected: clean build; every grep count is `0`.

- [ ] **Step 6: Commit**

```bash
git add -A README.md docs CHANGELOG.md
git commit -m "docs: drop pre-release wording, archive planning docs, add CHANGELOG

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase 1 — Core API

### Task 5: Allowlist reaches every strategy (A1)

**Files:**
- Modify: `pygarble/preprocessing.py` (add `scrubbed`)
- Modify: `pygarble/strategies/base.py:113-116`
- Create: `tests/test_allowlist.py`

**Interfaces:**
- Produces: `TextFeatures.scrubbed -> str` (same length as `text`, allowlisted tokens replaced by spaces). `BaseStrategy._evaluate_features` feeds `scrubbed` to `applicable` and `_predict_proba_impl`.

- [ ] **Step 1: Write failing tests**

```python
"""Allowlist must suppress evidence in every strategy, not just four."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy
from pygarble.preprocessing import TextFeatures

TEXT = "please ping asdfgh about the deploy"


def test_scrubbed_blanks_allowlisted_tokens_and_keeps_offsets():
    features = TextFeatures(TEXT, frozenset({"asdfgh"}))
    assert len(features.scrubbed) == len(TEXT)
    assert features.scrubbed == "please ping        about the deploy"
    assert TextFeatures(TEXT).scrubbed == TEXT


def test_scrubbed_handles_punctuation_attached_to_token():
    text = "ping asdfgh, then (asdfgh) deploy"
    features = TextFeatures(text, frozenset({"asdfgh"}))
    assert features.scrubbed == "ping       , then (      ) deploy"


@pytest.mark.parametrize(
    "strategy",
    [
        Strategy.KEYBOARD_ADJACENCY,
        Strategy.KEYBOARD_PATTERN,
        Strategy.PATTERN_MATCHING,
        Strategy.PRONOUNCEABILITY,
        Strategy.LETTER_POSITION,
    ],
)
def test_allowlist_suppresses_text_level_strategies(strategy):
    allowed = GarbleDetector(strategy, allowlist=["asdfgh"])
    assert allowed.predict(TEXT) is False
    assert allowed.analyze(TEXT).score < 0.5


def test_default_profile_honours_allowlist():
    assert EnsembleDetector().predict(TEXT) is True
    detector = EnsembleDetector(allowlist=["asdfgh"])
    assert detector.predict(TEXT) is False
    assert detector.analyze(TEXT).garbled is False


def test_extended_profile_honours_allowlist():
    text = "the xqzvkj report was fine"
    detector = EnsembleDetector(profile="english_extended", allowlist=["xqzvkj"])
    assert detector.analyze(text).garbled is False


def test_fully_allowlisted_text_is_insufficient_evidence():
    detector = GarbleDetector(Strategy.KEYBOARD_ADJACENCY, allowlist=["asdfgh"])
    result = detector.analyze("asdfgh asdfgh")
    assert result.garbled is False
    assert result.status == "insufficient_evidence"
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_allowlist.py -q`
Expected: failures on `scrubbed` (AttributeError) and on the suppression tests.

- [ ] **Step 3: Add `scrubbed` to TextFeatures**

In `pygarble/preprocessing.py`, after the `folded` property:

```python
    @cached_property
    def scrubbed(self) -> str:
        """Text with allowlisted words blanked out, offsets preserved.

        Strategies that scan raw text (keyboard rows, phonotactics,
        regex patterns) receive this instead of ``text`` so an allowlisted
        token can never contribute evidence.
        """
        if not self.allowlist:
            return self.text
        chars = list(self.text)
        for token in self.tokens:
            if token.folded in self.allowlist:
                for index in range(token.start, token.end):
                    chars[index] = " "
        return "".join(chars)
```

- [ ] **Step 4: Use it in BaseStrategy**

Replace `_evaluate_features` in `pygarble/strategies/base.py`:

```python
    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        text = features.scrubbed
        if not text.strip():
            return Evidence(0.0, False, "insufficient_evidence")
        if not self.applicable(text):
            return Evidence(0.0, False, "insufficient_evidence")
        return Evidence(self._predict_proba_impl(text))
```

- [ ] **Step 5: Run the new tests and the full suite**

Run: `python -m pytest tests/test_allowlist.py -q && python -m pytest -q`
Expected: all pass. If a strategy in the parametrized list still flags `TEXT`, check whether it overrides `_evaluate_features` and bypasses `scrubbed`; fix that override to read `features.scrubbed` (or `features.novel`, which already excludes allowlisted tokens).

- [ ] **Step 6: CHANGELOG + commit**

Add under `### Fixed`: `- allowlist is now honoured by every strategy, not only the four that override feature evaluation.`

```bash
git add pygarble/preprocessing.py pygarble/strategies/base.py tests/test_allowlist.py CHANGELOG.md
git commit -m "fix: apply allowlist to every strategy via TextFeatures.scrubbed

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 6: Visible unknown-setting warnings, no ensemble fan-out (C2, C3)

**Files:**
- Modify: `pygarble/options.py:150-162`
- Modify: `pygarble/ensemble.py:95-126`
- Modify: `tests/test_english_api.py:97-99` (`test_unknown_settings_warn`)
- Modify: `tests/test_strategies.py:11-14` (remove no-op `entropy_threshold`)
- Create: `tests/test_ensemble.py`

**Interfaces:**
- Produces in `options.py`: `accepted_options(strategy: str) -> Optional[FrozenSet[str]]`, `unknown_options(strategy: str, options: Mapping[str, Any]) -> List[str]`, `validate_options(strategy, options)` now emits `FutureWarning` attributed to the first frame outside the `pygarble` package.
- Produces on `EnsembleDetector`: `self.strategy_kwargs: Dict[Strategy, Dict[str, Any]]`.

- [ ] **Step 1: Write failing tests** (`tests/test_ensemble.py`)

```python
"""EnsembleDetector construction contracts."""

import warnings

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


def test_unknown_setting_is_a_future_warning_at_the_call_site():
    with pytest.warns(FutureWarning, match="min_lenght") as record:
        GarbleDetector(Strategy.MARKOV_CHAIN, min_lenght=8)
    assert len(record) == 1
    assert record[0].filename == __file__


def test_shared_kwargs_reach_only_strategies_that_accept_them():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detector = EnsembleDetector(profile="english", min_word_length=3)
    by_name = {d.strategy: d for d in detector._detectors}
    assert by_name[Strategy.WORD_ANOMALY].kwargs == {"min_word_length": 3}
    assert by_name[Strategy.MARKOV_CHAIN].kwargs == {}
    assert detector.kwargs == {"min_word_length": 3}


def test_kwarg_no_member_accepts_warns_once():
    with pytest.warns(FutureWarning, match="min_lenght") as record:
        EnsembleDetector(profile="english", min_lenght=3)
    assert len(record) == 1
    assert record[0].filename == __file__


def test_strategy_kwargs_unknown_key_warns_once_at_call_site():
    with pytest.warns(FutureWarning, match="typo") as record:
        EnsembleDetector(
            profile="english",
            strategy_kwargs={Strategy.MARKOV_CHAIN: {"typo": 1}},
        )
    assert len(record) == 1
    assert record[0].filename == __file__


def test_strategy_kwargs_are_retained_for_introspection():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN],
        strategy_kwargs={Strategy.MARKOV_CHAIN: {"min_length": 6}},
    )
    assert detector.strategy_kwargs == {Strategy.MARKOV_CHAIN: {"min_length": 6}}
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_ensemble.py -q`
Expected: 5 failed (DeprecationWarning category, 4 warnings instead of 1, missing attribute).

- [ ] **Step 3: Rewrite the bottom of options.py**

Replace everything from `def validate_options` to end of file with:

```python
def accepted_options(strategy: str) -> Optional[FrozenSet[str]]:
    """Settings a strategy class accepts, or None if unrestricted."""
    accepted = PARAMETERS.get(strategy)
    return None if accepted is None else frozenset(accepted)


def unknown_options(strategy: str, options: Mapping[str, Any]) -> List[str]:
    accepted = accepted_options(strategy)
    if accepted is None:
        return []
    return sorted(set(options) - accepted)


def _external_stacklevel() -> int:
    """Stack level of the first frame outside the pygarble package."""
    package = os.path.dirname(os.path.abspath(__file__))
    frame = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(
        frame.f_code.co_filename
    ).startswith(package):
        frame = frame.f_back
        level += 1
    return level


def warn_unknown_options(strategy: str, unknown: List[str]) -> None:
    if not unknown:
        return
    warnings.warn(
        f"Unknown settings for {strategy}: {', '.join(unknown)}. "
        "Unknown settings will become errors in a future release.",
        FutureWarning,
        stacklevel=_external_stacklevel(),
    )


def validate_options(strategy: str, options: Mapping[str, Any]) -> None:
    warn_unknown_options(strategy, unknown_options(strategy, options))
```

Update the imports at the top of `options.py`:

```python
import os
import sys
import warnings
from typing import Any, FrozenSet, List, Mapping, Optional
```

- [ ] **Step 4: Filter kwargs in EnsembleDetector.__init__**

In `pygarble/ensemble.py`, add imports `from .options import accepted_options, unknown_options, warn_unknown_options` and `from .registry import STRATEGY_MAP, Strategy` (replacing the existing registry import). Replace lines 95-126 (from `options: Dict[...] = dict(` through `self.kwargs = dict(kwargs)`) with:

```python
        options: Dict[Strategy, Dict[str, Any]] = {
            (Strategy(key) if isinstance(key, str) else key): dict(value)
            for key, value in (strategy_kwargs or {}).items()
        }
        if any(strategy not in strategies for strategy in options):
            raise ValueError(
                "strategy_kwargs contains a strategy not selected"
            )
        class_names = {
            strategy: STRATEGY_MAP[strategy].__name__
            for strategy in strategies
        }
        accepted_by_any = set()
        for name in class_names.values():
            accepted = accepted_options(name)
            accepted_by_any |= (
                set(kwargs) if accepted is None else set(accepted)
            )
        warn_unknown_options(
            "EnsembleDetector (no selected strategy accepts them)",
            sorted(set(kwargs) - accepted_by_any),
        )
        for strategy, member_options in options.items():
            warn_unknown_options(
                class_names[strategy],
                unknown_options(class_names[strategy], member_options),
            )
        words = (
            list(allowlist)
            if allowlist is not None and not isinstance(allowlist, str)
            else allowlist
        )
        self._detectors = []
        for strategy in strategies:
            accepted = accepted_options(class_names[strategy])
            shared = {
                key: value
                for key, value in kwargs.items()
                if accepted is None or key in accepted
            }
            member = dict(shared, **options.get(strategy, {}))
            member = {
                key: value
                for key, value in member.items()
                if accepted is None or key in accepted
            }
            self._detectors.append(
                GarbleDetector(
                    strategy,
                    threshold,
                    threads,
                    allowlist=words,
                    max_input_length=max_input_length,
                    timeout_per_text=timeout_per_text,
                    strategy_kwargs=member,
                )
            )
        first = self._detectors[0]
        self.threshold = first.threshold
        self.threads = first.threads
        self.max_input_length = first.max_input_length
        self.timeout_per_text = first.timeout_per_text
        self.allowlist = first.allowlist
        self.kwargs = dict(kwargs)
        self.strategy_kwargs = options
```

Note: `strategy_kwargs` unknown keys are warned once here and then filtered, so `GarbleDetector` never re-warns.

- [ ] **Step 5: Update the two legacy tests**

`tests/test_english_api.py` `test_unknown_settings_warn`: change `DeprecationWarning` to `FutureWarning`.

`tests/test_strategies.py:11-14`: change to `detector = GarbleDetector(Strategy.ENTROPY_BASED)` (the `entropy_threshold` kwarg was a no-op).

- [ ] **Step 6: Run everything**

Run: `python -m pytest -q -W error::FutureWarning`
Expected: all pass with zero warnings (the `-W error` proves no stray unknown-setting warnings remain in the suite).

- [ ] **Step 7: Commit**

```bash
git add pygarble/options.py pygarble/ensemble.py tests/test_ensemble.py tests/test_english_api.py tests/test_strategies.py
git commit -m "fix: unknown settings warn visibly at the call site; ensemble forwards kwargs per member

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 7: Detector and ensemble input hygiene (C1, C4, C6, C7, C10, C11)

**Files:**
- Modify: `pygarble/ensemble.py:80-94` (weights), `:134-145` (weighted mean)
- Modify: `pygarble/detector.py:45-50` (string strategy), `:65` (dead alias)
- Modify: `pygarble/validation.py:11-17` (OverflowError)
- Modify: `tests/test_ensemble.py` (append)

- [ ] **Step 1: Append failing tests to tests/test_ensemble.py**

```python
@pytest.mark.parametrize("voting", ["majority", "any", "all", "average"])
def test_weights_without_weighted_voting_is_an_error(voting):
    with pytest.raises(ValueError, match="weights"):
        EnsembleDetector(
            strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_ANOMALY],
            voting=voting,
            weights=[1, 0],
        )


def test_strategy_kwargs_accepts_string_keys():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN],
        strategy_kwargs={"markov_chain": {"min_length": 6}},
    )
    assert detector._detectors[0]._strategy_instance.min_length == 6


def test_detector_accepts_strategy_name_string():
    assert GarbleDetector("markov_chain").strategy is Strategy.MARKOV_CHAIN
    with pytest.raises(ValueError):
        GarbleDetector("no_such_strategy")


@pytest.mark.parametrize("kwargs", [{"threshold": 10**400}, {"threads": 10**400}])
def test_huge_ints_raise_value_error_not_overflow(kwargs):
    with pytest.raises(ValueError):
        GarbleDetector(Strategy.MARKOV_CHAIN, **kwargs)


def test_weighted_mean_matches_plain_formula():
    detector = EnsembleDetector(
        strategies=[Strategy.MARKOV_CHAIN, Strategy.WORD_ANOMALY],
        voting="weighted",
        weights=[3, 1],
    )
    analysis = detector.analyze("hello qxzjkwpv")
    scores = {s.strategy: s.score for s in analysis.signals if s.applicable}
    expected = (3 * scores["markov_chain"] + 1 * scores["word_anomaly"]) / 4
    assert analysis.score == pytest.approx(expected)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_ensemble.py -q`
Expected: the new tests fail (no error raised, `TypeError`, `OverflowError`).

- [ ] **Step 3: Implement**

`pygarble/ensemble.py`: after the `if self.voting == "weighted" and weights is None:` check add:

```python
        if self.voting != "weighted" and weights is not None:
            raise ValueError("weights are only used when voting='weighted'")
```

Replace the weighted branch of `_aggregate`:

```python
        if self.voting == "weighted":
            total = sum(weight for _, weight in pairs)
            score = (
                sum(signal.score * weight for signal, weight in pairs) / total
            )
```

`pygarble/detector.py`: replace lines 45-50 with:

```python
        if isinstance(strategy, str):
            strategy = Strategy(strategy)
        if not isinstance(strategy, Strategy):
            if isinstance(strategy, Enum):
                raise NotImplementedError(
                    f"Strategy {strategy} is not implemented"
                )
            raise TypeError("strategy must be a Strategy enum member or name")
```

Delete line 65 (`_validate_batch = staticmethod(validate_batch)`) and remove `validate_batch` from the import list if nothing else uses it in that file (the allowlist validation at line 54 does use it, so keep the import).

`pygarble/validation.py` `finite_number`:

```python
def finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number
```

`positive_int` for `threads=10**400`: `positive_int` accepts any int; `ThreadPoolExecutor(max_workers=10**400)` would fail later. Add to `positive_int` after the existing check:

```python
    if value > 2**31 - 1:
        raise ValueError(f"{name} is too large")
```

- [ ] **Step 4: Run**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- weights passed with a non-weighted voting mode now raise ValueError instead of being ignored.` and `- huge integers for numeric options raise ValueError instead of OverflowError.`

```bash
git add pygarble/ensemble.py pygarble/detector.py pygarble/validation.py tests/test_ensemble.py CHANGELOG.md
git commit -m "fix: reject ignored weights, accept strategy names, ValueError on overflow

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 8: Legacy strategy path, spelling alias, documentation of limits (C5, C8, C9, C12)

**Files:**
- Modify: `pygarble/strategies/base.py:29-47`
- Modify: `pygarble/strategies/pronounceability.py` (end of file), `pygarble/strategies/__init__.py`, `pygarble/registry.py`
- Modify: `pygarble/detector.py` (docstring), `pygarble/ensemble.py` (docstring), `docs/api.rst`
- Modify: `tests/test_ensemble.py` (append), create `tests/test_strategy_base.py`

- [ ] **Step 1: Confirm no strategy overrides `_predict_impl`**

Run: `grep -rn "def _predict_impl" pygarble/strategies/ | grep -v base.py`
Expected: no output. (If any appear, keep `_predict_impl` dispatch for that class and note it in the commit.)

- [ ] **Step 2: Write failing tests** (`tests/test_strategy_base.py`)

```python
"""Direct BaseStrategy use goes through the same evaluate() contract."""

import pytest

from pygarble.strategies import (
    FunctionWordDensityStrategy,
    PronouncabilityStrategy,
    PronounceabilityStrategy,
)


def test_legacy_predict_proba_respects_applicability():
    strategy = FunctionWordDensityStrategy()
    assert strategy.applicable("hi") is False
    assert strategy.predict_proba("hi") == 0.0
    assert strategy.predict("hi") is False


def test_legacy_predict_agrees_with_evaluate():
    from pygarble.preprocessing import TextFeatures

    strategy = FunctionWordDensityStrategy()
    text = "xkrf plmq bvzt nwsd jghc trbn mkpl qwer asdf zxcv poiu lkjh mnbv"
    assert strategy.predict_proba(text) == strategy.evaluate(
        TextFeatures(text)
    ).score


def test_pronounceability_alias_is_same_class():
    assert PronounceabilityStrategy is PronouncabilityStrategy
```

- [ ] **Step 3: Route legacy methods through evaluate**

In `pygarble/strategies/base.py` replace `predict` and `predict_proba`:

```python
    def predict(self, text: str) -> bool:
        return self.predict_proba(text) >= 0.5

    def predict_proba(self, text: str) -> float:
        self._validate_input(text)
        return self.evaluate(TextFeatures(text)).score
```

Delete `_predict_impl` (lines 118-121) since nothing overrides it.

- [ ] **Step 4: Add the alias**

At the end of `pygarble/strategies/pronounceability.py`:

```python
# Correctly spelled name; the misspelled class name is kept for
# backward compatibility.
PronounceabilityStrategy = PronouncabilityStrategy
```

In `pygarble/strategies/__init__.py`: add `"PronounceabilityStrategy": "pronounceability",` to `_EXPORTS`, add `"PronounceabilityStrategy",` to `__all__`, and add under `TYPE_CHECKING`: `from .pronounceability import PronounceabilityStrategy as PronounceabilityStrategy`.

- [ ] **Step 5: Document the two limits**

`pygarble/detector.py` `GarbleDetector.__init__`: add a docstring:

```python
        """Single-strategy detector.

        timeout_per_text only bounds work submitted to the thread pool,
        which is used for batches of 10 or more when ``threads`` is 2 or
        more. Single strings and small batches run inline and cannot be
        interrupted.
        """
```

`pygarble/ensemble.py` `_aggregate`: extend the existing `predict_proba` docstring to: `"""Heuristic aggregate. Under voting='majority' the decision counts member votes, so Analysis.garbled can be True while Analysis.score is below threshold."""` and add the same sentence to the `EnsembleDetector` section of `docs/api.rst`.

- [ ] **Step 6: Run**

Run: `python -m pytest -q && python -m sphinx -b html -W --keep-going -q docs /tmp/pygarble-docs`
Expected: all pass, clean docs. If any legacy test asserted a direct-strategy `predict_proba` that differs now because of applicability, the test was encoding the bug; update it to use text the strategy considers applicable.

- [ ] **Step 7: Commit**

```bash
git add pygarble tests/test_strategy_base.py docs/api.rst
git commit -m "fix: legacy strategy predict goes through evaluate(); add PronounceabilityStrategy alias

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase 2 — Strategies in shipped profiles

### Task 9: WordAnomaly scores only novel words (A2)

**Files:**
- Modify: `pygarble/strategies/word_anomaly.py:51-89`
- Create: `tests/test_strategy_word_anomaly.py`

- [ ] **Step 1: Failing tests**

```python
"""WordAnomaly: dictionary acronyms are not anomalies."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "Please enable DHCP on the router",
        "Our auditor is KPMG",
        "Watch it on HGTV tonight",
        "Use JDBC to connect",
    ],
)
def test_dictionary_acronyms_are_clean(text):
    assert GarbleDetector(Strategy.WORD_ANOMALY).predict(text) is False
    assert EnsembleDetector().predict(text) is False


def test_single_mashed_token_still_registers():
    detector = GarbleDetector(Strategy.WORD_ANOMALY)
    assert detector.predict("order confirmed asdkjfhq thanks") is True
    assert detector.predict("order confirmed successfully thanks") is False


def test_fraction_is_over_all_scoreable_words():
    detector = GarbleDetector(Strategy.WORD_ANOMALY)
    # 1 bad of 4 words * weight 2.0 = 0.5
    assert detector.score("order confirmed asdkjfhq thanks") == pytest.approx(0.5)


def test_structured_only_text_is_not_applicable():
    detector = GarbleDetector(Strategy.WORD_ANOMALY)
    result = detector.analyze("https://example.com/a1b2 12345 v2.0")
    assert result.status == "insufficient_evidence"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_strategy_word_anomaly.py -q` → first test fails.

- [ ] **Step 3: Implement**

Replace lines 51-89 of `word_anomaly.py`:

```python
    def applicable(self, text: str) -> bool:
        self._validate_input(text)
        return bool(self._scoreable_tokens(TextFeatures(text)))

    def _scoreable_tokens(self, features: TextFeatures) -> List[Token]:
        return [
            token
            for token in features.tokens
            if not token.structured
            and token.folded.isascii()
            and token.folded not in features.allowlist
            and len(token.folded) >= self.min_word_length
        ]

    def _word_log_prob(self, word: str) -> float:
        return word_log_probability(word)

    def _evaluate_features(self, features: TextFeatures) -> Evidence:
        words = self._scoreable_tokens(features)
        if not words:
            return Evidence(0.0, False, "insufficient_words")
        # Only words the dictionary cannot vouch for can be anomalous;
        # the fraction is still taken over every scoreable word so one
        # bad token in a short sentence registers without dictionary
        # acronyms (DHCP, KPMG) ever counting against the text.
        novel = {(token.start, token.end) for token in features.novel}
        bad = [
            token
            for token in words
            if (token.start, token.end) in novel
            and self._word_log_prob(token.folded)
            < self.word_log_prob_threshold
        ]
        score = min(1.0, len(bad) / len(words) * self.anomaly_weight)
        spans = tuple(
            Span(token.start, token.end, "anomalous_word") for token in bad
        )
        return Evidence(score, True, "anomalous_word_fraction", spans)

    def _predict_proba_impl(self, text: str) -> float:
        return self._evaluate_features(TextFeatures(text)).score
```

Update the import: `from ..preprocessing import TextFeatures, Token`.

- [ ] **Step 4: Run full suite** `python -m pytest -q` → all pass.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- WordAnomaly (default profile) no longer flags dictionary acronyms such as DHCP, KPMG, HGTV.`

```bash
git add pygarble/strategies/word_anomaly.py tests/test_strategy_word_anomaly.py CHANGELOG.md
git commit -m "fix: WordAnomaly scores only novel words so dictionary acronyms stay clean

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 10: Mojibake short-input and threshold docs (A3, D14)

**Files:**
- Modify: `pygarble/strategies/mojibake.py:13` (misplaced coding line), `:56-62` (docstring), `:105` (`except`), `:186-222`
- Create: `tests/test_strategy_mojibake.py`

- [ ] **Step 1: Failing tests**

```python
"""Mojibake: replacement characters count at any length."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize("text", ["�", "a�", "ab�"])
def test_replacement_char_is_detected_regardless_of_length(text):
    detector = GarbleDetector(Strategy.MOJIBAKE)
    assert detector.score(text) >= 0.8
    assert detector.predict(text) is True


def test_replacement_check_can_be_disabled():
    detector = GarbleDetector(Strategy.MOJIBAKE, check_replacement_char=False)
    assert detector.score("a�") == 0.0


@pytest.mark.parametrize("text", ["Café crème brûlée", "Sí, señor", "NÃO É"])
def test_accented_text_is_clean(text):
    assert GarbleDetector(Strategy.MOJIBAKE).predict(text) is False
```

- [ ] **Step 2: Run** → first test fails for the two short inputs.

- [ ] **Step 3: Implement**

In `_predict_proba_impl`, move the replacement-character block above the length guard:

```python
    def _predict_proba_impl(self, text: str) -> float:
        """Compute garble probability based on mojibake detection."""
        scores = []

        # U+FFFD is unambiguous at any length.
        if self.check_replacement_char:
            replacement_count = self._count_replacement_chars(text)
            if replacement_count > 0:
                scores.append(min(1.0, 0.8 + replacement_count * 0.05))

        if len(text) >= 3:
            pattern_count = self._count_mojibake_patterns(text)
            if pattern_count >= self.pattern_threshold:
                scores.append(min(1.0, 0.7 + pattern_count * 0.1))

            byte_density = self._high_byte_density(text)
            if byte_density >= self.ratio_threshold:
                scores.append(min(1.0, byte_density * 5))

            if self._check_double_encoding(text):
                scores.append(0.95)

        return max(scores) if scores else 0.0
```

Rename `_has_high_byte_density` → `_high_byte_density` (it returns a float); keep `_has_high_byte_density = _high_byte_density` as a class-level alias one line below the method for compatibility.

Fix the `ratio_threshold` docstring (around line 60) to: `ratio_threshold : float, optional. Density of mojibake lead/tail sequences (per character) at or above which density evidence is reported; the score is density * 5, so 0.1 maps to 0.5. Default is 0.05.`

Delete the misplaced `# -*- coding` line (~line 13). Change `except Exception` at ~line 105 to `except UnicodeDecodeError`.

- [ ] **Step 4: Run** `python -m pytest -q` → all pass.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- Mojibake detects U+FFFD in inputs shorter than three characters.`

```bash
git add pygarble/strategies/mojibake.py tests/test_strategy_mojibake.py CHANGELOG.md
git commit -m "fix: Mojibake checks replacement characters before the length guard

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 11: PatternMatching consonant cluster on novel words, letters-only repeats (B1, B2)

**Files:**
- Modify: `pygarble/strategies/pattern_matching.py:8-23`, `:46-73`
- Create: `tests/test_strategy_pattern_matching.py`

- [ ] **Step 1: Failing tests**

```python
"""PatternMatching: real words and round numbers are not patterns."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "The strengths of the plan are clear",
        "It was worthwhile after all",
        "A catchphrase and a birthplace",
        "We raised 10000 dollars for the school",
    ],
)
def test_english_with_consonant_runs_or_round_numbers_is_clean(text):
    assert GarbleDetector(Strategy.PATTERN_MATCHING).score(text) < 0.5
    assert EnsembleDetector(profile="english_extended").predict(text) is False


@pytest.mark.parametrize("text", ["asdfghjkl", "AAAAAAA", "xkrfplmqbvzt"])
def test_true_patterns_still_fire(text):
    assert GarbleDetector(Strategy.PATTERN_MATCHING).predict(text) is True


def test_custom_consonant_cluster_override_is_honoured():
    detector = GarbleDetector(
        Strategy.PATTERN_MATCHING,
        patterns={"consonant_cluster": r"[bcdfghjklmnpqrstvwxz]{3,}"},
    )
    # "xkr" is a novel word with a 3-consonant run under the custom rule
    assert detector.predict("xkr") is True
    # dictionary words are never fed to consonant_cluster
    assert detector.predict("strengths") is False
```

- [ ] **Step 2: Run** → first test fails.

- [ ] **Step 3: Implement**

Change `"repeated_chars": r"([a-zA-Z0-9])\1{3,}",` to `"repeated_chars": r"([a-zA-Z])\1{3,}",`. Update the class comment to say digits are covered by `long_numbers` (weak) only.

Replace `_predict_proba_impl`:

```python
    # consonant_cluster only judges words the dictionary cannot vouch for;
    # "strengths", "catchphrase" and "worthwhile" are real English.
    NOVEL_ONLY_PATTERNS = {"consonant_cluster"}

    def _predict_proba_impl(self, text: str) -> float:
        if not self._compiled_patterns:
            return 0.0

        # URL/email tokens legitimately contain consonant runs ("https")
        # and symbol clusters ("://")
        text = " ".join(
            t
            for t in text.split()
            if "://" not in t
            and "@" not in t
            and not t.lower().startswith("www.")
        )
        novel_text = None

        strong = weak = 0
        for name, pattern in self._compiled_patterns.items():
            target = text
            if name in self.NOVEL_ONLY_PATTERNS:
                if novel_text is None:
                    novel_text = " ".join(self._novel_words(text))
                target = novel_text
            if pattern.search(target):
                if name in self.WEAK_PATTERNS:
                    weak += 1
                else:
                    strong += 1

        if strong == 0 and weak == 0:
            return 0.0
        if strong == 0:
            # Weak evidence alone never crosses the default threshold
            return min(0.3 + 0.08 * (weak - 1), 0.45)
        return min(0.65 + 0.08 * (strong - 1 + weak), 1.0)
```

- [ ] **Step 4: Run** `python -m pytest -q` → all pass. If `tests/test_fix_regressions_c1.py:139` ("AAAAAAA qwerty !!!### 123456789") now scores differently, check that `many > one_ish >= 0.6` still holds (it should: `repeated_chars` on `AAAAAAA`, `keyboard_row_qwerty`, `special_chars`, `long_numbers`).

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- PatternMatching (english_extended) no longer flags real words with five-consonant runs or round numbers like 10000.`

```bash
git add pygarble/strategies/pattern_matching.py tests/test_strategy_pattern_matching.py CHANGELOG.md
git commit -m "fix: PatternMatching consonant clusters only on novel words; repeats are letters-only

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 12: Repetition digits and dominance, LocalAnomaly window, KeyboardPattern per-word (B3, B4, B5, D13)

**Files:**
- Modify: `pygarble/strategies/repetition.py:61-64` (unreachable), `:72-74`, `:131-135`
- Modify: `pygarble/strategies/local_anomaly.py:22-26`
- Modify: `pygarble/strategies/keyboard_pattern.py:69-130`
- Create: `tests/test_strategy_repetition.py`, `tests/test_strategy_keyboard_pattern.py`, `tests/test_strategy_local_anomaly.py`

- [ ] **Step 1: Failing tests**

`tests/test_strategy_repetition.py`:

```python
"""Repetition: digits and short emphasis are not repetition evidence."""

import pytest

from pygarble import EnsembleDetector, GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "One million is written 1000000",
        "very very very good",
        "Order 000000123 shipped",
    ],
)
def test_digit_runs_and_short_emphasis_are_clean(text):
    assert GarbleDetector(Strategy.REPETITION).score(text) < 0.5
    assert EnsembleDetector(profile="english_extended").predict(text) is False


@pytest.mark.parametrize(
    "text", ["aaaaaaaaaa", "abababababab", "test test test", "no no no no no"]
)
def test_documented_repetition_still_fires(text):
    assert GarbleDetector(Strategy.REPETITION).predict(text) is True
```

`tests/test_strategy_keyboard_pattern.py`:

```python
"""KeyboardPattern: repeated bigrams are judged per novel word."""

from pygarble import GarbleDetector, Strategy


def test_repeated_dictionary_words_are_clean():
    detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
    assert detector.score("Go go go") < 0.5
    assert detector.score("no no no") < 0.5


def test_repeated_bigram_inside_a_novel_word_fires():
    detector = GarbleDetector(Strategy.KEYBOARD_PATTERN)
    assert detector.score("xkxkxkxk") >= 0.5


def test_keyboard_rows_still_fire():
    assert GarbleDetector(Strategy.KEYBOARD_PATTERN).predict("asdfghjkl")
```

`tests/test_strategy_local_anomaly.py`:

```python
"""LocalAnomaly: window configuration must be usable."""

import pytest

from pygarble import GarbleDetector, Strategy


def test_window_of_one_is_rejected():
    with pytest.raises(ValueError, match="window_words"):
        GarbleDetector(Strategy.LOCAL_ANOMALY, window_words=1)


def test_window_span_is_emitted_for_dense_corruption():
    detector = GarbleDetector(Strategy.LOCAL_ANOMALY, window_words=2)
    result = detector.analyze("hello xqzkvbwq qzxkvjwp world")
    reasons = {span.reason for span in result.spans}
    assert "corrupt_token_window" in reasons
```

- [ ] **Step 2: Run** → the digit/emphasis, keyboard, and window tests fail.

- [ ] **Step 3: Implement Repetition**

Delete the unreachable checks at lines 61-64 (`parameter_value` already rejects `< 1`). Change the char pattern to letters only:

```python
        self._repeated_char_pattern = re.compile(
            r"([a-z])\1{" + str(self.max_char_repeat) + r",}"
        )
```

Update the comment above it: digit runs (order numbers, round amounts) are never repetition evidence. Replace the dominance block in `_check_word_repetition`:

```python
        # A single token dominating the text: either the whole text is one
        # repeated word ("test test test") or, with five or more words, the
        # top token holds at least 60% ("no no no no yes"). Three- and
        # four-word emphasis ("very very very good") is ordinary English.
        top_count = max(Counter(words).values())
        top_ratio = top_count / len(words)
        if top_count >= 3 and (
            top_count == len(words) or (len(words) >= 5 and top_ratio >= 0.6)
        ):
            return min(1.0, top_ratio)
```

Note for the changelog: "No, no, no!" (three identical words) is still flagged by design; the fix targets mixed emphasis.

- [ ] **Step 4: Implement LocalAnomaly**

After `self.window_words = positive_int(...)`:

```python
        if self.window_words < 2:
            raise ValueError("window_words must be at least 2")
```

- [ ] **Step 5: Implement KeyboardPattern**

Add at module level after `COMMON_TRIGRAMS`: `_REPEATED_BIGRAM = re.compile(r"(..)\1{2,}")`. Rewrite the class so words are computed once:

```python
class KeyboardPatternStrategy(BaseStrategy):
    def _get_trigrams(self, text: str) -> List[str]:
        return self._trigrams(self._novel_words(text))

    @staticmethod
    def _trigrams(words: List[str]) -> List[str]:
        # Per word so trigrams never span word boundaries.
        trigrams: List[str] = []
        for word in words:
            trigrams.extend(word[i : i + 3] for i in range(len(word) - 2))
        return trigrams

    def _get_keyboard_pattern_ratio(self, text: str) -> float:
        return self._keyboard_ratio(self._get_trigrams(text))

    @staticmethod
    def _keyboard_ratio(trigrams: List[str]) -> float:
        if not trigrams:
            return 0.0
        return sum(1 for tg in trigrams if tg in KEYBOARD_SEQUENCES) / len(
            trigrams
        )

    def _get_common_trigram_ratio(self, text: str) -> float:
        return self._common_ratio(self._get_trigrams(text))

    @staticmethod
    def _common_ratio(trigrams: List[str]) -> float:
        if not trigrams:
            return 0.0
        return sum(1 for tg in trigrams if tg in COMMON_TRIGRAMS) / len(
            trigrams
        )

    def _has_repeated_bigram_pattern(self, text: str) -> bool:
        return self._repeated_bigram(self._novel_words(text))

    @staticmethod
    def _repeated_bigram(words: List[str]) -> bool:
        # Judged inside a single novel word; "go go go" is English.
        return any(
            len(word) >= 6 and _REPEATED_BIGRAM.search(word) for word in words
        )

    def _predict_proba_impl(self, text: str) -> float:
        words = self._novel_words(text)
        trigrams = self._trigrams(words)
        keyboard_score = min(self._keyboard_ratio(trigrams) / 0.3, 1.0)

        # The common-trigram deficit only means something when there are
        # enough trigrams to expect hits from a 50-item list.
        common_score = 0.0
        if len(trigrams) >= 15:
            confidence = min(1.0, len(trigrams) / 28.0)
            common_score = (
                max(0.0, 1.0 - (self._common_ratio(trigrams) / 0.15))
                * confidence
            )

        repeated_score = 0.5 if self._repeated_bigram(words) else 0.0
        return min(
            max(keyboard_score, common_score * 0.7, repeated_score), 1.0
        )
```

The old text-taking helpers are kept as thin wrappers because `tests/test_fix_regressions_c1.py` and others may call them.

- [ ] **Step 6: Run** `python -m pytest -q` → all pass. If a legacy test asserts a 4-word emphatic phrase or a digit run is flagged by Repetition, it encoded the bug: update it to a 5-word example or letters.

- [ ] **Step 7: CHANGELOG + commit**

Under `### Fixed`: `- Repetition ignores digit runs and 3-4 word emphasis ("very very very good").`, `- LocalAnomaly rejects window_words=1, which could never emit a span.`, `- KeyboardPattern judges repeated bigrams per novel word, so "go go go" is clean.`

```bash
git add pygarble/strategies/repetition.py pygarble/strategies/local_anomaly.py pygarble/strategies/keyboard_pattern.py tests/test_strategy_repetition.py tests/test_strategy_keyboard_pattern.py tests/test_strategy_local_anomaly.py CHANGELOG.md
git commit -m "fix: repetition ignores digits and short emphasis; keyboard bigrams per word; window >= 2

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase 3 — Opt-in strategies

### Task 13: LetterPosition novel-word filter and zero threshold (D1, D2)

**Files:**
- Modify: `pygarble/strategies/letter_position.py` (`NEVER_START` entries `cz`, `sv`, `vl`; `__init__`; `_extract_words`)
- Create: `tests/test_strategy_letter_position.py`

- [ ] **Step 1: Failing tests**

```python
"""LetterPosition: dictionary words and acronyms are exempt."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "Czech svelte vlog",
        "Save as PDF or JPG",
        "speed 60 mph at 3000 rpm",
        "Gnocchi for dinner",
    ],
)
def test_real_words_and_acronyms_are_clean(text):
    assert GarbleDetector(Strategy.LETTER_POSITION).predict(text) is False


def test_novel_violations_still_fire():
    detector = GarbleDetector(Strategy.LETTER_POSITION)
    assert detector.predict("wordj endq") is True
    assert detector.predict("xjword bwtext") is True


def test_zero_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold"):
        GarbleDetector(Strategy.LETTER_POSITION, threshold=0.0)
```

- [ ] **Step 2: Run** → the first and third tests fail.

- [ ] **Step 3: Implement**

In `__init__` after `self.threshold = unit_interval(...)`:

```python
        if self.threshold <= 0.0:
            raise ValueError("threshold must be greater than 0.0")
```

Replace `_extract_words`:

```python
    def _extract_words(self, text: str) -> list:
        """Novel lowercase words long enough to judge.

        Dictionary words, short acronyms (PDF, JPG), likely proper nouns
        (Czech) and structured tokens are never positional violations.
        """
        return [
            word
            for word in self._novel_words(text, skip_titlecase=True)
            if len(word) >= self.min_word_length
        ]
```

Remove `"cz"`, `"sv"`, `"vl"` from `NEVER_START` (they are real English onsets: Czech, svelte, Vlad).

- [ ] **Step 4: Run** `python -m pytest -q` → all pass.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- LetterPosition scores only novel words and rejects threshold=0 (previously ZeroDivisionError).`

```bash
git add pygarble/strategies/letter_position.py tests/test_strategy_letter_position.py CHANGELOG.md
git commit -m "fix: LetterPosition judges novel words only; reject zero threshold

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 14: Pronounceability onset table and min_word_length (D5, D6, dead code)

**Files:**
- Modify: `pygarble/strategies/pronounceability.py` (`VALID_ONSET_CLUSTERS`, `__init__` default, `_predict_proba_impl` filter, delete `_extract_consonant_clusters`)
- Create: `tests/test_strategy_pronounceability.py`

- [ ] **Step 1: Failing tests**

```python
"""Pronounceability: real onsets are valid; min_word_length is honoured."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "word",
    ["squonk", "squib", "sphinxes", "chlorinate", "schlep", "ghoulish",
     "sclerotic", "phlox"],
)
def test_valid_english_onsets_are_not_violations(word):
    assert GarbleDetector(Strategy.PRONOUNCEABILITY).score(word) < 0.5


def test_min_word_length_is_honoured():
    strict = GarbleDetector(Strategy.PRONOUNCEABILITY, min_word_length=2)
    lenient = GarbleDetector(Strategy.PRONOUNCEABILITY)
    assert strict.score("xkq bkx") > 0.0
    assert lenient.score("xkq bkx") == 0.0


def test_gibberish_still_fires():
    assert GarbleDetector(Strategy.PRONOUNCEABILITY).predict("bkxq tpfk vzjk")


def test_dead_helper_removed():
    from pygarble.strategies.pronounceability import PronouncabilityStrategy

    assert not hasattr(PronouncabilityStrategy, "_extract_consonant_clusters")
```

- [ ] **Step 2: Run** → onset, min_word_length and dead-helper tests fail.

- [ ] **Step 3: Implement**

In `VALID_ONSET_CLUSTERS`: remove `"qu"` and `"squ"` (the `u` is a vowel so these never match; `"q"` is already present), add `"sq"`, `"gh"`, `"sph"`, `"chl"`, `"scl"`, `"phl"`, `"schl"`.

In `__init__` change the `min_word_length` default from 3 to 4 (matches the previous effective behaviour) and update its docstring: `min_word_length : int, optional. Shortest word judged (default 4; 2-3 letter unknowns are usually abbreviations).`

In `_predict_proba_impl` replace

```python
        novel = [w for w in novel if len(w) >= 4]
```

with

```python
        novel = [w for w in novel if len(w) >= self.min_word_length]
```

Delete the `_extract_consonant_clusters` method (around lines 541-558).

- [ ] **Step 4: Run** `python -m pytest -q` → all pass.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- Pronounceability accepts sq-, sph-, chl-, scl-, phl-, schl-, gh- onsets and honours min_word_length (default now 4, the previous effective value).`

```bash
git add pygarble/strategies/pronounceability.py tests/test_strategy_pronounceability.py CHANGELOG.md
git commit -m "fix: Pronounceability onset table and min_word_length

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 15: FunctionWordDensity and WordCollocation (D7, D8, D9, D10) + shared FUNCTION_WORDS

**Files:**
- Create: `pygarble/data/function_words.py`
- Modify: `pygarble/data/__init__.py` (lazy export map + `__all__`)
- Modify: `pygarble/preprocessing.py` (add `title_case_ratio`)
- Modify: `pygarble/strategies/function_word_density.py:43-...` (table), `:216-240`
- Modify: `pygarble/strategies/word_collocation.py:408-470`
- Modify: `pygarble/strategies/zipf_conformity.py:16, 90`
- Create: `tests/test_strategy_function_words.py`

- [ ] **Step 1: Failing tests**

```python
"""Function-word density and collocation: casing and short words."""

import pytest

from pygarble import GarbleDetector, Strategy
from pygarble.data import FUNCTION_WORDS
from pygarble.preprocessing import title_case_ratio

ALL_CAPS_MASH = (
    "XKRF PLMQ BVZT NWSD JGHC TRBN MKPL QWRT ZXCV POIU "
    "LKJH MNBV GHJK TYUI VBNM RTYU DFGH CVBN ERTY XCVB"
)  # 20 tokens: zero-collocation text needs 20+ words to cross 0.5
PLAIN_13 = (
    "researchers analyzed thousands of samples collected across multiple "
    "regions during several recent years"
)


def test_function_word_table_is_shared():
    assert "the" in FUNCTION_WORDS
    assert "a" in FUNCTION_WORDS and "i" in FUNCTION_WORDS


def test_title_case_ratio_excludes_all_caps():
    assert title_case_ratio("Alice Bob Carol") == 1.0
    assert title_case_ratio("XKRF PLMQ") == 0.0
    assert title_case_ratio("I am") == pytest.approx(0.5)


def test_single_letter_function_words_count():
    detector = GarbleDetector(Strategy.FUNCTION_WORD_DENSITY)
    assert detector._strategy_instance._tokenize("I am a cat") == [
        "i", "am", "a", "cat"
    ]


def test_all_caps_mash_is_not_title_case_exempt():
    assert GarbleDetector(Strategy.FUNCTION_WORD_DENSITY).score(ALL_CAPS_MASH) >= 0.5
    assert GarbleDetector(Strategy.WORD_COLLOCATION).score(ALL_CAPS_MASH) >= 0.5


def test_plain_sentence_without_listed_collocations_is_clean():
    assert GarbleDetector(Strategy.WORD_COLLOCATION).score(PLAIN_13) < 0.5


def test_curly_apostrophe_keeps_contractions_whole():
    strategy = GarbleDetector(Strategy.WORD_COLLOCATION)._strategy_instance
    assert strategy._tokenize("don’t stop") == ["don't", "stop"]
```

- [ ] **Step 2: Run** → import of `FUNCTION_WORDS` fails.

- [ ] **Step 3: Move FUNCTION_WORDS to data**

Read `sed -n 40,120p pygarble/strategies/function_word_density.py` to capture the full `FUNCTION_WORDS = frozenset(...)` literal. Create `pygarble/data/function_words.py`:

```python
"""English function words (determiners, prepositions, pronouns, ...)."""

FUNCTION_WORDS = frozenset(
    {
        # paste the exact word list from FunctionWordDensityStrategy here
    }
)
```

Read `cat pygarble/data/__init__.py` and add `"FUNCTION_WORDS": "function_words"` to the lazy export mapping and `"FUNCTION_WORDS"` to `__all__` following the pattern used for `DEFAULT_LOG_PROB`.

In `function_word_density.py` replace the class-level literal with:

```python
    FUNCTION_WORDS = FUNCTION_WORDS
```

with a module import `from ..data import FUNCTION_WORDS`. (Class attribute kept for compatibility.) In `zipf_conformity.py` replace the `FunctionWordDensityStrategy` import with `from ..data import ENGLISH_WORDS, FUNCTION_WORDS` and line ~90 with `if any(w in FUNCTION_WORDS for w in words):`.

- [ ] **Step 4: Add title_case_ratio to preprocessing.py**

```python
def title_case_ratio(text: str) -> float:
    """Fraction of alphabetic tokens that are Capitalized but not ALL-CAPS.

    Name lists and headlines are mostly Title Case; a shouted mash of
    consonants is not.
    """
    tokens = [t for t in text.split() if any(c.isalpha() for c in t)]
    if not tokens:
        return 0.0
    titled = 0
    for token in tokens:
        letters = [c for c in token if c.isalpha()]
        if letters[0].isupper() and not (
            len(letters) > 1 and all(c.isupper() for c in letters)
        ):
            titled += 1
    return titled / len(tokens)
```

- [ ] **Step 5: FunctionWordDensity**

```python
    def _tokenize(self, text: str) -> List[str]:
        """Lowercase alphabetic words; function words are kept at any
        length so "a" and "I" count."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return [
            w
            for w in words
            if len(w) >= self.min_word_length or w in self.FUNCTION_WORDS
        ]
```

Replace the title-case block inside `if function_count == 0:`:

```python
            if title_case_ratio(text) >= 0.6:
                return 0.3
```

and import `from ..preprocessing import title_case_ratio`.

- [ ] **Step 6: WordCollocation**

```python
    def _tokenize(self, text: str) -> List[str]:
        """Lowercase alphabetic words, contractions kept whole.

        Curly apostrophes are normalised so "don’t" and "don't" tokenize
        the same way.
        """
        tokens = re.findall(r"[a-zA-Z']+", text.replace("’", "'").lower())
        return [t.strip("'") for t in tokens if t.strip("'")]

    def _title_case_ratio(self, text: str) -> float:
        return title_case_ratio(text)
```

Replace the zero-collocation ladder:

```python
        if hit_count == 0:
            # Name lists and headlines legitimately contain zero
            # collocations.
            if self._title_case_ratio(text) >= 0.6:
                return 0.3
            # A missing collocation is weak evidence on its own: ordinary
            # prose can run 12-19 words without hitting the table. Only
            # 20+ words with no hit at all crosses the decision line.
            if len(words) >= 20:
                return 0.6
            if len(words) >= self.zero_collocation_min_words:
                return 0.45
            return 0.3
```

- [ ] **Step 7: Run** `python -m pytest -q`. If `tests/test_word_level_strategies.py:269-272` asserts `> 0.5` on a 12-19-word zero-collocation gibberish sentence, extend that text to 20+ words (the strategy now demands more evidence). All else should pass.

- [ ] **Step 8: CHANGELOG + commit**

Under `### Fixed`: `- FunctionWordDensity counts "a" and "I"; Title Case exemption no longer covers ALL-CAPS; WordCollocation needs 20+ words with no collocation to cross 0.5 and handles curly apostrophes.`

```bash
git add pygarble/data/function_words.py pygarble/data/__init__.py pygarble/preprocessing.py pygarble/strategies/function_word_density.py pygarble/strategies/word_collocation.py pygarble/strategies/zipf_conformity.py tests/test_strategy_function_words.py CHANGELOG.md
git commit -m "fix: function-word and collocation casing, short words, apostrophes; share FUNCTION_WORDS

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 16: HexString base64 heuristics (D11)

**Files:**
- Modify: `pygarble/strategies/hex_string.py:53` (unreachable check), `:115-131`
- Create: `tests/test_strategy_hex_string.py`

- [ ] **Step 1: Failing tests**

```python
"""HexString: paths and identifiers are not base64."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "/usr/local/bin/python3",
        "src/components/HeaderView",
        "getUserAccountBalanceV2",
        "some/path/withDigits1/x",
    ],
)
def test_paths_and_identifiers_are_clean(text):
    assert GarbleDetector(Strategy.HEX_STRING).score(text) < 0.5


@pytest.mark.parametrize(
    "text",
    [
        "aGVsbG8gd29ybGQgdGhpcyBpcw==",
        "U29tZSByYW5kb20gYmFzZTY0IHN0cmluZw",
        "4f8a9b2c1d3e5f6a7b8c9d0e",
    ],
)
def test_real_base64_and_hex_still_fire(text):
    assert GarbleDetector(Strategy.HEX_STRING).score(text) >= 0.5
```

- [ ] **Step 2: Run** → first test fails on three inputs.

- [ ] **Step 3: Implement**

Replace `_is_base64_like`:

```python
    def _is_base64_like(self, text: str) -> bool:
        """Base64-typical evidence, not merely the base64 alphabet.

        Paths ("/usr/local/bin") and camelCase identifiers share the
        alphabet; real base64 has a length that is never 1 mod 4, is
        padded or symbol-bearing, and mixes cases roughly evenly.
        """
        text = text.strip()
        if not self._base64_pattern.match(text):
            return False
        if text.startswith(("/", "./", "../")):
            return False
        if len(text) % 4 == 1:
            return False
        if "=" in text:
            return True
        letters = [c for c in text if c.isalpha()]
        upper = sum(1 for c in letters if c.isupper())
        if letters and upper / len(letters) < 0.2:
            return False
        if any(c in "+/" for c in text):
            return True
        has_digit = any(c.isdigit() for c in text)
        return has_digit and 0 < upper < len(letters)
```

Delete the unreachable `min_hex_length < 1` check near line 53 if `positive_int`/`parameter_value` already guards it (confirm with `sed -n 45,56p`).

- [ ] **Step 4: Run** `python -m pytest -q` → all pass.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- HexString no longer classifies file paths or camelCase identifiers as base64.`

```bash
git add pygarble/strategies/hex_string.py tests/test_strategy_hex_string.py CHANGELOG.md
git commit -m "fix: HexString base64 check rejects paths and identifiers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 17: VowelRatio validation and short inputs (D12)

**Files:**
- Modify: `pygarble/strategies/vowel_ratio.py` (add `__init__`, `applicable`, rewrite scoring; delete `_has_consonant_cluster`)
- Modify: `pygarble/options.py` (`VowelRatioStrategy` gains `"min_length"`)
- Create: `tests/test_strategy_vowel_ratio.py`

- [ ] **Step 1: Failing tests**

```python
"""VowelRatio: parameters are validated; tiny inputs abstain."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_vowel_ratio": 5.0},
        {"max_vowel_ratio": -1},
        {"consonant_cluster_len": -3},
        {"min_vowel_ratio": 0.7, "max_vowel_ratio": 0.6},
        {"min_length": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GarbleDetector(Strategy.VOWEL_RATIO, **kwargs)


@pytest.mark.parametrize("text", ["a", "I", "Hmm", "Shh", "Mr. Ng"])
def test_tiny_inputs_abstain(text):
    detector = GarbleDetector(Strategy.VOWEL_RATIO)
    assert detector.predict(text) is False
    assert detector.analyze(text).status == "insufficient_evidence"


def test_real_signals_still_fire():
    detector = GarbleDetector(Strategy.VOWEL_RATIO)
    assert detector.predict("bcdfghjklmnpqrstvwxyz") is True
    assert detector.predict("aeiouaeiou") is True
    assert detector.predict("hello world") is False
```

- [ ] **Step 2: Run** → validation and tiny-input tests fail.

- [ ] **Step 3: Implement**

Add `"min_length",` to `PARAMETERS["VowelRatioStrategy"]` in `options.py` (keep the list sorted).

In `vowel_ratio.py` add imports `from typing import Any, FrozenSet` and `from ..validation import positive_int, unit_interval`; drop the `parameter_value` import. Add:

```python
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_vowel_ratio = unit_interval(
            "min_vowel_ratio", kwargs.get("min_vowel_ratio", 0.15)
        )
        self.max_vowel_ratio = unit_interval(
            "max_vowel_ratio", kwargs.get("max_vowel_ratio", 0.65)
        )
        if self.min_vowel_ratio > self.max_vowel_ratio:
            raise ValueError("min_vowel_ratio must not exceed max_vowel_ratio")
        self.consonant_cluster_len = positive_int(
            "consonant_cluster_len", kwargs.get("consonant_cluster_len", 4)
        )
        # Ratios on one to three letters are noise ("I", "Hmm", "Mr. Ng").
        self.min_length = positive_int(
            "min_length", kwargs.get("min_length", 4)
        )

    def _letters(self, text: str) -> int:
        """Letters in words of three or more letters; "Mr", "Ng",
        "I" are abbreviations, not evidence."""
        total = 0
        for word in self._filter_acronyms(text).split():
            count = sum(1 for c in word if c.isalpha())
            if count >= 3:
                total += count
        return total

    def applicable(self, text: str) -> bool:
        self._validate_input(text)
        return self._letters(text) >= self.min_length
```

Delete `_has_consonant_cluster`. In `_get_vowel_ratio` and `_get_max_consonant_run`, skip words with fewer than 3 alphabetic characters (abbreviations such as "Mr", "Ng"):

```python
        for word in text.lower().split():
            if sum(1 for c in word if c.isalpha()) < 3:
                continue
```

Rewrite `_predict_proba_impl` to use the validated attributes:

```python
    def _predict_proba_impl(self, text: str) -> float:
        text = self._filter_acronyms(text)
        if self._letters(text) < self.min_length:
            return 0.0
        ratio = self._get_vowel_ratio(text)
        if ratio == 0.0 and self._get_max_consonant_run(text) == 0:
            return 0.0  # every word was a short abbreviation

        ratio_score = 0.0
        if ratio < self.min_vowel_ratio:
            ratio_score = (self.min_vowel_ratio - ratio) / self.min_vowel_ratio
        elif ratio > self.max_vowel_ratio:
            denominator = 1.0 - self.max_vowel_ratio
            ratio_score = (
                (ratio - self.max_vowel_ratio) / denominator
                if denominator > 0
                else 1.0
            )

        run = self._get_max_consonant_run(text)
        cluster_score = 0.0
        if run >= self.consonant_cluster_len:
            cluster_score = min((run - self.consonant_cluster_len) / 4, 1.0)

        return min(max(ratio_score, cluster_score), 1.0)
```

- [ ] **Step 4: Run** `python -m pytest -q` → all pass. `tests/test_new_features.py:35` (`predict("") is False`) is unaffected.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- VowelRatio validates its parameters, abstains below 4 letters (new min_length option), and ignores 1-2 letter abbreviations.`

```bash
git add pygarble/strategies/vowel_ratio.py pygarble/options.py tests/test_strategy_vowel_ratio.py CHANGELOG.md
git commit -m "fix: VowelRatio validates parameters and abstains on tiny inputs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 18: UnicodeScript mixed-word detection (D3, D4)

**Files:**
- Modify: `pygarble/strategies/unicode_script.py` (`_is_mixed_script_word`, `_count_homoglyphs`, `_count_mixed_script_words`, docstring)
- Create: `tests/test_strategy_unicode_script.py`

- [ ] **Step 1: Failing tests**

```python
"""UnicodeScript: only confusable-script mixing inside a word is spoofing."""

import pytest

from pygarble import GarbleDetector, Strategy


@pytest.mark.parametrize(
    "text",
    [
        "私はiPhoneを使います",
        "我用Python写代码",
        "The α-helix is stable",
        "cells were 5 μm wide",
        "E=mc² with Δt",
        "Hello, World! 123",
    ],
)
def test_natural_script_mixing_is_clean(text):
    assert GarbleDetector(Strategy.UNICODE_SCRIPT).score(text) < 0.5


@pytest.mark.parametrize("text", ["pаypal", "gооgle login", "аpple"])
def test_confusable_mixing_inside_a_word_fires(text):
    assert GarbleDetector(Strategy.UNICODE_SCRIPT).score(text) >= 0.5
```

- [ ] **Step 2: Run** → first test fails on the CJK and scientific inputs.

- [ ] **Step 3: Implement**

Add near the top of the module: `import re` and

```python
# Only these scripts have look-alike letters that make one word visually
# ambiguous. CJK, Arabic, Hebrew or Devanagari next to Latin inside one
# whitespace-delimited chunk (Japanese with an English brand, units like
# "μm") is ordinary writing, not spoofing.
CONFUSABLE_SCRIPTS = frozenset({"Latin", "Cyrillic", "Greek"})
_WORD_BOUNDARY = re.compile(r"[\W_]+")
```

Add a helper and rewrite the two word-level methods:

```python
    def _words(self, text: str) -> List[str]:
        """Split on whitespace, punctuation and digits so "α-helix" and
        "E=mc²" are judged part by part."""
        return [w for w in _WORD_BOUNDARY.split(text) if w]

    def _is_mixed_script_word(self, word: str) -> bool:
        """A word mixing confusable scripts (Latin/Cyrillic/Greek)."""
        alpha_chars = [c for c in word if c.isalpha()]
        if len(alpha_chars) < 3:
            return False
        scripts = set()
        for char in alpha_chars:
            script = self._get_script(char)
            script = COMPATIBLE_SCRIPT_GROUPS.get(script, script)
            if script in CONFUSABLE_SCRIPTS:
                scripts.add(script)
        return len(scripts) > 1

    def _count_homoglyphs(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = Counter()
        for word in self._words(text):
            if not self._is_mixed_script_word(word):
                continue
            for char in word:
                if char in HOMOGLYPHS:
                    _, script = HOMOGLYPHS[char]
                    counts[script] += 1
        return dict(counts)

    def _count_mixed_script_words(self, text: str) -> int:
        return sum(1 for w in self._words(text) if self._is_mixed_script_word(w))
```

Add `List` to the typing import. In the class docstring document: `check_homoglyphs / homoglyph_threshold add a homoglyph-count signal on top of the mixed-script-word signal; with the defaults both signals produce the same score for a single spoofed word, so disabling check_homoglyphs only matters when homoglyph_threshold is raised above the number of look-alike letters present.`

- [ ] **Step 4: Run** `python -m pytest -q` → all pass. `tests/test_new_strategies.py:410-422` use mixed inputs; if one of them mixed CJK+Latin in a single word and expected `True`, it encoded D3: change its input to a Latin/Cyrillic mix.

- [ ] **Step 5: CHANGELOG + commit**

Under `### Fixed`: `- UnicodeScript no longer flags CJK text with embedded Latin words or scientific units; only Latin/Cyrillic/Greek mixing inside a word counts.`

```bash
git add pygarble/strategies/unicode_script.py tests/test_strategy_unicode_script.py CHANGELOG.md
git commit -m "fix: UnicodeScript judges only confusable-script mixing inside a word

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase 4 — Cleanup

### Task 19: Dead code, unreachable checks, table dedupe, tokenizer sharing

**Files:**
- Modify: `pygarble/strategies/log_likelihood_ratio.py` (delete `_extract_bigrams`; keep `_average_llr` only if a test imports it: `grep -rn _average_llr tests/`)
- Modify: `pygarble/strategies/markov_chain.py` (delete `_compute_log_probability`, unreachable check ~line 67)
- Modify: `pygarble/strategies/consonant_sequence.py` (delete `_get_max_consonant_run`, `_count_violations` if unused: `grep -rn` both in `pygarble tests`)
- Modify: `pygarble/strategies/keyboard_adjacency.py:35-36` (unused `_ADJACENT`, `_ROW_OF`)
- Modify: `pygarble/strategies/ngram_frequency.py:59`, `hex_string.py:53`, `symbol_ratio.py:71` (unreachable checks)
- Modify: `pygarble/strategies/rare_trigram.py:127-138` (duplicates), `:28` (comment)
- Modify: `pygarble/preprocessing.py` (add `ascii_alpha_words`)
- Modify: `pygarble/strategies/affix_detection.py:137`, `zipf_conformity.py:77`, `word_lookup.py:66` (use the helper)
- Modify: `pygarble/strategies/letter_frequency.py` (tokenize once)

- [ ] **Step 1: Write a guard test** (`tests/test_contracts.py`, first content; Task 20 extends it)

```python
"""Contracts every registered strategy must satisfy."""

import pytest

from pygarble import GarbleDetector, Strategy
from pygarble.preprocessing import ascii_alpha_words
from pygarble.strategies.rare_trigram import RareTrigramStrategy


def test_ascii_alpha_words_matches_legacy_regex():
    text = "Hello, wörld! it's 42 x-ray"
    assert ascii_alpha_words(text) == ["hello", "w", "rld", "it", "s", "x", "ray"]


def test_rare_trigram_table_has_no_duplicates():
    table = RareTrigramStrategy.RARE_TRIGRAMS  # adjust to the real attribute
    assert len(table) == len(set(table))
```

Read `sed -n 100,145p pygarble/strategies/rare_trigram.py` first to learn whether the table is a set literal (then the duplicate test must instead assert the nine listed trigrams appear once in the source text) and adjust the test accordingly:

```python
def test_rare_trigram_source_lists_each_entry_once():
    import inspect
    import re
    from pygarble.strategies import rare_trigram

    source = inspect.getsource(rare_trigram)
    entries = re.findall(r'"([a-z]{3})"', source)
    dupes = {e for e in entries if entries.count(e) > 1}
    assert not dupes
```

- [ ] **Step 2: Run** → fails (no `ascii_alpha_words`, duplicates present).

- [ ] **Step 3: Implement the helper**

In `preprocessing.py`:

```python
_ASCII_ALPHA = re.compile(r"[a-zA-Z]+")


def ascii_alpha_words(text: str) -> List[str]:
    """Lowercase ASCII-letter runs, the tokenizer several strategies share."""
    return _ASCII_ALPHA.findall(text.lower())
```

Replace `re.findall(r"[a-zA-Z]+", text.lower())` in `affix_detection.py:137` and `zipf_conformity.py:77` with `ascii_alpha_words(text)`, and `re.findall(r"[a-zA-Z]+", folded)` in `word_lookup.py:66` with `ascii_alpha_words(folded)` (it lowercases; confirm `folded` is already lowercase so behaviour is identical). Remove the now-unused `re` imports if flake8 flags them.

- [ ] **Step 4: Dead code and unreachable checks**

For each item, first `grep -rn "<name>" pygarble tests` and delete only when the sole hit is the definition. Delete the `if self.x < 1: raise ValueError` blocks that follow a `parameter_value`/`positive_int` call in `markov_chain.py`, `ngram_frequency.py`, `hex_string.py`, `symbol_ratio.py`. Delete the second occurrence of each duplicated trigram in `rare_trigram.py` and reword the `# Q without u patterns` comment to `# Rare-letter sandwiches (q?q, x?x, z?z) and qq* runs`.

- [ ] **Step 5: LetterFrequency tokenize once**

Read `grep -n "_novel_words" pygarble/strategies/letter_frequency.py`. Change `_predict_proba_impl` to call `self._novel_words(text)` once and pass the list into the helpers it feeds (change those helpers to take `words: List[str]` and keep thin `text`-taking wrappers only if tests call them: `grep -rn "LetterFrequencyStrategy()\._" tests/`).

- [ ] **Step 6: Verify identical behaviour**

Run: `python -m pytest -q && python regression/evaluate.py --split all --output /tmp/after.json && git show HEAD:regression/english_results.json > /tmp/before.json && python - <<'EOF'
import json
a=json.load(open('/tmp/before.json')); b=json.load(open('/tmp/after.json'))
print("identical" if a==b else "DIFFERS")
EOF`
Expected: all pass. The results may say `DIFFERS` because of Phase 2/3 fixes already landed; what matters is that this task changes nothing further. To check that precisely, run the evaluator once *before* Step 3 into `/tmp/pre19.json` and compare `/tmp/after.json` against it: must be `identical`.

- [ ] **Step 7: Commit**

```bash
git add pygarble tests/test_contracts.py
git commit -m "refactor: remove dead helpers and unreachable checks, dedupe trigram table, share ASCII tokenizer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase 5 — Tests

### Task 20: Registry-wide contract tests, profile tests, duplicate removal

**Files:**
- Modify: `tests/test_contracts.py` (extend)
- Modify: `tests/test_ensemble.py` (append profile tests)
- Delete duplicates: `tests/test_core.py:11,16` (keep `test_strategies.py:36,41`), `tests/test_new_features.py:106` (keep `test_edge_cases.py:102`), `tests/test_new_strategies.py:56,105,161` (keep `test_edge_cases.py:138,152,162`), `tests/test_v7_strategies.py:108` (keep `:80`), `tests/test_fix_regressions_a.py:73` (keep `:62`)

- [ ] **Step 1: Append to tests/test_contracts.py**

```python
ALL = list(Strategy)


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t", "a", "👍👏🙏", "你好世界", "Привет мир", "x" * 20000],
    ids=["empty", "spaces", "ws", "one", "emoji", "cjk", "cyrillic", "long"],
)
def test_every_strategy_survives_edge_inputs(strategy, text):
    detector = GarbleDetector(strategy)
    score = detector.score(text)
    assert 0.0 <= score <= 1.0
    assert detector.predict(text) is (score >= 0.5)
    analysis = detector.analyze(text)
    assert analysis.score == score
    if not text.strip():
        assert analysis.status == "insufficient_evidence"


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
def test_every_strategy_rejects_non_strings(strategy):
    detector = GarbleDetector(strategy)
    for bad in [None, 0, b"x", [1]]:
        with pytest.raises(TypeError):
            detector.predict(bad)


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
def test_every_strategy_is_listed_in_options(strategy):
    from pygarble.options import PARAMETERS
    from pygarble.registry import STRATEGY_MAP

    assert STRATEGY_MAP[strategy].__name__ in PARAMETERS


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: s.value)
def test_instances_do_not_share_state(strategy):
    a = GarbleDetector(strategy, allowlist=["qxzjkwpv"])
    b = GarbleDetector(strategy)
    assert a.allowlist != b.allowlist
    assert a._strategy_instance is not b._strategy_instance
```

- [ ] **Step 2: Append profile tests to tests/test_ensemble.py**

```python
from pygarble.ensemble import PROFILES


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_constructs_and_votes_any(profile):
    detector = EnsembleDetector(profile=profile)
    assert detector.profile == profile
    assert detector.voting == "any"
    assert [d.strategy for d in detector._detectors] == list(PROFILES[profile])


def test_custom_strategy_list_votes_majority():
    detector = EnsembleDetector(strategies=list(PROFILES["english"]))
    assert detector.profile == "custom"
    assert detector.voting == "majority"


def test_unknown_profile_and_profile_plus_strategies_are_errors():
    with pytest.raises(ValueError, match="unknown profile"):
        EnsembleDetector(profile="nope")
    with pytest.raises(ValueError, match="either profile or strategies"):
        EnsembleDetector(profile="english", strategies=[Strategy.MARKOV_CHAIN])


@pytest.mark.parametrize(
    "profile,text,expected",
    [
        ("english", "The quick brown fox jumps over the lazy dog", False),
        ("english", "qxzjkwpv bnmqwer zxcvbnm", True),
        ("english_extended", "The strengths of the plan are clear", False),
        ("legacy", "hello world", False),
        ("corruption", "CafÃ© crÃ¨me", True),
        ("corruption", "Café crème", False),
        ("spoofing", "pаypal login", True),
        ("spoofing", "paypal login", False),
    ],
)
def test_profile_decisions(profile, text, expected):
    assert EnsembleDetector(profile=profile).predict(text) is expected
```

- [ ] **Step 3: Remove the exact duplicates**

For each pair in the Files list, open both and confirm the bodies are identical before deleting the second copy. Run `python -m pytest --co -q | tail -1` before and after to record the count.

- [ ] **Step 4: Run** `python -m pytest -q -W error::FutureWarning` → all pass, no warnings.

- [ ] **Step 5: Commit**

```bash
git add tests
git commit -m "test: registry-wide contract tests, profile coverage, drop exact duplicates

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Phase 6 — Re-baseline and release readiness

### Task 21: Regression re-baseline, full verification, CHANGELOG finalisation

**Files:**
- Modify: `regression/english_results.json`, `regression/benchmark_results.json`, `regression/benchmark_results.txt`, `CHANGELOG.md`, `docs/strategies.rst` (generated)

- [ ] **Step 1: Regenerate the evaluation and compare holdout metrics**

```bash
git show 56b04f5:regression/english_results.json > /tmp/baseline.json
python regression/evaluate.py --split all --output regression/english_results.json
python - <<'EOF'
import json
a=json.load(open('/tmp/baseline.json')); b=json.load(open('regression/english_results.json'))
def find(d, key):
    if isinstance(d, dict):
        for k,v in d.items():
            if k==key: yield v
            yield from find(v,key)
    elif isinstance(d, list):
        for v in d: yield from find(v,key)
print("baseline f1s:", list(find(a,"f1"))[:6])
print("current  f1s:", list(find(b,"f1"))[:6])
EOF
```
Expected: holdout F1 for the `english` profile is >= baseline. If any profile's holdout F1 dropped, stop and report which challenge cases flipped (`python regression/evaluate.py --split holdout --details`) before continuing.

- [ ] **Step 2: Regenerate benchmark results and strategy docs**

```bash
python regression/benchmark.py
python scripts/update_strategy_docs.py
git diff --stat regression docs/strategies.rst
```

- [ ] **Step 3: Full verification gate**

```bash
black --check pygarble tests scripts regression && isort --check-only pygarble tests scripts regression && flake8 pygarble tests scripts regression && mypy pygarble && python -m pytest -q -W error::FutureWarning && python scripts/update_strategy_docs.py --check && python scripts/generate_data.py --check && python -m sphinx -b html -W --keep-going -q docs /tmp/pygarble-docs && python -m build -q --outdir /tmp/pygarble-final && echo ALL-GREEN
```
Expected: `ALL-GREEN`.

- [ ] **Step 4: Verify every README/doc snippet still runs**

```bash
python - <<'EOF'
import re, subprocess, sys, pathlib
blocks = []
for path in ["README.md", *map(str, pathlib.Path("docs").glob("*.rst"))]:
    text = pathlib.Path(path).read_text()
    blocks += re.findall(r"```python\n(.*?)```", text, re.S)
    blocks += re.findall(r".. code-block:: python\n\n((?:    .*\n|\n)+)", text)
ok = 0
for block in blocks:
    code = "\n".join(l[4:] if l.startswith("    ") else l for l in block.splitlines())
    r = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, env={"PYTHONPATH": "."})
    if r.returncode: print("FAILED:\n", code, r.stderr); sys.exit(1)
    ok += 1
print("snippets ok:", ok)
EOF
```
Expected: `snippets ok: N` with no failures.

- [ ] **Step 5: Finalise CHANGELOG**

Set the heading to `## [0.9.0] - <today's date>` and make sure every `### Fixed` line added by Tasks 5-18 is present. Add under `### Changed`: `- Regression baseline regenerated; ensemble holdout F1 <before> -> <after>.` with the numbers from Step 1.

- [ ] **Step 6: Commit**

```bash
git add regression CHANGELOG.md docs/strategies.rst
git commit -m "chore: re-baseline regression results for 0.9.0 fixes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 7: Manual follow-ups for the maintainer (not automated)**

- Switch GitHub Pages source to "GitHub Actions" in repo settings (spec F8). Optional CLI: `gh api -X PUT repos/brightertiger/pygarble/pages -f build_type=workflow`.
- Migrate `release.yml` to PyPI Trusted Publishing (drop `PYPI_API_TOKEN`); needs a one-time PyPI setting.
- Regenerate `examples/getting_started.ipynb` against the 0.9 API.

---

## Deferred (tracked, not in this release)

- Consolidate the five bigram log-prob scorers (Markov, LLR, Entropy, WordAnomaly, LocalAnomaly) into `scoring.py` with a word-selection policy.
- Merge the three keyboard detectors; one shared consonant-run / vowel definition.
- Score-calibration policy across strategies (when weak evidence may cross 0.5).
- `DEFAULT_LOG_PROB` -10 → table minimum (needs data regeneration + manifest).
- Full test-file re-layout by subject.
- `timeout_per_text` enforcement for inline calls (would need a subprocess or signal-based design).
