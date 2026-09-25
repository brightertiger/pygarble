# pygarble 0.9.x Audit Findings (spec for the cleanup release)

Date: 2026-09-26. Baseline: 618 tests pass, flake8 and mypy clean, 90% coverage.
Four independent read-only audits (core API, large strategies, remaining
strategies + data, infra/tests/docs) were consolidated here. This document is
the *spec* for `docs/superpowers/plans/2026-09-26-cleanup-and-refactor.md`.

## Release constraints

- **Backward compatibility is required.** No public name may be removed or
  renamed: `GarbleDetector`, `EnsembleDetector`, `Strategy`, every
  `Strategy` member, every `*Strategy` class (including the misspelled
  `PronouncabilityStrategy`), `Analysis`, `Signal`, `Span`, `PROFILES`,
  `STRATEGY_MAP`, every accepted kwarg in `pygarble/options.py`.
- Default ensemble voting per profile must not change ("any" for named
  profiles, "majority" for custom lists).
- Scores may change where the change fixes a documented false positive or
  false negative. Every such change is recorded in `CHANGELOG.md`.
- The package stays pure Python, zero runtime dependencies, Python >= 3.8.
- Determinism across processes must hold (existing test enforces it).

## A. Default-profile bugs (affect `EnsembleDetector()` with no config)

| # | Location | Defect | Verified trigger |
|---|---|---|---|
| A1 | `strategies/base.py:113-116`, `:97` | Allowlist dropped: default `_evaluate_features` passes only `features.text`; `_novel_words` builds a fresh `TextFeatures` with no allowlist. Only 4 strategies honor it. | `EnsembleDetector(allowlist=["asdfgh"]).analyze("please ping asdfgh about the deploy").garbled` is `True` |
| A2 | `strategies/word_anomaly.py:66` | Scores all ASCII words instead of novel ones, so dictionary acronyms (dhcp, kpmg, hgtv, jdbc, cnbc, xvii) below -4.6 flag the whole ensemble under "any" voting. | `EnsembleDetector().predict("Please enable DHCP on the router")` is `True` |
| A3 | `strategies/mojibake.py:190` | Length guard (`< 3`) runs before the U+FFFD check. | `"a�"` scores 0.0, `"ab�"` scores 0.85 |

## B. `english_extended` profile bugs

| # | Location | Defect | Verified trigger |
|---|---|---|---|
| B1 | `strategies/pattern_matching.py:21` | `consonant_cluster` (5+ consonants) is a strong pattern and matches real words. | "The strengths of the plan are clear" → ensemble `True` |
| B2 | `strategies/pattern_matching.py:11` | `repeated_chars` counts digits as strong evidence. | "We raised 10000 dollars for the school" → `True` |
| B3 | `strategies/repetition.py:73` | Repeated-char regex counts digits; 6+ zeros crosses 0.5. | "One million is written 1000000" → 0.6 |
| B4 | `strategies/repetition.py:134` | Word dominance fires on 3-word emphatic phrases. | "No, no, no!" → 1.0 |
| B5 | `strategies/local_anomaly.py:54` | Window threshold `max(2, (w+1)//2)` means `window_words=1` never emits a span. | `LocalAnomalyStrategy(window_words=1)` on "xqzkvbw qzxkvjw" → no spans |

## C. Core API defects

| # | Location | Defect |
|---|---|---|
| C1 | `ensemble.py:83-94` | `weights` silently ignored unless `voting="weighted"`. |
| C2 | `options.py:156-161` | Unknown-kwarg warning is a `DeprecationWarning` (hidden by default outside `__main__`), `stacklevel=3` lands in `detector.py:63`, and the message wrongly tells single-strategy users to use `strategy_kwargs`. |
| C3 | `ensemble.py:115-116` | Shared `**kwargs` fan out to every member; a strategy-specific kwarg warns once per member that lacks it (4 warnings for `min_word_length=3`). Will become 4 errors when the promise in C2 lands. |
| C4 | `validation.py:14` | `float(10**400)` raises `OverflowError`, not the promised `ValueError`. |
| C5 | `validation.py:79-82` | `timeout_per_text` is accepted but ignored for single strings, `threads in (None, 1)`, and batches under 10. |
| C6 | `ensemble.py:98` | String keys in `strategy_kwargs` (`{"markov_chain": {...}}`) fail with the misleading "contains a strategy not selected". |
| C7 | `detector.py:45-50` | String `strategy` raises `TypeError` with no coercion. |
| C8 | `strategies/base.py:29-47` | Legacy `BaseStrategy.predict`/`predict_proba` bypass `evaluate()`: skip applicability, the score-range check, and the allowlist. |
| C9 | `registry.py:59`, `options.py:103`, `strategies/__init__.py` | Class is misspelled `PronouncabilityStrategy`; enum is `PRONOUNCEABILITY`. |
| C10 | `ensemble.py:134-145` | Weighted mean divides by `maximum` which cancels algebraically; the `maximum == 0` guard is unreachable. |
| C11 | `detector.py:65`, `validation.py:32` (core use) | Dead: `_validate_batch` alias; `parameter_value` unused in core. |
| C12 | `ensemble.py:152-157` | Under `majority`, `Analysis.garbled` can be `True` while `Analysis.score < threshold`. Documented in `predict_proba` docstring only. |

## D. Opt-in strategy defects

| # | Location | Defect | Verified trigger |
|---|---|---|---|
| D1 | `letter_position.py:448` | `threshold=0.0` passes validation then divides by zero. | `LetterPositionStrategy(threshold=0.0).predict_proba("jbx")` → `ZeroDivisionError` |
| D2 | `letter_position.py:415` | Skips the novel-word filter; `NEVER_START` contains real onsets `cz`, `sv`, `vl`. | "Czech svelte vlog" → 1.0; "Save as PDF or JPG" → 0.833 |
| D3 | `unicode_script.py:176,211` | Splits on whitespace only, so CJK+Latin and scientific units count as mixed-script words. | "私はiPhoneを使います", "cells were 5 μm wide", "The α-helix is stable" → 0.85 |
| D4 | `unicode_script.py:238-251` | `check_homoglyphs` / `homoglyph_threshold` have no observable effect. | "pаypal" → 0.85 with every setting |
| D5 | `pronounceability.py:71,81` | Onsets `qu`/`squ` never match because `u` is a vowel; `sph`, `chl`, `scl`, `phl`, `schl`, `gh` missing. | "squid", "sphinx", "chlorinate", "schlep", "ghoulish" → 1.0 |
| D6 | `pronounceability.py:683` | `min_word_length` below 4 is silently ignored by a hard-coded filter. | `min_word_length=2` on "xkq bkx" → 0.0 |
| D7 | `function_word_density.py:217` | "a" and "I" never counted: length filter runs before the function-word lookup. | `_tokenize("I am a cat")` → `['am','cat']` |
| D8 | `function_word_density.py:238`, `word_collocation.py:460` | Title Case exemption also exempts ALL-CAPS gibberish (first char upper). | 15-token all-caps mash → 0.3; lowercase → 0.9 |
| D9 | `word_collocation.py:462-465` | Zero-collocation 12-19 words scores 0.6, above decision. | 13-word plain sentence → `True` |
| D10 | `word_collocation.py:410` | Only ASCII apostrophe keeps contractions whole. | "don’t" → `['don','t']` |
| D11 | `hex_string.py:70,118` | Base64 check accepts paths and camelCase with a digit. | "/usr/local/bin/python3", "getUserAccountBalanceV2" → 0.9 |
| D12 | `vowel_ratio.py:83-87` | Three params read raw per call, never range-checked; 1-3 letter inputs saturate. | `min_vowel_ratio=5.0` accepted; "I", "Hmm", "Mr. Ng" → 1.0 |
| D13 | `keyboard_pattern.py:100` | Repeated-bigram check joins letters across words. | "Go go go" → 0.5 |
| D14 | `mojibake.py:60-62` vs `:212` | Docstring says text at `ratio_threshold` is garbled; density at threshold maps to 0.25. | `"a"*19 + "\x85"` → 0.25, `predict` False |

## E. Duplication and dead code

- Tokenizer reimplemented ~7 ways: `[a-zA-Z]+` (affix, zipf, word_lookup, function_word_density), `[a-zA-Z']+` (word_collocation), isalpha loops (letter_position, consonant_sequence, vowel_pattern), split+isalpha (bigram_probability, entropy), `[a-z0-9]+` (keyboard_adjacency, repetition), `.split()` (pronounceability, unicode_script). `TextFeatures.tokens` / `ascii_words` exist for this.
- Dead code: `pronounceability._extract_consonant_clusters`, `log_likelihood_ratio._extract_bigrams`, `markov_chain._compute_log_probability`, `consonant_sequence._get_max_consonant_run` + `_count_violations`, `vowel_ratio._has_consonant_cluster`, `keyboard_adjacency` module-level `_ADJACENT`/`_ROW_OF` (lines 35-36), `detector._validate_batch`.
- Unreachable range checks after `parameter_value`/`positive_int` already reject `< 1`: `markov_chain.py:67`, `ngram_frequency.py:59`, `repetition.py:61-64`, `hex_string.py:53`, `symbol_ratio.py:71`.
- `rare_trigram.py:127-138`: nine trigrams listed twice (`qjq qxq qzq xjx xqx xzx zjx zqx zxj`).
- `zipf_conformity.py:16` imports `FunctionWordDensityStrategy` only for `FUNCTION_WORDS`.
- `mojibake.py:13` misplaced coding line; `except Exception` at `:105` should be `UnicodeDecodeError`.
- `keyboard_pattern.py` tokenizes 3x per call, `letter_frequency.py` 3x; these are the two slowest strategies.
- Deferred (not this release): consolidating the five bigram log-prob scorers, three keyboard detectors, and the score-calibration policy. Changing `DEFAULT_LOG_PROB` requires regenerating data + manifest.

## F. Release plumbing

| # | Location | Defect |
|---|---|---|
| F1 | `README.md:15`, `docs/{index,api,installation,migration,quickstart,examples}.rst` | Eight places say 0.9.0 is "upcoming / not on PyPI". README becomes the PyPI long description. |
| F2 | `.bumpversion.cfg:2` | Says 0.3.1; `publish.yml` and `test-pypi.yml` run `bump2version` and fail. Both also check out the pre-bump SHA. `publish-current.yml` can publish with no tag. `release.yml` is the correct path. |
| F3 | `pyproject.toml:10` | Author email is the placeholder `ujjwal@example.com`, live on PyPI. Real values are in `pygarble/__init__.py:2-3`. |
| F4 | `tests/test_english_api.py:319` | Imports `regression.evaluate`; sdist ships `tests/` but not `regression/`, so sdist test run fails 1/618. |
| F5 | `pyproject.toml:2` | `setuptools>=45` but PEP 621 `[project]` needs `>=61`. |
| F6 | Version hard-coded in `pyproject.toml:7`, `pygarble/__init__.py:1`, `docs/conf.py:19-20`. `release.yml:46` checks tag against pyproject only. |
| F7 | `.github/workflows/test.yml:5-7` triggers on `master`/`develop` (don't exist); `setup-python@v4` and `@v5` mixed. |
| F8 | GitHub Pages source is still "legacy branch" so Jekyll and the Sphinx action race on every push (Sphinx wins by ~5 s). Repo setting, not code. |
| F9 | `.github/instructions.txt` references `ci.yml`, `version-bump.yml`, `setup.py`, FastText, "44 tests". Entirely stale. |
| F10 | `docs/audit-plan.md`, `docs/implementation.md` are internal planning docs presented as current; not built by Sphinx (no Markdown extension). |
| F11 | No `CHANGELOG.md`. `docs/conf.py:16` copyright 2025. `examples/getting_started.ipynb` shows 0.8 API and stale outputs. |
| F12 | `Makefile` `lint` omits black/isort; `format` omits isort. `requirements-dev.txt` duplicates `[dev]` extra and adds sphinx; extra lacks sphinx/build/twine. |
| F13 | `regression/benchmark.py:495` has no argparse; any invocation (even `--help`) overwrites tracked `benchmark_results.*`. Committed results are from 0.8.0. |
| F14 | Classifiers lack 3.13 though CI passes on it. |

## G. Test suite

- 49 tests assert only `isinstance`, `0 <= p <= 1`, `len == n`, or `is not None`.
- No test iterates over `Strategy`; profiles `english_extended`, `legacy`, `spoofing` have zero tests; "unknown profile" and "profile + strategies" errors untested.
- `tests/test_strategies.py:13` passes `entropy_threshold=2.0`, a no-op that produces the suite's only warning.
- Duplicate bodies: `test_core.py:11,16` = `test_strategies.py:36,41`; `test_edge_cases.py:102` = `test_new_features.py:106`; `test_edge_cases.py:138/152/162` = `test_new_strategies.py:56/105/161`; `test_v7_strategies.py:80` = `:108`; `test_fix_regressions_a.py:62` = `:73`; `predict("") is False` asserted 20 times in 7 files.
- Never executed: `letter_frequency.py:90-110,163-175` (chi-squared path), `local_anomaly.py:57` (window span), `zipf_conformity.py:114-132`.
- `test_english_api.py:278` depends on a 0.05 s sleep vs 0.001 s timeout.
- File layout is by date written (`fix_regressions_a/b/c1/c2`, `v7`, `new_*`, `precision`), not by subject. Full re-layout is deferred; this release adds contract tests and removes exact duplicates.

## Things to preserve

- Frozen dataclasses, lazy data loading via `__getattr__`, deterministic span ordering, `evaluate()` score-range enforcement.
- Hash-pinned reproducible data tables and `generate_data.py --check` in CI.
- Executable docs (every snippet runs) and generated `strategies.rst` with `--check`.
- `release.yml` shape and `test_english_api.py` style.
