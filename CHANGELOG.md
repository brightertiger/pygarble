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
- allowlist is now honoured by every strategy, not only the four that
  override feature evaluation.
- weights passed with a non-weighted voting mode now raise ValueError instead
  of being ignored.
- huge integers for numeric options raise ValueError instead of
  OverflowError.
- calling a strategy's `predict`/`predict_proba` directly now goes through
  `evaluate()`, so text the strategy deems not applicable scores 0.0.
- WordAnomaly (default profile) no longer flags dictionary acronyms such as
  DHCP, KPMG, HGTV.
- Mojibake detects U+FFFD in inputs shorter than three characters.
- PatternMatching (english_extended) no longer flags real words with
  five-consonant runs or round numbers like 10000; repeated-character and
  alternating patterns are letters-only.
- Repetition ignores digit runs and 3-4 word emphasis ("very very very
  good"). A text that is one repeated word ("No, no, no!") is still flagged
  by design.
- LocalAnomaly rejects window_words=1, which could never emit a span.
- KeyboardPattern judges repeated bigrams per novel word, so "go go go" is
  clean.
- LetterPosition scores only novel words and rejects threshold=0 (previously
  ZeroDivisionError).
- Pronounceability accepts sq-, sph-, chl-, scl-, phl-, schl-, gh- onsets and
  honours min_word_length (default now 4, the previous effective value).
- FunctionWordDensity counts "a" and "I"; Title Case exemption no longer
  covers ALL-CAPS; WordCollocation needs 20+ words with no collocation to
  cross 0.5 and handles curly apostrophes.
- HexString no longer classifies file paths or camelCase identifiers as base64.

### Removed
- Five legacy strategies (see docs/migration.rst).

## [0.8.0] - 2026-07-04
- Previous release; see git history.
