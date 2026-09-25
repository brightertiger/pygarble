# Changelog

All notable changes to pygarble are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Command-line interface: `pygarble check|score|analyze` reads files or
  stdin, supports --profile/--strategy/--threshold/--allowlist,
  text/TSV/JSONL output, JSON --field mode, and exit code 1 when any input
  is garbled.
- `llm_output` profile (repetition, control characters, mojibake, local
  anomaly): a deterministic pre-check for degenerate model output that
  stays quiet on code and technical prose.
- `pygarble.calibrate(detector, garbled, clean)` sweeps thresholds over
  labeled samples and recommends one by F1 or by a maximum
  false-positive rate; `pygarble calibrate` does the same from files.
- Language-neutral JSON copies of the word, bigram and trigram tables under
  `pygarble/data/` (repo and sdist only), hashed in the manifest, as the
  shared source for ports.

## [0.9.0] - 2026-09-26

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
- weights passed with a non-weighted voting mode now emit a FutureWarning
  (they were silently ignored); this will become an error in a future
  release.
- Regression baseline regenerated; ensemble holdout F1 0.943 -> 0.943
  (`english` profile, unchanged; `english_extended` 0.982 -> 0.982).
  Benchmark (1644-case legacy dataset) ensemble F1 0.919 -> 0.932.

### Fixed
- allowlist is now honoured by every strategy, not only the four that
  override feature evaluation.
- huge integers for numeric options raise ValueError instead of
  OverflowError.
- calling a strategy's `predict`/`predict_proba` directly now goes through
  `evaluate()`, so text the strategy deems not applicable scores 0.0.
- WordAnomaly (default profile) no longer flags dictionary acronyms such as
  DHCP, KPMG, HGTV; ALL-CAPS tokens of six letters or fewer are treated as
  acronyms and never flagged by this strategy.
- Mojibake detects U+FFFD in inputs shorter than three characters.
- PatternMatching (english_extended) no longer flags real words with
  five-consonant runs or round numbers like 10000; repeated-character and
  alternating patterns are letters-only.
- Repetition ignores digit runs and 3-4 word emphasis ("very very very
  good"). A text that is one repeated word ("No, no, no!") is still flagged
  by design.
- LocalAnomaly: a one-word window now emits corrupt_token_window spans
  (previously only severe_local_anomaly).
- KeyboardPattern judges repeated bigrams per novel word, so "go go go" is
  clean.
- LetterPosition scores only novel words and rejects threshold=0 (previously
  ZeroDivisionError); Title-Case and ALL-CAPS tokens are exempt as likely
  proper nouns and acronyms.
- Pronounceability accepts sq-, sph-, chl-, scl-, phl-, schl-, gh- onsets and
  honours min_word_length (default now 4, the previous effective value).
- FunctionWordDensity counts "a" and "I"; Title Case exemption no longer
  covers ALL-CAPS; WordCollocation needs 20+ words with no collocation to
  cross 0.5 and handles curly apostrophes. Lists of ALL-CAPS acronyms no
  longer receive the Title-Case exemption.
- HexString no longer classifies file paths or camelCase identifiers as base64.
- VowelRatio validates its parameters, abstains below 4 letters (new
  min_length option), and ignores 1-2 letter abbreviations.
- UnicodeScript no longer flags CJK text with embedded Latin words,
  hyphen/digit-separated symbols such as "α-helix" or "μm", or any word
  shorter than three letters; only Latin/Cyrillic/Greek mixing inside one
  word of three or more letters counts. Greek letters attached to three or
  more Latin letters (e.g. "μmol") are still flagged.

### Removed
- Five legacy strategies (see docs/migration.rst).

## [0.8.0] - 2026-07-04
- Previous release; see git history.
