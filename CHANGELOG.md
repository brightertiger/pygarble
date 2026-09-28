# Changelog

All notable changes to pygarble are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Four English-reference strategies, `CROSS_PARSING`,
  `PRIMED_COMPRESSION`, `NGRAM_RANK` and `PERMUTATION_TEST`, which compare
  the lowercase ASCII words of a text with frequent English (cross parsing,
  preset-dictionary deflate, character n-gram ranks and a letter-shuffle
  permutation test). Each is standardised against a synthetic English null
  and scored as the median over 127-character windows; they need at least
  8 letters and are not members of any existing profile.
  `PRIMED_COMPRESSION` sizes come from the platform's zlib, so its scores
  can differ slightly between zlib builds.
- `EnsembleDetector(voting="fisher")` with keyword-only `fisher_alpha`
  (default 0.001). Each applicable member's score is read against that
  strategy's scores on a synthetic English null to give a tail p-value, and
  Fisher's method combines them; the score is 0.5 when the combined p-value
  equals `fisher_alpha`. It is opt-in and heuristic: members are correlated
  and the null is synthetic, so `fisher_alpha` is not a guaranteed
  false-positive rate. Passing `fisher_alpha` with another voting mode emits
  a `FutureWarning`; a strategy listed twice raises `ValueError` under
  `fisher` only.
- `SCORE_NULL_TAILS` in `pygarble.data`: per-strategy score thresholds on
  the synthetic English null, loaded only for Fisher voting.
- Opt-in `english_fusion` profile: `WORD_LOOKUP`, `LOG_LIKELIHOOD_RATIO` and
  `CROSS_PARSING` with `fisher` voting by default. Its members are
  English-reference methods, so meaningful text in other languages written
  in Latin letters can be flagged, and text in other scripts gets no signal
  (it reads as clean, which is not evidence that it is meaningful).
  Existing profiles, their voting and the `EnsembleDetector()` default are
  unchanged; the golden corpus gains `english_fusion` rows only.
- Documentation discovery metadata, a generated sitemap and `llms.txt`,
  optional Search Console verification, and a publishing/discoverability guide.
- Root contributor and security guides, issue templates, and package metadata
  links for open-source contributors.
- `pygarble.screening`: a dedicated secrets, PII and profanity API, with
  explicit optional backends for phonenumberslite, python-stdnum,
  detect-secrets and a separately installed Gitleaks executable.
- Per-instance custom detectors, kind filtering, source-free backend errors,
  and shared span validation/redaction. Existing top-level APIs retain their
  defaults; importing screening does not load the gibberish engine.
- `pygarble-screen` / `python -m pygarble.screening`: whole-document scanning
  and redaction, findings-only scan output, and a configurable input limit.
  This avoids the legacy line CLI's exposed multiline key bodies and echoed
  source text. The legacy CLI is retained unchanged.
- Optional-backend CI, offline Python integration tests, Unicode/CRLF offset
  checks and subprocess error/timeout tests.

### Fixed
- NHS/phone overlap filtering now uses a sorted sweep rather than a pairwise
  search, keeping dense PII documents from causing quadratic work.

### Changed
- Release uploads use PyPI Trusted Publishing with the `pypi` GitHub
  environment; the project owner must configure that publisher before tagging
  the next release. Existing distribution files are no longer silently skipped.
- Updated the README, quick start, API reference, CLI and installation
  guides for the separate modules and optional backends. Added architecture
  and migration guidance covering compatibility pointers and unreleased APIs.
- Gibberish implementations now live under `pygarble.gibberish`; native
  PII, profanity and secret detectors and their rules live under
  `pygarble.screening`. Old module paths and top-level exports remain
  compatibility pointers to the same implementations. Scanner defaults,
  lazy strategy loading and CLI behavior are unchanged.

## [0.11.0] - 2026-09-26

### Added
- `Scanner`, `scan()` and `redact()`: one call that screens text for
  secrets, PII, profanity and gibberish and returns findings with kind,
  span, confidence and reason. Findings never carry the matched text.
- Secrets detector: known vendor prefixes (AWS, GitHub, GitLab, Slack,
  Stripe, Google, OpenAI, Anthropic, Hugging Face, npm, PyPI, SendGrid),
  JWTs, private key blocks, credentials in URLs, bearer tokens,
  keyword-plus-entropy generic secrets, and an opt-in
  `high_entropy_string` rule (`secrets_without_context=True`).
- PII detector: email, phone, credit card (Luhn), IBAN (mod-97), IPv4/IPv6,
  plus locale packs for the US (SSN), UK (National Insurance and NHS
  numbers) and India (Aadhaar with Verhoeff, PAN), each with its national
  phone formats.
- Profanity detector: an attributed word list with leetspeak, elongation,
  embedded, masked and spaced obfuscation handling and an allowlist.
- Redaction in placeholder, mask and partial modes.
- `pygarble scan` and `pygarble redact` commands.
- JSON copies of the rule tables for ports, a golden scan corpus with a CI
  check, a clean-corpus false-positive gate and a throughput script.
- Throughput work on the rule detectors: literal prefilters, an ASCII fast
  path and a token cache. `regression/throughput.py` reports MB/s per
  category, with `--chunk-bytes` for document-sized inputs.
- Documentation pages for screening and redaction, secrets, PII and
  profanity; the README now leads with screening.
- A vendor token or JWT inside a bearer token or URL credentials is also
  reported as its own nested finding; selecting kinds only filters the
  default output.
- `Scanner` raises `ValueError` when a selection leaves nothing to scan or
  redact; the CLI rejects empty `--categories`, `--kinds` and `--locales`.

### Notes
- The gibberish API is unchanged. This release is additive.

## [0.10.0] - 2026-09-26

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
- Golden corpus `regression/golden.jsonl` (challenge cases and edge inputs
  x every profile) with a CI check; ports in other languages are held to
  it.
- README rewritten around use cases, plus new documentation pages for the
  command line and threshold calibration.

### Notes
- 0.9.0 was prepared but never published to PyPI; users upgrading from 0.8.0 should read both entries.

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
