# Changelog

All notable changes to pygarble are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **Docs: switched to the sphinx-immaterial theme.** The site now matches the
  Material for MkDocs look of the sister project: navigation tabs, a light
  and dark palette toggle, search suggestions, code copy buttons and a cookie
  consent banner. Optional Google Analytics now uses the theme's native
  integration and loads only after consent. The `docs` extra installs
  `sphinx-immaterial==0.13.9` on Python 3.10+ in place of `sphinx-rtd-theme`.
- **Docs: the sitemap has `lastmod` dates and the build writes a
  `robots.txt`.** Dates come from each source file's last commit. The robots
  file is served under `/pygarble/`, which Google does not read; it mirrors
  the sister project's site.
- **The golden regression corpus has new inputs.** They are a seeded
  600-row sample (seed 20261011) of the English rows of the Apache-2.0
  Hugging Face `Sowmya15/gibberish_multilingual` dataset, 300 random strings
  and 300 tweets, followed by the unchanged edge inputs. They replace the
  authored 132-case challenge set. `paper/regression/golden.jsonl` and
  `golden_scan.jsonl` were regenerated, and ports must reproduce the new
  files. `paper/regression/evaluate.py` and the challenge files
  (`english_challenge.json`, its checksum, `benchmark_data.json` and
  `label_overrides.json`) were removed.

## [0.11.0] - Unreleased

0.11.0 is the first release since 0.8.0 and it is a **breaking release**
relative to 0.8.0. Versions 0.9.0 and 0.10.0 were development milestones
that were never published to PyPI; their notes are merged into this entry.
Earlier drafts of this file listed `WORD_ANOMALY`, `LOG_LIKELIHOOD_RATIO` and
`KEYBOARD_ADJACENCY` as 0.9.0 additions and the removal of five legacy
strategies as a 0.9.0 change; all of that shipped in 0.8.0. The upgrade
guide (`docs/migration.rst`, "Upgrading from 0.8.0 to 0.11.0") explains how
to find and handle each change below.

### Breaking changes since 0.8.0

Gibberish scores are not reproducible across the 0.8.0 boundary. No setting
restores 0.8.0 output exactly; compare old and new decisions on your own
inputs before upgrading.

- **Scores changed in every 0.8.0 strategy.** Strategies share one
  tokeniser. A whitespace-separated chunk that contains a digit, `://`, `@`
  or a backslash, starts with `www.`, `data:` or a path prefix, or has a
  lowercase-to-uppercase step (camelCase) is treated as structured, and its
  letters are not scored as words. Most JWTs, base64 blobs, GitHub-style
  tokens, PEM private key blocks, URLs, email addresses, paths and code
  identifiers are therefore no longer flagged as gibberish; screen them with
  the new `secrets` and `pii` detectors instead. Affected strategies include
  `MARKOV_CHAIN`, `LOG_LIKELIHOOD_RATIO`, `NGRAM_FREQUENCY`, `WORD_ANOMALY`,
  `KEYBOARD_ADJACENCY`, `KEYBOARD_PATTERN` and `LETTER_FREQUENCY`. Some
  inputs move the other way: zero-width-joined letters and homoglyph domains
  such as `pаypal.com` (Cyrillic `а`) can now be flagged. The per-strategy
  fixes under Fixed change further scores. `LOG_LIKELIHOOD_RATIO`, and every
  ensemble containing it, also differs from 0.8.0 in the last one or two
  floating-point digits, which breaks exact-equality golden files. There is
  no escape hatch.
- **The default `EnsembleDetector()` has six members instead of three.**
  `MOJIBAKE`, `KEYBOARD_ADJACENCY` and `CONTROL_CHARACTERS` joined
  `MARKOV_CHAIN`, `LOG_LIKELIHOOD_RATIO` and `WORD_ANOMALY`, still with `any`
  voting. Keyboard runs such as `qwerty`, mojibake, control characters and
  some long URLs can now be flagged, and `.strategies` and `.weights` have
  six entries. Escape hatch: `EnsembleDetector(profile="legacy")` selects the
  0.8.0 members; its scores still follow the tokeniser change above.
- **Weighted voting with three weights on the default set raises
  `ValueError`.** `EnsembleDetector(voting="weighted", weights=[a, b, c])`
  worked in 0.8.0. The message gives both lengths and points to the legacy
  profile. Escape hatch:
  `EnsembleDetector(profile="legacy", voting="weighted", weights=[a, b, c])`.
- **Length alone no longer makes text garbled.** In 0.8.0 every strategy
  scored text with a single token over 1,000 characters as 1.0 and garbled.
  That rule is now opt-in. Escape hatch: pass `max_string_length=1000`,
  which classifies long non-URL tokens as garbled. To reject oversized input
  instead, set `max_input_length`.
- **`threshold=0.0` no longer flags every non-empty text,** and
  `applicable("")` and `applicable("!!!")` return `False`. Empty or wholly
  inapplicable input predicts `False`; `analyze()` reports
  `insufficient_evidence`.
- **Some invalid arguments raise different exceptions.** A wrongly typed
  `threshold`, `threads`, `threshold_per_char` or weight raises `ValueError`
  where 0.8.0 raised `TypeError`. A huge integer weight such as `10**400`
  raises `ValueError` where 0.8.0 raised `OverflowError`. An unknown
  strategy name raises
  `ValueError` instead of `AttributeError`, and `None` or a number as the
  strategy raises `TypeError` instead of `AttributeError`.
  `EnsembleDetector(strategies=[None])` raises `ValueError` naming the bad
  member. An invalid custom regex for `pattern_matching` raises `ValueError`
  naming the pattern, chained from `re.error`, instead of `re.error`.
- **Some values that 0.8.0 accepted now raise.** `threshold=True`,
  `threads=2.5`, NaN weights and a non-bool `override_defaults` raise
  `ValueError`. So does a huge integer such as `10**400` for any numeric
  option: in 0.8.0 `threads`, `timeout_per_text`, `max_string_length` and
  `min_length` accepted it, as did settings 0.8.0 ignored, such as `n`. `applicable(None)` and a batch with a non-string entry, such
  as `predict([None])`, raise `TypeError`; batches are validated before any
  work starts.
- **New `FutureWarning`s.** Unknown strategy settings, and `weights` passed
  with a voting mode other than `weighted`, warn at the call site. Test
  suites run with `-W error` or `filterwarnings = error` fail on them. Escape
  hatch: remove the setting, or filter `FutureWarning` from `pygarble`.
- **`Strategy` and `STRATEGY_MAP` have 32 entries instead of 26.** Code
  that iterates either to build "all strategies" also gets the six new ones.
  `STRATEGY_MAP` is still a plain `dict` and, as in 0.8.0, detectors use its
  entries.

0.8.0 import paths keep working with a `DeprecationWarning` (see
Deprecated), and detectors pickled by 0.8.0 load and predict.

### Added

Gibberish detection:
- `EnsembleDetector` profiles: `english` (the default set), `english_extended`,
  `legacy`, `corruption`, `spoofing`, `llm_output` and the opt-in
  `english_fusion`. `llm_output` (repetition, control characters, mojibake,
  local anomaly) is a deterministic pre-check for degenerate model output
  that stays quiet on code and technical prose.
- `analyze()` with spans, status and per-strategy signals, and `score()` as
  an alias of `predict_proba`.
- `allowlist`, honoured by every strategy; `strategy_kwargs`;
  `max_input_length`, enforced by `analyze()`, `predict()` and
  `applicable()`. `timeout_per_text`, which 0.8.0 read from `**kwargs`, is a
  named, validated keyword.
- `GarbleDetector`, `EnsembleDetector` and `strategy_kwargs` accept strategy
  names as strings.
- Strategies `CONTROL_CHARACTERS` and `LOCAL_ANOMALY`. `CONTROL_CHARACTERS`
  also flags text made only of controls that `str.strip()` removes (VT, FF,
  U+001C-U+001F, NEL), scoring it 0.9. `LOCAL_ANOMALY` emits
  `corrupt_token_window` spans, including for a one-word window.
- Four English-reference strategies, `CROSS_PARSING`,
  `PRIMED_COMPRESSION`, `NGRAM_RANK` and `PERMUTATION_TEST`, which compare
  the lowercase ASCII words of a text with frequent English (cross parsing,
  preset-dictionary deflate, character n-gram ranks and a letter-shuffle
  permutation test). Each is standardised against a synthetic English null
  and scored as the median over 127-character windows; they need at least
  8 letters and are not members of any profile except `english_fusion`
  (`CROSS_PARSING`). `PRIMED_COMPRESSION` sizes come from the platform's
  zlib, so its scores can differ slightly between zlib builds.
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
- The correctly spelled `PronounceabilityStrategy` alias.
- `pygarble.calibrate(detector, garbled, clean)` sweeps thresholds over
  labeled samples and recommends one by F1 or by a maximum false-positive
  rate, in O(N log N). `pygarble calibrate` does the same from files.

Screening:
- `Scanner`, `scan()` and `redact()`: one call that screens text for
  secrets, PII, profanity and gibberish and returns findings with kind,
  span, confidence and reason. Findings never carry the matched text.
  `scan()` and `redact()` validate their options on every call, and pass a
  dict-valued option to `Scanner` uncached.
- Secrets detector: known vendor prefixes (AWS, GitHub, GitLab, Slack,
  Stripe, Google, OpenAI, Anthropic, Hugging Face, npm, PyPI, SendGrid),
  JWTs, private key blocks, credentials in URLs, bearer tokens,
  keyword-plus-entropy generic secrets, and an opt-in
  `high_entropy_string` rule (`secrets_without_context=True`). A vendor
  token or JWT inside a bearer token or URL credentials is also reported as
  its own nested finding; selecting kinds only filters the default output.
- PII detector: email, phone, credit card (Luhn), IBAN (mod-97), IPv4/IPv6,
  plus locale packs for the US (SSN), UK (National Insurance and NHS
  numbers) and India (Aadhaar with Verhoeff, PAN), each with its national
  phone formats.
- Profanity detector: an attributed word list with leetspeak, elongation,
  embedded, masked and spaced obfuscation handling and an allowlist.
- Redaction in placeholder, mask and partial modes.
- `Scanner` raises `ValueError` when a selection leaves nothing to scan or
  redact.
- `pygarble.screening`: a dedicated secrets, PII and profanity API that does
  not load the gibberish engine, with per-instance custom detectors, kind
  filtering, source-free backend errors and shared span validation and
  redaction.
- Optional local screening backends, each enabled explicitly: phonenumberslite
  (`pygarble[phones]`), python-stdnum (`pygarble[stdnum]`), detect-secrets
  (`pygarble[secrets]`), all three (`pygarble[screening]`), and a separately
  installed Gitleaks executable. A backend whose package is installed but
  fails to import says why. On Python 3.8 the `stdnum` and `screening`
  extras also install setuptools, which python-stdnum needs.
- Throughput work on the rule detectors: literal prefilters, an ASCII fast
  path, a token cache and sorted-sweep overlap filtering, so dense PII
  documents do not cause quadratic work.

Command line:
- `pygarble check|score|analyze` (also `python -m pygarble`) reads files or
  stdin, supports `--profile`, `--strategy`, `--threshold` and
  `--allowlist`, text/TSV/JSONL output and JSON `--field` mode, and exits 1
  when any input is garbled. `pygarble calibrate`, `pygarble scan` and
  `pygarble redact` complete the line-oriented CLI; empty `--categories`,
  `--kinds` and `--locales` are rejected. Text and TSV rows print a tab or
  any character `str.splitlines()` breaks on inside the text as a Python
  escape (`\t`, `\n`, `\r`, `\x0b`, `\x0c`, `\x1c`, `\x1d`, `\x1e`,
  `\x85`, `\u2028`, `\u2029`), so one text is always one row. Backslashes
  are not escaped; JSONL is the lossless format.
- `pygarble-screen` (also `python -m pygarble.screening`): whole-document
  scanning and redaction, findings-only scan output and a configurable input
  limit, so multiline key bodies are redacted and source text is not echoed.
  It exits 0 when a reader such as `head` closes the pipe.

Data, regression checks and documentation:
- Language-neutral JSON copies of the word, bigram, trigram, secrets, PII
  and profanity tables under `pygarble/data/` (repository and sdist only),
  hashed in the manifest, as the shared source for ports.
- A golden corpus of frozen detector outputs for every profile and a golden
  scan corpus, both checked in CI, plus a clean-corpus false-positive gate
  and a throughput script (`paper/regression/`).
- Documentation for the command line, calibration, screening and
  redaction, secrets, PII, profanity, standalone screening, architecture and
  upgrading; discovery metadata, a generated sitemap and `llms.txt`, and
  optional Search Console verification; root contributor and security
  guides, issue templates and package metadata links.
- Optional-backend CI, offline Python integration tests, Unicode/CRLF offset
  checks and subprocess error and timeout tests.

### Changed
- Gibberish implementations live under `pygarble.gibberish`; native PII,
  profanity and secret detectors and their rules live under
  `pygarble.screening`. `pygarble`, `pygarble.core`, `pygarble.strategies`
  and the other old module paths point to the same classes, enums and
  caches. New pickles use the canonical paths and may not load in 0.8.0.
- Ensemble-level `**kwargs` are forwarded only to member strategies that
  accept them, with one warning for a key that no selected member accepts.
- Calling a strategy's `predict`/`predict_proba` directly goes through
  `evaluate()`, so text the strategy deems not applicable scores 0.0.
- `WORD_LOOKUP`'s `unknown_threshold` now affects the score. The default
  0.5 keeps the 0.8.0 mapping; other values change results.
- At the 0.9.0 milestone the regression baseline was regenerated: ensemble
  holdout F1 0.943 (`english`) and 0.982 (`english_extended`); benchmark
  (1644-case legacy dataset) ensemble F1 0.919 -> 0.932 against 0.8.0.
- CI tests Python 3.8 to 3.13, pins every action to a commit SHA, and keeps
  jobs that need Python 3.8 or 3.9 on `ubuntu-24.04`, because
  `ubuntu-latest` moves to 26.04, which has no builds for those versions.
  The generated-data check caches the pinned word list. The data generator
  retries a failed download once and reports a network fault separately
  from a checksum mismatch.
- Releases are published from a `v*` tag by one workflow. Before uploading
  it runs the tests on Python 3.8 and 3.12, lint, the golden checks and a
  smoke test of the built wheel. Uploads use PyPI Trusted Publishing with
  the `pypi` GitHub environment, so the project owner must configure that
  publisher before tagging; existing distribution files are no longer
  silently skipped. The Docker image is pushed to GHCR only after the PyPI
  upload succeeds. The separate Docker workflow, including its manual
  trigger, is removed.
- Development installs pin flake8 7.x and isort 9.x, which CI lints with;
  isort 5.13 is used on Python 3.8 and 3.9. `make lint` and `make format`
  cover `docs/_ext`, as CI does.
- Generated data tables cite `paper/scripts/generate_data.py` and
  `paper/scripts/data_curation.json` in their headers.

### Deprecated
- 0.8.0 import paths work with a `DeprecationWarning` naming the current
  home, and will be removed in a future release: `BaseStrategy` and the 26
  0.8.0 strategy classes from `pygarble.core` (use
  `pygarble.gibberish.strategies`), `BIGRAM_LOG_PROBS` and
  `DEFAULT_LOG_PROB` from the `markov_chain`, `word_anomaly` and
  `log_likelihood_ratio` strategy modules (use `pygarble.data`), and
  `FunctionWordDensityStrategy` from `zipf_conformity` (use
  `pygarble.gibberish.strategies`).
- Unknown strategy settings emit a `FutureWarning` at the call site and will
  become errors in a future release.
- `weights` passed with a voting mode other than `weighted` emit a
  `FutureWarning` (they were silently ignored); this will become an error
  in a future release.

### Removed
- `pygarble.core` no longer re-exports incidental names that 0.8.0 exposed
  through its own imports, such as the `typing` helpers `Callable` and
  `Dict` and the `concurrent` module. Import them from the standard library.

### Fixed
- `GarbleDetector` and `EnsembleDetector` objects pickled by 0.8.0 load and
  predict. A 0.8.0 default ensemble reports profile `legacy`.
- A bare string passed where a list of names belongs (`EnsembleDetector`
  strategies, allowlists, PII and secret kinds, locales, and `Scanner`
  `categories`, `backends`, `kinds` and `exclude_kinds`) raises an error
  that is both a `TypeError` and a `ValueError`, with a message saying to
  pass a list. `EnsembleDetector("markov_chain")` no longer reports `'m' is
  not a valid Strategy`.
- `WORD_ANOMALY` no longer flags dictionary acronyms such as DHCP, KPMG and
  HGTV; ALL-CAPS tokens of six letters or fewer are treated as acronyms and
  never flagged by this strategy.
- `MOJIBAKE` detects U+FFFD in inputs shorter than three characters.
- `PATTERN_MATCHING` no longer flags real words with five-consonant runs or
  round numbers like 10000; repeated-character and alternating patterns are
  letters-only.
- `REPETITION` ignores digit runs and 3-4 word emphasis ("very very very
  good"). A text that is one repeated word ("No, no, no!") is still flagged
  by design.
- `KEYBOARD_PATTERN` judges repeated bigrams per novel word, so "go go go"
  is clean.
- `LETTER_POSITION` scores only novel words and rejects `threshold=0`
  (previously `ZeroDivisionError`); Title-Case and ALL-CAPS tokens are
  exempt as likely proper nouns and acronyms.
- `PRONOUNCEABILITY` accepts sq-, sph-, chl-, scl-, phl-, schl- and gh-
  onsets and honours `min_word_length` (default 4, the previous effective
  value). It is linear in the length of a long consonant run; it was
  quadratic, about 10 s for one 1M-character token.
- `FUNCTION_WORD_DENSITY` counts "a" and "I"; its Title Case exemption no
  longer covers ALL-CAPS, so lists of ALL-CAPS acronyms are not exempt.
  `WORD_COLLOCATION` needs 20+ words with no collocation to cross 0.5 and
  handles curly apostrophes.
- `HEX_STRING` no longer classifies file paths or camelCase identifiers as
  base64.
- `VOWEL_RATIO` validates its parameters, abstains below 4 letters (new
  `min_length` option), and ignores 1-2 letter abbreviations.
- `UNICODE_SCRIPT` no longer flags CJK text with embedded Latin words,
  hyphen/digit-separated symbols such as "α-helix" or "μm", or any word
  shorter than three letters; only Latin/Cyrillic/Greek mixing inside one
  word of three or more letters counts. Greek letters attached to three or
  more Latin letters (e.g. "μmol") are still flagged.

## [0.8.0] - 2026-07-04
- Previous release; see git history.
