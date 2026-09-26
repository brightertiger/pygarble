# pygarble 0.11.0 "First Line of Defence" Design

Date: 2026-09-26. Author: Claude, on the maintainer's decision to grow pygarble from a gibberish detector into a deterministic, zero-dependency text screener, keeping the `pygarble` name.

## Goal

One call that screens a text for the things rules can catch reliably, says what it found and where, and returns a redacted copy. Four categories: **secrets**, **PII**, **profanity**, **gibberish**. Callers run it first, at scale, and reserve NLP models for what it misses.

## Why

Research on 2026-09-26 (npm and PyPI download stats, competitor source and issue trackers):

- The LLM guardrail stacks (LLM Guard, Guardrails AI) wrap heavy dependencies for each check: a 268 MB DistilBERT model for gibberish, Presidio plus spaCy for PII, detect-secrets for secrets, transformer classifiers for toxicity. There is no zero-dependency, deterministic, millisecond alternative that covers the deterministic subset in one package.
- The PyPI download leader for gibberish gets its traffic transitively from detect-secrets, which uses gibberish detection as a secrets filter. Secrets and gibberish are already adjacent in practice.
- pygarble already has the explainable span-and-reason output model, deterministic tests, a CLI, and a golden corpus. The new categories reuse all of it.

## Constraints

- Backward compatible with 0.10.0: no public name removed or renamed; `GarbleDetector`, `EnsembleDetector`, every profile, kwarg and default unchanged. Everything here is additive.
- Pure Python, `dependencies = []`, Python >= 3.8, black line length 79, isort, flake8, mypy `disallow_untyped_defs`, determinism preserved. `python -m pytest -q -W error::FutureWarning` stays green.
- Rules only. Nothing learned at runtime, nothing downloaded. Word and pattern tables ship in the package as `.py` modules; JSON copies are generated for ports and excluded from the wheel, as the 0.10.0 tables are.
- Findings never carry matched text, so a logged report cannot leak a secret. Offsets are Unicode code points.
- Every README and docs snippet executes under the existing snippet runner.

## Non-goals

Names, postal addresses and free-text dates of birth (need NLP). Toxicity or hate speech beyond a word list. Non-English profanity. Secret verification against providers. Training or custom corpora. These are documented as "what to send to the model after this pass".

## Package layout

```
pygarble/
  findings.py            Finding, ScanReport, Redaction
  scanner.py             Scanner, scan(), redact(), DEFAULT_CATEGORIES
  redaction.py           merge regions, render placeholders
  secrets/
    __init__.py          SecretsDetector, detect()
    patterns.py          KNOWN_PATTERNS table (source of truth)
    entropy.py           shannon(), charset classification, placeholder filter
  pii/
    __init__.py          PIIDetector, detect(), LOCALES
    checksums.py         luhn(), iban_mod97(), verhoeff(), nhs_mod11()
    patterns.py          GENERIC and per-locale pattern tables
  profanity/
    __init__.py          ProfanityDetector, detect()
    normalize.py         normalize_token(), collapse_runs(), LEET_MAP
    wordlist.py          PROFANITY_STRONG, PROFANITY_MILD, EMBEDDED (source of truth)
  data/
    secrets.json, pii.json, profanity.json   generated copies (sdist only)
```

`pygarble/__init__.py` exports `Scanner`, `scan`, `redact`, `Finding`, `ScanReport`, `Redaction`, `SecretsDetector`, `PIIDetector`, `ProfanityDetector`. Subpackages import lazily so `import pygarble` stays cheap.

## Shared model

```python
@dataclass(frozen=True)
class Finding:
    category: str      # "secrets" | "pii" | "profanity" | "gibberish"
    kind: str          # "aws_access_key_id", "email", "profanity", "garbled", ...
    start: int         # code point offset, inclusive
    end: int           # exclusive
    confidence: float  # 0.0..1.0, fixed per rule (see tiers)
    reason: str        # short machine-readable why: "luhn_valid", "known_prefix", "strong", ...

@dataclass(frozen=True)
class ScanReport:
    findings: Tuple[Finding, ...]   # sorted by (start, end, category, kind)
    flagged: bool                   # any finding with confidence >= min_confidence
    length: int                     # len(text)
    def by_category(self) -> Dict[str, Tuple[Finding, ...]]
    def kinds(self) -> Tuple[str, ...]          # distinct kinds, sorted
    def to_dict(self) -> Dict[str, Any]         # JSON-serialisable, no text

@dataclass(frozen=True)
class Redaction:
    text: str                       # the redacted copy
    findings: Tuple[Finding, ...]   # the findings that were redacted
    count: int                      # regions replaced (after merging overlaps)
```

**Confidence tiers** are fixed per rule and documented, not tuned:

| Tier | Meaning | Examples |
|---|---|---|
| 1.0 | checksum-verified or unique vendor prefix | Luhn card, IBAN mod-97, Aadhaar Verhoeff, `AKIA…`, `ghp_…`, private key block |
| 0.9 | unambiguous structure without checksum | email, E.164 phone, JWT with decodable header, credentials in URL, PAN, NI number |
| 0.8 | structural, some ambiguity | national phone formats, SSN with separators, bearer token, elongated, embedded or spaced profanity |
| 0.9 | see above | unambiguous masked profanity such as `f*ck` |
| 0.7 | weak or context-dependent | IPv4/IPv6, mild profanity |
| 0.6 | contextual entropy | `password=…` with high-entropy value, ambiguous wildcard profanity |
| score | gibberish | the ensemble score |

`min_confidence` (default 0.5) decides `flagged`; findings below it are still returned so callers can inspect them.

## Scanner

```python
class Scanner:
    def __init__(
        self,
        categories: Iterable[str] = ("secrets", "pii", "profanity", "gibberish"),
        *,
        min_confidence: float = 0.5,
        kinds: Optional[Iterable[str]] = None,       # restrict to these kinds
        exclude_kinds: Iterable[str] = (),
        locales: Iterable[str] = ("us", "uk", "in"),  # PII locale packs
        profile: str = "english",                     # gibberish profile
        threshold: float = 0.5,                       # gibberish threshold
        allowlist: Optional[Iterable[str]] = None,    # gibberish allowlist
        profanity_allowlist: Optional[Iterable[str]] = None,
        secrets_without_context: bool = False,        # standalone high-entropy strings
        max_input_length: Optional[int] = None,       # None = unlimited for rules
    ) -> None
    def scan(self, text: str) -> ScanReport
    def scan_batch(self, texts: Sequence[str]) -> List[ScanReport]
    def iter_scan(self, texts: Iterable[str]) -> Iterator[ScanReport]
    def redact(self, text: str, *, mode: str = "placeholder",
               placeholder: str = "[{KIND}]", mask_char: str = "*",
               categories: Optional[Iterable[str]] = None) -> Redaction
```

Module-level `scan(text, **kwargs)` and `redact(text, **kwargs)` build a `Scanner` per distinct kwargs and cache it (small LRU keyed by the sorted kwargs), so the ten-second-start snippet is one line.

Validation: `text` must be `str` (`TypeError` otherwise, via the existing `process_input`); unknown category, kind or locale → `ValueError` listing the valid names; `min_confidence` and `threshold` via `unit_interval`. The gibberish category delegates to `EnsembleDetector(profile=..., threshold=..., allowlist=...)` and applies its own `max_input_length` defaults; the rule categories scan the whole text unless `max_input_length` is set.

**Gibberish as a category.** When the ensemble says garbled, one `Finding(category="gibberish", kind="garbled", start=0, end=len(text), confidence=score, reason=status)` is emitted. Not garbled → no finding. Gibberish findings are never redacted (a whole-text placeholder is useless); callers wanting spans use `EnsembleDetector.analyze` directly, and the docs say so.

**Determinism.** Findings are sorted by `(start, end, category, kind)`. Every detector is a pure function of its inputs and tables.

## Redaction

`Scanner.redact` scans, keeps findings with `confidence >= min_confidence` in the chosen categories (default: all but gibberish), merges overlapping or touching regions, and replaces each region right to left so offsets stay valid.

- `mode="placeholder"`: region becomes `placeholder.format(KIND=kind.upper(), kind=kind, category=category)`; default `[EMAIL]`, `[AWS_ACCESS_KEY_ID]`, `[PROFANITY]`. A merged region takes the kind of its highest-confidence finding, ties to the earliest.
- `mode="mask"`: region becomes `mask_char * (end - start)`, length preserved.
- `mode="partial"`: like mask but keeps the last four characters for kinds in `REVEAL_LAST_FOUR = {credit_card, phone, iban, ssn_us, nhs_number, aadhaar}`; all other kinds are fully masked.

`Redaction.text` for a text with no findings is the input unchanged. Redaction is deterministic and idempotent for placeholder mode (placeholders contain nothing the rules match).

## Secrets

`SecretsDetector(kinds=None, exclude_kinds=(), without_context=False)`.

**Known-prefix patterns** (`patterns.py`, one entry per kind: `kind`, `regex`, `confidence`, `reason="known_prefix"`, `vectors` = at least one positive and one negative example). v1 kinds:

| kind | shape | conf |
|---|---|---|
| aws_access_key_id | `(AKIA\|ASIA\|ABIA\|ACCA)[0-9A-Z]{16}` | 1.0 |
| aws_secret_access_key | `aws` within 20 chars of `secret`/`key`, then a 40-char `[A-Za-z0-9/+=]` value | 0.9 |
| github_token | `(ghp\|gho\|ghu\|ghs\|ghr)_[A-Za-z0-9]{36,}` or `github_pat_[A-Za-z0-9_]{22,}` | 1.0 |
| gitlab_token | `glpat-[A-Za-z0-9_\-]{20,}` | 1.0 |
| slack_token | `xox[abprs]-[0-9A-Za-z\-]{10,}` | 1.0 |
| slack_webhook | `hooks.slack.com/services/T…/B…/…` | 1.0 |
| stripe_key | `(sk\|rk)_(live\|test)_[A-Za-z0-9]{16,}` | 1.0 live, 0.8 test |
| google_api_key | `AIza[0-9A-Za-z_\-]{35}` | 1.0 |
| openai_api_key | `sk-(proj-\|svcacct-)?…T3BlbkFJ…` or legacy `sk-[A-Za-z0-9]{48}` | 1.0 / 0.9 |
| anthropic_api_key | `sk-ant-(api\|admin)\d{2}-[A-Za-z0-9_\-]{80,}` | 1.0 |
| huggingface_token | `hf_[A-Za-z0-9]{34}` | 1.0 |
| npm_token | `npm_[A-Za-z0-9]{36}` | 1.0 |
| pypi_token | `pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,}` | 1.0 |
| sendgrid_key | `SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}` | 1.0 |
| jwt | three base64url segments, first two starting `eyJ`; header must base64-decode to JSON containing `"alg"` | 1.0 (0.8 if header fails to decode) |
| private_key | `-----BEGIN (RSA\|EC\|DSA\|OPENSSH\|PGP\|ENCRYPTED )?PRIVATE KEY( BLOCK)?-----`; span extends to the matching END marker when present | 1.0 |
| url_credentials | `scheme://user:password@host`; span covers `user:password` | 0.9 |
| bearer_token | `Bearer <20+ token chars>` (case-insensitive) | 0.8 |

All known patterns compile into one alternation with named groups; one `finditer` pass per text. Boundaries use lookarounds, not `\b`, because tokens contain `-` and `_`.

**Contextual entropy** (`generic_secret`, conf 0.6, reason `"keyword_entropy"`): a keyword `password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|auth[_-]?token|client[_-]?secret|private[_-]?key` (case-insensitive, whole word) followed by optional whitespace, `:` or `=` or `=>`, optional quote, then a value of 8+ non-space non-quote characters. The value is a finding when its Shannon entropy in bits per character is at or above the threshold for its charset: hex (`[0-9a-fA-F]`) 3.0, base64 (`[A-Za-z0-9+/=_-]`) 4.5, anything else 3.5 (thresholds follow detect-secrets). Placeholders are rejected before entropy: values matching `^[<{$%]`, `^x+$`, `^\*+$`, or in `{changeme, password, secret, example, null, none, true, false, todo, redacted}` (case-insensitive), or containing `example` or `placeholder`. The finding span is the value only.

**Standalone high entropy** (`high_entropy_string`, conf 0.5, off by default via `without_context`): any `[A-Za-z0-9+/=_-]{32,}` token meeting the base64 threshold, or `[0-9a-fA-F]{32,}` meeting the hex threshold, excluding tokens already covered by another finding. Documented as noisy; for secret-scanning pipelines, not chat.

`detect(text, **kwargs) -> Tuple[Finding, ...]` is the functional form on every detector.

## PII

`PIIDetector(kinds=None, exclude_kinds=(), locales=("us", "uk", "in"))`. `LOCALES = {"us", "uk", "in"}`; unknown locale → `ValueError`.

**Generic kinds:**

| kind | rule | conf | reason |
|---|---|---|---|
| email | `[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}` bounded by non-address chars | 0.9 | `structure` |
| phone | E.164 `\+[1-9]\d{6,14}` | 0.9 | `e164` |
| phone | national: US `(\(\d{3}\)\s?\|\d{3}[\s.-])\d{3}[\s.-]\d{4}`; UK `0\d{4}\s?\d{6}`, `0\d{3}\s?\d{3}\s?\d{4}`; IN `(\+91[\s-]?\|0)?[6-9]\d{4}[\s-]?\d{5}`; 7 to 15 digits total, not part of a longer digit run, not matching a date/timestamp shape | 0.8 | `national_<locale>` |
| credit_card | 13-19 digits with optional single spaces or hyphens between groups, Luhn valid, IIN maps to a brand (visa, mastercard, amex, discover, jcb, diners, rupay, maestro); Luhn failure or unknown brand → no finding | 1.0 | `luhn_<brand>` |
| iban | `[A-Z]{2}\d{2}[A-Z0-9]{11,30}` (spaces allowed every four), country in the length table, mod-97 == 1 | 1.0 | `mod97` |
| ipv4 | four octets 0-255 bounded by non-digit-non-dot; excludes version-like `1.2.3.4` only when preceded by `v` | 0.7 | `structure` |
| ipv6 | RFC 4291 forms including `::` compression and IPv4 tail, bounded by non-hex-non-colon | 0.7 | `structure` |

**Locale packs:**

| locale | kind | rule | conf |
|---|---|---|---|
| us | ssn_us | `AAA-GG-SSSS` with `-` or space; area not 000, 666, 900-999; group not 00; serial not 0000 | 0.8; 0.6 when bare 9 digits preceded within 30 chars by `ssn` or `social security` |
| uk | nino | two letters (excluding D, F, I, Q, U, V in either, O in second; prefixes BG GB NK KN TN NT ZZ rejected), six digits, suffix A-D, spaces optional | 0.9 |
| uk | nhs_number | ten digits as `NNN NNN NNNN`, or bare when preceded within 30 characters by `nhs`; mod-11 check digit valid | 0.9 |
| in | aadhaar | twelve digits as `NNNN NNNN NNNN`, or bare when preceded within 30 characters by `aadhaar`/`uidai`; first digit 2-9, Verhoeff valid | 1.0 |
| in | pan | `[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]` | 0.9 |

Phone national formats are locale-gated too. Each PII rule compiles to its own regex (backreferences and per-kind validators make one alternation impractical); all start with a literal, digit class or lookbehind. Checksums live in `checksums.py` as pure functions with their own unit tests: `luhn(digits)`, `iban_mod97(iban)`, `verhoeff(digits)`, `nhs_mod11(digits)`.

**Overlap rule within PII:** when a credit card and a phone match the same digits, the checksum-verified kind wins and the other is dropped. In general, two findings with identical spans keep the higher confidence; nested findings of different kinds are both kept (redaction merges them).

## Profanity

`ProfanityDetector(allowlist=None, tiers=("strong", "mild"), obfuscation=True)`.

**Word list** (`wordlist.py`): `PROFANITY_STRONG` (slurs and hard profanity, including common compounds like `bullshit`, `dumbass`, `motherfucker`) and `PROFANITY_MILD` (`damn`, `crap`, `bloody` and similar; `hell` is deliberately excluded). Seed source: the LDNOOBW English list (CC-BY 4.0, Shutterstock), filtered to remove non-profane sexual-health and anatomical terms that would flag ordinary text, plus curated additions. The data module header and `docs/profanity.rst` carry the attribution. Entries are stored normalised (casefolded, diacritics folded). Multi-word entries are stored as tuples and matched as token sequences. `EMBEDDED` is the small subset of strong words (four letters or more, verified not to occur inside any word in `ENGLISH_WORDS`) that are also matched inside longer tokens, so `fuckwit` and `shitposting` are found without a Scunthorpe problem.

**Tokenisation:** `\w+(?:['’]\w+)*` over the original text, offsets preserved.

**Normalisation** (`normalize.py`), applied to each token: casefold → NFKD and strip combining marks → leetspeak map (`0→o 1→i 3→e 4→a 5→s 7→t 8→b @→a $→s !→i |→l +→t`) → keep letters only for comparison. `collapse_runs(token, to)` collapses runs of the same letter to length `to`.

**Matching, in order, per token:**

1. Exact: normalised token in strong or mild list → conf 1.0 / 0.7, reason `"strong"` / `"mild"`.
2. Elongation: if the token contains a run of three or more of the same letter, test `collapse_runs(t, 2)` and `collapse_runs(t, 1)` against the lists → conf 0.8, reason `"elongated"`. Tokens without such a run never use collapsed forms, so `as` never matches `ass`.
3. Embedded: token contains an `EMBEDDED` word as a substring → conf 0.8, reason `"embedded"`, span is the whole token.
4. Wildcard (obfuscation=True): token from the *raw* text matches `[\w*#@$!]+` with at least one of `* # @ $ !` and at least two letters; each symbol is treated as a one-letter wildcard; if exactly one strong-list word of the same length matches → conf 0.9, reason `"masked"`; if more than one matches → conf 0.6, reason `"masked_ambiguous"`; if the wildcard pattern also fits an ordinary word in `ENGLISH_WORDS` (e.g. `sh*t` → `shot`) confidence drops to 0.6. Bare-digit checksum forms need a keyword because a random number passes mod-11 or Verhoeff about one time in ten. `dick` is not listed because it is a common name; its compounds are. Wildcard is never applied to the mild list.
5. Spaced (obfuscation=True): a run of three or more consecutive single-letter tokens separated only by spaces, dots or hyphens is joined and tested as an exact token → conf 0.8, reason `"spaced"`, span covers the run.

Multi-word entries are checked as sliding windows over the normalised token stream before single-token matching. The allowlist (normalised) suppresses any finding whose span text normalises to an allowlisted entry. `kind` is always `"profanity"`; the tier is in `reason` and drives confidence.

## CLI

Two new subcommands sharing the 0.10.0 input machinery (`inputs`, `-t/--text`, `--field`, streaming `read_lines`, UTF-8 replace/backslashreplace, BrokenPipe handling, exit 2 on errors):

- `pygarble scan [inputs] [--categories LIST] [--kinds LIST] [--exclude-kinds LIST] [--locales LIST] [--min-confidence F] [--profile P] [--threshold F] [--format text|tsv|jsonl] [--show-matches]`. Text output per line: `flagged\t<comma-separated kinds>\t<text>` or `clean\t\t<text>`. TSV: `flagged(0/1)\tcount\tkinds\ttext`. JSONL: `{"text":…, "flagged":…, "findings":[{"category","kind","start","end","confidence","reason"}]}`; `--show-matches` adds `"match"` with the matched substring (off by default so logs stay clean). Exit 1 if any line was flagged, else 0.
- `pygarble redact [inputs] [same selection flags] [--mode placeholder|mask|partial] [--placeholder TEMPLATE]`. Prints the redacted line per input. With `--field`, the field is redacted in place and the object echoed. Exit 0, or 2 on error.

## Data tables and ports

`scripts/generate_data.py` gains writers for `secrets.json` (list of `{kind, regex, confidence, reason, vectors}`), `pii.json` (patterns, locale tables, IBAN lengths, IIN table) and `profanity.json` (`strong`, `mild`, `embedded`, `leet_map`, `attribution`). Same byte-reproducible JSON settings as 0.10.0, manifest hashes cover them, `--check` verifies them, `exclude-package-data` keeps them out of the wheel. Regexes are written as Python `re` syntax; the docs note which constructs a port must support (lookbehind, named groups, no Unicode property escapes are used).

## Regression and quality gates

- `regression/scan_vectors.json`: per-kind positive and hard-negative vectors with expected findings (kind, start, end, confidence). Consumed by `tests/test_scan_vectors.py`. Every kind has at least three positives and three negatives.
- `regression/clean_corpus/`: `code.py`, `code.js`, `config.json`, `logs.txt`, `prose.txt` (ordinary text, technical prose, changelog-style text). `tests/test_clean_corpus.py` asserts no secrets, PII or profanity finding at or above 0.8 in any file, with a small documented exception list (e.g. an IP in a log line is expected at 0.7 and below the bar).
- `regression/golden_scan.jsonl` + `.sha256`: frozen `Scanner.scan` output (findings only, no text) for every scan vector, every clean-corpus line and the 0.10.0 golden inputs. `regression/golden.py` gains `--scan` handling or a sibling `golden_scan.py` with the same `--write`/`--check` contract; CI runs the check.
- `regression/throughput.py`: builds a deterministic 10 MB synthetic corpus (prose with planted findings), times `scan` per category and all together, and prints MB/s and findings/s. `--json` output; the README table is pasted from it with the machine noted. No target is promised; the number is published.
- Existing gates unchanged: black, isort, flake8, mypy, `pytest -W error::FutureWarning`, `update_strategy_docs.py --check`, `generate_data.py --check`, `golden.py --check`, sphinx `-W`, `python -m build`.

## Docs and release

- README: pitch becomes "deterministic, zero-dependency first line of defence for text: secrets, PII, profanity and gibberish, with redaction"; ten-second start shows `scan` and `redact`; a "What it catches / what it doesn't" table; the throughput table; the gibberish sections condensed under a "Gibberish" heading.
- New `docs/screening.rst` (Scanner, findings, redaction, confidence tiers), `docs/secrets.rst`, `docs/pii.rst`, `docs/profanity.rst` (with attribution), added to the toctree after `quickstart`. `docs/api.rst` gains the new classes. `docs/cli.rst` gains `scan` and `redact`. `docs/contributing.rst` covers the vectors, clean corpus, golden scan file and throughput script.
- `pyproject.toml`: description updated; keywords gain `pii`, `secrets`, `profanity`, `redaction`, `guardrails`, `llm`; classifier `Topic :: Security` added.
- `__version__ = "0.11.0"`, CHANGELOG `## [0.11.0] - 2026-09-26` with Added (Scanner and redaction, secrets, PII with US/UK/IN packs, profanity, CLI scan/redact, JSON tables, golden scan corpus, throughput script) and Notes (findings never carry text; gibberish API unchanged). `docs/migration.rst` gains a 0.11.0 paragraph: no breaking changes.
- Branch pushed, one PR opened, phases visible as commit groups. Tagging and publishing are left to the maintainer.

## Testing summary

- Unit: checksums; entropy and placeholder filter; normalisation and each profanity matching rule; redaction merging, modes and idempotence; Scanner validation errors and kind/locale selection; lazy import (`import pygarble` does not import subpackages).
- Vectors: `test_scan_vectors.py` over `scan_vectors.json`.
- False positives: `test_clean_corpus.py`.
- Golden: `test_golden_scan.py`.
- CLI: `main(argv)` with `capsys` for `scan` and `redact`, all formats, `--field`, `--show-matches`, exit codes.
- Docs: snippet runner covers the new pages.
- Determinism: scanning the same text twice, and in a batch vs alone, yields equal findings.

## Phases

1. Shared model (`findings.py`, `scanner.py`, `redaction.py`) with the gibberish category and the secrets detector; scan vectors and clean corpus scaffolding; unit tests.
2. PII detector with checksums and locale packs; vectors extended.
3. Profanity detector with normalisation and obfuscation rules; word list with attribution; vectors extended.
4. CLI `scan`/`redact`, JSON tables and manifest, golden scan corpus and CI, throughput script, README and docs, release metadata.
