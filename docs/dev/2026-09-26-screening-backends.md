# Screening module and optional backends

## Scope and compatibility

Introduce `pygarble.screening` for secrets, PII and profanity. Its `Scanner`
never invokes gibberish detection. Keep the existing top-level scanner and
detector imports compatible, including their defaults and result types.
Share collection and redaction logic rather than maintaining two engines.
Existing detector implementations remain reusable from both APIs.

## Implementation plan

1. Extract a typed internal scan engine and publish a screening detector
   protocol. Add explicit per-instance backend selection and custom detector
   injection, without global registration or automatic plugin discovery.
2. Add optional phone extraction with `phonenumberslite` and an explicit
   catalogue of identifier patterns validated locally by `python-stdnum`.
   Keep extraction and validation distinct; preserve original string offsets.
3. Add `detect-secrets` string plugins without verification, global settings
   mutation or model filters. Preserve entropy filtering when those plugins
   are explicitly selected. Report provider provenance in finding reasons.
4. Add a Gitleaks stdin adapter with a subprocess timeout, isolated default
   configuration, private temporary reports and validated original-text spans.
   Require an explicitly installed executable; never download it at runtime.
5. Add `python -m pygarble.screening` / `pygarble-screen` for document scanning
   and redaction. Default scan output contains findings only. Bound input size,
   preserve multiline keys and report backend failures as errors, never clean
   scans. Keep the legacy CLI compatible and document the difference.
6. Replace quadratic NHS/phone overlap filtering with a sorted sweep.
7. Add compatibility, Unicode/redaction, offline, failure-path and real-backend
   integration tests. Run the base suite without extras, the optional suite
   with extras and Gitleaks, formatting, typing, generated-data/golden checks,
   documentation and packaging checks. Add an optional-backend CI job.
8. Push the working branch, open a PR and check CI. Merge only when requested.

## Runtime contract

- Core installation has no runtime dependencies; optional packages load only
  when explicitly selected. No LLMs, model downloads or network verification.
- `Finding`, `ScanReport` and `Redaction` retain their existing structure.
  Backend findings carry no source text; their reasons identify the provider.
- Backend scores are fixed evidence tiers, not calibrated probabilities.
- Kind/category selection applies to every backend. Errors propagate with
  source-free messages, and malformed backend spans cannot reach redaction.
- Gitleaks runs once per document. For throughput, reuse Python scanners and
  submit documents rather than launching an executable per short line.
- Broader profanity semantics, names/addresses, automatic blocking policies
  and a separate distribution are outside this change.

## Local validation

- Base suite: 1,672 passed; 10 optional-backend tests skipped without extras.
- Optional suite: 58 passed with phonenumberslite 9.0.40, python-stdnum 2.2,
  detect-secrets 1.5.0 and Gitleaks 8.30.1, on Python 3.12/macOS ARM64.
- Black, isort, flake8 and mypy pass. Generated strategy docs, data tables,
  918 golden gibberish rows and 621 golden scan rows reproduce.
- Sphinx HTML builds with warnings treated as errors. Wheel and sdist build
  and pass twine checks. A clean environment installs the wheel with
  `--no-deps` and exercises screening, the console entry point and legacy
  gibberish APIs outside the checkout.
- CI adds real-backend jobs on Python 3.8 and 3.12 with a checksummed,
  pinned Gitleaks binary. These complement the existing base-package matrix.
