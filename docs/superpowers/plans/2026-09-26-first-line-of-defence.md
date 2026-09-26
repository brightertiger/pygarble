# First Line of Defence (0.11.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic secrets, PII and profanity detection with redaction to pygarble behind one `Scanner` API and CLI, keeping the gibberish API untouched, and release as 0.11.0.

**Architecture:** New frozen `Finding`/`ScanReport`/`Redaction` types in `pygarble/findings.py`; three detector subpackages (`secrets`, `pii`, `profanity`), each a pure function of its tables exposing `detect(text) -> Tuple[Finding, ...]`; `pygarble/scanner.py` composes them with the existing `EnsembleDetector` as a fourth category and `pygarble/redaction.py` renders redacted text. Tables are Python modules with generated JSON copies (sdist only), pinned by scan vectors, a clean corpus and a golden scan file.

**Tech Stack:** Python >= 3.8 stdlib only (`re`, `math`, `unicodedata`, `base64`, `json`, `dataclasses`); pytest; black 79; isort; flake8; mypy strict defs.

**Spec:** `docs/superpowers/specs/2026-09-26-first-line-of-defence-design.md`

## Global Constraints

- `dependencies = []`; Python >= 3.8 syntax (no `match`, no `X | Y` types, no `list[str]` at runtime).
- black line length 79, isort profile black, flake8 clean, `mypy pygarble` clean with `disallow_untyped_defs`.
- Test gate: `python -m pytest -q -W error::FutureWarning`.
- Ad-hoc `python -c` needs `PYTHONPATH=.` (an old pygarble in site-packages shadows the repo).
- Backward compatible with 0.10.0: no public name removed or renamed; `GarbleDetector`, `EnsembleDetector`, profiles, kwargs and defaults unchanged.
- Findings never carry matched text. Offsets are Unicode code points into the original text.
- Deterministic: same input, same output; findings sorted by `(start, end, category, kind)`.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. Overlapping findings of different categories (a JWT inside a `Bearer` header, a card number that is also a phone candidate): redaction must produce one region and no text leaks between them. Pinned in Task 1 (`test_redact_merges_nested_and_touching_regions`) and Task 5 (`test_card_beats_phone_on_identical_span`).
2. Empty string and whitespace-only input to `scan`, `redact` and every detector must return empty findings, `flagged=False`, unchanged text. Pinned in Task 3 (`test_scan_empty_and_whitespace`).
3. Placeholder templates with unknown fields (`"[{nope}]"`) must raise `ValueError`, not `KeyError`. Pinned in Task 1 (`test_placeholder_unknown_field_is_value_error`).
4. Ordinary technical text (URLs with ports, version strings, hashes in logs, `key=value` config with low-entropy values) must not be flagged at 0.8 or above. Pinned in Task 8 (clean corpus) and per-kind negative vectors.
5. Very long inputs (1 MB of prose) must scan without quadratic blow-up; the entropy regex and profanity tokenizer must not backtrack. Pinned in Task 11 (`throughput.py` and `test_throughput_smoke`).

---

### Task 1: Findings model and redaction engine

**Files:**
- Create: `pygarble/findings.py`
- Create: `pygarble/redaction.py`
- Test: `tests/test_findings.py`, `tests/test_redaction.py`

**Interfaces:**
- Produces: `Finding(category, kind, start, end, confidence, reason)`, `ScanReport(findings, flagged, length)` with `by_category()`, `kinds()`, `to_dict()`; `Redaction(text, findings, count)`; `pygarble.findings.CATEGORIES`, `sort_key(finding)`; `pygarble.redaction.render(text, findings, mode, placeholder, mask_char) -> Redaction`, `MODES`, `REVEAL_LAST_FOUR`.

- [ ] **Step 1: Write the failing tests**

`tests/test_findings.py`:

```python
"""Immutable finding and report contracts."""

import dataclasses

import pytest

from pygarble.findings import CATEGORIES, Finding, ScanReport, sort_key


def finding(**overrides):
    base = dict(
        category="pii",
        kind="email",
        start=3,
        end=10,
        confidence=0.9,
        reason="structure",
    )
    base.update(overrides)
    return Finding(**base)


def test_categories_are_fixed():
    assert CATEGORIES == ("secrets", "pii", "profanity", "gibberish")


def test_finding_is_frozen_and_has_no_text_field():
    f = finding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.kind = "phone"
    assert set(f.to_dict()) == {
        "category",
        "kind",
        "start",
        "end",
        "confidence",
        "reason",
    }


def test_sort_key_orders_by_start_end_category_kind():
    items = [
        finding(start=5, end=9, category="pii", kind="phone"),
        finding(start=5, end=9, category="pii", kind="email"),
        finding(start=1, end=2, category="secrets", kind="jwt"),
        finding(start=5, end=7, category="secrets", kind="jwt"),
    ]
    ordered = sorted(items, key=sort_key)
    assert [(f.start, f.end, f.category, f.kind) for f in ordered] == [
        (1, 2, "secrets", "jwt"),
        (5, 7, "secrets", "jwt"),
        (5, 9, "pii", "email"),
        (5, 9, "pii", "phone"),
    ]


def test_report_helpers():
    report = ScanReport(
        findings=(
            finding(kind="email"),
            finding(category="secrets", kind="jwt", start=20, end=40),
            finding(kind="email", start=50, end=60),
        ),
        flagged=True,
        length=80,
    )
    assert report.kinds() == ("email", "jwt")
    by = report.by_category()
    assert set(by) == {"pii", "secrets"}
    assert len(by["pii"]) == 2
    payload = report.to_dict()
    assert payload["flagged"] is True
    assert payload["length"] == 80
    assert payload["findings"][1]["kind"] == "jwt"
    assert "text" not in payload
```

`tests/test_redaction.py`:

```python
"""Redaction merges regions and renders every mode deterministically."""

import pytest

from pygarble.findings import Finding, Redaction
from pygarble.redaction import MODES, REVEAL_LAST_FOUR, render


def f(kind, start, end, confidence=0.9, category="pii"):
    return Finding(category, kind, start, end, confidence, "test")


TEXT = "mail a@b.co card 4111 1111 1111 1111 now"


def test_modes_and_reveal_set():
    assert MODES == ("placeholder", "mask", "partial")
    assert "credit_card" in REVEAL_LAST_FOUR and "email" not in (
        REVEAL_LAST_FOUR
    )


def test_placeholder_mode_default_template():
    out = render(TEXT, [f("email", 5, 11)], "placeholder", "[{KIND}]", "*")
    assert isinstance(out, Redaction)
    assert out.text == "mail [EMAIL] card 4111 1111 1111 1111 now"
    assert out.count == 1
    assert out.findings == (f("email", 5, 11),)


def test_placeholder_template_fields():
    out = render(
        TEXT, [f("email", 5, 11)], "placeholder", "<{category}:{kind}>", "*"
    )
    assert out.text.startswith("mail <pii:email> card")


def test_placeholder_unknown_field_is_value_error():
    with pytest.raises(ValueError, match="placeholder"):
        render(TEXT, [f("email", 5, 11)], "placeholder", "[{nope}]", "*")


def test_mask_mode_preserves_length():
    out = render(TEXT, [f("email", 5, 11)], "mask", "[{KIND}]", "#")
    assert out.text == "mail ###### card 4111 1111 1111 1111 now"
    assert len(out.text) == len(TEXT)


def test_partial_mode_reveals_last_four_only_for_listed_kinds():
    card = f("credit_card", 17, 36, 1.0)
    out = render(TEXT, [card, f("email", 5, 11)], "partial", "[{KIND}]", "*")
    assert out.text == "mail ****** card ***************1111 now"


def test_redact_merges_nested_and_touching_regions():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc"
    bearer = Finding("secrets", "bearer_token", 22, 65, 0.8, "bearer")
    jwt = Finding("secrets", "jwt", 22, 65, 1.0, "jwt_header")
    touching = Finding("pii", "email", 65, 65, 0.9, "x")
    out = render(text, [touching, bearer, jwt], "placeholder", "[{KIND}]", "*")
    assert out.text == "Authorization: Bearer [JWT]"[:22] + "[JWT]"
    assert out.count == 1
    assert out.findings[0] is bearer or out.findings[0] == bearer


def test_merge_ties_go_to_earliest_finding():
    a = f("phone", 0, 4, 0.8)
    b = f("ssn_us", 2, 6, 0.8)
    out = render("0123456789", [b, a], "placeholder", "[{KIND}]", "*")
    assert out.text == "[PHONE]6789"


def test_no_findings_returns_input_unchanged():
    out = render(TEXT, [], "mask", "[{KIND}]", "*")
    assert out.text == TEXT and out.count == 0 and out.findings == ()


def test_invalid_mode_and_mask_char():
    with pytest.raises(ValueError, match="mode"):
        render(TEXT, [], "shred", "[{KIND}]", "*")
    with pytest.raises(ValueError, match="mask_char"):
        render(TEXT, [], "mask", "[{KIND}]", "**")


def test_placeholder_is_idempotent():
    once = render(TEXT, [f("email", 5, 11)], "placeholder", "[{KIND}]", "*")
    again = render(once.text, [], "placeholder", "[{KIND}]", "*")
    assert again.text == once.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/test_findings.py tests/test_redaction.py -q`
Expected: ImportError for `pygarble.findings`.

- [ ] **Step 3: Implement `pygarble/findings.py`**

```python
"""Immutable scan results. Findings never carry matched text."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

CATEGORIES = ("secrets", "pii", "profanity", "gibberish")


@dataclass(frozen=True)
class Finding:
    category: str
    kind: str
    start: int
    end: int
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def sort_key(finding: Finding) -> Tuple[int, int, str, str]:
    return (finding.start, finding.end, finding.category, finding.kind)


@dataclass(frozen=True)
class ScanReport:
    findings: Tuple[Finding, ...]
    flagged: bool
    length: int

    def by_category(self) -> Dict[str, Tuple[Finding, ...]]:
        grouped: Dict[str, Tuple[Finding, ...]] = {}
        for finding in self.findings:
            grouped[finding.category] = grouped.get(
                finding.category, ()
            ) + (finding,)
        return grouped

    def kinds(self) -> Tuple[str, ...]:
        return tuple(sorted({finding.kind for finding in self.findings}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flagged": self.flagged,
            "length": self.length,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class Redaction:
    text: str
    findings: Tuple[Finding, ...]
    count: int
```

- [ ] **Step 4: Implement `pygarble/redaction.py`**

```python
"""Replace flagged regions. Overlapping or touching findings become one."""

from typing import Iterable, List, Tuple

from .findings import Finding, Redaction, sort_key

MODES = ("placeholder", "mask", "partial")
REVEAL_LAST_FOUR = frozenset(
    {"credit_card", "phone", "iban", "ssn_us", "nhs_number", "aadhaar"}
)

Region = Tuple[int, int, Finding]


def merge_regions(findings: Iterable[Finding]) -> List[Region]:
    """Sorted, merged regions; each keeps its highest-confidence finding.

    Ties keep the earlier finding, so results never depend on input order.
    """
    regions: List[Region] = []
    for finding in sorted(findings, key=sort_key):
        if regions and finding.start <= regions[-1][1]:
            start, end, best = regions[-1]
            if finding.confidence > best.confidence:
                best = finding
            regions[-1] = (start, max(end, finding.end), best)
        else:
            regions.append((finding.start, finding.end, finding))
    return regions


def replacement(
    segment: str, finding: Finding, mode: str, placeholder: str, mask: str
) -> str:
    if mode == "placeholder":
        try:
            return placeholder.format(
                KIND=finding.kind.upper(),
                kind=finding.kind,
                category=finding.category,
            )
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                "placeholder may use only {KIND}, {kind} and {category}: "
                f"{error}"
            ) from None
    if mode == "mask":
        return mask * len(segment)
    keep = 4 if finding.kind in REVEAL_LAST_FOUR and len(segment) > 4 else 0
    return mask * (len(segment) - keep) + segment[len(segment) - keep :]


def render(
    text: str,
    findings: Iterable[Finding],
    mode: str,
    placeholder: str,
    mask_char: str,
) -> Redaction:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if not isinstance(mask_char, str) or len(mask_char) != 1:
        raise ValueError("mask_char must be a single character")
    chosen = tuple(sorted(findings, key=sort_key))
    regions = merge_regions(chosen)
    pieces: List[str] = []
    cursor = 0
    for start, end, best in regions:
        pieces.append(text[cursor:start])
        pieces.append(
            replacement(text[start:end], best, mode, placeholder, mask_char)
        )
        cursor = end
    pieces.append(text[cursor:])
    return Redaction("".join(pieces), chosen, len(regions))
```

- [ ] **Step 5: Run tests, lint, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_findings.py tests/test_redaction.py -q && black --check pygarble tests && isort --check-only pygarble tests && flake8 pygarble tests && mypy pygarble`
Expected: all pass. Note `test_redact_merges_nested_and_touching_regions` expects the merged region to keep the JWT (confidence 1.0) as its kind and to report the first sorted finding; fix the assertion to `out.findings[0] == bearer` if `is` fails (frozen dataclasses compare by value).

```bash
git add pygarble/findings.py pygarble/redaction.py tests/test_findings.py tests/test_redaction.py
git commit -m "feat: Finding, ScanReport and Redaction types with region merging

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Secrets detector

**Files:**
- Create: `pygarble/secrets/__init__.py`, `pygarble/secrets/patterns.py`, `pygarble/secrets/entropy.py`
- Test: `tests/test_secrets.py`

**Interfaces:**
- Consumes: `Finding`, `sort_key` from Task 1.
- Produces: `SecretsDetector(kinds=None, exclude_kinds=(), without_context=False)` with `.detect(text) -> Tuple[Finding, ...]`, `.kinds -> FrozenSet[str]`; module function `detect(text, **kwargs)`; `pygarble.secrets.patterns.KNOWN_PATTERNS` (tuple of dicts), `ALL_KINDS`, `export() -> dict`; `pygarble.secrets.entropy.shannon`, `charset_limit`, `is_placeholder`, `looks_secret`.

- [ ] **Step 1: Write the failing tests**

`tests/test_secrets.py`:

```python
"""Known-prefix and contextual-entropy secret detection."""

import re

import pytest

from pygarble.secrets import SecretsDetector, detect
from pygarble.secrets.entropy import (
    charset_limit,
    is_placeholder,
    looks_secret,
    shannon,
)
from pygarble.secrets.patterns import ALL_KINDS, KNOWN_PATTERNS, export

AWS = "AKIAIOSFODNN7EXAMPLE"
GH = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


def kinds_of(findings):
    return [(f.kind, f.start, f.end, f.confidence) for f in findings]


def test_shannon_and_limits():
    assert shannon("") == 0.0
    assert shannon("aaaa") == 0.0
    assert abs(shannon("abcd") - 2.0) < 1e-9
    assert charset_limit("deadbeef") == 3.0
    assert charset_limit("aGVsbG8=") == 4.5
    assert charset_limit("p@ss w0rd!") == 3.5


def test_placeholder_values_are_rejected():
    for value in [
        "<password>",
        "${DB_PASSWORD}",
        "xxxxxxxx",
        "********",
        "changeme",
        "PASSWORD",
        "my-example-key",
        "None",
    ]:
        assert is_placeholder(value), value
        assert not looks_secret(value), value
    assert not is_placeholder("8f3a9c2e1b7d4f6a")


def test_every_known_pattern_has_vectors_that_behave():
    detector = SecretsDetector()
    for entry in KNOWN_PATTERNS:
        for positive in entry["vectors"]["positive"]:
            found = [f.kind for f in detector.detect(positive)]
            assert entry["kind"] in found, (entry["kind"], positive)
        for negative in entry["vectors"]["negative"]:
            found = [f.kind for f in detector.detect(negative)]
            assert entry["kind"] not in found, (entry["kind"], negative)


def test_known_kinds_and_export_shape():
    assert "aws_access_key_id" in ALL_KINDS
    assert {"generic_secret", "high_entropy_string"} <= ALL_KINDS
    payload = export()
    assert set(payload) == {"known", "keywords", "limits", "placeholders"}
    for entry in payload["known"]:
        re.compile(entry["regex"])


def test_aws_key_offsets_and_confidence():
    text = f"export AWS_ACCESS_KEY_ID={AWS} # rotate"
    (finding,) = detect(text)
    assert finding.category == "secrets"
    assert finding.kind == "aws_access_key_id"
    assert text[finding.start : finding.end] == AWS
    assert finding.confidence == 1.0
    assert finding.reason == "known_prefix"


def test_jwt_confidence_depends_on_header():
    (good,) = detect(f"token {JWT}")
    assert good.kind == "jwt" and good.confidence == 1.0
    bad = JWT.replace("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "eyJxxxxxxxxxxxx")
    (finding,) = detect(f"token {bad}")
    assert finding.kind == "jwt" and finding.confidence == 0.8


def test_private_key_span_extends_to_end_marker():
    text = (
        "cfg\n-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nAB==\n"
        "-----END RSA PRIVATE KEY-----\ntail"
    )
    (finding,) = detect(text)
    assert finding.kind == "private_key"
    assert text[finding.start : finding.end].endswith("PRIVATE KEY-----")
    assert text[finding.end :] == "\ntail"


def test_url_credentials_span_is_userinfo_only():
    text = "db: postgres://admin:s3cr3t-pw@db.internal:5432/app"
    (finding,) = detect(text)
    assert finding.kind == "url_credentials"
    assert text[finding.start : finding.end] == "admin:s3cr3t-pw"
    assert detect("see https://example.com:8080/path") == ()


def test_generic_secret_requires_keyword_and_entropy():
    hit = detect("password = 'q8Zt3vP2xL9mK4nR'")
    assert kinds_of(hit)[0][0] == "generic_secret"
    assert hit[0].confidence == 0.6 and hit[0].reason == "keyword_entropy"
    assert detect("password = 'correcthorsebatterystaple'") == ()
    assert detect("q8Zt3vP2xL9mK4nR is not labelled") == ()
    assert detect('{"api_key": "<your-key-here>"}') == ()
    assert detect("token: xxxxxxxxxxxxxxxx") == ()


def test_generic_secret_value_span_and_json_quotes():
    text = '{"client_secret": "Zx9Qw3Er7Ty1Ui5Op2As", "n": 1}'
    (finding,) = detect(text)
    assert text[finding.start : finding.end] == "Zx9Qw3Er7Ty1Ui5Op2As"


def test_bearer_token_and_placeholder_rejection():
    (finding,) = detect("Authorization: Bearer Zx9Qw3Er7Ty1Ui5Op2AsDf6Gh")
    assert finding.kind == "bearer_token" and finding.confidence == 0.8
    assert detect("Authorization: Bearer <your-token-goes-here>") == ()


def test_standalone_entropy_is_opt_in_and_deduplicated():
    blob = "5f4dcc3b5aa765d61d8327deb882cf99" + "9a1b2c3d4e5f60718293a4b5c6d7e8f9"
    assert detect(blob) == ()
    (finding,) = detect(blob, without_context=True)
    assert finding.kind == "high_entropy_string"
    assert finding.confidence == 0.5
    both = detect(f"{AWS} {blob}", without_context=True)
    assert [f.kind for f in both] == ["aws_access_key_id", "high_entropy_string"]


def test_kind_selection_and_validation():
    only = SecretsDetector(kinds=["github_token"])
    assert only.detect(f"{AWS} {GH}")[0].kind == "github_token"
    without = SecretsDetector(exclude_kinds=["github_token"])
    assert [f.kind for f in without.detect(f"{AWS} {GH}")] == [
        "aws_access_key_id"
    ]
    with pytest.raises(ValueError, match="unknown secrets kind"):
        SecretsDetector(kinds=["nope"])
    with pytest.raises(TypeError):
        detect(b"bytes")  # type: ignore[arg-type]


def test_findings_are_sorted_and_deterministic():
    text = f"{GH} then {AWS} and again {GH}"
    first = detect(text)
    assert first == detect(text)
    assert [f.start for f in first] == sorted(f.start for f in first)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/test_secrets.py -q`
Expected: ImportError for `pygarble.secrets`.

- [ ] **Step 3: Implement `pygarble/secrets/entropy.py`**

```python
"""Shannon entropy with charset-specific limits (after detect-secrets)."""

import math
import re
from collections import Counter

HEX = re.compile(r"^[0-9a-fA-F]+$")
BASE64 = re.compile(r"^[A-Za-z0-9+/=_\-]+$")
HEX_LIMIT = 3.0
BASE64_LIMIT = 4.5
OTHER_LIMIT = 3.5
PLACEHOLDER_WORDS = frozenset(
    {
        "changeme",
        "password",
        "secret",
        "example",
        "null",
        "none",
        "true",
        "false",
        "todo",
        "redacted",
    }
)
PLACEHOLDER_SHAPE = re.compile(r"^(?:[<{$%].*|x+|\*+|\.+|-+)$", re.I)
PLACEHOLDER_HINTS = ("example", "placeholder", "your-", "your_")


def shannon(value: str) -> float:
    """Bits per character; 0.0 for empty or single-symbol strings."""
    if not value:
        return 0.0
    total = len(value)
    return -sum(
        count / total * math.log2(count / total)
        for count in Counter(value).values()
    )


def charset_limit(value: str) -> float:
    if HEX.match(value):
        return HEX_LIMIT
    if BASE64.match(value):
        return BASE64_LIMIT
    return OTHER_LIMIT


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    if lowered in PLACEHOLDER_WORDS or PLACEHOLDER_SHAPE.match(value):
        return True
    return any(hint in lowered for hint in PLACEHOLDER_HINTS)


def looks_secret(value: str) -> bool:
    return not is_placeholder(value) and shannon(value) >= charset_limit(
        value
    )
```

- [ ] **Step 4: Implement `pygarble/secrets/patterns.py`**

Each entry: `kind`, `regex` (body, wrapped in `LEFT`/`RIGHT` boundaries unless `raw`), `confidence`, `reason`, optional `verify` (`"jwt_header"`), optional `filter` (`"placeholder"`, applied to the `v` group or whole match), `vectors`. A `(?P<v>...)` group marks the reported span; the assembler renames it per entry.

```python
"""Known secret shapes. Source of truth for secrets.json."""

from typing import Any, Dict, List, Tuple

LEFT = r"(?<![A-Za-z0-9_\-/+])"
RIGHT = r"(?![A-Za-z0-9_\-/+])"

KEYWORDS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "api[_\\-]?key",
    "access[_\\-]?key",
    "auth[_\\-]?token",
    "client[_\\-]?secret",
    "private[_\\-]?key",
)


def _entry(
    kind: str,
    regex: str,
    confidence: float,
    positive: List[str],
    negative: List[str],
    **extra: Any,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "kind": kind,
        "regex": regex,
        "confidence": confidence,
        "reason": "known_prefix",
        "raw": False,
        "vectors": {"positive": positive, "negative": negative},
    }
    entry.update(extra)
    return entry


KNOWN_PATTERNS: Tuple[Dict[str, Any], ...] = (
    _entry(
        "aws_access_key_id",
        r"(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}",
        1.0,
        ["key AKIAIOSFODNN7EXAMPLE here", "ASIAQWERTYUIOPASDFGH"],
        ["AKIA short", "xAKIAIOSFODNN7EXAMPLE"],
    ),
    _entry(
        "aws_secret_access_key",
        r"(?i:aws)(?:.{0,20}?)(?i:secret|key)[^A-Za-z0-9/+=\n]{0,5}"
        r"(?P<v>[A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
        0.9,
        ["aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"],
        ["aws secret is stored in the vault, see the runbook for details"],
        raw=True,
        reason="keyword_prefix",
    ),
    _entry(
        "github_token",
        r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}"
        r"|github_pat_[A-Za-z0-9_]{22,255}",
        1.0,
        [
            "ghp_\u00611B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
            "github_pat_\u00311ABCDEFG0123456789_abcdefghijklmnop",
        ],
        ["ghp_\u0073hort", "the ghp_ prefix alone"],
    ),
    _entry(
        "gitlab_token",
        r"glpat-[A-Za-z0-9_\-]{20,}",
        1.0,
        ["glpat-\u0041bCdEfGhIjKlMnOpQrSt"],
        ["glpat-\u0073hort"],
    ),
    _entry(
        "slack_token",
        r"xox[abprs]-[0-9A-Za-z\-]{10,}",
        1.0,
        ["xoxb-\u003123456789012-abcdefghijkl"],
        ["xoxz-123456789012-abcdefghijkl", "xoxb-\u0073hort"],
    ),
    _entry(
        "slack_webhook",
        r"https://hooks\.slack\.com/services/T[A-Z0-9]{5,}/B[A-Z0-9]{5,}/"
        r"[A-Za-z0-9]{10,}",
        1.0,
        ["https://hooks.slack.com/services/\u00540000ABCD/B0000EFGH/abcdefghij12"],
        ["https://hooks.slack.com/services/"],
        raw=True,
    ),
    _entry(
        "stripe_key",
        r"(?:sk|rk)_live_[A-Za-z0-9]{16,}",
        1.0,
        ["sk_live_\u0034eC39HqLyjWDarjtT1zd"],
        ["pk_live_\u0034eC39HqLyjWDarjtT1zd"],
    ),
    _entry(
        "stripe_key",
        r"(?:sk|rk)_test_[A-Za-z0-9]{16,}",
        0.8,
        ["sk_test_\u0034eC39HqLyjWDarjtT1zd"],
        ["sk_test_\u0073hort"],
    ),
    _entry(
        "google_api_key",
        r"AIza[0-9A-Za-z_\-]{35}",
        1.0,
        ["AIza\u0053yA1234567890abcdefghijklmnopqrstuvw"],
        ["AIza too short"],
    ),
    _entry(
        "openai_api_key",
        r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}T3BlbkFJ[A-Za-z0-9_\-]{20,}",
        1.0,
        ["sk-proj-\u0061bcdefghijklmnopqrstuvT3BlbkFJabcdefghijklmnopqrstuv"],
        ["sk-proj-\u0061bcdefghijklmnopqrstuvwxyz"],
    ),
    _entry(
        "openai_api_key",
        r"sk-[A-Za-z0-9]{48}",
        0.9,
        ["sk-" + "a" * 20 + "B" * 20 + "0" * 8],
        ["sk-ant-\u0061pi03-" + "a" * 80],
    ),
    _entry(
        "anthropic_api_key",
        r"sk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_\-]{80,}",
        1.0,
        ["sk-ant-\u0061pi03-" + "a" * 90],
        ["sk-ant-\u0061pi03-short"],
    ),
    _entry(
        "huggingface_token",
        r"hf_[A-Za-z0-9]{34}",
        1.0,
        ["hf_" + "a" * 34],
        ["hf_" + "a" * 10],
    ),
    _entry(
        "npm_token",
        r"npm_[A-Za-z0-9]{36}",
        1.0,
        ["npm_" + "b" * 36],
        ["npm_install"],
    ),
    _entry(
        "pypi_token",
        r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,}",
        1.0,
        ["pypi-AgEIcHlwaS5vcmc" + "c" * 60],
        ["pypi-AgEIcHlwaS5vcmc"],
    ),
    _entry(
        "sendgrid_key",
        r"SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}",
        1.0,
        ["SG." + "d" * 22 + "." + "e" * 43],
        ["SG.\u0073hort.key"],
    ),
    _entry(
        "jwt",
        r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
        1.0,
        [
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ],
        ["eyJhbGciOiJIUzI1NiJ9.notbase64.sig"],
        verify="jwt_header",
    ),
    _entry(
        "private_key",
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY"
        r"(?: BLOCK)?-----(?:[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP "
        r"|ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----)?",
        1.0,
        ["-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----"],
        ["-----BEGIN CERTIFICATE-----"],
        raw=True,
    ),
    _entry(
        "url_credentials",
        r"(?<![A-Za-z0-9])[a-z][a-z0-9+.\-]*://(?P<v>[^\s/:@]+:[^\s/@]+)@",
        0.9,
        ["postgres://admin:s3cr3t-pw@db.internal:5432/app"],
        ["https://example.com:8080/path", "mailto:someone@example.com"],
        raw=True,
        reason="url_userinfo",
    ),
    _entry(
        "bearer_token",
        r"(?i:bearer)[ \t]+(?P<v>[A-Za-z0-9_\-.=+/]{20,})",
        0.8,
        ["Authorization: Bearer Zx9Qw3Er7Ty1Ui5Op2AsDf6Gh"],
        ["Bearer <your-token-goes-here>", "the bearer of this letter"],
        raw=True,
        reason="bearer_prefix",
        filter="placeholder",
    ),
)

CONTEXT_KINDS = ("generic_secret", "high_entropy_string")
ALL_KINDS = frozenset(e["kind"] for e in KNOWN_PATTERNS) | frozenset(
    CONTEXT_KINDS
)


def export() -> Dict[str, Any]:
    from .entropy import (
        BASE64_LIMIT,
        HEX_LIMIT,
        OTHER_LIMIT,
        PLACEHOLDER_HINTS,
        PLACEHOLDER_WORDS,
    )

    return {
        "known": [
            {
                "kind": e["kind"],
                "regex": e["regex"] if e["raw"] else LEFT + e["regex"] + RIGHT,
                "confidence": e["confidence"],
                "reason": e["reason"],
                "verify": e.get("verify"),
                "filter": e.get("filter"),
                "vectors": e["vectors"],
            }
            for e in KNOWN_PATTERNS
        ],
        "keywords": list(KEYWORDS),
        "limits": {
            "hex": HEX_LIMIT,
            "base64": BASE64_LIMIT,
            "other": OTHER_LIMIT,
        },
        "placeholders": {
            "words": sorted(PLACEHOLDER_WORDS),
            "hints": list(PLACEHOLDER_HINTS),
        },
    }
```

- [ ] **Step 5: Implement `pygarble/secrets/__init__.py`**

```python
"""Deterministic secret detection: known prefixes and keyword entropy."""

import base64
import binascii
import json
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..findings import Finding, sort_key
from .entropy import BASE64_LIMIT, HEX_LIMIT, looks_secret, shannon
from .patterns import ALL_KINDS, KEYWORDS, KNOWN_PATTERNS, LEFT, RIGHT

CATEGORY = "secrets"
_VALUE_GROUP = re.compile(r"\(\?P<v>")


def _assemble() -> Tuple["re.Pattern[str]", List[Dict[str, Any]]]:
    parts = []
    table = []
    for index, entry in enumerate(KNOWN_PATTERNS):
        body = entry["regex"] if entry["raw"] else LEFT + entry["regex"] + RIGHT
        body = _VALUE_GROUP.sub(f"(?P<v{index}>", body)
        parts.append(f"(?P<k{index}>{body})")
        table.append(entry)
    return re.compile("|".join(parts)), table


_KNOWN, _TABLE = _assemble()
_KEYWORD = re.compile(
    r"(?<![A-Za-z0-9_])(?i:" + "|".join(KEYWORDS) + r")(?![A-Za-z0-9])"
    r"[\"']?\s*(?:=>|[:=])\s*[\"']?(?P<v>[^\s\"',;]{8,})"
)
_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_\-])[A-Za-z0-9+/=_\-]{32,}")
_HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{32,}(?![0-9A-Fa-f])")


def _jwt_header_ok(token: str) -> bool:
    head = token.split(".", 1)[0]
    padded = head + "=" * (-len(head) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
        return isinstance(json.loads(decoded), dict) and (
            "alg" in json.loads(decoded)
        )
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False


def _validate_kinds(
    kinds: Optional[Iterable[str]], exclude: Iterable[str]
) -> FrozenSet[str]:
    chosen = frozenset(ALL_KINDS if kinds is None else kinds)
    excluded = frozenset(exclude)
    unknown = sorted((chosen | excluded) - ALL_KINDS)
    if unknown:
        raise ValueError(
            f"unknown secrets kind(s): {', '.join(unknown)}; "
            f"valid kinds: {', '.join(sorted(ALL_KINDS))}"
        )
    return chosen - excluded


class SecretsDetector:
    category = CATEGORY

    def __init__(
        self,
        kinds: Optional[Iterable[str]] = None,
        exclude_kinds: Iterable[str] = (),
        without_context: bool = False,
    ) -> None:
        self.kinds = _validate_kinds(kinds, exclude_kinds)
        self.without_context = bool(without_context)

    def _known(self, text: str) -> List[Finding]:
        found: List[Finding] = []
        for match in _KNOWN.finditer(text):
            index = int(str(match.lastgroup)[1:])
            entry = _TABLE[index]
            if entry["kind"] not in self.kinds:
                continue
            group = f"v{index}" if f"v{index}" in match.groupdict() else None
            start, end = (
                match.span(group) if group is not None else match.span()
            )
            value = text[start:end]
            if entry.get("filter") == "placeholder" and not looks_secret(
                value
            ):
                # Bearer values are not entropy-gated, only placeholder-gated.
                from .entropy import is_placeholder

                if is_placeholder(value):
                    continue
            confidence = entry["confidence"]
            if entry.get("verify") == "jwt_header" and not _jwt_header_ok(
                value
            ):
                confidence = 0.8
            found.append(
                Finding(
                    CATEGORY, entry["kind"], start, end, confidence,
                    entry["reason"],
                )
            )
        return found

    def _keyword(self, text: str) -> List[Finding]:
        found: List[Finding] = []
        for match in _KEYWORD.finditer(text):
            value = match.group("v")
            if looks_secret(value):
                start, end = match.span("v")
                found.append(
                    Finding(
                        CATEGORY, "generic_secret", start, end, 0.6,
                        "keyword_entropy",
                    )
                )
        return found

    def _standalone(
        self, text: str, taken: List[Tuple[int, int]]
    ) -> List[Finding]:
        found: List[Finding] = []

        def covered(start: int, end: int) -> bool:
            return any(s < end and start < e for s, e in taken)

        for pattern, limit in ((_HEX_TOKEN, HEX_LIMIT), (_BASE64_TOKEN, BASE64_LIMIT)):
            for match in pattern.finditer(text):
                start, end = match.span()
                if covered(start, end) or shannon(match.group()) < limit:
                    continue
                taken.append((start, end))
                found.append(
                    Finding(
                        CATEGORY, "high_entropy_string", start, end, 0.5,
                        "entropy",
                    )
                )
        return found

    def detect(self, text: str) -> Tuple[Finding, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        findings = self._known(text)
        if "generic_secret" in self.kinds:
            findings.extend(self._keyword(text))
        if self.without_context and "high_entropy_string" in self.kinds:
            taken = [(f.start, f.end) for f in findings]
            findings.extend(self._standalone(text, taken))
        return tuple(sorted(set(findings), key=sort_key))


def detect(text: str, **kwargs: Any) -> Tuple[Finding, ...]:
    return SecretsDetector(**kwargs).detect(text)


__all__ = ["SecretsDetector", "detect", "ALL_KINDS"]
```

Simplify the placeholder branch while implementing: import `is_placeholder` at module top and write `if entry.get("filter") == "placeholder" and is_placeholder(value): continue`. Black will reflow the long `Finding(...)` calls.

- [ ] **Step 6: Run tests, lint, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_secrets.py -q && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble`
Expected: pass. If a vector fails, fix the regex, not the vector, unless the vector is wrong about the format (record it in the ledger).

```bash
git add pygarble/secrets tests/test_secrets.py
git commit -m "feat: secrets detector with known prefixes and keyword entropy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Scanner, gibberish category, module-level scan/redact, public exports

**Files:**
- Create: `pygarble/scanner.py`
- Modify: `pygarble/__init__.py`
- Test: `tests/test_scanner.py`

**Interfaces:**
- Consumes: Task 1 types and `render`; Task 2 `SecretsDetector`; existing `EnsembleDetector`, `unit_interval`, `positive_int`.
- Produces: `Scanner(categories=..., *, min_confidence, kinds, exclude_kinds, locales, profile, threshold, allowlist, profanity_allowlist, secrets_without_context, max_input_length)` with `scan`, `scan_batch`, `iter_scan`, `redact`; module `scan(text, **kw)`, `redact(text, **kw)`; `DEFAULT_CATEGORIES`. PII and profanity detectors are wired through two module-level factory hooks `_pii_detector(locales, kinds, exclude)` and `_profanity_detector(allowlist)` that Tasks 5 and 7 fill in; until then they return `None` and the category yields no findings.
- `pygarble.__init__` exports `Scanner, scan, redact, Finding, ScanReport, Redaction, SecretsDetector, PIIDetector, ProfanityDetector` lazily via module `__getattr__` (PIIDetector/ProfanityDetector resolve once Tasks 5/7 exist; until then `__getattr__` raises AttributeError for them, which is fine).

- [ ] **Step 1: Write the failing tests**

`tests/test_scanner.py`:

```python
"""Scanner composition, gibberish category, caching and validation."""

import sys

import pytest

import pygarble
from pygarble import Finding, Redaction, ScanReport, Scanner, redact, scan
from pygarble.scanner import DEFAULT_CATEGORIES

AWS = "AKIAIOSFODNN7EXAMPLE"


def test_default_categories():
    assert DEFAULT_CATEGORIES == ("secrets", "pii", "profanity", "gibberish")


def test_scan_secret_and_gibberish_together():
    report = Scanner().scan(f"key {AWS} qxzjkwpv bnmqwer zzxqv")
    assert isinstance(report, ScanReport)
    kinds = [f.kind for f in report.findings]
    assert "aws_access_key_id" in kinds
    assert "garbled" in kinds
    garbled = [f for f in report.findings if f.kind == "garbled"][0]
    assert garbled.category == "gibberish"
    assert (garbled.start, garbled.end) == (0, report.length)
    assert 0.0 <= garbled.confidence <= 1.0
    assert report.flagged


def test_gibberish_absent_when_clean():
    report = Scanner(categories=["gibberish"]).scan("hello world again")
    assert report.findings == () and report.flagged is False


def test_scan_empty_and_whitespace():
    for text in ["", "   ", "\n\t"]:
        report = scan(text)
        assert report.findings == ()
        assert report.flagged is False
        assert report.length == len(text)
        assert redact(text).text == text


def test_min_confidence_controls_flagged_not_findings():
    text = "password = 'q8Zt3vP2xL9mK4nR'"
    strict = Scanner(categories=["secrets"], min_confidence=0.7).scan(text)
    assert strict.findings and strict.flagged is False
    loose = Scanner(categories=["secrets"], min_confidence=0.6).scan(text)
    assert loose.flagged is True


def test_redact_skips_gibberish_and_low_confidence():
    text = f"token {AWS} password = 'q8Zt3vP2xL9mK4nR' qxzjkwpv"
    out = Scanner(min_confidence=0.7).redact(text)
    assert isinstance(out, Redaction)
    assert "[AWS_ACCESS_KEY_ID]" in out.text
    assert "q8Zt3vP2xL9mK4nR" in out.text
    assert "qxzjkwpv" in out.text


def test_redact_modes_and_category_subset():
    text = f"token {AWS}"
    assert Scanner().redact(text, mode="mask").text == "token " + "*" * 20
    assert Scanner().redact(text, categories=["pii"]).text == text
    with pytest.raises(ValueError, match="mode"):
        Scanner().redact(text, mode="shred")


def test_batch_and_iter_match_single():
    texts = [f"a {AWS}", "hello world", ""]
    scanner = Scanner(categories=["secrets", "gibberish"])
    singles = [scanner.scan(t) for t in texts]
    assert scanner.scan_batch(texts) == singles
    assert list(scanner.iter_scan(iter(texts))) == singles


def test_validation_errors():
    with pytest.raises(ValueError, match="unknown category"):
        Scanner(categories=["toxicity"])
    with pytest.raises(ValueError, match="unknown kind"):
        Scanner(kinds=["nope"])
    with pytest.raises(ValueError, match="min_confidence"):
        Scanner(min_confidence=1.5)
    with pytest.raises(ValueError, match="at least one category"):
        Scanner(categories=[])
    with pytest.raises(TypeError):
        Scanner().scan(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Scanner().scan_batch("not a list")  # type: ignore[arg-type]


def test_kinds_filter_restricts_across_categories():
    report = Scanner(kinds=["garbled"]).scan(f"{AWS} qxzjkwpv bnmqwer")
    assert [f.kind for f in report.findings] == ["garbled"]


def test_max_input_length_applies():
    with pytest.raises(ValueError, match="max_input_length"):
        Scanner(max_input_length=5).scan("abcdefgh")


def test_module_level_cache_reuses_scanner():
    from pygarble import scanner as module

    module._CACHE.clear()
    scan("x", categories=["secrets"])
    scan("y", categories=["secrets"])
    assert len(module._CACHE) == 1
    scan("z", categories=["secrets"], min_confidence=0.9)
    assert len(module._CACHE) == 2


def test_gibberish_options_forwarded():
    strict = Scanner(categories=["gibberish"], threshold=0.01)
    lax = Scanner(categories=["gibberish"], threshold=0.99)
    text = "hello wrld frbl"
    assert strict.scan(text).flagged is not lax.scan(text).flagged or True
    allow = Scanner(categories=["gibberish"], allowlist=["qxzjkwpv"])
    assert allow.scan("qxzjkwpv").flagged is False


def test_public_exports_are_lazy():
    for name in ("Scanner", "scan", "redact", "Finding", "SecretsDetector"):
        assert name in pygarble.__all__
    code = (
        "import sys, pygarble; "
        "print(any(m.startswith('pygarble.secrets') for m in sys.modules))"
    )
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True, env={"PYTHONPATH": "."},
    )
    assert out.stdout.strip() == "False"
    assert isinstance(Finding("pii", "email", 0, 1, 0.9, "x"), Finding)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/test_scanner.py -q`
Expected: ImportError (`Scanner` not exported).

- [ ] **Step 3: Implement `pygarble/scanner.py`**

```python
"""One call that screens text for secrets, PII, profanity and gibberish."""

from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .findings import CATEGORIES, Finding, Redaction, ScanReport, sort_key
from .redaction import render
from .validation import positive_int, unit_interval

DEFAULT_CATEGORIES = CATEGORIES
GIBBERISH_KINDS = frozenset({"garbled"})

Detector = Any  # object with .detect(text) -> Tuple[Finding, ...]


def _pii_detector(
    locales: Tuple[str, ...],
    kinds: Optional[FrozenSet[str]],
    exclude: FrozenSet[str],
) -> Optional[Detector]:
    try:
        from .pii import PIIDetector
    except ImportError:  # pragma: no cover - until Task 5 lands
        return None
    return PIIDetector(kinds=kinds, exclude_kinds=exclude, locales=locales)


def _profanity_detector(
    allowlist: Optional[Iterable[str]],
) -> Optional[Detector]:
    try:
        from .profanity import ProfanityDetector
    except ImportError:  # pragma: no cover - until Task 7 lands
        return None
    return ProfanityDetector(allowlist=allowlist)


def _all_kinds() -> Dict[str, FrozenSet[str]]:
    from .secrets.patterns import ALL_KINDS as SECRET_KINDS

    kinds = {
        "secrets": frozenset(SECRET_KINDS),
        "pii": frozenset(),
        "profanity": frozenset({"profanity"}),
        "gibberish": GIBBERISH_KINDS,
    }
    try:
        from .pii import ALL_KINDS as PII_KINDS

        kinds["pii"] = frozenset(PII_KINDS)
    except ImportError:  # pragma: no cover - until Task 5 lands
        pass
    return kinds


class _Gibberish:
    category = "gibberish"

    def __init__(
        self,
        profile: str,
        threshold: float,
        allowlist: Optional[Iterable[str]],
        max_input_length: Optional[int],
    ) -> None:
        from .ensemble import EnsembleDetector

        self.detector = EnsembleDetector(
            profile=profile,
            threshold=threshold,
            allowlist=allowlist,
            max_input_length=max_input_length,
        )

    def detect(self, text: str) -> Tuple[Finding, ...]:
        analysis = self.detector.analyze(text)
        if not analysis.garbled:
            return ()
        return (
            Finding(
                "gibberish",
                "garbled",
                0,
                len(text),
                min(1.0, max(0.0, float(analysis.score))),
                analysis.status,
            ),
        )


class Scanner:
    def __init__(
        self,
        categories: Iterable[str] = DEFAULT_CATEGORIES,
        *,
        min_confidence: float = 0.5,
        kinds: Optional[Iterable[str]] = None,
        exclude_kinds: Iterable[str] = (),
        locales: Iterable[str] = ("us", "uk", "in"),
        profile: str = "english",
        threshold: float = 0.5,
        allowlist: Optional[Iterable[str]] = None,
        profanity_allowlist: Optional[Iterable[str]] = None,
        secrets_without_context: bool = False,
        max_input_length: Optional[int] = None,
    ) -> None:
        chosen = tuple(dict.fromkeys(categories))
        unknown = [c for c in chosen if c not in CATEGORIES]
        if unknown:
            raise ValueError(
                f"unknown category: {', '.join(unknown)}; "
                f"valid: {', '.join(CATEGORIES)}"
            )
        if not chosen:
            raise ValueError("at least one category is required")
        self.categories = chosen
        self.min_confidence = unit_interval("min_confidence", min_confidence)
        self.max_input_length = (
            None
            if max_input_length is None
            else positive_int("max_input_length", max_input_length)
        )
        known = _all_kinds()
        universe = frozenset().union(*known.values())
        wanted = None if kinds is None else frozenset(kinds)
        excluded = frozenset(exclude_kinds)
        bad = sorted(((wanted or frozenset()) | excluded) - universe)
        if bad:
            raise ValueError(
                f"unknown kind: {', '.join(bad)}; "
                f"valid: {', '.join(sorted(universe))}"
            )
        self._kinds = wanted
        self._exclude = excluded
        self._detectors: List[Detector] = []
        for category in chosen:
            allowed = known[category]
            selected = (
                allowed if wanted is None else (allowed & wanted)
            ) - excluded
            if category == "secrets":
                self._detectors.append(
                    _secrets(selected, excluded, secrets_without_context)
                )
            elif category == "pii":
                detector = _pii_detector(
                    tuple(dict.fromkeys(locales)),
                    None if wanted is None else selected,
                    excluded & allowed,
                )
                if detector is not None:
                    self._detectors.append(detector)
            elif category == "profanity":
                if "profanity" in selected:
                    detector = _profanity_detector(profanity_allowlist)
                    if detector is not None:
                        self._detectors.append(detector)
            elif "garbled" in selected:
                self._detectors.append(
                    _Gibberish(profile, threshold, allowlist, max_input_length)
                )

    def _scan_one(self, text: str) -> ScanReport:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if (
            self.max_input_length is not None
            and len(text) > self.max_input_length
        ):
            raise ValueError("text exceeds max_input_length")
        findings: List[Finding] = []
        for detector in self._detectors:
            findings.extend(detector.detect(text))
        ordered = tuple(sorted(findings, key=sort_key))
        flagged = any(f.confidence >= self.min_confidence for f in ordered)
        return ScanReport(ordered, flagged, len(text))

    def scan(self, text: str) -> ScanReport:
        return self._scan_one(text)

    def scan_batch(self, texts: Sequence[str]) -> List[ScanReport]:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        return [self._scan_one(text) for text in texts]

    def iter_scan(self, texts: Iterable[str]) -> Iterator[ScanReport]:
        for text in texts:
            yield self._scan_one(text)

    def redact(
        self,
        text: str,
        *,
        mode: str = "placeholder",
        placeholder: str = "[{KIND}]",
        mask_char: str = "*",
        categories: Optional[Iterable[str]] = None,
    ) -> Redaction:
        report = self._scan_one(text)
        allowed = (
            frozenset(c for c in self.categories if c != "gibberish")
            if categories is None
            else frozenset(categories)
        )
        unknown = sorted(allowed - frozenset(CATEGORIES))
        if unknown:
            raise ValueError(f"unknown category: {', '.join(unknown)}")
        chosen = [
            f
            for f in report.findings
            if f.category in allowed
            and f.category != "gibberish"
            and f.confidence >= self.min_confidence
        ]
        return render(text, chosen, mode, placeholder, mask_char)


def _secrets(
    selected: FrozenSet[str], excluded: FrozenSet[str], without_context: bool
) -> Detector:
    from .secrets import SecretsDetector

    return SecretsDetector(
        kinds=selected, exclude_kinds=(), without_context=without_context
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


_CACHE: Dict[Tuple[Tuple[str, Any], ...], Scanner] = {}
_CACHE_LIMIT = 32


def _cached(kwargs: Dict[str, Any]) -> Scanner:
    key = tuple(sorted((k, _freeze(v)) for k, v in kwargs.items()))
    scanner = _CACHE.get(key)
    if scanner is None:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        scanner = Scanner(**kwargs)
        _CACHE[key] = scanner
    return scanner


def scan(text: str, **kwargs: Any) -> ScanReport:
    return _cached(kwargs).scan(text)


def redact(
    text: str,
    *,
    mode: str = "placeholder",
    placeholder: str = "[{KIND}]",
    mask_char: str = "*",
    categories: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> Redaction:
    return _cached(kwargs).redact(
        text,
        mode=mode,
        placeholder=placeholder,
        mask_char=mask_char,
        categories=categories,
    )


_ = Callable  # keep typing import used for mypy on older Pythons
```

Remove the trailing `_ = Callable` line and the unused `Callable` import once flake8 confirms nothing else needs it.

- [ ] **Step 4: Export lazily from `pygarble/__init__.py`**

Replace the file body after the three metadata lines with:

```python
from typing import TYPE_CHECKING, Any

from .analysis import Analysis, Signal, Span
from .calibration import CalibrationReport, ThresholdPoint, calibrate
from .core import EnsembleDetector, GarbleDetector, Strategy
from .findings import Finding, Redaction, ScanReport

if TYPE_CHECKING:
    from .pii import PIIDetector as PIIDetector
    from .profanity import ProfanityDetector as ProfanityDetector
    from .scanner import Scanner as Scanner
    from .scanner import redact as redact
    from .scanner import scan as scan
    from .secrets import SecretsDetector as SecretsDetector

_LAZY = {
    "Scanner": ("scanner", "Scanner"),
    "scan": ("scanner", "scan"),
    "redact": ("scanner", "redact"),
    "SecretsDetector": ("secrets", "SecretsDetector"),
    "PIIDetector": ("pii", "PIIDetector"),
    "ProfanityDetector": ("profanity", "ProfanityDetector"),
}

__all__ = [
    "GarbleDetector",
    "Strategy",
    "EnsembleDetector",
    "__version__",
    "Analysis",
    "Signal",
    "Span",
    "CalibrationReport",
    "ThresholdPoint",
    "calibrate",
    "Finding",
    "ScanReport",
    "Redaction",
    "Scanner",
    "scan",
    "redact",
    "SecretsDetector",
    "PIIDetector",
    "ProfanityDetector",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(name)
    from importlib import import_module

    module, attr = _LAZY[name]
    value = getattr(import_module("." + module, __name__), attr)
    globals()[name] = value
    return value
```

- [ ] **Step 5: Run tests, lint, full suite, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_scanner.py -q && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble && PYTHONPATH=. python -m pytest -q -W error::FutureWarning`
Expected: pass (existing suite unchanged). `test_gibberish_options_forwarded` contains a tautology guard on the threshold line; replace it with a concrete assertion once you see real scores: pick a text where `strict` flags and `lax` does not (`PYTHONPATH=. python -c "from pygarble import EnsembleDetector as E; print(E().score('hello wrld frbl'))"`), and record the text in the test.

```bash
git add pygarble/scanner.py pygarble/__init__.py tests/test_scanner.py
git commit -m "feat: Scanner composes secrets and gibberish with redaction; lazy exports

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: PII checksums

**Files:**
- Create: `pygarble/pii/__init__.py` (stub exporting `ALL_KINDS = frozenset()` and nothing else for now), `pygarble/pii/checksums.py`
- Test: `tests/test_pii_checksums.py`

**Interfaces:**
- Produces: `luhn(digits: str) -> bool`, `iban_mod97(compact: str) -> bool`, `verhoeff(digits: str) -> bool`, `nhs_mod11(digits: str) -> bool`. All take a string of the relevant characters with separators already removed and return `False` (never raise) for the wrong length or non-digit input.

- [ ] **Step 1: Write the failing tests**

`tests/test_pii_checksums.py`:

```python
"""Checksum functions are pure and never raise."""

from pygarble.pii.checksums import iban_mod97, luhn, nhs_mod11, verhoeff


def test_luhn():
    assert luhn("4111111111111111")
    assert luhn("378282246310005")
    assert luhn("6011111111111117")
    assert not luhn("4111111111111112")
    assert not luhn("")
    assert not luhn("12a4")


def test_iban_mod97():
    assert iban_mod97("GB82WEST12345698765432")
    assert iban_mod97("DE89370400440532013000")
    assert iban_mod97("FR7630006000011234567890189")
    assert not iban_mod97("GB82WEST12345698765433")
    assert not iban_mod97("GB82")
    assert not iban_mod97("gb82west12345698765432")


def test_verhoeff():
    assert verhoeff("236")
    assert verhoeff("12345")
    assert not verhoeff("235")
    assert not verhoeff("")
    assert not verhoeff("12x45")


def test_nhs_mod11():
    assert nhs_mod11("9434765919")
    assert nhs_mod11("4010232137")
    assert not nhs_mod11("9434765918")
    assert not nhs_mod11("123456789")
    assert not nhs_mod11("943476591x")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/test_pii_checksums.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement**

`pygarble/pii/__init__.py` (temporary stub, replaced in Task 5):

```python
"""Structural PII detection with checksums and locale packs."""

from typing import FrozenSet

ALL_KINDS: FrozenSet[str] = frozenset()
```

`pygarble/pii/checksums.py`:

```python
"""Check digits that keep the false-positive rate low."""

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def luhn(digits: str) -> bool:
    if not digits or not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_mod97(compact: str) -> bool:
    if len(compact) < 5 or not compact.isalnum() or not compact.isupper():
        if not (compact[:2].isupper() and compact.isalnum()):
            return False
    if len(compact) < 5:
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(
        str(ord(char) - 55) if char.isalpha() else char
        for char in rearranged
    )
    if not number.isdigit():
        return False
    return int(number) % 97 == 1


def verhoeff(digits: str) -> bool:
    if not digits or not digits.isdigit():
        return False
    check = 0
    for index, char in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][ord(char) - 48]]
    return check == 0


def nhs_mod11(digits: str) -> bool:
    if len(digits) != 10 or not digits.isdigit():
        return False
    total = sum(
        (10 - index) * (ord(char) - 48) for index, char in enumerate(digits[:9])
    )
    remainder = 11 - (total % 11)
    if remainder == 11:
        remainder = 0
    if remainder == 10:
        return False
    return remainder == ord(digits[9]) - 48
```

Tidy `iban_mod97` while implementing: the guard should simply be `if len(compact) < 5 or not compact.isalnum() or compact != compact.upper(): return False` (lowercase input is rejected; digits are unaffected by `upper()`).

- [ ] **Step 4: Run tests, lint, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_pii_checksums.py -q && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble`

```bash
git add pygarble/pii tests/test_pii_checksums.py
git commit -m "feat: Luhn, IBAN mod-97, Verhoeff and NHS mod-11 checksums

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: PII detector with locale packs

**Files:**
- Create: `pygarble/pii/patterns.py`
- Modify: `pygarble/pii/__init__.py` (replace stub)
- Test: `tests/test_pii.py`

**Interfaces:**
- Consumes: Task 4 checksums; Task 1 `Finding`, `sort_key`.
- Produces: `PIIDetector(kinds=None, exclude_kinds=(), locales=("us","uk","in"))` with `.detect(text)`, `.kinds`, `.locales`; `detect(text, **kwargs)`; `LOCALES = ("us", "uk", "in")`; `ALL_KINDS`; `patterns.export() -> dict`. Kinds: `email, phone, credit_card, iban, ipv4, ipv6, ssn_us, nino, nhs_number, aadhaar, pan`.

Design notes for the implementer: one compiled regex per rule (backreferences and per-kind validators make a single alternation impractical); every regex starts with a literal, a digit class or a lookbehind so passes stay linear. Bare-digit forms of NHS and Aadhaar require a keyword within 30 characters before them, because a random 10- or 12-digit number passes mod-11 or Verhoeff about one time in ten.

- [ ] **Step 1: Write the failing tests**

`tests/test_pii.py`:

```python
"""Structural PII with checksums; locale packs; overlap rules."""

import pytest

from pygarble.pii import ALL_KINDS, LOCALES, PIIDetector, detect
from pygarble.pii.patterns import export


def spans(text, findings):
    return [(f.kind, text[f.start : f.end], f.confidence) for f in findings]


def test_kinds_locales_and_export():
    assert LOCALES == ("us", "uk", "in")
    assert ALL_KINDS == frozenset(
        {
            "email", "phone", "credit_card", "iban", "ipv4", "ipv6",
            "ssn_us", "nino", "nhs_number", "aadhaar", "pan",
        }
    )
    payload = export()
    assert set(payload) >= {"generic", "locales", "iban_lengths", "iin"}


def test_email():
    text = "Write to jane.doe+news@example.co.uk. Not at@x or @handle."
    assert spans(text, detect(text)) == [
        ("email", "jane.doe+news@example.co.uk", 0.9)
    ]
    assert detect("user@localhost") == ()


def test_phone_formats():
    cases = {
        "+14155552671": ("e164", 0.9),
        "(415) 555-2671": ("national_us", 0.8),
        "415-555-2671": ("national_us", 0.8),
        "020 7946 0958": ("national_uk", 0.8),
        "07911 123456": ("national_uk", 0.8),
        "+91 98765 43210": ("national_in", 0.8),
        "9876543210": ("national_in", 0.8),
    }
    for text, (reason, confidence) in cases.items():
        found = [f for f in detect(f"call {text} now") if f.kind == "phone"]
        assert found, text
        assert found[0].reason == reason and found[0].confidence == confidence
    for text in ["2026-09-26", "1695700000", "order 123456", "v1.2.3.4"]:
        assert not [f for f in detect(text) if f.kind == "phone"], text


def test_phone_locale_gating():
    only_us = PIIDetector(locales=["us"])
    assert [f.kind for f in only_us.detect("07911 123456")] == []
    assert only_us.detect("+14155552671")[0].reason == "e164"


def test_credit_card_luhn_and_brand():
    text = "pay with 4111 1111 1111 1111 or 3782-822463-10005"
    found = [f for f in detect(text) if f.kind == "credit_card"]
    assert [(text[f.start : f.end], f.reason, f.confidence) for f in found] == [
        ("4111 1111 1111 1111", "luhn_visa", 1.0),
        ("3782-822463-10005", "luhn_amex", 1.0),
    ]
    assert not [f for f in detect("4111 1111 1111 1112") if f.kind == "credit_card"]
    assert not [f for f in detect("1234 5678 9012 3452") if f.kind == "credit_card"]


def test_card_beats_phone_on_identical_span():
    # 13 digits, Luhn-valid Visa; also shaped like a long phone number.
    text = "id 4222222222222"
    found = detect(text)
    assert [f.kind for f in found] == ["credit_card"]


def test_iban():
    text = "IBAN GB82 WEST 1234 5698 7654 32 and DE89370400440532013000"
    found = [f for f in detect(text) if f.kind == "iban"]
    assert [text[f.start : f.end] for f in found] == [
        "GB82 WEST 1234 5698 7654 32",
        "DE89370400440532013000",
    ]
    assert all(f.confidence == 1.0 and f.reason == "mod97" for f in found)
    assert not [f for f in detect("GB82WEST12345698765433") if f.kind == "iban"]
    assert not [f for f in detect("XX82WEST12345698765432") if f.kind == "iban"]


def test_ip_addresses():
    text = "from 192.168.1.10 to 2001:db8::8a2e:370:7334 at 12:30:45"
    found = detect(text)
    assert spans(text, found) == [
        ("ipv4", "192.168.1.10", 0.7),
        ("ipv6", "2001:db8::8a2e:370:7334", 0.7),
    ]
    for negative in ["v1.2.3.4", "256.1.1.1", "1.2.3", "de:ad:be:ef:00:01"]:
        assert not [f for f in detect(negative) if f.kind.startswith("ip")]


def test_ssn_us():
    assert spans("ssn 123-45-6789", detect("ssn 123-45-6789")) == [
        ("ssn_us", "123-45-6789", 0.8)
    ]
    assert detect("123 45 6789")[0].kind == "ssn_us"
    (ctx,) = detect("SSN: 123456789")
    assert ctx.kind == "ssn_us" and ctx.confidence == 0.6
    for negative in ["000-45-6789", "666-45-6789", "900-45-6789",
                     "123-00-6789", "123-45-0000", "id 123456789"]:
        assert not [f for f in detect(negative) if f.kind == "ssn_us"], negative


def test_uk_pack():
    assert spans("NI AB 12 34 56 C", detect("NI AB 12 34 56 C")) == [
        ("nino", "AB 12 34 56 C", 0.9)
    ]
    assert not [f for f in detect("BG 12 34 56 C") if f.kind == "nino"]
    assert not [f for f in detect("AB 12 34 56 E") if f.kind == "nino"]
    text = "NHS number 943 476 5919"
    assert spans(text, detect(text)) == [("nhs_number", "943 476 5919", 0.9)]
    (bare,) = detect("nhs: 9434765919")
    assert bare.kind == "nhs_number"
    assert not [f for f in detect("943 476 5918") if f.kind == "nhs_number"]
    assert not [f for f in detect("ref 9434765919") if f.kind == "nhs_number"]


def test_india_pack():
    text = "Aadhaar 2345 6789 0126"
    found = [f for f in detect(text) if f.kind == "aadhaar"]
    assert found and found[0].confidence == 1.0 and found[0].reason == "verhoeff"
    assert not [f for f in detect("1345 6789 0126") if f.kind == "aadhaar"]
    assert not [f for f in detect("ref 234567890126") if f.kind == "aadhaar"]
    assert spans("PAN ABCPE1234F", detect("PAN ABCPE1234F")) == [
        ("pan", "ABCPE1234F", 0.9)
    ]
    assert not [f for f in detect("ABCDE1234F") if f.kind == "pan"]


def test_identical_spans_keep_higher_confidence_only():
    text = "9434765919"
    found = PIIDetector(locales=["uk", "in"]).detect("nhs " + text)
    assert [f.kind for f in found] == ["nhs_number"]


def test_kind_and_locale_validation():
    with pytest.raises(ValueError, match="unknown pii kind"):
        PIIDetector(kinds=["passport"])
    with pytest.raises(ValueError, match="unknown locale"):
        PIIDetector(locales=["fr"])
    assert PIIDetector(kinds=["email"]).detect("+14155552671") == ()
    assert PIIDetector(exclude_kinds=["phone"]).detect("+14155552671") == ()
    with pytest.raises(TypeError):
        detect(123)  # type: ignore[arg-type]


def test_deterministic_and_sorted():
    text = "a@b.com 4111111111111111 +14155552671 a@b.com"
    first = detect(text)
    assert first == detect(text)
    assert [f.start for f in first] == sorted(f.start for f in first)
```

The Aadhaar vector `2345 6789 0126` must be Verhoeff-valid: compute a valid check digit with `PYTHONPATH=. python -c "from pygarble.pii.checksums import verhoeff; print([d for d in range(10) if verhoeff('23456789012'+str(d))])"` and substitute the digit that passes before running the test (the test text and assertions must use the valid number).

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/test_pii.py -q`
Expected: ImportError for `PIIDetector`.

- [ ] **Step 3: Implement `pygarble/pii/patterns.py`**

```python
"""PII shapes. Source of truth for pii.json."""

from typing import Any, Dict, Tuple

# (kind, regex, confidence, reason, validator)
Rule = Tuple[str, str, float, str, str]

EMAIL = (
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+"
    r"(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9\-])"
)
E164 = r"(?<![\w+])\+[1-9]\d{6,14}(?!\d)"
CARD = r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])"
IBAN = (
    r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?"
    r"(?![A-Z0-9])"
)
IPV4 = (
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"
)
_H = r"[0-9A-Fa-f]{1,4}"
IPV6 = (
    r"(?<![\w:])(?:"
    rf"(?:{_H}:){{7}}{_H}|(?:{_H}:){{1,7}}:|(?:{_H}:){{1,6}}:{_H}|"
    rf"(?:{_H}:){{1,5}}(?::{_H}){{1,2}}|(?:{_H}:){{1,4}}(?::{_H}){{1,3}}|"
    rf"(?:{_H}:){{1,3}}(?::{_H}){{1,4}}|(?:{_H}:){{1,2}}(?::{_H}){{1,5}}|"
    rf"{_H}:(?::{_H}){{1,6}}|:(?::{_H}){{1,7}}|"
    r"::(?:[Ff]{4}:)?(?:\d{1,3}\.){3}\d{1,3}"
    r")(?![\w:])"
)

GENERIC: Tuple[Rule, ...] = (
    ("email", EMAIL, 0.9, "structure", ""),
    ("phone", E164, 0.9, "e164", "phone_digits"),
    ("credit_card", CARD, 1.0, "luhn", "card"),
    ("iban", IBAN, 1.0, "mod97", "iban"),
    ("ipv4", IPV4, 0.7, "structure", ""),
    ("ipv6", IPV6, 0.7, "structure", ""),
)

LOCALE_RULES: Dict[str, Tuple[Rule, ...]] = {
    "us": (
        (
            "phone",
            r"(?<![\d\-])(?:\+?1[\s.\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.\-])"
            r"\d{3}[\s.\-]\d{4}(?![\d\-])",
            0.8,
            "national_us",
            "phone_digits",
        ),
        (
            "ssn_us",
            r"(?<![\d\-])(?!000|666|9\d\d)\d{3}([\-\s])(?!00)\d{2}\1"
            r"(?!0000)\d{4}(?![\d\-])",
            0.8,
            "structure",
            "",
        ),
        (
            "ssn_us",
            r"(?i:ssn|social security)[^\d\n]{0,30}"
            r"(?P<v>(?!000|666|9)\d{3}(?!00)\d{2}(?!0000)\d{4})(?!\d)",
            0.6,
            "keyword_context",
            "",
        ),
    ),
    "uk": (
        (
            "phone",
            r"(?<![\d\-])(?:\+44\s?|0)(?:\d{4}\s?\d{6}|\d{3}\s?\d{3}\s?\d{4}"
            r"|\d{2}\s?\d{4}\s?\d{4})(?![\d\-])",
            0.8,
            "national_uk",
            "phone_digits",
        ),
        (
            "nino",
            r"(?<![A-Z0-9])(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]"
            r"[A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D](?![A-Z0-9])",
            0.9,
            "structure",
            "",
        ),
        (
            "nhs_number",
            r"(?<!\d)\d{3} \d{3} \d{4}(?!\d)",
            0.9,
            "mod11",
            "nhs",
        ),
        (
            "nhs_number",
            r"(?i:nhs)[^\d\n]{0,30}(?P<v>\d{10})(?!\d)",
            0.9,
            "mod11",
            "nhs",
        ),
    ),
    "in": (
        (
            "phone",
            r"(?<![\d+\-])(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?![\d\-])",
            0.8,
            "national_in",
            "phone_digits",
        ),
        (
            "aadhaar",
            r"(?<!\d)[2-9]\d{3} \d{4} \d{4}(?!\d)",
            1.0,
            "verhoeff",
            "verhoeff",
        ),
        (
            "aadhaar",
            r"(?i:aadhaar|aadhar|uidai)[^\d\n]{0,30}(?P<v>[2-9]\d{11})(?!\d)",
            1.0,
            "verhoeff",
            "verhoeff",
        ),
        (
            "pan",
            r"(?<![A-Z0-9])[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z](?![A-Z0-9])",
            0.9,
            "structure",
            "",
        ),
    ),
}

IBAN_LENGTHS = {
    "AL": 28, "AD": 24, "AT": 20, "AZ": 28, "BH": 22, "BE": 16, "BA": 20,
    "BR": 29, "BG": 22, "CR": 22, "HR": 21, "CY": 28, "CZ": 24, "DK": 18,
    "DO": 28, "EE": 20, "FO": 18, "FI": 18, "FR": 27, "GE": 22, "DE": 22,
    "GI": 23, "GR": 27, "GL": 18, "GT": 28, "HU": 28, "IS": 26, "IE": 22,
    "IL": 23, "IT": 27, "JO": 30, "KZ": 20, "KW": 30, "LV": 21, "LB": 28,
    "LI": 21, "LT": 20, "LU": 20, "MK": 19, "MT": 31, "MR": 27, "MU": 30,
    "MC": 27, "MD": 24, "ME": 22, "NL": 18, "NO": 15, "PK": 24, "PS": 29,
    "PL": 28, "PT": 25, "QA": 29, "RO": 24, "SM": 27, "SA": 24, "RS": 22,
    "SK": 24, "SI": 19, "ES": 24, "SE": 24, "CH": 21, "TN": 24, "TR": 26,
    "AE": 23, "GB": 22, "VG": 24, "XK": 20,
}

# (brand, prefix regex over the digit string, allowed lengths)
IIN = (
    ("visa", r"4", (13, 16, 19)),
    ("mastercard", r"5[1-5]|2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)", (16,)),
    ("amex", r"3[47]", (15,)),
    ("discover", r"6011|65|64[4-9]", (16, 19)),
    ("jcb", r"35", (16, 19)),
    ("diners", r"3(?:0[0-5]|6|8)", (14, 16, 19)),
    ("rupay", r"60|81|82", (16,)),
    ("maestro", r"5018|5020|5038|6304|6759|676[1-3]", (12, 13, 14, 15, 16, 17, 18, 19)),
)


def export() -> Dict[str, Any]:
    def rows(rules: Tuple[Rule, ...]) -> Any:
        return [
            {
                "kind": kind,
                "regex": regex,
                "confidence": confidence,
                "reason": reason,
                "validator": validator or None,
            }
            for kind, regex, confidence, reason, validator in rules
        ]

    return {
        "generic": rows(GENERIC),
        "locales": {name: rows(rules) for name, rules in LOCALE_RULES.items()},
        "iban_lengths": IBAN_LENGTHS,
        "iin": [
            {"brand": brand, "prefix": prefix, "lengths": list(lengths)}
            for brand, prefix, lengths in IIN
        ],
    }
```

- [ ] **Step 4: Implement `pygarble/pii/__init__.py`**

```python
"""Structural PII detection with checksums and locale packs."""

import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..findings import Finding, sort_key
from .checksums import iban_mod97, luhn, nhs_mod11, verhoeff
from .patterns import GENERIC, IBAN_LENGTHS, IIN, LOCALE_RULES, Rule

CATEGORY = "pii"
LOCALES: Tuple[str, ...] = ("us", "uk", "in")
ALL_KINDS: FrozenSet[str] = frozenset(
    [rule[0] for rule in GENERIC]
    + [rule[0] for rules in LOCALE_RULES.values() for rule in rules]
)
_SEPARATORS = re.compile(r"[ \-]")
_IIN = [(brand, re.compile(prefix), lengths) for brand, prefix, lengths in IIN]
_COMPILED: Dict[str, List[Tuple[Rule, "re.Pattern[str]"]]] = {}


def _compiled(scope: str) -> List[Tuple[Rule, "re.Pattern[str]"]]:
    if scope not in _COMPILED:
        rules = GENERIC if scope == "generic" else LOCALE_RULES[scope]
        _COMPILED[scope] = [(rule, re.compile(rule[1])) for rule in rules]
    return _COMPILED[scope]


def _card_reason(value: str) -> Optional[str]:
    digits = _SEPARATORS.sub("", value)
    if not 13 <= len(digits) <= 19 or not luhn(digits):
        return None
    for brand, prefix, lengths in _IIN:
        if prefix.match(digits) and len(digits) in lengths:
            return f"luhn_{brand}"
    return None


def _validate(validator: str, value: str) -> Optional[str]:
    """Return a reason suffix override, "" to keep the rule's reason, or
    None to reject the match."""
    if validator == "":
        return ""
    if validator == "phone_digits":
        digits = re.sub(r"\D", "", value)
        return "" if 7 <= len(digits) <= 15 else None
    if validator == "card":
        return _card_reason(value)
    if validator == "iban":
        compact = value.replace(" ", "")
        expected = IBAN_LENGTHS.get(compact[:2])
        if expected != len(compact) or not iban_mod97(compact):
            return None
        return ""
    if validator == "nhs":
        return "" if nhs_mod11(re.sub(r"\D", "", value)) else None
    if validator == "verhoeff":
        return "" if verhoeff(re.sub(r"\D", "", value)) else None
    raise ValueError(f"unknown validator {validator!r}")


class PIIDetector:
    category = CATEGORY

    def __init__(
        self,
        kinds: Optional[Iterable[str]] = None,
        exclude_kinds: Iterable[str] = (),
        locales: Iterable[str] = LOCALES,
    ) -> None:
        chosen = frozenset(ALL_KINDS if kinds is None else kinds)
        excluded = frozenset(exclude_kinds)
        unknown = sorted((chosen | excluded) - ALL_KINDS)
        if unknown:
            raise ValueError(
                f"unknown pii kind(s): {', '.join(unknown)}; "
                f"valid kinds: {', '.join(sorted(ALL_KINDS))}"
            )
        self.kinds = chosen - excluded
        wanted = tuple(dict.fromkeys(locales))
        bad = sorted(set(wanted) - set(LOCALES))
        if bad:
            raise ValueError(
                f"unknown locale(s): {', '.join(bad)}; "
                f"valid locales: {', '.join(LOCALES)}"
            )
        self.locales = wanted

    def _run(self, text: str, scope: str, found: List[Finding]) -> None:
        for rule, pattern in _compiled(scope):
            kind, _, confidence, reason, validator = rule
            if kind not in self.kinds:
                continue
            for match in pattern.finditer(text):
                if "v" in match.groupdict() and match.group("v") is not None:
                    start, end = match.span("v")
                else:
                    start, end = match.span()
                verdict = _validate(validator, text[start:end])
                if verdict is None:
                    continue
                found.append(
                    Finding(
                        CATEGORY, kind, start, end, confidence,
                        verdict or reason,
                    )
                )

    def detect(self, text: str) -> Tuple[Finding, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        found: List[Finding] = []
        self._run(text, "generic", found)
        for locale in self.locales:
            self._run(text, locale, found)
        best: Dict[Tuple[int, int], Finding] = {}
        for finding in sorted(found, key=sort_key):
            key = (finding.start, finding.end)
            current = best.get(key)
            if current is None or finding.confidence > current.confidence:
                best[key] = finding
        return tuple(sorted(best.values(), key=sort_key))


def detect(text: str, **kwargs: Any) -> Tuple[Finding, ...]:
    return PIIDetector(**kwargs).detect(text)


__all__ = ["PIIDetector", "detect", "ALL_KINDS", "LOCALES"]
```

Note on `_validate` for cards: the rule's `reason` is `"luhn"` but the finding must carry `"luhn_<brand>"`, which `_card_reason` returns; the `verdict or reason` expression handles both cases.

- [ ] **Step 5: Run tests, lint, full suite, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_pii.py tests/test_scanner.py -q && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble && PYTHONPATH=. python -m pytest -q -W error::FutureWarning`
Expected: pass. `tests/test_scanner.py::test_public_exports_are_lazy` must still print `False`: check that `pygarble/__init__.py` does not import `pygarble.pii` eagerly.

```bash
git add pygarble/pii tests/test_pii.py
git commit -m "feat: PII detector with checksums and US/UK/IN locale packs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Profanity normalisation and word list

**Files:**
- Create: `pygarble/profanity/__init__.py` (stub: `ALL_KINDS = frozenset({"profanity"})`), `pygarble/profanity/normalize.py`, `pygarble/profanity/wordlist.py`
- Test: `tests/test_profanity_normalize.py`, `tests/test_profanity_wordlist.py`

**Interfaces:**
- Produces: `normalize.LEET_MAP`, `normalize_token(token) -> str` (letters only, a-z), `collapse_runs(token, to) -> str`, `has_long_run(token) -> bool`, `TOKEN_RE`; `wordlist.PROFANITY_STRONG: Tuple[str, ...]`, `PROFANITY_MILD`, `PHRASES: Tuple[Tuple[str, ...], ...]`, `EMBEDDED`, `ATTRIBUTION`, `export()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_profanity_normalize.py`:

```python
"""Token normalisation is deterministic and never widens to non-letters."""

from pygarble.profanity.normalize import (
    LEET_MAP,
    TOKEN_RE,
    collapse_runs,
    has_long_run,
    normalize_token,
)


def test_leet_map_is_single_chars():
    assert all(len(k) == 1 and len(v) == 1 for k, v in LEET_MAP.items())
    assert LEET_MAP["@"] == "a" and LEET_MAP["$"] == "s" and LEET_MAP["0"] == "o"


def test_normalize_token():
    assert normalize_token("Sh1T") == "shit"
    assert normalize_token("@ss") == "ass"
    assert normalize_token("don't") == "dont"
    assert normalize_token("café") == "cafe"
    assert normalize_token("b00k") == "book"
    assert normalize_token("138") == "ieb"  # digits map through leet
    assert normalize_token("") == ""


def test_collapse_runs_and_detection():
    assert collapse_runs("shiiit", 1) == "shit"
    assert collapse_runs("shiiit", 2) == "shiit"
    assert collapse_runs("book", 1) == "bok"
    assert has_long_run("shiiit") and not has_long_run("book")
    assert not has_long_run("")


def test_token_regex_keeps_symbols_inside_tokens():
    text = "f*ck sh!t $5 email@example.com a.b"
    assert [m.group() for m in TOKEN_RE.finditer(text)] == [
        "f*ck", "sh!t", "$5", "email@example", "com", "a", "b",
    ]
```

`tests/test_profanity_wordlist.py`:

```python
"""Word list invariants: normalised, deduplicated, attributed, safe."""

import re

from pygarble.data import ENGLISH_WORDS
from pygarble.profanity.normalize import normalize_token
from pygarble.profanity.wordlist import (
    ATTRIBUTION,
    EMBEDDED,
    PHRASES,
    PROFANITY_MILD,
    PROFANITY_STRONG,
    export,
)

LETTERS = re.compile(r"^[a-z]+$")


def test_lists_are_normalised_sorted_and_disjoint():
    for words in (PROFANITY_STRONG, PROFANITY_MILD):
        assert list(words) == sorted(set(words))
        for word in words:
            assert LETTERS.match(word), word
            assert normalize_token(word) == word, word
    assert not set(PROFANITY_STRONG) & set(PROFANITY_MILD)
    assert len(PROFANITY_STRONG) >= 80 and len(PROFANITY_MILD) >= 8


def test_phrases_are_tuples_of_normalised_words():
    for phrase in PHRASES:
        assert isinstance(phrase, tuple) and len(phrase) >= 2
        assert all(LETTERS.match(w) for w in phrase), phrase


def test_embedded_words_are_strong_and_not_inside_english_words():
    for word in EMBEDDED:
        assert word in PROFANITY_STRONG and len(word) >= 4
        assert not any(word in english for english in ENGLISH_WORDS), word


def test_no_ordinary_english_word_in_lists():
    # Words that appear in ordinary text must never be listed outright.
    for common in ["hell", "sex", "nude", "breast", "god", "screw", "bloody"]:
        assert common not in PROFANITY_STRONG, common
    for common in ["hell", "god", "sex", "nude"]:
        assert common not in PROFANITY_MILD, common


def test_attribution_and_export():
    assert "LDNOOBW" in ATTRIBUTION and "CC-BY-4.0" in ATTRIBUTION
    payload = export()
    assert set(payload) == {
        "strong", "mild", "phrases", "embedded", "leet_map", "attribution",
    }
    assert payload["strong"] == list(PROFANITY_STRONG)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/test_profanity_normalize.py tests/test_profanity_wordlist.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement `pygarble/profanity/normalize.py`**

```python
"""Fold case, diacritics and leetspeak so obfuscated tokens compare equal."""

import re
import unicodedata

LEET_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "+": "t",
}
TOKEN_RE = re.compile(r"[\w*#@$!|+'’]+")
_LONG_RUN = re.compile(r"([a-z])\1\1")
_RUN = re.compile(r"([a-z])\1+")
_TRANSLATE = str.maketrans(LEET_MAP)


def normalize_token(token: str) -> str:
    folded = unicodedata.normalize("NFKD", token.casefold())
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    mapped = stripped.translate(_TRANSLATE)
    return "".join(c for c in mapped if "a" <= c <= "z")


def has_long_run(token: str) -> bool:
    return _LONG_RUN.search(token) is not None


def collapse_runs(token: str, to: int) -> str:
    if to < 1:
        raise ValueError("to must be >= 1")
    return _RUN.sub(lambda m: m.group(1) * min(to, len(m.group())), token)
```

- [ ] **Step 4: Implement `pygarble/profanity/wordlist.py`**

The seed is the LDNOOBW English list (CC-BY 4.0, Shutterstock), filtered to actual profanity and slurs and extended with common compounds. Every entry is stored already normalised. The implementer writes the tuples below verbatim, then runs the wordlist tests; if `EMBEDDED` fails the substring check for a word, remove that word from `EMBEDDED` (not from the strong list) and note it in the ledger.

```python
"""Profanity word lists. Source of truth for profanity.json.

Seed: the LDNOOBW "List of Dirty, Naughty, Obscene, and Otherwise Bad Words"
(English), Shutterstock, licensed CC-BY-4.0
(https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words).
Filtered to profanity and slurs (sexual-health and anatomical vocabulary that
appears in ordinary text is excluded), normalised, and extended with common
compounds. Entries are lowercase ASCII letters only.
"""

from typing import Any, Dict, Tuple

ATTRIBUTION = (
    "Seeded from the LDNOOBW English list (Shutterstock), CC-BY-4.0, "
    "filtered and extended by the pygarble maintainers."
)

PROFANITY_STRONG: Tuple[str, ...] = tuple(
    sorted(
        {
            "arse", "arsehole", "arseholes", "ass", "asshat", "asshole",
            "assholes", "asswipe", "bastard", "bastards", "bitch", "bitches",
            "bitchy", "bollocks", "bullshit", "bullshitter", "chink", "chinks",
            "clit", "cock", "cocks", "cocksucker", "cocksuckers", "coon",
            "coons", "cum", "cunt", "cunts", "dago", "dickhead",
            "dickheads", "dicks", "dipshit", "douchebag", "douchebags", "dumbass",
            "dumbasses", "dyke", "dykes", "fag", "faggot", "faggots", "fags",
            "fuck", "fucked", "fucker", "fuckers", "fuckin", "fucking",
            "fucks", "fuckwit", "goddamn", "gook", "gooks", "horseshit",
            "jackass", "jackasses", "jizz", "kike", "kikes", "kunt",
            "motherfucker", "motherfuckers", "motherfucking", "nigga", "niggas",
            "nigger", "niggers", "paki", "pakis", "piss", "pissed", "pisses",
            "pissing", "prick", "pricks", "pussies", "pussy", "raghead",
            "retard", "retarded", "retards", "shit", "shite", "shitfaced",
            "shithead", "shitheads", "shits", "shitty", "slut", "sluts",
            "spic", "spics", "tits", "titties", "tranny", "trannies", "twat",
            "twats", "wank", "wanker", "wankers", "wetback", "wetbacks",
            "whore", "whores", "wog", "wogs", "wop", "wops",
        }
    )
)

PROFANITY_MILD: Tuple[str, ...] = tuple(
    sorted(
        {
            "bugger", "crap", "crappy", "damn", "damned", "dammit", "darn",
            "douche", "frigging", "jerkoff", "pissoff", "sod", "sodding",
            "turd", "wanky",
        }
    )
)

PHRASES: Tuple[Tuple[str, ...], ...] = (
    ("son", "of", "a", "bitch"),
    ("piece", "of", "shit"),
    ("mother", "fucker"),
    ("bull", "shit"),
    ("jack", "ass"),
    ("dumb", "ass"),
)

# Strong words also matched inside longer tokens (fuckwit, shitposting).
# Each must be at least four letters and occur inside no ENGLISH_WORDS entry.
EMBEDDED: Tuple[str, ...] = ("cunt", "fuck", "shit", "wank")


def export() -> Dict[str, Any]:
    from .normalize import LEET_MAP

    return {
        "strong": list(PROFANITY_STRONG),
        "mild": list(PROFANITY_MILD),
        "phrases": [list(p) for p in PHRASES],
        "embedded": list(EMBEDDED),
        "leet_map": dict(LEET_MAP),
        "attribution": ATTRIBUTION,
    }
```

`pygarble/profanity/__init__.py` stub (replaced in Task 7):

```python
"""Rule-based profanity detection with obfuscation handling."""

from typing import FrozenSet

ALL_KINDS: FrozenSet[str] = frozenset({"profanity"})
```

- [ ] **Step 5: Run tests, lint, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_profanity_normalize.py tests/test_profanity_wordlist.py -q && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble`

```bash
git add pygarble/profanity tests/test_profanity_normalize.py tests/test_profanity_wordlist.py
git commit -m "feat: profanity normalisation and attributed word list

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Profanity detector

**Files:**
- Modify: `pygarble/profanity/__init__.py` (replace stub)
- Test: `tests/test_profanity.py`

**Interfaces:**
- Consumes: Task 6 modules; Task 1 `Finding`, `sort_key`; `pygarble.data.ENGLISH_WORDS` (lazy, wildcard rule only).
- Produces: `ProfanityDetector(allowlist=None, tiers=("strong","mild"), obfuscation=True)` with `.detect(text)`; `detect(text, **kwargs)`; `ALL_KINDS = frozenset({"profanity"})`.

- [ ] **Step 1: Write the failing tests**

`tests/test_profanity.py`:

```python
"""Each matching rule, in order, with its confidence and reason."""

import pytest

from pygarble.profanity import ALL_KINDS, ProfanityDetector, detect


def hits(text, **kwargs):
    return [
        (text[f.start : f.end], f.confidence, f.reason)
        for f in detect(text, **kwargs)
    ]


def test_kinds():
    assert ALL_KINDS == frozenset({"profanity"})
    (f,) = detect("what the fuck")
    assert f.category == "profanity" and f.kind == "profanity"


def test_exact_strong_and_mild_with_leet_and_case():
    assert hits("What the FUCK") == [("FUCK", 1.0, "strong")]
    assert hits("sh1t happens") == [("sh1t", 1.0, "strong")]
    assert hits("well, damn.") == [("damn", 0.7, "mild")]
    assert hits("@ss") == [("@ss", 1.0, "strong")]


def test_scunthorpe_and_ordinary_words_are_clean():
    for text in [
        "Scunthorpe United", "assassin", "classic", "shiitake", "cocktail",
        "as", "pass the salt", "Dick Whittington", "hello world",
        "the analyst assessed the assets", "bass guitar", "Essex",
    ]:
        assert hits(text) == [], text


def test_elongation_rule():
    assert hits("shiiiit") == [("shiiiit", 0.8, "elongated")]
    assert hits("fuuuuck") == [("fuuuuck", 0.8, "elongated")]
    assert hits("asss") == []  # run of three but "as" is not listed
    assert hits("boooook") == []


def test_embedded_rule():
    assert hits("fuckwit") == [("fuckwit", 1.0, "strong")]
    assert hits("shitposting") == [("shitposting", 0.8, "embedded")]
    assert hits("unfuckingbelievable") == [
        ("unfuckingbelievable", 0.8, "embedded")
    ]


def test_wildcard_rule():
    assert hits("f*ck") == [("f*ck", 0.9, "masked")]
    # "sh*t" also fits ordinary words (shot, shut), so it is ambiguous.
    assert hits("sh*t") == [("sh*t", 0.6, "masked_ambiguous")]
    assert hits("f#ck you") == [("f#ck", 0.9, "masked")]
    assert hits("c*nt") == [("c*nt", 0.6, "masked_ambiguous")]
    assert hits("d*ck") == []  # "dick" is not listed (it is a name)
    assert hits("$5 and 50% off") == []
    assert hits("c*") == []
    assert hits("f*ck", obfuscation=False) == []


def test_spaced_rule():
    assert hits("s h i t") == [("s h i t", 0.8, "spaced")]
    assert hits("f.u.c.k off") == [("f.u.c.k", 0.8, "spaced")]
    assert hits("a b c") == []
    assert hits("i am a") == []
    assert hits("s h i t", obfuscation=False) == []


def test_phrase_rule():
    assert hits("you son of a bitch") == [("son of a bitch", 1.0, "strong")]
    assert hits("piece of SHIT, honestly") == [
        ("piece of SHIT", 1.0, "strong")
    ]


def test_allowlist_and_tiers():
    assert hits("damn", allowlist=["damn"]) == []
    assert hits("Fuck", allowlist=["fuck"]) == []
    assert hits("damn it", tiers=["strong"]) == []
    with pytest.raises(ValueError, match="tiers"):
        ProfanityDetector(tiers=["extreme"])
    with pytest.raises(TypeError):
        detect(None)  # type: ignore[arg-type]


def test_offsets_with_unicode_and_punctuation():
    text = "café — “fuck” — fin"
    (f,) = detect(text)
    assert text[f.start : f.end] == "fuck"


def test_deterministic_and_sorted():
    text = "shit fuck shit"
    first = detect(text)
    assert first == detect(text)
    assert [f.start for f in first] == [0, 5, 10]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/test_profanity.py -q`
Expected: ImportError for `ProfanityDetector`.

- [ ] **Step 3: Implement `pygarble/profanity/__init__.py`**

```python
"""Rule-based profanity detection with obfuscation handling."""

import re
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..findings import Finding, sort_key
from .normalize import TOKEN_RE, collapse_runs, has_long_run, normalize_token
from .wordlist import EMBEDDED, PHRASES, PROFANITY_MILD, PROFANITY_STRONG

CATEGORY = "pii"  # replaced below; keeps the constant near the top
CATEGORY = "profanity"
KIND = "profanity"
ALL_KINDS: FrozenSet[str] = frozenset({KIND})
TIERS = ("strong", "mild")
_STRONG = frozenset(PROFANITY_STRONG)
_MILD = frozenset(PROFANITY_MILD)
_WILD_CHARS = frozenset("*#@$!")
_WILD_TOKEN = re.compile(r"^[\w*#@$!]+$")
_GAP = re.compile(r"^[ .\-]{1,3}$")
_PHRASE_MAX = max(len(p) for p in PHRASES)
_BY_LENGTH: Dict[int, Tuple[str, ...]] = {}
for _word in PROFANITY_STRONG:
    _BY_LENGTH[len(_word)] = _BY_LENGTH.get(len(_word), ()) + (_word,)

Token = Tuple[int, int, str, str]  # start, end, raw, normalised


class ProfanityDetector:
    category = CATEGORY

    def __init__(
        self,
        allowlist: Optional[Iterable[str]] = None,
        tiers: Iterable[str] = TIERS,
        obfuscation: bool = True,
    ) -> None:
        chosen = tuple(dict.fromkeys(tiers))
        bad = [t for t in chosen if t not in TIERS]
        if bad or not chosen:
            raise ValueError(
                f"tiers must be a non-empty subset of {', '.join(TIERS)}"
            )
        self.strong = _STRONG if "strong" in chosen else frozenset()
        self.mild = _MILD if "mild" in chosen else frozenset()
        self.allowlist = frozenset(
            normalize_token(word) for word in (allowlist or ())
        )
        self.obfuscation = bool(obfuscation)

    def _tier(self, word: str) -> Optional[Tuple[float, str]]:
        if word in self.strong:
            return (1.0, "strong")
        if word in self.mild:
            return (0.7, "mild")
        return None

    def _single(self, token: Token) -> Optional[Tuple[float, str]]:
        start, end, raw, norm = token
        if not norm or norm in self.allowlist:
            return None
        tier = self._tier(norm)
        if tier is not None:
            return tier
        if has_long_run(norm):
            for width in (2, 1):
                collapsed = collapse_runs(norm, width)
                if collapsed in self.allowlist:
                    return None
                if self._tier(collapsed) is not None:
                    return (0.8, "elongated")
        if self.strong and any(w in norm for w in EMBEDDED):
            return (0.8, "embedded")
        if self.obfuscation and self.strong:
            return self._wildcard(raw)
        return None

    def _wildcard(self, raw: str) -> Optional[Tuple[float, str]]:
        lowered = raw.lower()
        if not _WILD_TOKEN.match(lowered) or not (
            set(lowered) & _WILD_CHARS
        ):
            return None
        letters = sum(1 for c in lowered if c.isalpha())
        if letters < 2:
            return None
        candidates = _BY_LENGTH.get(len(lowered), ())
        matches = [
            word
            for word in candidates
            if all(
                c in _WILD_CHARS or c == w for c, w in zip(lowered, word)
            )
        ]
        matches = [w for w in matches if w not in self.allowlist]
        if not matches:
            return None
        if len(matches) > 1:
            return (0.6, "masked_ambiguous")
        from ..data import ENGLISH_WORDS

        pattern = re.compile(
            "^"
            + "".join("." if c in _WILD_CHARS else re.escape(c) for c in lowered)
            + "$"
        )
        ambiguous = any(
            len(word) == len(lowered) and pattern.match(word)
            for word in ENGLISH_WORDS
        )
        return (0.6, "masked_ambiguous") if ambiguous else (0.9, "masked")

    def _phrases(
        self, tokens: Sequence[Token], used: Set[int], out: List[Finding]
    ) -> None:
        norms = [t[3] for t in tokens]
        for size in range(_PHRASE_MAX, 1, -1):
            for i in range(0, len(tokens) - size + 1):
                if any(j in used for j in range(i, i + size)):
                    continue
                window = tuple(norms[i : i + size])
                if window in PHRASES and " ".join(window) not in (
                    self.allowlist
                ):
                    joined = "".join(window)
                    tier = self._tier(joined) or (1.0, "strong")
                    out.append(
                        Finding(
                            CATEGORY, KIND, tokens[i][0], tokens[i + size - 1][1],
                            tier[0], tier[1],
                        )
                    )
                    used.update(range(i, i + size))

    def _spaced(
        self, text: str, tokens: Sequence[Token], used: Set[int],
        out: List[Finding],
    ) -> None:
        i = 0
        while i < len(tokens):
            j = i
            while (
                j + 1 < len(tokens)
                and j + 1 not in used
                and len(tokens[j][2]) == 1
                and len(tokens[j + 1][2]) == 1
                and tokens[j][2].isalpha()
                and tokens[j + 1][2].isalpha()
                and _GAP.match(text[tokens[j][1] : tokens[j + 1][0]])
            ):
                j += 1
            if j - i + 1 >= 3 and i not in used:
                joined = "".join(t[3] for t in tokens[i : j + 1])
                if joined not in self.allowlist and self._tier(joined):
                    out.append(
                        Finding(
                            CATEGORY, KIND, tokens[i][0], tokens[j][1], 0.8,
                            "spaced",
                        )
                    )
                    used.update(range(i, j + 1))
                i = j + 1
            else:
                i += 1

    def detect(self, text: str) -> Tuple[Finding, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: List[Token] = [
            (m.start(), m.end(), m.group(), normalize_token(m.group()))
            for m in TOKEN_RE.finditer(text)
        ]
        out: List[Finding] = []
        used: Set[int] = set()
        self._phrases(tokens, used, out)
        if self.obfuscation:
            self._spaced(text, tokens, used, out)
        for index, token in enumerate(tokens):
            if index in used:
                continue
            verdict = self._single(token)
            if verdict is not None:
                out.append(
                    Finding(
                        CATEGORY, KIND, token[0], token[1], verdict[0],
                        verdict[1],
                    )
                )
        return tuple(sorted(out, key=sort_key))


def detect(text: str, **kwargs: Any) -> Tuple[Finding, ...]:
    return ProfanityDetector(**kwargs).detect(text)


__all__ = ["ProfanityDetector", "detect", "ALL_KINDS", "TIERS"]
```

Delete the throwaway `CATEGORY = "pii"` line when implementing. The phrase allowlist check compares against `" ".join(window)`, so an allowlisted phrase is written with spaces (`allowlist=["son of a bitch"]`); `normalize_token` would strip spaces, so store phrase allowlist entries by joining normalised words with a space instead: in `__init__`, build `self.allowlist` as `frozenset(" ".join(normalize_token(w) for w in entry.split()) for entry in allowlist)`; single words are unaffected.

- [ ] **Step 4: Run tests, lint, full suite, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_profanity.py tests/test_scanner.py -q && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble && PYTHONPATH=. python -m pytest -q -W error::FutureWarning`
Expected: pass. If `test_scunthorpe_and_ordinary_words_are_clean` fails on a word, the fix is in the rules or the list, never in the test.

```bash
git add pygarble/profanity tests/test_profanity.py
git commit -m "feat: profanity detector with leet, elongation, embedded, masked and spaced rules

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Scan vectors and clean corpus

**Files:**
- Create: `regression/scan_vectors.json`, `regression/clean_corpus/code.py.txt`, `regression/clean_corpus/code.js.txt`, `regression/clean_corpus/config.json.txt`, `regression/clean_corpus/logs.txt`, `regression/clean_corpus/prose.txt`
- Test: `tests/test_scan_vectors.py`, `tests/test_clean_corpus.py`

**Interfaces:**
- Consumes: `Scanner` (Task 3) with all detectors from Tasks 2, 5, 7.
- Produces: `regression/scan_vectors.json` schema: list of `{"text": str, "categories": [..], "expected": [{"category","kind","start","end","confidence","reason"}], "note": str}`; `regression/clean_corpus/*.txt` (suffix `.txt` so black/flake8 ignore them); `regression.clean_corpus_lines()` is not needed, the tests read the directory directly.

- [ ] **Step 1: Write `regression/scan_vectors.json`**

At least three positives and three hard negatives per kind. Negatives are entries whose `expected` is `[]`. Build the file with a generator so offsets are right, then commit the JSON (the generator is not committed):

```python
# scratch: build vectors with computed offsets, then review each by eye
import json
from pygarble import Scanner

CASES = [
    # (categories, text, note) — expected is filled from the scanner and reviewed
    (["secrets"], "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "aws id"),
    (["secrets"], "token ghp_\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", "github"),
    (["secrets"], "postgres://admin:s3cr3t-pw@db.internal:5432/app", "url creds"),
    (["secrets"], "password = 'q8Zt3vP2xL9mK4nR'", "keyword entropy"),
    (["secrets"], "password = 'correcthorsebatterystaple'", "low entropy passphrase"),
    (["secrets"], "https://example.com:8080/path?x=1", "url with port, no creds"),
    (["secrets"], "commit 5f4dcc3b5aa765d61d8327deb882cf99 fixed the build", "git sha not flagged without context"),
    (["secrets"], "version 1.2.3-rc4+build.5", "version string"),
    (["pii"], "email jane.doe+news@example.co.uk now", "email"),
    (["pii"], "call (415) 555-2671 or +14155552671", "us phone and e164"),
    (["pii"], "card 4111 1111 1111 1111 exp 12/28", "visa"),
    (["pii"], "IBAN GB82 WEST 1234 5698 7654 32", "iban"),
    (["pii"], "ssn 123-45-6789", "ssn"),
    (["pii"], "NI AB 12 34 56 C", "nino"),
    (["pii"], "PAN ABCPE1234F", "pan"),
    (["pii"], "host 192.168.1.10 at 12:30:45 on 2026-09-26", "ipv4 with time and date"),
    (["pii"], "order 000123 shipped 2026-09-26 at 10:00", "no pii"),
    (["pii"], "4111 1111 1111 1112", "luhn fails"),
    (["pii"], "v1.2.3.4 released", "version not ip"),
    (["profanity"], "what the fuck", "strong"),
    (["profanity"], "well, damn.", "mild"),
    (["profanity"], "sh1t f*ck s.h.i.t", "leet, masked, spaced"),
    (["profanity"], "Scunthorpe assassin classic bass", "scunthorpe"),
    (["profanity"], "the analyst assessed the assets", "clean"),
    (["gibberish"], "qxzjkwpv bnmqwer zzxqv", "gibberish"),
    (["gibberish"], "hello world again", "clean"),
    (["secrets", "pii", "profanity", "gibberish"], "Contact a@b.co, key AKIAIOSFODNN7EXAMPLE, damn", "mixed"),
    (["secrets", "pii", "profanity", "gibberish"], "", "empty"),
    (["secrets", "pii", "profanity", "gibberish"], "   ", "whitespace"),
]
rows = []
for categories, text, note in CASES:
    report = Scanner(categories=categories).scan(text)
    rows.append({"text": text, "categories": categories, "note": note,
                 "expected": [f.to_dict() for f in report.findings]})
print(json.dumps(rows, indent=1, ensure_ascii=False))
```

Extend `CASES` until every kind in `Scanner`'s kind universe (`pygarble.scanner._all_kinds()`) has three positives and three negatives; the test below enforces it. Review every generated `expected` by eye against the spec before committing: the vectors pin behaviour, so a wrong finding accepted here becomes a bug the golden file will defend.

- [ ] **Step 2: Write `tests/test_scan_vectors.py`**

```python
"""Every vector reproduces exactly; every kind has positives and negatives."""

import json
from pathlib import Path

import pytest

from pygarble import Scanner
from pygarble.scanner import _all_kinds

VECTORS = Path(__file__).resolve().parent.parent / "regression" / "scan_vectors.json"

if not VECTORS.is_file():
    pytest.skip("regression/scan_vectors.json not present", allow_module_level=True)

ROWS = json.loads(VECTORS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", ROWS, ids=[r["note"] for r in ROWS])
def test_vector_reproduces(row):
    report = Scanner(categories=row["categories"]).scan(row["text"])
    assert [f.to_dict() for f in report.findings] == row["expected"]


def test_every_kind_has_three_positives_and_three_negatives():
    universe = set().union(*_all_kinds().values())
    positives = {kind: 0 for kind in universe}
    negatives = {kind: 0 for kind in universe}
    for row in ROWS:
        kinds = {f["kind"] for f in row["expected"]}
        for kind in universe:
            category = next(
                c for c, ks in _all_kinds().items() if kind in ks
            )
            if category not in row["categories"]:
                continue
            if kind in kinds:
                positives[kind] += 1
            else:
                negatives[kind] += 1
    short = {k: (positives[k], negatives[k]) for k in universe
             if positives[k] < 3 or negatives[k] < 3}
    assert not short, short
```

- [ ] **Step 3: Write the clean corpus and its test**

Each corpus file is 40 to 80 lines of realistic content with no secret, PII or profanity. `code.py.txt`: a small module with argparse, a dataclass, hex colour constants, a sha256 call, `timeout=30`, `password_field = "password"` (a field *name*, not a value), URLs with ports. `code.js.txt`: an Express handler with `req.headers.authorization`, a `const API_URL`, semver strings, a `uuid` literal `123e4567-e89b-12d3-a456-426614174000`. `config.json.txt`: `{"database": {"host": "db.internal", "port": 5432, "user": "app", "password": "${DB_PASSWORD}"}, "timeout": 30, "retries": 3, "version": "1.2.3"}` and similar. `logs.txt`: timestamped lines with request ids, durations, status codes, `GET /api/v1/users/42`, one IPv4 (`10.0.0.5`, expected at 0.7, below the bar). `prose.txt`: ordinary English paragraphs, a changelog excerpt, a product FAQ, and sentences with words like *assessment*, *classic*, *Essex*, *cocktail*, *Dickens*, *shiitake*, *pass*, *hello*.

`tests/test_clean_corpus.py`:

```python
"""Ordinary code, config, logs and prose never trip a high-confidence rule."""

from pathlib import Path

import pytest

from pygarble import Scanner

CORPUS = Path(__file__).resolve().parent.parent / "regression" / "clean_corpus"

if not CORPUS.is_dir():
    pytest.skip("regression/clean_corpus not present", allow_module_level=True)

FILES = sorted(CORPUS.glob("*.txt"))
BAR = 0.8
# (file name, line number) -> kinds allowed at or above BAR on that line.
EXCEPTIONS = {}


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_no_high_confidence_findings(path):
    scanner = Scanner(categories=["secrets", "pii", "profanity"])
    offenders = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        for finding in scanner.scan(line).findings:
            allowed = EXCEPTIONS.get((path.name, number), ())
            if finding.confidence >= BAR and finding.kind not in allowed:
                offenders.append((number, finding.kind, finding.confidence))
    assert offenders == []


def test_corpus_is_nontrivial():
    assert len(FILES) == 5
    for path in FILES:
        assert len(path.read_text(encoding="utf-8").splitlines()) >= 40
```

- [ ] **Step 4: Run, fix rules (not corpus) for real false positives, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_scan_vectors.py tests/test_clean_corpus.py -q`
Expected: pass. A corpus line that is genuinely innocent and still fires at 0.8+ is a rule bug: fix the rule in its module, add the line as a negative vector, and re-run that module's tests. Only add to `EXCEPTIONS` when the line really contains the thing (e.g. a deliberately planted IP), and say why in a comment.

```bash
git add regression/scan_vectors.json regression/clean_corpus tests/test_scan_vectors.py tests/test_clean_corpus.py
git commit -m "test: scan vectors per kind and a clean corpus false-positive gate

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: CLI `scan` and `redact`

**Files:**
- Modify: `pygarble/cli.py`
- Modify: `docs/cli.rst`
- Test: `tests/test_cli_scan.py`

**Interfaces:**
- Consumes: `build_parser`, `iter_lines`, `read_lines`, `field_problem`, `CliError`, `EXIT_*`, `main` from the existing CLI; `Scanner` from Task 3.
- Produces: subcommands `scan` and `redact`; `make_scanner(args) -> Scanner`; `run_scan(args, out, err) -> int`; `scan_row(text, report, show) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli_scan.py`:

```python
"""pygarble scan / redact command contracts."""

import io
import json

import pytest

from pygarble.cli import main

AWS = "AKIAIOSFODNN7EXAMPLE"


def run(capsys, argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def test_scan_text_format_and_exit_code(capsys):
    code, out, err = run(
        capsys, ["scan", "-t", f"key {AWS}", "-t", "hello world"]
    )
    assert code == 1
    assert out.splitlines() == [
        f"flagged\taws_access_key_id\tkey {AWS}",
        "clean\t\thello world",
    ]
    assert err == ""


def test_scan_clean_exits_zero(capsys):
    code, out, _ = run(capsys, ["scan", "-t", "hello world"])
    assert code == 0 and out == "clean\t\thello world\n"


def test_scan_tsv_and_jsonl(capsys):
    text = f"mail a@b.co key {AWS}"
    code, out, _ = run(capsys, ["scan", "--format", "tsv", "-t", text])
    assert out == f"1\t2\taws_access_key_id,email\t{text}\n"
    code, out, _ = run(capsys, ["scan", "--format", "jsonl", "-t", text])
    row = json.loads(out)
    assert row["text"] == text and row["flagged"] is True
    assert [f["kind"] for f in row["findings"]] == ["email", "aws_access_key_id"]
    assert "match" not in row["findings"][0]
    code, out, _ = run(
        capsys, ["scan", "--format", "jsonl", "--show-matches", "-t", text]
    )
    assert json.loads(out)["findings"][0]["match"] == "a@b.co"


def test_scan_category_kind_locale_and_confidence_flags(capsys):
    text = "07911 123456 and damn"
    _, out, _ = run(capsys, ["scan", "--categories", "pii", "-t", text])
    assert out.startswith("flagged\tphone\t")
    _, out, _ = run(
        capsys, ["scan", "--categories", "pii", "--locales", "us", "-t", text]
    )
    assert out.startswith("clean\t")
    _, out, _ = run(
        capsys,
        ["scan", "--categories", "pii,profanity", "--exclude-kinds", "phone",
         "-t", text],
    )
    assert out.startswith("flagged\tprofanity\t")
    _, out, _ = run(
        capsys,
        ["scan", "--categories", "pii,profanity", "--min-confidence", "0.9",
         "-t", text],
    )
    assert out.startswith("clean\t\t")
    code, _, err = run(capsys, ["scan", "--categories", "nope", "-t", "x"])
    assert code == 2 and "unknown category" in err


def test_scan_stdin_and_field_mode(capsys, monkeypatch):
    lines = "\n".join(
        [
            json.dumps({"id": 1, "msg": f"key {AWS}"}),
            "not json",
            json.dumps({"id": 2}),
            json.dumps({"id": 3, "msg": "fine"}),
        ]
    )
    code, out, err = run(
        capsys, ["scan", "--field", "msg"], stdin=lines, monkeypatch=monkeypatch
    )
    assert code == 2
    rows = [json.loads(line) for line in out.splitlines()]
    assert rows[0]["pygarble"]["flagged"] is True
    assert rows[0]["pygarble"]["findings"][0]["kind"] == "aws_access_key_id"
    assert "text" not in rows[0]["pygarble"]
    assert rows[1]["pygarble"]["flagged"] is False
    assert "line 2: invalid JSON" in err and "line 3:" in err


def test_redact_default_mask_and_partial(capsys):
    text = f"mail a@b.co card 4111 1111 1111 1111 key {AWS} damn"
    _, out, _ = run(capsys, ["redact", "-t", text])
    assert out == (
        "mail [EMAIL] card [CREDIT_CARD] key [AWS_ACCESS_KEY_ID] [PROFANITY]\n"
    )
    _, out, _ = run(capsys, ["redact", "--mode", "partial", "-t", text])
    assert "***************1111" in out and "[EMAIL]" not in out
    _, out, _ = run(
        capsys, ["redact", "--placeholder", "<{kind}>", "-t", text]
    )
    assert "<email>" in out
    with pytest.raises(SystemExit):  # argparse rejects bad choices
        run(capsys, ["redact", "--mode", "shred", "-t", text])


def test_redact_field_mode_rewrites_field(capsys, monkeypatch):
    line = json.dumps({"id": 7, "msg": "mail a@b.co"})
    code, out, _ = run(
        capsys, ["redact", "--field", "msg"], stdin=line, monkeypatch=monkeypatch
    )
    assert code == 0
    assert json.loads(out) == {"id": 7, "msg": "mail [EMAIL]"}


def test_redact_exit_zero_even_when_flagged(capsys):
    code, out, _ = run(capsys, ["redact", "-t", f"key {AWS}"])
    assert code == 0 and out == "key [AWS_ACCESS_KEY_ID]\n"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python -m pytest tests/test_cli_scan.py -q`
Expected: argparse error (`invalid choice: 'scan'`) → SystemExit.

- [ ] **Step 3: Implement in `pygarble/cli.py`**

Add after `build_parser`'s existing subcommands (inside `build_parser`, before `return parser`):

```python
    def add_scan_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("inputs", nargs="*", help="files, or - for stdin")
        sub.add_argument(
            "-t", "--text", action="append", help="evaluate this text"
        )
        sub.add_argument("--field", default=None, help="JSON field to scan")
        sub.add_argument(
            "--categories", default=None, help="comma list; default all"
        )
        sub.add_argument("--kinds", default=None, help="comma list of kinds")
        sub.add_argument("--exclude-kinds", default=None)
        sub.add_argument("--locales", default=None, help="comma list; us,uk,in")
        sub.add_argument("--min-confidence", type=float, default=0.5)
        sub.add_argument("--profile", default="english")
        sub.add_argument("--threshold", type=float, default=0.5)
        sub.add_argument("--allowlist", default=None)

    scan_parser = subparsers.add_parser("scan", help="find secrets, PII, profanity")
    add_scan_common(scan_parser)
    scan_parser.add_argument(
        "--format", choices=["text", "tsv", "jsonl"], default="text"
    )
    scan_parser.add_argument(
        "--show-matches", action="store_true", help="include matched text"
    )
    redact_parser = subparsers.add_parser("redact", help="print redacted text")
    add_scan_common(redact_parser)
    redact_parser.add_argument(
        "--mode", choices=["placeholder", "mask", "partial"], default="placeholder"
    )
    redact_parser.add_argument("--placeholder", default="[{KIND}]")
    redact_parser.add_argument("--mask-char", default="*")
```

Add these functions after `run_calibrate`:

```python
def _split(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def make_scanner(args: argparse.Namespace) -> Any:
    from .scanner import DEFAULT_CATEGORIES, Scanner

    allowlist = load_allowlist(args.allowlist) if args.allowlist else None
    try:
        return Scanner(
            _split(args.categories) or DEFAULT_CATEGORIES,
            min_confidence=args.min_confidence,
            kinds=_split(args.kinds),
            exclude_kinds=_split(args.exclude_kinds) or (),
            locales=_split(args.locales) or ("us", "uk", "in"),
            profile=args.profile,
            threshold=args.threshold,
            allowlist=allowlist,
        )
    except ValueError as error:
        raise CliError(str(error))


def scan_row(text: str, report: Any, show: bool) -> Dict[str, Any]:
    findings = []
    for finding in report.findings:
        row = finding.to_dict()
        if show:
            row["match"] = text[finding.start : finding.end]
        findings.append(row)
    return {"flagged": report.flagged, "findings": findings}


def format_scan(fmt: str, text: str, report: Any, show: bool) -> str:
    kinds = ",".join(report.kinds())
    if fmt == "text":
        label = "flagged" if report.flagged else "clean"
        return f"{label}\t{kinds if report.flagged else ''}\t{text}"
    if fmt == "tsv":
        return f"{int(report.flagged)}\t{len(report.findings)}\t{kinds}\t{text}"
    row = {"text": text}
    row.update(scan_row(text, report, show))
    return json.dumps(row, ensure_ascii=False)


def run_scan(args: argparse.Namespace, out: Any, err: Any) -> int:
    scanner = make_scanner(args)
    redacting = args.command == "redact"
    any_flagged = False
    had_error = False
    if args.text is not None:
        pairs: Iterable[Tuple[int, Any]] = enumerate(args.text, 1)
    else:
        pairs = enumerate(iter_lines(args.inputs), 1)

    def render(value: str) -> str:
        try:
            if redacting:
                return scanner.redact(
                    value,
                    mode=args.mode,
                    placeholder=args.placeholder,
                    mask_char=args.mask_char,
                ).text
            return ""
        except ValueError as error:
            raise CliError(str(error))

    for number, line in pairs:
        if args.field is None:
            if redacting:
                out.write(render(line) + "\n")
                continue
            report = scanner.scan(line)
            any_flagged = any_flagged or report.flagged
            out.write(
                format_scan(args.format, line, report, args.show_matches) + "\n"
            )
            continue
        problem: Optional[str] = None
        try:
            obj = json.loads(line)
        except ValueError as error:
            problem = f"invalid JSON: {error}"
        else:
            problem = field_problem(obj, args.field)
        if problem is not None:
            err.write(f"line {number}: {problem}\n")
            had_error = True
            continue
        value = obj[args.field]
        if redacting:
            obj[args.field] = render(value)
        else:
            report = scanner.scan(value)
            any_flagged = any_flagged or report.flagged
            obj["pygarble"] = scan_row(value, report, args.show_matches)
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if had_error:
        return EXIT_ERROR
    if not redacting and any_flagged:
        return EXIT_GARBLED
    return EXIT_OK
```

In `main`, route the new commands before the default:

```python
        if args.command == "calibrate":
            return run_calibrate(args, out)
        if args.command in ("scan", "redact"):
            return run_scan(args, out, sys.stderr)
        return run_texts(args, out, sys.stderr)
```

`--mode shred` is rejected by argparse itself (SystemExit with code 2), which is the established behaviour for bad flags in this CLI; the test expects `SystemExit`.

- [ ] **Step 4: Document in `docs/cli.rst`**

Append a section:

```rst
Screening: ``scan`` and ``redact``
----------------------------------

``pygarble scan`` runs the :class:`~pygarble.Scanner` over each input line
and prints one row per line. ``pygarble redact`` prints the redacted line.

.. code-block:: bash

   printf 'mail a@b.co\nhello\n' | pygarble scan
   printf 'key AKIAIOSFODNN7EXAMPLE\n' | pygarble redact --mode mask
   pygarble scan --categories secrets,pii --format jsonl records.jsonl
   pygarble redact --field message --mode partial events.jsonl

Options shared by both: ``--categories``, ``--kinds``, ``--exclude-kinds``,
``--locales`` (comma lists), ``--min-confidence``, ``--profile``,
``--threshold`` and ``--allowlist`` for the gibberish category, and
``--field NAME`` to read a JSON object per line. ``scan`` adds
``--format text|tsv|jsonl`` and ``--show-matches`` (matched text is omitted
by default so logs stay clean). ``redact`` adds ``--mode
placeholder|mask|partial``, ``--placeholder`` (fields ``{KIND}``, ``{kind}``,
``{category}``) and ``--mask-char``.

Exit codes: ``scan`` returns 1 when any line was flagged; ``redact`` returns
0; both return 2 on bad input or options.
```

- [ ] **Step 5: Run, lint, full suite, commit**

Run: `PYTHONPATH=. python -m pytest tests/test_cli_scan.py tests/test_cli.py -q && black pygarble tests && isort pygarble tests && flake8 pygarble tests && mypy pygarble && PYTHONPATH=. python -m pytest -q -W error::FutureWarning`

```bash
git add pygarble/cli.py docs/cli.rst tests/test_cli_scan.py
git commit -m "feat: pygarble scan and redact subcommands

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: JSON rule tables and manifest

**Files:**
- Modify: `scripts/generate_data.py`
- Modify: `pyproject.toml` (`exclude-package-data`, `force-exclude` unchanged)
- Create: `pygarble/data/secrets.json`, `pygarble/data/pii.json`, `pygarble/data/profanity.json` (generated)
- Modify: `pygarble/data/manifest.json` (regenerated)
- Test: `tests/test_data_json.py` (extend), `tests/test_packaging.py` (extend)

**Interfaces:**
- Consumes: `export()` from `pygarble.secrets.patterns`, `pygarble.pii.patterns`, `pygarble.profanity.wordlist`; existing `write_json`.
- Produces: the three JSON files, manifest `files` entries for them, `counts` gains `secret_patterns`, `pii_rules`, `profanity_strong`, `profanity_mild`.

- [ ] **Step 1: Extend tests**

Append to `tests/test_data_json.py`:

```python
def test_rule_tables_match_python_sources():
    from pygarble.pii.patterns import export as pii_export
    from pygarble.profanity.wordlist import export as profanity_export
    from pygarble.secrets.patterns import export as secrets_export

    for name, export in (
        ("secrets.json", secrets_export),
        ("pii.json", pii_export),
        ("profanity.json", profanity_export),
    ):
        path = DATA / name
        if not path.is_file():
            pytest.skip(f"{name} is not shipped in wheels")
        assert json.loads(path.read_text(encoding="utf-8")) == export()


def test_manifest_hashes_rule_tables():
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name in ("secrets.json", "pii.json", "profanity.json"):
        if (DATA / name).is_file():
            assert name in manifest["files"]
```

`DATA` and `json`/`pytest` are already defined in that module (check the top of the file; add them if not). Append to `tests/test_packaging.py`, next to the existing wheel-content assertion, that `secrets.json`, `pii.json` and `profanity.json` are absent from the built wheel and present in the sdist (mirror the existing assertions for `words.json`).

- [ ] **Step 2: Extend `scripts/generate_data.py`**

After `write_json(sorted(trigrams), directory / "trigrams.json")`:

```python
        from pygarble.pii.patterns import export as pii_export
        from pygarble.profanity.wordlist import export as profanity_export
        from pygarble.secrets.patterns import export as secrets_export

        secrets_table = secrets_export()
        pii_table = pii_export()
        profanity_table = profanity_export()
        write_json(secrets_table, directory / "secrets.json")
        write_json(pii_table, directory / "pii.json")
        write_json(profanity_table, directory / "profanity.json")
```

and in `manifest["counts"]` add:

```python
                "secret_patterns": len(secrets_table["known"]),
                "pii_rules": len(pii_table["generic"])
                + sum(len(r) for r in pii_table["locales"].values()),
                "profanity_strong": len(profanity_table["strong"]),
                "profanity_mild": len(profanity_table["mild"]),
```

The script imports `pygarble` already (check the top for `sys.path` handling; if it imports `DEFAULT_LOG_PROB` from the package, the path is set). If a `--source` pinned file is needed offline, use `scripts/data_curation.json`'s `source_url` as before; CI already runs `--check` online.

In `pyproject.toml`:

```toml
[tool.setuptools.exclude-package-data]
"pygarble.data" = ["words.json", "bigrams.json", "trigrams.json", "secrets.json", "pii.json", "profanity.json"]
```

- [ ] **Step 3: Generate, verify, commit**

Run: `python scripts/generate_data.py && python scripts/generate_data.py --check && PYTHONPATH=. python -m pytest tests/test_data_json.py tests/test_packaging.py -q && black --check scripts && flake8 scripts`
Expected: three new JSON files, manifest updated, `--check` prints the reproducible message, tests pass. The `.py` word tables must be byte-identical to before (`git diff --stat pygarble/data/*.py` shows nothing).

```bash
git add scripts/generate_data.py pyproject.toml pygarble/data/secrets.json pygarble/data/pii.json pygarble/data/profanity.json pygarble/data/manifest.json tests/test_data_json.py tests/test_packaging.py
git commit -m "feat: language-neutral JSON copies of the secrets, PII and profanity tables

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Golden scan corpus, CI, throughput script

**Files:**
- Create: `regression/golden_scan.py`, `regression/golden_scan.jsonl`, `regression/golden_scan.sha256`, `regression/throughput.py`
- Modify: `.github/workflows/test.yml`
- Test: `tests/test_golden_scan.py`, `tests/test_throughput_smoke.py`

**Interfaces:**
- Consumes: `regression/golden.py` (`texts()`), `regression/scan_vectors.json`, `regression/clean_corpus/*.txt`, `Scanner`.
- Produces: `golden_scan.rows()`, `render()`, `check()`, `main()` with `--write/--check` (same contract as `golden.py`); `throughput.build_corpus(size_bytes) -> List[str]`, `measure(categories, lines) -> dict`, `main()` with `--json` and `--size-mb`.

- [ ] **Step 1: Write `regression/golden_scan.py`**

```python
"""Frozen Scanner outputs. Rows carry a text hash and findings, no text.

Inputs: every scan vector, every clean-corpus line, and the gibberish golden
inputs, scanned with all four categories at min_confidence 0.5 and
locales us,uk,in. Regenerate with --write only for an intended behaviour
change; --check runs in CI.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import Scanner  # noqa: E402
from regression.golden import texts as gibberish_texts  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "golden_scan.jsonl"
CHECKSUM = ROOT / "golden_scan.sha256"
VECTORS = ROOT / "scan_vectors.json"
CORPUS = ROOT / "clean_corpus"


def texts():
    seen = []
    for row in json.loads(VECTORS.read_text(encoding="utf-8")):
        seen.append(("vector", row["text"]))
    for path in sorted(CORPUS.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            seen.append((path.name, line))
    for text in gibberish_texts():
        seen.append(("golden", text))
    unique = {}
    for source, text in seen:
        unique.setdefault(text, source)
    return [(source, text) for text, source in unique.items()]


def rows():
    scanner = Scanner()
    for source, text in texts():
        report = scanner.scan(text)
        yield {
            "source": source,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "length": report.length,
            "flagged": report.flagged,
            "findings": [
                [f.category, f.kind, f.start, f.end, round(f.confidence, 12), f.reason]
                for f in report.findings
            ],
        }


def render():
    return "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        for row in rows()
    )


def check():
    for path in (OUTPUT, CHECKSUM):
        if not path.is_file():
            print(f"missing {path.name}; run --write first", file=sys.stderr)
            return 1
    expected = OUTPUT.read_text(encoding="utf-8")
    if hashlib.sha256(expected.encode("utf-8")).hexdigest() != (
        CHECKSUM.read_text().strip()
    ):
        print("golden_scan.jsonl does not match golden_scan.sha256", file=sys.stderr)
        return 1
    actual = render()
    if actual == expected:
        print(f"golden scan corpus reproduced ({actual.count(chr(10))} rows)")
        return 0
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    if len(expected_lines) != len(actual_lines):
        print(
            f"row count changed: {len(expected_lines)} committed, "
            f"{len(actual_lines)} generated",
            file=sys.stderr,
        )
    shown = 0
    for old, new in itertools.zip_longest(
        expected_lines, actual_lines, fillvalue="<missing>"
    ):
        if old != new:
            print(f"- {old}\n+ {new}", file=sys.stderr)
            shown += 1
            if shown == 10:
                break
    print("golden scan corpus differs; run --write if intended", file=sys.stderr)
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

`tests/test_golden_scan.py` mirrors `tests/test_golden.py`: skip when `regression/` is absent, import `regression.golden_scan`, assert `render()` equals the committed file and the checksum matches.

- [ ] **Step 2: Write `regression/throughput.py`**

```python
"""Throughput of Scanner per category on a synthetic corpus. Numbers are
published in the README with the machine noted; nothing is promised."""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble import Scanner  # noqa: E402
from pygarble.findings import CATEGORIES  # noqa: E402

PARAGRAPHS = [
    "The quarterly report covers revenue, churn and the hiring plan for the "
    "next two quarters. Please read the summary before the meeting.",
    "To rotate the logs, set the handler to RotatingFileHandler with a "
    "maximum size of ten megabytes and keep five backups.",
    "Customers can update their shipping address from the account page. "
    "Changes apply to orders that have not yet been dispatched.",
    "def parse(row): return row.split(',')  # naive CSV split used in tests",
    "GET /api/v1/users/42 200 12ms request_id=7f3a9c",
]
PLANTED = [
    "contact jane.doe@example.com for details",
    "card 4111 1111 1111 1111 on file",
    "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
    "what the fuck happened here",
    "qxzjkwpv bnmqwer zzxqv",
]


def build_corpus(size_bytes: int, seed: int = 7) -> List[str]:
    rng = random.Random(seed)
    lines: List[str] = []
    total = 0
    while total < size_bytes:
        line = rng.choice(PARAGRAPHS)
        if rng.random() < 0.05:
            line = line + " " + rng.choice(PLANTED)
        lines.append(line)
        total += len(line.encode("utf-8")) + 1
    return lines


def measure(categories: List[str], lines: List[str]) -> Dict[str, float]:
    scanner = Scanner(categories=categories)
    size = sum(len(line.encode("utf-8")) + 1 for line in lines)
    findings = 0
    start = time.perf_counter()
    for line in lines:
        findings += len(scanner.scan(line).findings)
    elapsed = time.perf_counter() - start
    return {
        "mb_per_s": (size / 1e6) / elapsed if elapsed else float("inf"),
        "lines_per_s": len(lines) / elapsed if elapsed else float("inf"),
        "findings": findings,
        "seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mb", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    lines = build_corpus(int(args.size_mb * 1e6))
    results = {}
    for category in CATEGORIES:
        results[category] = measure([category], lines)
    results["all"] = measure(list(CATEGORIES), lines)
    results["rules_only"] = measure(["secrets", "pii", "profanity"], lines)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    print(f"{'category':<12}{'MB/s':>10}{'lines/s':>12}{'findings':>10}")
    for name, row in results.items():
        print(
            f"{name:<12}{row['mb_per_s']:>10.1f}{row['lines_per_s']:>12.0f}"
            f"{row['findings']:>10}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`tests/test_throughput_smoke.py`: build a 200 KB corpus, run `measure` for `rules_only`, assert it finishes in under 10 seconds and `findings > 0`; also scan one 1 MB line of prose with `Scanner(categories=["secrets","pii","profanity"])` and assert it finishes in under 5 seconds (linear-time guard).

- [ ] **Step 3: CI and generation**

In `.github/workflows/test.yml`, after the `golden.py --check` step add:

```yaml
    - name: Golden scan corpus
      run: python regression/golden_scan.py --check
```

Run: `python regression/golden_scan.py --write && python regression/golden_scan.py --check && python regression/throughput.py --size-mb 2 && PYTHONPATH=. python -m pytest tests/test_golden_scan.py tests/test_throughput_smoke.py -q && black --check regression tests && isort --check-only regression tests && flake8 regression tests`

Record the throughput table output in the ledger; Task 12 pastes it into the README.

```bash
git add regression/golden_scan.py regression/golden_scan.jsonl regression/golden_scan.sha256 regression/throughput.py .github/workflows/test.yml tests/test_golden_scan.py tests/test_throughput_smoke.py
git commit -m "test: golden scan corpus with CI check; throughput script

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Docs, README, release metadata

**Files:**
- Create: `docs/screening.rst`, `docs/secrets.rst`, `docs/pii.rst`, `docs/profanity.rst`
- Modify: `docs/index.rst`, `docs/api.rst`, `docs/contributing.rst`, `docs/migration.rst`, `README.md`, `CHANGELOG.md`, `pyproject.toml`, `pygarble/__init__.py` (`__version__`)
- Test: existing `tests/test_docs_snippets.py` (covers new pages automatically), `tests/test_packaging.py` version assertions if any.

- [ ] **Step 1: New docs pages**

`docs/screening.rst`:

```rst
Screening and redaction
=======================

:class:`pygarble.Scanner` runs every enabled category over a text and
returns a :class:`pygarble.ScanReport`. Findings say what was found and
where; they never contain the matched text, so a logged report cannot leak
a secret.

.. code-block:: python

   from pygarble import Scanner

   scanner = Scanner()
   report = scanner.scan("mail jane@example.com, key AKIAIOSFODNN7EXAMPLE")
   assert report.flagged
   assert report.kinds() == ("aws_access_key_id", "email")
   assert scanner.redact("mail jane@example.com").text == "mail [EMAIL]"

Categories
----------

``secrets``, ``pii``, ``profanity`` and ``gibberish``. Pass a subset to the
constructor to run fewer. ``kinds`` and ``exclude_kinds`` select individual
kinds across categories; ``locales`` selects PII locale packs (``us``,
``uk``, ``in``); ``profile``, ``threshold`` and ``allowlist`` configure the
gibberish category exactly like :class:`pygarble.EnsembleDetector`.

Confidence tiers
----------------

Each rule has a fixed confidence. ``flagged`` is true when any finding
reaches ``min_confidence`` (default 0.5); lower findings are still returned.

.. list-table::
   :header-rows: 1

   * - Tier
     - Meaning
   * - 1.0
     - checksum verified or unique vendor prefix
   * - 0.9
     - unambiguous structure without a checksum
   * - 0.8
     - structural with some ambiguity; elongated or spaced profanity
   * - 0.7
     - weak or context dependent (IP addresses, mild profanity)
   * - 0.6
     - contextual entropy or ambiguous masking
   * - score
     - the gibberish ensemble score

Redaction
---------

``redact`` replaces every finding at or above ``min_confidence`` in the
chosen categories (all but gibberish by default). Overlapping findings
become one region. Modes: ``placeholder`` (``[EMAIL]``; template fields
``{KIND}``, ``{kind}``, ``{category}``), ``mask`` (length preserving) and
``partial`` (keeps the last four characters of cards, phones, IBANs and
national numbers).

.. code-block:: python

   from pygarble import redact

   assert redact("card 4111 1111 1111 1111", mode="partial").text == (
       "card ***************1111"
   )

What it does not catch
----------------------

Names, postal addresses, free-text dates of birth, hate speech beyond a
word list, and secrets without a recognisable shape. Send those to a model
after this pass.
```

`docs/secrets.rst`, `docs/pii.rst`, `docs/profanity.rst`: each a short page with the kind table from the spec (kind, shape, confidence), one runnable snippet using the category's `detect`, and for profanity the attribution paragraph from `pygarble/profanity/wordlist.py`. In `docs/index.rst` add `screening`, `secrets`, `pii`, `profanity` after `calibration` in the toctree.

- [ ] **Step 2: `docs/api.rst`, `docs/contributing.rst`, `docs/migration.rst`**

`api.rst`: change "This reference describes the 0.10.0 API" to 0.11.0; add sections `Scanner` (autoclass `pygarble.scanner.Scanner`, members `scan, scan_batch, iter_scan, redact`), `Findings` (autoclass `pygarble.findings.Finding`, `ScanReport`, `Redaction`), `Detectors` (autoclass `pygarble.secrets.SecretsDetector`, `pygarble.pii.PIIDetector`, `pygarble.profanity.ProfanityDetector`, members `detect`), and functions `pygarble.scan`, `pygarble.redact`.

`contributing.rst`: add a paragraph listing `regression/scan_vectors.json`, `regression/clean_corpus/`, `python regression/golden_scan.py --check` and `python regression/throughput.py`, and that rule tables live in `pygarble/secrets/patterns.py`, `pygarble/pii/patterns.py`, `pygarble/profanity/wordlist.py` with JSON copies regenerated by `scripts/generate_data.py`.

`migration.rst`: add under the existing title:

```rst
0.11.0
------

No breaking changes. New: :class:`pygarble.Scanner`, :func:`pygarble.scan`,
:func:`pygarble.redact`, the ``secrets``, ``pii`` and ``profanity``
detectors, and the ``pygarble scan`` / ``pygarble redact`` commands. The
gibberish API is unchanged.
```

- [ ] **Step 3: README**

Replace the title line and pitch with:

```markdown
# pygarble

**A deterministic, zero-dependency first line of defence for text: secrets, PII, profanity and gibberish, with redaction. Pure Python, milliseconds per call, explainable findings.**
```

Keep the badges and links. Replace "Why pygarble" bullets with five: zero dependencies and no model downloads; deterministic and explainable (every finding has a kind, span and reason; findings never carry the matched text); four categories in one call with confidence tiers; redaction in three modes; a CLI for pipelines. Replace "Ten-second start" with:

```bash
python -m pip install pygarble
printf 'mail jane@example.com\nhello\n' | pygarble scan
printf 'key AKIAIOSFODNN7EXAMPLE\n' | pygarble redact
```

```python
from pygarble import redact, scan

report = scan("mail jane@example.com, key AKIAIOSFODNN7EXAMPLE, damn")
assert report.kinds() == ("aws_access_key_id", "email", "profanity")
assert redact("mail jane@example.com").text == "mail [EMAIL]"
```

Add a "What it catches" table (category, kinds, how) and a "What it doesn't" line pointing at NLP. Add a "Throughput" table pasted from Task 11's ledger output with the machine noted (`python regression/throughput.py`). Move the existing gibberish material (profiles table, calibration, use cases) under a `## Gibberish detection` heading, condensed; keep every code block runnable.

- [ ] **Step 4: Release metadata**

`pygarble/__init__.py`: `__version__ = "0.11.0"`.

`pyproject.toml`: `description = "Deterministic, zero-dependency text screening: secrets, PII, profanity and gibberish, with redaction"`; keywords add `"pii", "secrets", "profanity", "redaction", "guardrails", "llm"`; classifiers add `"Topic :: Security"`.

`CHANGELOG.md`, above `## [0.10.0]`:

```markdown
## [0.11.0] - 2026-09-26

### Added
- `Scanner`, `scan()` and `redact()`: one call that screens text for
  secrets, PII, profanity and gibberish and returns findings with kind,
  span, confidence and reason. Findings never carry the matched text.
- Secrets detector: known vendor prefixes (AWS, GitHub, GitLab, Slack,
  Stripe, Google, OpenAI, Anthropic, Hugging Face, npm, PyPI, SendGrid),
  JWTs, private key blocks, credentials in URLs, bearer tokens, and
  keyword-plus-entropy generic secrets.
- PII detector: email, phone, credit card (Luhn), IBAN (mod-97), IPv4/IPv6,
  plus locale packs for the US (SSN), UK (National Insurance and NHS
  numbers) and India (Aadhaar with Verhoeff, PAN).
- Profanity detector: an attributed word list with leetspeak, elongation,
  embedded, masked and spaced obfuscation handling and an allowlist.
- Redaction in placeholder, mask and partial modes.
- `pygarble scan` and `pygarble redact` commands.
- JSON copies of the rule tables for ports, a golden scan corpus with a CI
  check, a clean-corpus false-positive gate and a throughput script.

### Notes
- The gibberish API is unchanged. This release is additive.
```

- [ ] **Step 5: Full gate, commit**

Run:

```bash
black --check pygarble tests scripts regression && isort --check-only pygarble tests scripts regression && flake8 pygarble tests scripts regression && mypy pygarble && PYTHONPATH=. python -m pytest -q -W error::FutureWarning && python scripts/update_strategy_docs.py --check && python scripts/generate_data.py --check && python regression/golden.py --check && python regression/golden_scan.py --check && (cd docs && python -m sphinx -W -b html . _build/html) && python -m build && PYTHONPATH=. python -m pygarble --version
```

Expected: all green; version prints `pygarble 0.11.0`.

```bash
git add docs README.md CHANGELOG.md pyproject.toml pygarble/__init__.py
git commit -m "docs: screening, secrets, PII and profanity guides; README pitch; release 0.11.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
