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
        r"(?i:aws)(?:.{0,20}?)(?i:secret|key)[^A-Za-z0-9/+\n]{0,5}"
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
        ["ghp_short", "the ghp_ prefix alone"],
    ),
    _entry(
        "gitlab_token",
        r"glpat-[A-Za-z0-9_\-]{20,}",
        1.0,
        ["glpat-\u0041bCdEfGhIjKlMnOpQrSt"],
        ["glpat-short"],
    ),
    _entry(
        "slack_token",
        r"xox[abprs]-[0-9A-Za-z\-]{10,}",
        1.0,
        ["xoxb-\u003123456789012-abcdefghijkl"],
        ["xoxz-123456789012-abcdefghijkl", "xoxb-short"],
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
        ["sk_test_short"],
    ),
    _entry(
        "google_api_key",
        r"AIza[0-9A-Za-z_\-]{35}",
        1.0,
        ["AIza\u0053yA1234567890abcdefghijklmnopqrstuv"],
        ["AIza too short"],
    ),
    _entry(
        "openai_api_key",
        r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"
        r"T3BlbkFJ[A-Za-z0-9_\-]{20,}",
        1.0,
        ["sk-proj-\u0061bcdefghijklmnopqrstuvT3BlbkFJabcdefghijklmnopqrstuv"],
        ["sk-proj-\u0061bcdefghijklmnopqrstuvwxyz"],
    ),
    _entry(
        "openai_api_key",
        r"sk-[A-Za-z0-9]{48}",
        0.9,
        ["sk-" + "a" * 20 + "B" * 20 + "0" * 8],
        ["sk-\u0061nt-api03-" + "a" * 80],
    ),
    _entry(
        "anthropic_api_key",
        r"sk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_\-]{80,}",
        1.0,
        ["sk-\u0061nt-api03-" + "a" * 90],
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
        ["SG.short.key"],
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
        SHORT_CLASSES,
        SHORT_ENTROPY,
        SHORT_MAX_LENGTH,
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
            "short_max_length": SHORT_MAX_LENGTH,
            "short_entropy": SHORT_ENTROPY,
            "short_classes": SHORT_CLASSES,
        },
        "placeholders": {
            "words": sorted(PLACEHOLDER_WORDS),
            "hints": list(PLACEHOLDER_HINTS),
        },
    }
