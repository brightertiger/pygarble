"""Known-prefix and contextual-entropy secret detection."""

import re
import time

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
    bad = JWT.replace(
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "eyJxxxxxxxxxxxx"
    )
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
    blob = (
        "5f4dcc3b5aa765d61d8327deb882cf99" + "9a1b2c3d4e5f60718293a4b5c6d7e8f9"
    )
    assert detect(blob) == ()
    (finding,) = detect(blob, without_context=True)
    assert finding.kind == "high_entropy_string"
    assert finding.confidence == 0.5
    both = detect(f"{AWS} {blob}", without_context=True)
    assert [f.kind for f in both] == [
        "aws_access_key_id",
        "high_entropy_string",
    ]


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


@pytest.mark.parametrize(
    "value, expected",
    [
        ("q8Zt3vP2xL9mK4nR", True),
        ("8f3a9c2e1b7d4f6a", True),
        ("abcdefgh", False),
        ("correcthorsebatterystaple", False),
        ("xxxxxxxxxxxxxxxx", False),
    ],
)
def test_looks_secret_short_and_long_regimes(value, expected):
    assert looks_secret(value) is expected


def test_short_values_flag_only_with_keyword_and_class_mix():
    assert [f.kind for f in detect("api_key = 8f3a9c2e1b7d4f6a")] == [
        "generic_secret"
    ]
    assert detect("password = abcdefgh") == ()


def test_export_carries_short_value_rule():
    limits = export()["limits"]
    assert limits["short_max_length"] == 22
    assert limits["short_entropy"] == 3.0
    assert limits["short_classes"] == 3


@pytest.mark.parametrize(
    "text",
    [
        "a." * 50000,
        "a-" * 50000,
        "a_" * 50000,
        "com.example.service.module." * 4000,
        "-----BEGIN PRIVATE KEY-----\n" * 3500,
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: x (y)\n" * 2000,
        "password=" * 12000,
        "password" * 12000,
    ],
)
def test_adversarial_inputs_scan_in_linear_time(text):
    detector = SecretsDetector()
    started = time.perf_counter()
    detector.detect(text)
    assert time.perf_counter() - started < 0.5


def test_truncated_private_key_stops_at_blank_line():
    text = (
        "cfg\n-----BEGIN RSA PRIVATE KEY-----\n"
        "Proc-Type: 4,ENCRYPTED\nMIIEowIBAAKCAQEA\nAB==\n\nnext"
    )
    (finding,) = detect(text)
    assert finding.kind == "private_key"
    assert finding.start == text.index("-----BEGIN")
    assert finding.end == text.index("\n\nnext")


def test_truncated_private_key_at_end_of_text():
    text = "-----BEGIN PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n"
    (finding,) = detect(text)
    assert (finding.start, finding.end) == (0, len(text) - 1)


def test_pgp_armor_headers_stay_inside_the_span():
    text = (
        "key:\n-----BEGIN PGP PRIVATE KEY BLOCK-----\n"
        "Version: GnuPG v2.0.22 (GNU/Linux)\n"
        "Comment: user@example.com\n\nlQOYBFx0AB==\n=Xy9z\n"
        "-----END PGP PRIVATE KEY BLOCK-----\ntail"
    )
    (finding,) = detect(text)
    assert finding.kind == "private_key"
    assert finding.start == text.index("-----BEGIN")
    assert finding.end == text.index("\ntail")


@pytest.mark.parametrize(
    "text",
    [
        "DB_PASSWORD=q8Zt3vP2xL9mK4nR",
        "GITHUB_TOKEN=q8Zt3vP2xL9mK4nR",
        "SECRET_KEY = 'q8Zt3vP2xL9mK4nR'",
        "export MY_API_KEY=q8Zt3vP2xL9mK4nR",
        "aws_secret = q8Zt3vP2xL9mK4nR",
    ],
)
def test_env_style_keyword_names(text):
    (finding,) = detect(text)
    assert finding.kind == "generic_secret"
    assert text[finding.start : finding.end] == "q8Zt3vP2xL9mK4nR"


def test_generic_value_stops_at_query_and_call_syntax():
    text = "password=q8Zt3vP2xL9mK4nR&user=bob"
    (finding,) = detect(text)
    assert text[finding.start : finding.end] == "q8Zt3vP2xL9mK4nR"
    assert detect("token = generate_token(user_id)") == ()


def test_kind_arguments_reject_bare_strings():
    message = "kinds must be an iterable of kind names, not a string"
    with pytest.raises(ValueError, match=message):
        SecretsDetector(kinds="jwt")
    with pytest.raises(ValueError, match=message):
        SecretsDetector(exclude_kinds="jwt")


def test_keyword_finding_dropped_when_known_prefix_covers_value():
    aws_value = "Hk9pQ2wR7tY4uI1oP3aS6dF8gJ0kL5zX2cV7bN4m"
    text = "AWS_SECRET_KEY: " + aws_value
    found = detect(text)
    assert [(f.kind, text[f.start : f.end]) for f in found] == [
        ("aws_secret_access_key", aws_value)
    ]
    gho = "gho_" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0Pp9Oo8Nn"
    text = "GH_TOKEN=" + gho
    found = detect(text)
    assert [(f.kind, text[f.start : f.end]) for f in found] == [
        ("github_token", gho)
    ]
    # With no known prefix the keyword finding still stands.
    assert [f.kind for f in detect("password = 'q8Zt3vP2xL9mK4nR'")] == [
        "generic_secret"
    ]
