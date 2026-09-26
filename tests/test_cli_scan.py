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
    assert [f["kind"] for f in row["findings"]] == [
        "email",
        "aws_access_key_id",
    ]
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
        [
            "scan",
            "--categories",
            "pii,profanity",
            "--exclude-kinds",
            "phone",
            "-t",
            text,
        ],
    )
    assert out.startswith("flagged\tprofanity\t")
    _, out, _ = run(
        capsys,
        [
            "scan",
            "--categories",
            "pii,profanity",
            "--min-confidence",
            "0.9",
            "-t",
            text,
        ],
    )
    assert out.startswith("clean\t\t")
    code, _, err = run(capsys, ["scan", "--categories", "nope", "-t", "x"])
    assert code == 2 and "unknown category" in err


def test_scan_bad_locale_and_kind_exit_two(capsys):
    code, out, err = run(capsys, ["scan", "--locales", "fr", "-t", "x"])
    assert code == 2 and out == "" and "unknown locale" in err
    code, out, err = run(capsys, ["redact", "--kinds", "nope", "-t", "x"])
    assert code == 2 and out == "" and "unknown kind" in err


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
        capsys,
        ["scan", "--field", "msg"],
        stdin=lines,
        monkeypatch=monkeypatch,
    )
    assert code == 2
    rows = [json.loads(line) for line in out.splitlines()]
    assert rows[0]["pygarble"]["flagged"] is True
    assert rows[0]["pygarble"]["findings"][0]["kind"] == "aws_access_key_id"
    assert "text" not in rows[0]["pygarble"]
    assert rows[1]["pygarble"]["flagged"] is False
    assert "line 2: invalid JSON" in err and "line 3:" in err


def test_scan_reads_files(capsys, tmp_path):
    path = tmp_path / "in.txt"
    path.write_text(f"key {AWS}\nhello world\n", encoding="utf-8")
    code, out, _ = run(capsys, ["scan", "--format", "tsv", str(path)])
    assert code == 1
    assert out.splitlines()[1] == "0\t0\t\thello world"


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


def test_redact_mask_char_and_bad_placeholder(capsys):
    _, out, _ = run(
        capsys, ["redact", "--mode", "mask", "--mask-char", "#", "-t", AWS]
    )
    assert out == "#" * len(AWS) + "\n"
    code, out, err = run(
        capsys, ["redact", "--placeholder", "{nope}", "-t", AWS]
    )
    assert code == 2 and out == "" and err.startswith("pygarble: ")


def test_redact_field_mode_rewrites_field(capsys, monkeypatch):
    line = json.dumps({"id": 7, "msg": "mail a@b.co"})
    code, out, _ = run(
        capsys,
        ["redact", "--field", "msg"],
        stdin=line,
        monkeypatch=monkeypatch,
    )
    assert code == 0
    assert json.loads(out) == {"id": 7, "msg": "mail [EMAIL]"}


def test_redact_field_mode_bad_line_exits_two(capsys, monkeypatch):
    lines = json.dumps({"id": 7, "msg": 5}) + "\n" + json.dumps({"msg": "ok"})
    code, out, err = run(
        capsys,
        ["redact", "--field", "msg"],
        stdin=lines,
        monkeypatch=monkeypatch,
    )
    assert code == 2
    assert json.loads(out) == {"msg": "ok"}
    assert "line 1: field 'msg' is not a string" in err


def test_redact_exit_zero_even_when_flagged(capsys):
    code, out, _ = run(capsys, ["redact", "-t", f"key {AWS}"])
    assert code == 0 and out == "key [AWS_ACCESS_KEY_ID]\n"
