"""Command-line interface contracts (no subprocess except --version)."""

import io
import json
import subprocess
import sys

from pygarble import __version__
from pygarble.cli import main


def run(capsys, argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def test_check_text_inputs(capsys):
    code, out, err = run(
        capsys, ["check", "-t", "hello world", "-t", "qxzjkwpv bnmqwer"]
    )
    assert code == 1
    assert out.splitlines() == [
        "clean\thello world",
        "garbled\tqxzjkwpv bnmqwer",
    ]
    assert err == ""


def test_check_all_clean_exits_zero(capsys):
    code, out, _ = run(capsys, ["check", "-t", "hello world"])
    assert code == 0
    assert out == "clean\thello world\n"


def test_check_reads_stdin_and_handles_blank_lines(capsys, monkeypatch):
    code, out, _ = run(
        capsys,
        ["check"],
        stdin="hello world\n\n   \nqxzjkwpv\n",
        monkeypatch=monkeypatch,
    )
    assert code == 1
    assert out.splitlines() == [
        "clean\thello world",
        "insufficient\t",
        "insufficient\t   ",
        "garbled\tqxzjkwpv",
    ]


def test_check_reads_files(capsys, tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("hello world\nasdfghjkl\n", encoding="utf-8")
    code, out, _ = run(capsys, ["check", str(path)])
    assert code == 1
    assert out.splitlines()[1] == "garbled\tasdfghjkl"


def test_score_format(capsys):
    code, out, _ = run(capsys, ["score", "-t", "hello world"])
    assert code == 0
    value, text = out.strip().split("\t")
    assert text == "hello world"
    assert 0.0 <= float(value) < 0.5


def test_tsv_format(capsys):
    _, out, _ = run(capsys, ["check", "--format", "tsv", "-t", "qxzjkwpv"])
    garbled, score, status, text = out.strip().split("\t")
    assert garbled == "1"
    assert float(score) >= 0.5
    assert status == "garbled"
    assert text == "qxzjkwpv"


def test_analyze_jsonl(capsys):
    code, out, _ = run(capsys, ["analyze", "-t", "please review qxzjkwpvm"])
    assert code == 0
    row = json.loads(out)
    assert row["text"] == "please review qxzjkwpvm"
    assert row["garbled"] is True
    assert row["profile"] == "english"
    assert {"start", "end", "reason"} <= set(row["spans"][0])
    assert {"strategy", "score", "applicable", "reason"} <= set(
        row["signals"][0]
    )


def test_profile_and_strategy_selection(capsys):
    _, out, _ = run(
        capsys, ["check", "--profile", "corruption", "-t", "CafÃ© au lait"]
    )
    assert out.startswith("garbled\t")
    _, out, _ = run(
        capsys,
        ["check", "--strategy", "control_characters", "-t", "hello\x00x"],
    )
    assert out.startswith("garbled\t")
    code, _, err = run(capsys, ["check", "--profile", "nope", "-t", "x"])
    assert code == 2 and "unknown profile" in err


def test_threshold_and_allowlist(capsys, tmp_path):
    words = tmp_path / "allow.txt"
    words.write_text("# domain words\nqxzjkwpv\n\n", encoding="utf-8")
    _, out, _ = run(
        capsys, ["check", "--allowlist", str(words), "-t", "hello qxzjkwpv"]
    )
    assert out.startswith("clean\t")
    # Obvious garble saturates at 1.0, so exercise the flag by lowering it
    # on a mid-scoring text (default ~0.31).
    text = "hello wrld frbl"
    _, out, _ = run(capsys, ["check", "-t", text])
    assert out.startswith("clean\t")
    _, out, _ = run(capsys, ["check", "--threshold", "0.2", "-t", text])
    assert out.startswith("garbled\t")


def test_field_mode(capsys, monkeypatch):
    lines = "\n".join(
        [
            json.dumps({"id": 1, "msg": "hello world"}),
            "not json",
            json.dumps({"id": 3}),
            json.dumps({"id": 4, "msg": 7}),
            json.dumps({"id": 5, "msg": "qxzjkwpv"}),
        ]
    )
    code, out, err = run(
        capsys,
        ["check", "--field", "msg"],
        stdin=lines,
        monkeypatch=monkeypatch,
    )
    rows = [json.loads(line) for line in out.splitlines()]
    assert [r["id"] for r in rows] == [1, 5]
    assert rows[0]["pygarble"]["garbled"] is False
    assert rows[1]["pygarble"]["garbled"] is True
    assert code == 2
    assert "line 2" in err and "line 3" in err and "line 4" in err


def test_missing_file_is_exit_2(capsys, tmp_path):
    code, _, err = run(capsys, ["check", str(tmp_path / "missing.txt")])
    assert code == 2 and "missing.txt" in err


def test_version_via_module():
    out = subprocess.check_output(
        [sys.executable, "-m", "pygarble", "--version"], text=True
    )
    assert out.strip() == f"pygarble {__version__}"


def test_help_does_not_import_data(capsys):
    program = (
        "import sys; from pygarble.cli import build_parser; "
        "build_parser(); "
        "assert 'pygarble.data.words' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_lone_surrogate_text_does_not_crash(capsys):
    code = main(["check", "-t", "ab\ud800cd"])
    out, _ = capsys.readouterr()
    assert code in (0, 1)
    assert "\\ud800" in out or "\ud800" in out


def test_invalid_utf8_stdin_via_module():
    result = subprocess.run(
        [sys.executable, "-m", "pygarble", "check"],
        input=b"ab\xffcd\n",
        capture_output=True,
    )
    assert result.returncode in (0, 1)
    assert result.stderr == b""
    assert result.stdout.decode("utf-8").endswith("ab\ufffdcd\n")


def test_invalid_utf8_file_is_replaced_not_crash(capsys, tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"caf\xe9\n")
    code, out, err = run(capsys, ["check", str(path)])
    # U+FFFD is itself corruption evidence, so the line is flagged (exit 1);
    # the contract is "no traceback, no exit 2".
    assert code == 1 and err == ""
    assert out.splitlines() == ["garbled\tcaf\ufffd"]


def test_bad_allowlist_path_is_exit_2(capsys, tmp_path):
    missing = str(tmp_path / "nope.txt")
    code, out, err = run(
        capsys, ["check", "--allowlist", missing, "-t", "hello"]
    )
    assert code == 2
    assert out == ""
    assert "cannot read allowlist" in err and "nope.txt" in err


def test_field_mode_error_messages(capsys, monkeypatch):
    lines = "\n".join(["[1, 2]", "{}", json.dumps({"msg": 3})])
    code, out, err = run(
        capsys,
        ["check", "--field", "msg"],
        stdin=lines,
        monkeypatch=monkeypatch,
    )
    assert code == 2 and out == ""
    assert err.splitlines() == [
        "line 1: not a JSON object",
        "line 2: missing field 'msg'",
        "line 3: field 'msg' is not a string",
    ]


def test_calibrate_subcommand(capsys, tmp_path):
    garbled = tmp_path / "g.txt"
    clean = tmp_path / "c.txt"
    garbled.write_text("qxzjkwpv bnmqwer\nasdfghjkl\n", encoding="utf-8")
    clean.write_text("hello world\nthe quick brown fox\n", encoding="utf-8")
    code, out, _ = run(
        capsys, ["calibrate", "--garbled", str(garbled), "--clean", str(clean)]
    )
    assert code == 0
    assert "recommended threshold:" in out.splitlines()[-1]
    code, out, _ = run(
        capsys,
        [
            "calibrate",
            "--garbled",
            str(garbled),
            "--clean",
            str(clean),
            "--format",
            "jsonl",
        ],
    )
    row = json.loads(out)
    assert 0.0 <= row["recommended"]["threshold"] <= 1.0
    assert row["objective"] == "f1"
