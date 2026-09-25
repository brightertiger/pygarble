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


def test_calibrate_max_fpr_implies_objective(capsys, tmp_path):
    garbled = tmp_path / "g.txt"
    clean = tmp_path / "c.txt"
    garbled.write_text("qxzjkwpv bnmqwer\nasdfghjkl\n", encoding="utf-8")
    clean.write_text("hello world\nthe quick brown fox\n", encoding="utf-8")
    argv = ["calibrate", "--garbled", str(garbled), "--clean", str(clean)]
    code, out, err = run(capsys, argv + ["--max-fpr", "0.0"])
    assert code == 0 and err == ""
    assert out.splitlines()[:2] == ["objective: max_fpr", "max fpr: 0.0"]
    code, out, _ = run(capsys, argv)
    assert code == 0
    assert out.splitlines()[0] == "objective: f1"


def test_calibrate_max_fpr_conflicts_with_explicit_f1(capsys, tmp_path):
    garbled = tmp_path / "g.txt"
    clean = tmp_path / "c.txt"
    garbled.write_text("asdfghjkl\n", encoding="utf-8")
    clean.write_text("hello world\n", encoding="utf-8")
    code, out, err = run(
        capsys,
        [
            "calibrate",
            "--garbled",
            str(garbled),
            "--clean",
            str(clean),
            "--objective",
            "f1",
            "--max-fpr",
            "0.1",
        ],
    )
    assert code == 2 and out == ""
    assert "--max-fpr requires --objective max_fpr" in err


def test_file_lines_split_on_newline_only(capsys, tmp_path):
    path = tmp_path / "in.txt"
    path.write_bytes(b"hello\x1cworld\n")
    code, out, _ = run(
        capsys, ["check", "--strategy", "control_characters", str(path)]
    )
    assert code == 1
    assert out == "garbled\thello\x1cworld\n"


def test_nel_and_separators_stay_on_one_line(capsys, tmp_path):
    path = tmp_path / "in.txt"
    text = "caf\u0085e x y\x0b\x0cz"
    path.write_text(text + "\n", encoding="utf-8")
    code, out, _ = run(capsys, ["score", str(path)])
    assert code == 0
    assert out.count("\n") == 1
    assert out.rstrip("\n").split("\t", 1)[1] == text


def test_crlf_lines_drop_carriage_return(capsys, tmp_path):
    path = tmp_path / "in.txt"
    path.write_bytes(b"hello world\r\nqxzjkwpv bnmqwer\r\n")
    code, out, _ = run(capsys, ["check", str(path)])
    assert code == 1
    assert out.splitlines() == [
        "clean\thello world",
        "garbled\tqxzjkwpv bnmqwer",
    ]
    assert "\r" not in out


def test_crlf_allowlist(capsys, tmp_path):
    allow = tmp_path / "allow.txt"
    allow.write_bytes(b"qxzjkwpv\r\n")
    code, out, _ = run(
        capsys,
        ["check", "--allowlist", str(allow), "-t", "qxzjkwpv"],
    )
    assert code == 0, out


def test_field_line_numbers_ignore_form_feed(capsys, tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_bytes(b'{"msg": "hi"} \x0c junk\n{}\n{"msg": "ok"}\n')
    code, out, err = run(capsys, ["check", "--field", "msg", str(path)])
    assert code == 2
    assert len(out.splitlines()) == 1
    errors = err.splitlines()
    assert len(errors) == 2
    assert errors[0].startswith("line 1: invalid JSON")
    assert errors[1] == "line 2: missing field 'msg'"


def test_stdin_splits_on_newline_only(capsys, monkeypatch):
    code, out, _ = run(
        capsys,
        ["check", "--strategy", "control_characters"],
        stdin="hello\x1cworld\n",
        monkeypatch=monkeypatch,
    )
    assert code == 1
    assert out == "garbled\thello\x1cworld\n"


def test_broken_pipe_is_quiet(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("hello world\n" * 5000, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "pygarble", "score", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    first = proc.stdout.readline()
    proc.stdout.close()
    stderr = proc.stderr.read()
    proc.stderr.close()
    assert proc.wait(timeout=120) == 0
    assert first.startswith(b"0.")
    assert stderr == b""
