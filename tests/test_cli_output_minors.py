"""CLI rows stay one line per text; a closed pipe is not a failed scan."""

import subprocess
import sys

from pygarble.cli import main


def test_cli_text_and_tsv_rows_escape_tabs_and_newlines(capsys):
    # F3: one input text is one output row with a fixed number of columns.
    text = "fine\tline\r\nasdkjh qwpoeiru zmxncbv"
    escaped = "fine\\tline\\r\\nasdkjh qwpoeiru zmxncbv"
    for argv, columns in (
        (["check", "-t", text], 2),
        (["score", "-t", text], 2),
        (["check", "--format", "tsv", "-t", text], 4),
        (["scan", "-t", text], 3),
        (["scan", "--format", "tsv", "-t", text], 4),
    ):
        main(argv)
        out = capsys.readouterr().out
        assert out.count("\n") == 1, argv
        fields = out.rstrip("\n").split("\t")
        assert len(fields) == columns, argv
        assert fields[-1] == escaped, argv


def test_screen_cli_exits_zero_when_reader_closes_pipe(tmp_path):
    # F4: `pygarble-screen scan ... | head` is not a failed scan.
    doc = tmp_path / "big.txt"
    doc.write_text(
        " ".join(f"user{i}@example.com" for i in range(20000)),
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "pygarble.screening", "scan"] + [str(doc)] * 4,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    proc.stdout.read(1)
    proc.stdout.close()
    err = proc.stderr.read().decode()
    assert proc.wait(timeout=120) == 0, err
    assert "screening failed" not in err
