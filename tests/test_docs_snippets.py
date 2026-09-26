"""Every Python example in README.md and docs/*.rst runs cleanly."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DOCS = ROOT / "docs"

if not README.is_file() or not DOCS.is_dir():
    pytest.skip("README.md or docs/ not present", allow_module_level=True)

MARKDOWN_BLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)
RST_DIRECTIVE = re.compile(r"^(\s*)\.\. code-block:: python\s*$")


def markdown_blocks(path):
    return MARKDOWN_BLOCK.findall(path.read_text(encoding="utf-8"))


def rst_blocks(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        match = RST_DIRECTIVE.match(lines[i])
        i += 1
        if not match:
            continue
        indent = len(match.group(1))
        # Skip directive options and the blank line before the body.
        while i < len(lines) and (
            not lines[i].strip() or lines[i].strip().startswith(":")
        ):
            i += 1
        body = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
            i += 1
        blocks.append(textwrap.dedent("\n".join(body)).strip() + "\n")
    return blocks


def collect():
    cases = []
    for index, code in enumerate(markdown_blocks(README)):
        cases.append(pytest.param(code, id=f"README.md:{index}"))
    for path in sorted(DOCS.glob("*.rst")):
        for index, code in enumerate(rst_blocks(path)):
            cases.append(pytest.param(code, id=f"docs/{path.name}:{index}"))
    return cases


CASES = collect()


def test_snippets_were_found():
    assert len(CASES) >= 25


@pytest.mark.parametrize("code", CASES)
def test_snippet_runs(code):
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        env={"PYTHONPATH": str(ROOT), "PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
