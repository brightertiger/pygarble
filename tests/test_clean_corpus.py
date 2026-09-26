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
