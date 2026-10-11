"""Gopher and C4 quality rules on RedPajama-V2 quality-signal definitions.

Signals follow togethercomputer/RedPajama-Data at commit 6d2cee9
(app/src/core/document.py, quality_signals/*.py); thresholds follow Rae et
al. (2021, Gopher) and Raffel et al. (2020, C4). Each function returns the
names of the rules a text fails.
"""

import re
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import FrozenSet, List, Sequence, Tuple

# nltk's WordPunctTokenizer is exactly this pattern.
WORD_PUNCT = re.compile(r"\w+|[^\w\s]+")
PUNCTUATION = str.maketrans("", "", string.punctuation)
LINES = re.compile(r"([^\n]*\n|[^\n]+$)")
SENTENCES = re.compile(r"\b[^.!?]+[.!?]*", flags=re.UNICODE)
ALPHA = re.compile(r"[a-zA-Z]")
BULLETS = tuple("•‣▶◀◦■□▪▫–")
ELLIPSIS = ("...", "…")
SYMBOLS = ("#", "...", "…")
TERMINAL = (".", "!", "?", "”")
GOPHER_STOP_WORDS = frozenset(
    ("the", "be", "to", "of", "and", "that", "have", "with")
)
TOP_NGRAM_LIMITS = ((2, 0.20), (3, 0.18), (4, 0.16))
DUPE_NGRAM_LIMITS = (
    (5, 0.15),
    (6, 0.14),
    (7, 0.13),
    (8, 0.12),
    (9, 0.11),
    (10, 0.10),
)
GOPHER_RULES = (
    (
        "word_count",
        "mean_word_length",
        "symbol_to_word_ratio",
        "bullet_lines",
        "ellipsis_lines",
        "no_alpha_words",
        "stop_words",
    )
    + tuple(f"top_{n}gram" for n, _ in TOP_NGRAM_LIMITS)
    + tuple(f"dupe_{n}gram" for n, _ in DUPE_NGRAM_LIMITS)
)
C4_RULES = (
    "terminal_punctuation",
    "words_per_line",
    "javascript",
    "sentences",
    "lorem_ipsum",
    "curly_bracket",
    "bad_words",
)


def normalize(text: str) -> str:
    """RedPajama-V2 normalisation: drop punctuation, lowercase, NFD."""
    text = text.translate(PUNCTUATION).lower().strip()
    return unicodedata.normalize("NFD", re.sub(r"\s+", " ", text))


def lines(text: str) -> List[str]:
    return [m.group(0) for m in LINES.finditer(text)]


def ngrams(words: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def top_ngram_fraction(words: Sequence[str], n: int) -> float:
    common = Counter(ngrams(words, n)).most_common(1)
    if not common or common[0][1] <= 1:
        return 0.0
    gram, count = common[0]
    return sum(map(len, gram)) * count / sum(map(len, words))


def dupe_ngram_fraction(words: Sequence[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = ngrams(words, n)
    dupes = {g for g, c in Counter(grams).items() if c > 1}
    marked = [0] * len(words)
    for i, gram in enumerate(grams):
        if gram in dupes:
            marked[i : i + n] = [1] * n
    total = sum(map(len, words))
    if not total:
        return 0.0
    return sum(len(w) * m for w, m in zip(words, marked)) / total


def gopher_failures(text: str) -> List[str]:
    norm = normalize(text).split()
    raw = WORD_PUNCT.findall(text)
    raw_lines = lines(text)
    failed = []
    if not 50 <= len(norm) <= 100_000:
        failed.append("word_count")
    mean_length = sum(map(len, norm)) / len(norm) if norm else None
    if mean_length is None or not 3 <= mean_length <= 10:
        failed.append("mean_word_length")
    symbols = sum(text.count(s) for s in SYMBOLS)
    if not raw or symbols / len(raw) > 0.1:
        failed.append("symbol_to_word_ratio")
    if raw_lines:
        bullets = sum(ln.lstrip().startswith(BULLETS) for ln in raw_lines)
        if bullets / len(raw_lines) > 0.9:
            failed.append("bullet_lines")
        ellipses = sum(ln.rstrip().endswith(ELLIPSIS) for ln in raw_lines)
        if ellipses / len(raw_lines) > 0.3:
            failed.append("ellipsis_lines")
    alpha = sum(bool(ALPHA.search(w)) for w in raw)
    if not raw or 1 - alpha / len(raw) > 0.2:
        failed.append("no_alpha_words")
    if len(GOPHER_STOP_WORDS.intersection(norm)) < 2:
        failed.append("stop_words")
    for n, limit in TOP_NGRAM_LIMITS:
        if top_ngram_fraction(norm, n) > limit:
            failed.append(f"top_{n}gram")
    for n, limit in DUPE_NGRAM_LIMITS:
        if dupe_ngram_fraction(norm, n) > limit:
            failed.append(f"dupe_{n}gram")
    return failed


def load_bad_words(path: Path) -> FrozenSet[str]:
    return frozenset(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def c4_failures(text: str, bad_words: FrozenSet[str]) -> List[str]:
    raw_lines = lines(text)
    norm_lines = [normalize(ln) for ln in raw_lines]
    norm = normalize(text)
    words = norm.split()
    failed = []
    if not raw_lines or not all(
        ln.rstrip().endswith(TERMINAL) for ln in raw_lines
    ):
        failed.append("terminal_punctuation")
    if not norm_lines or any(len(ln.split()) < 3 for ln in norm_lines):
        failed.append("words_per_line")
    if any("javascript" in ln.split() for ln in norm_lines):
        failed.append("javascript")
    if len(SENTENCES.findall(text)) < 5:
        failed.append("sentences")
    if "lorem ipsum" in norm:
        failed.append("lorem_ipsum")
    if "{" in text or "}" in text:
        failed.append("curly_bracket")
    if _has_bad_word(words, bad_words):
        failed.append("bad_words")
    return failed


def _has_bad_word(words: Sequence[str], bad_words: FrozenSet[str]) -> bool:
    sizes = {1 + w.count(" ") for w in bad_words}
    for n in sizes:
        if any(" ".join(g) in bad_words for g in ngrams(words, n)):
            return True
    return False
