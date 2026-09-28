#!/usr/bin/env python3
"""
Data generation script for pygarble.

Downloads pinned word frequency data from Peter Norvig's collection
and generates embedded lookup tables for the library.

Data source: https://norvig.com/ngrams/
See paper/scripts/data_curation.json for source attribution and provenance.

This script should be run at development time, not at runtime.
The generated files are committed to the repository.
"""

import argparse
import bisect
import hashlib
import json
import math
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

# URLs for data sources
NORVIG_WORD_FREQ_URL = "https://norvig.com/ngrams/count_1w.txt"

# Generated header paths are historical; retain them for byte reproducibility.
# Output paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "pygarble" / "data"

# Log probability for unseen bigrams, shared by the .py and .json tables
DEFAULT_LOG_PROB = -10.0

# Reference text for the non-parametric strategies (deflate window size)
REFERENCE_BYTES = 32768
NGRAM_RANK_SIZE = 1000

# Synthetic English null used to standardise the non-parametric statistics
STATISTIC_NULL_SEED = 20260929
STATISTIC_NULL_RANGES = ((8, 15), (16, 31), (32, 63), (64, 127))
NULL_TEXTS_PER_RANGE = 2000
NULL_MIN_LENGTH = 8
STATISTICS = (
    "cross_parsing",
    "primed_compression",
    "ngram_rank",
    "permutation_test",
)
TAIL_GRID = (0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001)


def download_word_frequencies(url: str) -> Dict[str, int]:
    """Download word frequency data from Norvig's site."""
    print(f"Downloading word frequencies from {url}...")

    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read().decode("utf-8")

    word_freq = {}
    for line in content.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 2:
            word, count = parts
            word = word.lower().strip()
            # Only include alphabetic words
            if word.isalpha() and len(word) >= 2:
                word_freq[word] = int(count)

    print(f"  Downloaded {len(word_freq)} words")
    return word_freq


def generate_english_words(
    word_freq: Dict[str, int], max_words: int = 50000
) -> Set[str]:
    """
    Generate set of common English words.

    Uses top N words by frequency, filtered for quality.
    """
    print(f"Generating English word set (top {max_words})...")

    # Sort by frequency and take top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    words = set()
    for word, freq in sorted_words[:max_words]:
        # Additional quality filters
        if len(word) >= 2 and word.isalpha():
            words.add(word)

    print(f"  Generated {len(words)} words")
    return words


def compute_bigram_probabilities(
    word_freq: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute log-probability transition matrix for character bigrams.

    Uses Laplace smoothing to handle unseen bigrams.
    Returns log probabilities for numerical stability.
    """
    print("Computing bigram transition probabilities...")

    # Count all bigrams weighted by word frequency
    bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    char_counts: Dict[str, int] = defaultdict(int)

    # Process words weighted by their frequency
    for word, freq in word_freq.items():
        word = word.lower()
        # Add word boundaries
        padded = " " + word + " "
        for i in range(len(padded) - 1):
            c1, c2 = padded[i], padded[i + 1]
            if c1.isalpha() or c1 == " ":
                if c2.isalpha() or c2 == " ":
                    bigram_counts[(c1, c2)] += freq
                    char_counts[c1] += freq

    # All possible characters (26 letters + space)
    chars = " abcdefghijklmnopqrstuvwxyz"
    vocab_size = len(chars)

    # Compute log probabilities with Laplace smoothing
    bigram_log_probs = {}
    smoothing = 1  # Laplace smoothing parameter

    for c1 in chars:
        total = char_counts.get(c1, 0) + smoothing * vocab_size
        for c2 in chars:
            count = bigram_counts.get((c1, c2), 0) + smoothing
            prob = count / total
            log_prob = math.log(prob)
            bigram_log_probs[c1 + c2] = round(log_prob, 6)

    print(f"  Generated {len(bigram_log_probs)} bigram probabilities")
    return bigram_log_probs


def compute_trigram_frequencies(
    word_freq: Dict[str, int], top_n: int = 2000
) -> Set[str]:
    """
    Compute set of most common character trigrams.
    """
    print(f"Computing top {top_n} trigram frequencies...")

    trigram_counts: Counter = Counter()

    for word, freq in word_freq.items():
        word = word.lower()
        if len(word) >= 3:
            for i in range(len(word) - 2):
                trigram = word[i : i + 3]
                if trigram.isalpha():
                    trigram_counts[trigram] += freq

    # Get top N trigrams
    top_trigrams = set(t for t, _ in trigram_counts.most_common(top_n))

    print(f"  Generated {len(top_trigrams)} common trigrams")
    return top_trigrams


def select_reference_words(
    words: Set[str], word_freq: Dict[str, int]
) -> List[str]:
    """Most frequent shipped words that fit the deflate window."""
    ordered = sorted(words, key=lambda w: (-word_freq.get(w, 0), w))
    selected: List[str] = []
    size = -1
    for word in ordered:
        if size + 1 + len(word) > REFERENCE_BYTES:
            break
        selected.append(word)
        size += 1 + len(word)
    print(f"  Selected {len(selected)} reference words")
    return selected


def compute_ngram_ranks(
    word_freq: Dict[str, int], top_n: int = NGRAM_RANK_SIZE
) -> List[str]:
    """Most frequent padded character 1- to 3-grams, in rank order."""
    print(f"Computing top {top_n} n-gram ranks...")
    counts: Dict[str, int] = defaultdict(int)
    for word, freq in word_freq.items():
        padded = " " + word + " "
        for n in (1, 2, 3):
            for i in range(len(padded) - n + 1):
                counts[padded[i : i + n]] += freq
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [gram for gram, _ in ordered[:top_n]]


def load_null_words(text: str) -> Tuple[List[str], List[int]]:
    """Null vocabulary in file order with cumulative counts."""
    words: List[str] = []
    cumulative: List[int] = []
    total = 0
    for line in text.splitlines():
        word, count = line.split("\t")
        if word.isalpha() and (len(word) >= 2 or word in ("a", "i")):
            total += int(count)
            words.append(word)
            cumulative.append(total)
    return words, cumulative


def synthetic_texts(
    words: Sequence[str],
    cumulative: Sequence[int],
    seed: int,
    ranges: Sequence[Tuple[int, int]],
    per_range: int = NULL_TEXTS_PER_RANGE,
) -> List[str]:
    """Frequency-weighted word salads with lengths drawn per range."""
    from pygarble.gibberish.measures import Lcg

    rng = Lcg(seed)
    total = cumulative[-1]

    def draw() -> str:
        return words[bisect.bisect_right(cumulative, rng.below(total))]

    texts = []
    for lo, hi in ranges:
        for _ in range(per_range):
            target = lo + rng.below(hi - lo + 1)
            text = draw()
            while True:
                word = draw()
                if len(text) + 1 + len(word) > target:
                    break
                text += " " + word
            texts.append(text)
    return texts


def quantile(values: Sequence[float], p: float) -> float:
    """Lower empirical quantile of sorted ``values``."""
    return values[min(len(values) - 1, int(p * len(values)))]


def compute_statistic_null(
    texts: Sequence[str],
    reference_words: Sequence[str],
    ngram_ranks: Sequence[str],
    bigrams: Dict[str, float],
) -> Dict[str, List[Tuple[float, float]]]:
    """Per-bucket (median, q99) of each statistic on the null texts."""
    from pygarble.gibberish import measures

    print("Computing statistic null...")
    reference = " ".join(reversed(reference_words))
    dictionary = reference.encode("ascii")
    ranks = {gram: rank for rank, gram in enumerate(ngram_ranks)}
    samples: Dict[str, List[List[float]]] = {
        name: [[] for _ in range(len(measures.BUCKET_EDGES) + 1)]
        for name in STATISTICS
    }
    for text in texts:
        b = measures.bucket(len(text))
        samples["cross_parsing"][b].append(
            measures.cross_parsing(text, reference)
        )
        samples["primed_compression"][b].append(
            measures.primed_compression(text, dictionary)
        )
        samples["ngram_rank"][b].append(
            measures.ngram_rank_distance(text, ranks)
        )
        samples["permutation_test"][b].append(
            measures.permutation_gap(text, bigrams, DEFAULT_LOG_PROB)
        )
    table: Dict[str, List[Tuple[float, float]]] = {}
    for name in STATISTICS:
        table[name] = []
        for values in samples[name]:
            values.sort()
            table[name].append(
                (
                    round(quantile(values, 0.5), 6),
                    round(quantile(values, 0.99), 6),
                )
            )
    for b, values in enumerate(samples["cross_parsing"]):
        print(f"  Bucket {b}: {len(values)} texts")
    return table


def write_words_file(words: Set[str], filepath: Path) -> None:
    """Write English words as a Python module."""
    print(f"Writing words to {filepath}...")

    # Sort for deterministic output
    sorted_words = sorted(words)

    content = '''"""
English word set for garble detection.

This file is auto-generated by scripts/generate_data.py
Data source: Peter Norvig's word frequency list (https://norvig.com/ngrams/)
Source provenance: scripts/data_curation.json

Do not edit this file manually.
"""

# fmt: off
ENGLISH_WORDS = frozenset({
'''

    # Write words in chunks for readability
    chunk_size = 10
    for i in range(0, len(sorted_words), chunk_size):
        chunk = sorted_words[i : i + chunk_size]
        line = "    " + ", ".join(f'"{w}"' for w in chunk) + ","
        content += line + "\n"

    content += """})
# fmt: on
"""

    filepath.write_text(content, encoding="utf-8")
    print(
        (
            "  Written "
            f"{len(words)}"
            " words ("
            f"{filepath.stat().st_size / 1024:.1f}"
            " KB)"
        )
    )


def write_bigrams_file(bigram_probs: Dict[str, float], filepath: Path) -> None:
    """Write bigram probabilities as a Python module."""
    print(f"Writing bigrams to {filepath}...")

    content = (
        '''"""
Character bigram log-probabilities for Markov chain garble detection.

This file is auto-generated by scripts/generate_data.py
Data source: Peter Norvig's word frequency list (https://norvig.com/ngrams/)
Source provenance: scripts/data_curation.json

Log probabilities are used for numerical stability.
Sum transition log scores to obtain the text log likelihood.

Do not edit this file manually.
"""

# Default log probability for unseen bigrams (very unlikely)
'''
        f"DEFAULT_LOG_PROB = {DEFAULT_LOG_PROB!r}\n"
        """
# fmt: off
BIGRAM_LOG_PROBS = {
"""
    )

    # Sort for deterministic output
    sorted_bigrams = sorted(bigram_probs.items())

    # Write in chunks
    chunk_size = 8
    for i in range(0, len(sorted_bigrams), chunk_size):
        chunk = sorted_bigrams[i : i + chunk_size]
        pairs = [f'"{k}": {v}' for k, v in chunk]
        content += "    " + ", ".join(pairs) + ",\n"

    content += """}
# fmt: on
"""

    filepath.write_text(content, encoding="utf-8")
    print(
        (
            "  Written "
            f"{len(bigram_probs)}"
            " bigrams ("
            f"{filepath.stat().st_size / 1024:.1f}"
            " KB)"
        )
    )


def write_trigrams_file(trigrams: Set[str], filepath: Path) -> None:
    """Write common trigrams as a Python module."""
    print(f"Writing trigrams to {filepath}...")

    sorted_trigrams = sorted(trigrams)

    content = '''"""
Common English character trigrams for garble detection.

This file is auto-generated by scripts/generate_data.py
Data source: Peter Norvig's word frequency list (https://norvig.com/ngrams/)
Source provenance: scripts/data_curation.json

Do not edit this file manually.
"""

# fmt: off
COMMON_TRIGRAMS = frozenset({
'''

    # Write in chunks
    chunk_size = 15
    for i in range(0, len(sorted_trigrams), chunk_size):
        chunk = sorted_trigrams[i : i + chunk_size]
        line = "    " + ", ".join(f'"{t}"' for t in chunk) + ","
        content += line + "\n"

    content += """})
# fmt: on
"""

    filepath.write_text(content, encoding="utf-8")
    print(
        (
            "  Written "
            f"{len(trigrams)}"
            " trigrams ("
            f"{filepath.stat().st_size / 1024:.1f}"
            " KB)"
        )
    )


def write_sequence_file(
    header: str, name: str, items: Sequence[str], filepath: Path
) -> None:
    """Write an ordered tuple of strings as a Python module."""
    print(f"Writing {name} to {filepath}...")
    content = f'''"""
{header}

This file is auto-generated by paper/scripts/generate_data.py
Data source: Peter Norvig's word frequency list (https://norvig.com/ngrams/)
Source provenance: paper/scripts/data_curation.json

Do not edit this file manually.
"""

# fmt: off
{name} = (
'''
    chunk_size = 10
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        content += "    " + ", ".join(json.dumps(s) for s in chunk) + ",\n"
    content += """)
# fmt: on
"""
    filepath.write_text(content, encoding="utf-8")
    print(f"  Written {len(items)} entries")


def python_literal(value: object, indent: str = "") -> str:
    """Render calibration values one entry per line, tuples for lists."""
    inner = indent + "    "
    if isinstance(value, dict):
        lines = [
            f"{inner}{json.dumps(key)}: {python_literal(item, inner)},\n"
            for key, item in value.items()
        ]
        return "{\n" + "".join(lines) + indent + "}"
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, float) for item in value):
            return repr(tuple(value))
        lines = [f"{inner}{python_literal(item, inner)},\n" for item in value]
        return "(\n" + "".join(lines) + indent + ")"
    return repr(value)


def write_calibration_file(
    calibration: Dict[str, object], filepath: Path
) -> None:
    """Write calibration tables, one upper-case constant per key."""
    print(f"Writing calibration to {filepath}...")
    content = '''"""
Synthetic English null calibration for the non-parametric strategies.

This file is auto-generated by paper/scripts/generate_data.py
Data source: Peter Norvig's word frequency list (https://norvig.com/ngrams/)
Source provenance: paper/scripts/data_curation.json

STATISTIC_NULL holds the (median, q99) of each raw statistic per length
bucket; TAIL_GRID lists the upper-tail probabilities of the tail tables.

Do not edit this file manually.
"""

# fmt: off
'''
    for key, value in calibration.items():
        content += f"{key.upper()} = {python_literal(value)}\n"
    content += "# fmt: on\n"
    filepath.write_text(content, encoding="utf-8")


def write_json(payload: object, filepath: Path) -> None:
    """Byte-reproducible JSON: sorted keys, compact, ASCII, final newline."""
    filepath.write_text(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Pinned count_1w.txt; downloads if omitted"
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify reproducible files without writing",
    )
    args = parser.parse_args()
    curation_path = SCRIPT_DIR / "data_curation.json"
    curation = json.loads(curation_path.read_text())
    if args.source:
        raw = args.source.read_bytes()
    else:
        with urllib.request.urlopen(
            curation["source_url"], timeout=30
        ) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != curation["source_sha256"]:
        raise ValueError(
            "source checksum mismatch; review provenance before updating"
        )
    word_freq = {}
    for line in raw.decode("utf-8").splitlines():
        word, count = line.split("\t")
        if word.isalpha() and len(word) >= 2:
            word_freq[word] = int(count)
    words = generate_english_words(word_freq, curation["max_words"])
    words.difference_update(curation["exclude_words"])
    words.update(curation["include_words"])
    bigrams = compute_bigram_probabilities(word_freq)
    trigrams = compute_trigram_frequencies(word_freq, curation["top_trigrams"])
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    reference_words = select_reference_words(words, word_freq)
    ngram_ranks = compute_ngram_ranks(word_freq)
    null_words, cumulative = load_null_words(raw.decode("utf-8"))
    null_texts = [
        text
        for text in synthetic_texts(
            null_words, cumulative, STATISTIC_NULL_SEED, STATISTIC_NULL_RANGES
        )
        if len(text) >= NULL_MIN_LENGTH
    ]
    calibration: Dict[str, object] = {
        "statistic_null": compute_statistic_null(
            null_texts, reference_words, ngram_ranks, bigrams
        ),
        "tail_grid": list(TAIL_GRID),
    }
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        write_words_file(words, directory / "words.py")
        write_bigrams_file(bigrams, directory / "bigrams.py")
        write_trigrams_file(trigrams, directory / "trigrams.py")
        write_json(sorted(words), directory / "words.json")
        write_json(
            {"default_log_prob": DEFAULT_LOG_PROB, "log_probs": bigrams},
            directory / "bigrams.json",
        )
        write_json(sorted(trigrams), directory / "trigrams.json")
        write_sequence_file(
            "Most frequent English words, joined as the reference text.",
            "REFERENCE_WORDS",
            reference_words,
            directory / "reference.py",
        )
        write_sequence_file(
            "Most frequent character 1- to 3-grams, in rank order.",
            "NGRAM_RANKS",
            ngram_ranks,
            directory / "ngram_ranks.py",
        )
        write_calibration_file(calibration, directory / "calibration.py")
        write_json(reference_words, directory / "reference.json")
        write_json(ngram_ranks, directory / "ngram_ranks.json")
        write_json(calibration, directory / "calibration.json")
        from pygarble.screening.pii.patterns import export as pii_export
        from pygarble.screening.profanity.wordlist import (
            export as profanity_export,
        )
        from pygarble.screening.secrets.patterns import (
            export as secrets_export,
        )

        secrets_table = secrets_export()
        pii_table = pii_export()
        profanity_table = profanity_export()
        write_json(secrets_table, directory / "secrets.json")
        write_json(pii_table, directory / "pii.json")
        write_json(profanity_table, directory / "profanity.json")
        manifest = {
            "model_version": "english-v2",
            "source_url": curation["source_url"],
            "source_sha256": curation["source_sha256"],
            "curation_sha256": hashlib.sha256(
                curation_path.read_bytes()
            ).hexdigest(),
            "counts": {
                "words": len(words),
                "bigrams": len(bigrams),
                "trigrams": len(trigrams),
                "secret_patterns": len(secrets_table["known"]),
                "pii_rules": len(pii_table["generic"])
                + sum(len(r) for r in pii_table["locales"].values()),
                "profanity_strong": len(profanity_table["strong"]),
                "profanity_mild": len(profanity_table["mild"]),
                "reference_words": len(reference_words),
                "ngram_ranks": len(ngram_ranks),
                "statistic_null_texts": len(null_texts),
            },
            "files": {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted(directory.iterdir())
                if file.suffix in (".py", ".json")
                and file.name != "manifest.json"
            },
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not args.check:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        for generated in sorted(directory.iterdir()):
            destination = args.output_dir / generated.name
            if args.check:
                if (
                    not destination.exists()
                    or generated.read_bytes() != destination.read_bytes()
                ):
                    raise ValueError(
                        f"generated artifact differs: {destination}"
                    )
            else:
                destination.write_bytes(generated.read_bytes())
    print(
        "Artifact verification passed" if args.check else "Artifacts generated"
    )


if __name__ == "__main__":
    main()
