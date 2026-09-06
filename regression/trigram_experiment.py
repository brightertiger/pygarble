"""Offline conditional-trigram experiment. No runtime fitting or downloads.

Builds a 27^3 float32 table from the pinned Norvig source, with add-one
smoothing and 20% bigram backoff. Output is an experimental artifact, not
implicitly included in the default ensemble.
"""

import argparse
import hashlib
import json
import math
import sys
from array import array
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pygarble.data import BIGRAM_LOG_PROBS

ALPHABET = " abcdefghijklmnopqrstuvwxyz"


def build(source: Path) -> array:
    curation = json.loads(
        (
            Path(__file__).resolve().parents[1] / "scripts/data_curation.json"
        ).read_text()
    )
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != curation["source_sha256"]:
        raise ValueError("source checksum mismatch")
    triples = Counter()
    contexts = Counter()
    for line in raw.decode("utf-8").splitlines():
        word, value = line.split("\t")
        if not word.isascii() or not word.isalpha() or len(word) < 2:
            continue
        frequency = int(value)
        padded = "  " + word + " "
        for i in range(len(padded) - 2):
            triple = padded[i : i + 3]
            triples[triple] += frequency
            contexts[triple[:2]] += frequency
    scores = array("f")
    for first in ALPHABET:
        for second in ALPHABET:
            context = first + second
            denominator = contexts[context] + 27
            for third in ALPHABET:
                trigram = (triples[context + third] + 1) / denominator
                bigram = math.exp(BIGRAM_LOG_PROBS[second + third])
                scores.append(math.log(0.8 * trigram + 0.2 * bigram))
    return scores


def mean_score(text, scores):
    from pygarble.preprocessing import TextFeatures

    indices = {char: i for i, char in enumerate(ALPHABET)}
    total = count = 0
    for token in TextFeatures(text).novel:
        if len(token.folded) < 4:
            continue
        padded = "  " + token.folded + " "
        for i in range(len(padded) - 2):
            triple = padded[i : i + 3]
            if all(char in indices for char in triple):
                total += scores[
                    indices[triple[0]] * 729
                    + indices[triple[1]] * 27
                    + indices[triple[2]]
                ]
            else:
                total -= 10.0
            count += 1
    return total / count if count else 0.0


def evaluate_table(scores):
    from pygarble import EnsembleDetector
    from regression.evaluate import challenge, metrics, reviewed_legacy

    detector = EnsembleDetector()
    result = {}
    for name, cases in [
        ("challenge_development", challenge("development")),
        ("legacy_reviewed", reviewed_legacy()),
    ]:
        baseline = [detector.predict(case["text"]) for case in cases]
        means = [mean_score(case["text"], scores) for case in cases]
        rows = []
        for threshold in [-3.0, -3.5, -4.0, -4.5, -5.0]:
            predictions = [mean < threshold for mean in means]
            rows.append(
                {
                    "threshold": threshold,
                    "standalone": metrics(cases, predictions),
                    "additional_tp": sum(
                        p and not b and c["expected"]
                        for p, b, c in zip(predictions, baseline, cases)
                    ),
                    "additional_fp": sum(
                        p and not b and not c["expected"]
                        for p, b, c in zip(predictions, baseline, cases)
                    ),
                }
            )
        result[name] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    scores = build(args.source)
    if scores.itemsize != 4:
        raise ValueError("float32 array representation required")
    if args.report:
        report = {
            "selected_threshold": -4.5,
            "selection": (
                "First threshold with zero standalone false positives on "
                "development challenge cases. No holdout tuning."
            ),
            "decision": (
                "Do not ship: no incremental detections over the English "
                "specialist ensemble at the selected threshold."
            ),
            "entries": len(scores),
            "raw_bytes": len(scores) * 4,
            "smoothing": (
                "add-one; 80% conditional trigram, 20% bigram backoff"
            ),
            "datasets": evaluate_table(scores),
        }
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    if sys.byteorder != "little":
        scores.byteswap()
    args.output.write_bytes(scores.tobytes())
    print(
        json.dumps(
            {
                "entries": len(scores),
                "bytes": args.output.stat().st_size,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
