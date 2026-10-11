"""Speed, import time, install size and offline use for every method.

Speed: the native decision call over all B test texts, one warm-up and five
timed runs in this process on one thread; median, in ms per 1,000 texts.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from . import data
from .methods import Method, specs
from .metrics import RESULTS
from .run import cpu, write_json

RUNS = 5
PYGARBLE_WHEEL_EXCLUDED = {
    "words.json",
    "bigrams.json",
    "trigrams.json",
    "secrets.json",
    "pii.json",
    "profanity.json",
    "reference.json",
    "ngram_ranks.json",
    "calibration.json",
}
# Files the method needs at run time beyond its installed distributions.
ASSETS = {
    "gibberish_detector": ("baselines/big.txt",),
    "c4_rules": ("baselines/ldnoobw_en.txt",),
}
NOTES = {
    "gibberish_detector": "size includes the training corpus big.txt",
    "gopher_rules": "implemented in this benchmark (rules.py)",
    "c4_rules": "implemented in this benchmark; size includes LDNOOBW list",
    "lingua": "language models bundled in the wheel",
    "nostril": "n-gram table bundled in the package",
}


def time_method(method: Method, texts: Sequence[str], runs: int = RUNS):
    method.decide(texts)
    seconds = []
    for _ in range(runs):
        start = time.perf_counter()
        method.decide(texts)
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds) / len(texts) * 1e6, seconds


def import_seconds(statement: str, runs: int = RUNS) -> float:
    code = (
        "import time; t = time.perf_counter(); "
        f"{statement}; print(time.perf_counter() - t)"
    )
    values = [
        float(
            subprocess.run(
                (sys.executable, "-c", code),
                cwd=data.REPOSITORY,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        for _ in range(runs)
    ]
    return statistics.median(values)


def requirement_names(name: str) -> List[str]:
    """Runtime requirements whose environment markers hold here."""
    from packaging.requirements import Requirement

    out = []
    for text in metadata.requires(name) or []:
        requirement = Requirement(text)
        marker = requirement.marker
        if marker is None or marker.evaluate({"extra": ""}):
            out.append(requirement.name)
    return out


def closure(names: Iterable[str]) -> Set[str]:
    seen: Set[str] = set()
    todo = list(names)
    while todo:
        name = todo.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        if key != "pygarble":
            todo += requirement_names(name)
    return seen


def distribution_bytes(name: str) -> int:
    if name == "pygarble":
        package = data.REPOSITORY / "pygarble"
        return sum(
            path.stat().st_size
            for path in package.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name not in PYGARBLE_WHEEL_EXCLUDED
        )
    total = 0
    for file in metadata.distribution(name).files or []:
        path = Path(str(file.locate()))
        if "__pycache__" not in path.parts and path.is_file():
            total += path.stat().st_size
    return total


def install_bytes(
    name: str, distributions: Sequence[str], root: Path
) -> Optional[int]:
    total = sum(distribution_bytes(d) for d in closure(distributions))
    total += sum((root / a).stat().st_size for a in ASSETS.get(name, ()))
    if name in ("gopher_rules", "c4_rules"):
        total += (Path(__file__).parent / "rules.py").stat().st_size
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    texts = [row.text for row in data.load(args.data_root)["B_test"]]
    every = specs(args.data_root)
    imports: Dict[str, float] = {}
    out = {}
    for name in args.methods or list(every):
        spec = every[name]
        print(f"footprint: {name}", flush=True)
        ms, runs = time_method(spec.build(), texts)
        if spec.imports not in imports:
            imports[spec.imports] = import_seconds(spec.imports)
        out[name] = {
            "ms_per_1000": round(ms, 3),
            "run_seconds": [round(s, 4) for s in runs],
            "import_s": round(imports[spec.imports], 4),
            "import_statement": spec.imports,
            "install_bytes": install_bytes(
                name, spec.distributions, args.data_root
            ),
            "offline": True,
            "note": NOTES.get(name, ""),
        }
    write_json(
        RESULTS / "footprint.json",
        {
            "texts": len(texts),
            "runs": RUNS,
            "cpu": cpu(),
            "python": sys.version.split()[0],
            "methods": out,
        },
    )
    from .report import write_report

    write_report(json.loads((RESULTS / "metrics.json").read_text()))


if __name__ == "__main__":
    main()
