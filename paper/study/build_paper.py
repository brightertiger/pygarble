"""Build review-ready Markdown, TeX and PDF from measured result tables."""

import argparse
import gzip
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render() -> str:
    from .full_paper import render as render_full

    return render_full()


def bundle_review() -> None:
    """Archive paper sources, excluding all raw comparison data."""
    names = [
        "manuscript.tex",
        "references.bib",
        "figures/full-comparison.pdf",
        "figures/full-languages.pdf",
    ]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in names:
            raw = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
    (ROOT / "review-source.tar.gz").write_bytes(
        gzip.compress(stream.getvalue(), mtime=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    (ROOT / "manuscript.md").write_text(render(), encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            "manuscript.md",
            "--citeproc",
            "--standalone",
            "--to",
            "latex",
            "--output",
            "manuscript.tex",
        ],
        cwd=ROOT,
        check=True,
    )
    if not args.skip_pdf:
        subprocess.run(["tectonic", "manuscript.tex"], cwd=ROOT, check=True)
    bundle_review()


if __name__ == "__main__":
    main()
