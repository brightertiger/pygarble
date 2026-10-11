"""Figure, LaTeX tables and number macros for the preprint.

Reads `results/metrics.json`, `results/footprint.json`, the A predictions
and the manifest; writes the results figure (PDF, and a 300 dpi PNG for
checking) to `paper/preprint/figures/` and the LaTeX fragments
`table_survey.tex`, `table_tweets.tex`, `table_footprint.tex` and
`numbers.tex` to `paper/preprint/tables/`. Nothing in the paper is typed by
hand. Run from the repository root:

    python -m paper.scripts.benchmark.figures
"""

import json
import logging
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
# fontTools reports cmr10's 1990s timestamps as suspicious; they are fine.
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402

from pygarble.gibberish.registry import Strategy  # noqa: E402

from .data import MANIFEST, REPOSITORY  # noqa: E402
from .metrics import PREDICTIONS, RESULTS, read_predictions  # noqa: E402

PREPRINT = REPOSITORY / "paper" / "preprint"
FIGURES = PREPRINT / "figures"
TABLES = PREPRINT / "tables"

# Table and figure rows, top to bottom; the tuned row is set apart.
UNTUNED = ("english", "english_fusion", "gibberish_detector", "lingua")
TUNED = ("english_tuned",)
LABELS = {
    "english": r"pygarble \texttt{english} (default)",
    "english_fusion": r"pygarble \texttt{english\_fusion}",
    "gibberish_detector": r"\texttt{gibberish-\allowbreak detector}",
    "lingua": r"\texttt{lingua}",
    "english_tuned": r"pygarble \texttt{english}, tuned (post hoc)",
}
PLAIN = {
    "english": "pygarble english (default)",
    # cmr10 has no underscore glyph; mathtext draws one.
    "english_fusion": r"pygarble english$\_$fusion",
    "gibberish_detector": "gibberish-detector",
    "lingua": "lingua",
    "english_tuned": "pygarble english, tuned\n(post hoc)",
}
# Runtime dependencies, read from the installed distributions: none of the
# three declares a requirement. gibberish-detector ships no model and the
# benchmark trains one from big.txt; lingua's models sit inside its wheel.
DEPENDENCIES = {
    "english": "none",
    "english_fusion": "none",
    "gibberish_detector": r"none$^a$",
    "lingua": r"none$^b$",
    "english_tuned": "none",
}
DEPENDENCY_NOTES = (
    r"$^a$\,Ships no model; trains one from a text corpus at start-up "
    r"(the benchmark uses the project's \texttt{big.txt}, counted in its "
    r"install size). "
    r"$^b$\,Bundles its language models in the wheel."
)
# Categorical slots 1 and 2 of the validated default palette, for the two
# Tweets metrics; the Survey Junk panel uses ink so colour is not needed.
PRECISION = "#2a78d6"
RECALL = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
METRICS = ("A_capture", "B_precision", "B_recall", "B_fpr")
# Methods whose joint misses on A are counted for the label-noise note.
NOISE_PANEL = UNTUNED
# One column of the two-column article, in inches (8.4 cm by 6 cm).
COLUMN = (8.4 / 2.54, 6.0 / 2.54)


def save(fig: Any, name: str) -> None:
    """No creation date or software stamp, so a rebuild gives the same
    bytes. The PNG copy is for checking the figure by eye."""
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            # Computer Modern, shipped with matplotlib, to match the paper
            # and keep the row labels narrow enough for one column.
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def positions() -> List[float]:
    """y positions, top to bottom, with a gap before the tuned row."""
    return [0.0, -1.0, -2.0, -3.0, -4.6]


# The dashed rule between the untuned rows and the tuned row.
RULE = -3.8


def interval(ax: Any, value: float, ci: List[float], y: float, colour: str):
    ax.errorbar(
        value,
        y,
        xerr=[[value - ci[0]], [ci[1] - value]],
        fmt="none",
        ecolor=colour,
        elinewidth=0.9,
        capsize=2,
    )


def figure_results(methods: Dict[str, Any]) -> None:
    """Survey Junk capture above, Tweets precision and recall below."""
    names = UNTUNED + TUNED
    ys = positions()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=COLUMN)
    for y, name in zip(ys, names):
        point = methods[name]["A_capture"]
        top.barh(y, point["value"], height=0.6, color="#86b6ef")
        interval(top, point["value"], point["ci"], y, INK)
        for key, colour, marker, dy in (
            ("B_precision", PRECISION, "o", 0.17),
            ("B_recall", RECALL, "s", -0.17),
        ):
            p = methods[name][key]
            bottom.scatter(
                p["value"], y + dy, s=12, marker=marker, color=colour, zorder=3
            )
            interval(bottom, p["value"], p["ci"], y + dy, colour)
    top.set_xlabel("Capture rate on Survey Junk", labelpad=2, loc="right")
    bottom.set_xlabel(
        "Precision and recall on Tweets vs. Random Strings",
        labelpad=2,
        loc="right",
    )
    top.set_xlim(0, 1.0)
    top.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    bottom.set_xlim(0.6, 1.0)
    bottom.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0], [".6", ".7", ".8", ".9", "1"])
    for ax in (top, bottom):
        ax.set_ylim(-5.1, 0.5)
        ax.set_yticks(ys, [PLAIN[n] for n in names])
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", pad=1.5)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.axhline(RULE, color=MUTED, linewidth=0.6, linestyle="--")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=PRECISION,
            markersize=4,
            label="precision",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=RECALL,
            markersize=4,
            label="recall",
        ),
    ]
    # Sits in the empty band left of the untuned points, clear of the data.
    bottom.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.55),
        frameon=False,
        handletextpad=0.2,
        borderaxespad=0.2,
        labelspacing=0.2,
    )
    fig.subplots_adjust(
        left=0.4, right=0.97, top=0.985, bottom=0.13, hspace=0.5
    )
    save(fig, "fig1_results")


def number(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def bracket(point: Dict[str, Any]) -> str:
    low, high = point["ci"]
    return f"[{low:.3f}, {high:.3f}]"


def stack(lines: List[str], align: str = "c", pos: str = "t") -> str:
    """Several lines in one cell, with only the packages already loaded."""
    body = r" \\ ".join(lines)
    spec = f"[{pos}]{{@{{}}{align}@{{}}}}"
    return rf"\begin{{tabular}}{spec}{body}\end{{tabular}}"


def rows(cells: Any) -> List[str]:
    """Body rows in table order, a rule before the tuned row."""
    lines: List[str] = []
    for name in UNTUNED + TUNED:
        if name in TUNED:
            lines.append(r"\midrule")
        lines.append(LABELS[name] + " & " + " & ".join(cells(name)) + r" \\")
    return lines


def survey_table(methods: Dict[str, Any]) -> str:
    """Capture rate on Survey Junk, every row of which is junk."""
    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.7cm}cc@{}}",
        r"\toprule",
        r"Method & Capture rate & 95\% CI \\",
        r"\midrule",
    ]
    lines += rows(
        lambda name: [
            number(methods[name]["A_capture"]["value"]),
            bracket(methods[name]["A_capture"]),
        ]
    )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def tweets_table(methods: Dict[str, Any]) -> str:
    """Precision, recall and FPR on Tweets vs. Random Strings, each with its
    95% CI on a second, smaller line."""

    def cells(name: str) -> List[str]:
        return [
            stack(
                [
                    number(methods[name][key]["value"]),
                    r"{\scriptsize " + bracket(methods[name][key]) + "}",
                ]
            )
            for key in ("B_precision", "B_recall", "B_fpr")
        ]

    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.4cm}"
        r"@{\hspace{4pt}}c@{\hspace{4pt}}c@{\hspace{4pt}}c@{}}",
        r"\toprule",
        r"Method & Precision & Recall & FPR \\",
        r"\midrule",
    ]
    body = rows(cells)
    # A little air between the two-line rows.
    lines += [
        line if line == r"\midrule" else line + r"[2pt]" for line in body
    ]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def footprint_table(footprint: Dict[str, Any]) -> str:
    """Speed, import time, install size and dependencies per method."""

    def head(*text: str) -> str:
        return stack(list(text), "c", "b")

    def cells(name: str) -> List[str]:
        row = footprint[name]
        return [
            f"{row['ms_per_1000']:.1f}",
            f"{row['import_s']:.3f}",
            f"{row['install_bytes'] / 1e6:.2f}",
            DEPENDENCIES[name],
        ]

    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.4cm}"
        r"@{\hspace{4pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}r"
        r"@{\hspace{4pt}}l@{}}",
        r"\toprule",
        "Method & "
        + " & ".join(
            [
                head("ms per", "1,000 texts"),
                head("Import", "time (s)"),
                head("Install", "size (MB)"),
                head("Runtime", "deps."),
            ]
        )
        + r" \\",
        r"\midrule",
    ]
    lines += rows(cells)
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{@{}>{\raggedright\arraybackslash}p{\linewidth}@{}}"
        r"{\footnotesize\rule{0pt}{2.5ex}" + DEPENDENCY_NOTES + r"} \\",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def a_misses(manifest: Dict) -> Dict[str, int]:
    flags = {
        name: read_predictions(
            PREDICTIONS / "A" / f"{name}.csv", manifest["A_ids"]
        )[1]
        for name in NOISE_PANEL
    }
    flagged_by_any = sum(flags.values())  # type: ignore[arg-type]
    return {
        "nAmissedfusion": int((~flags["english_fusion"]).sum()),
        "nAmissedall": int((flagged_by_any == 0).sum()),
    }


def macros(
    metrics: Dict[str, Any],
    footprint: Dict[str, Any],
    manifest: Dict,
    environment: Dict[str, Any],
) -> str:
    """\\res{method}{metric}, \\resci, \\respct, \\ms and dataset counts."""
    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\makeatletter",
    ]
    for name, method in metrics["methods"].items():
        for key in METRICS:
            point = method[key]
            value = point["value"]
            base = f"r@{name}@{key}"
            lines.append(
                rf"\expandafter\def\csname {base}\endcsname{{{number(value)}}}"
            )
            pct = "n/a" if value is None else f"{100 * value:.1f}"
            lines.append(
                rf"\expandafter\def\csname {base}@pct\endcsname{{{pct}}}"
            )
            ci = point["ci"]
            text = "n/a" if not ci else f"{ci[0]:.3f}--{ci[1]:.3f}"
            lines.append(
                rf"\expandafter\def\csname {base}@ci\endcsname{{{text}}}"
            )
    for name, row in footprint.items():
        lines.append(
            rf"\expandafter\def\csname ms@{name}\endcsname"
            rf"{{{row['ms_per_1000']:.1f}}}"
        )
        lines.append(
            rf"\expandafter\def\csname mb@{name}\endcsname"
            rf"{{{row['install_bytes'] / 1e6:.2f}}}"
        )
        lines.append(
            rf"\expandafter\def\csname imp@{name}\endcsname"
            rf"{{{row['import_s']:.3f}}}"
        )
    lines += [
        r"\newcommand{\res}[2]{\csname r@#1@#2\endcsname}",
        r"\newcommand{\respct}[2]{\csname r@#1@#2@pct\endcsname}",
        r"\newcommand{\resci}[2]{\csname r@#1@#2@ci\endcsname}",
        r"\newcommand{\ms}[1]{\csname ms@#1\endcsname}",
        r"\newcommand{\mb}[1]{\csname mb@#1\endcsname}",
        r"\newcommand{\importtime}[1]{\csname imp@#1\endcsname}",
        r"\makeatother",
    ]
    counts, notes = manifest["counts"], manifest["notes"]
    sources = manifest["sources"]
    share = notes["B_tweets_with_mojibake_share"]
    values = {
        "nA": f"{counts['A']:,}",
        "nArows": f"{sources['A']['rows_in_file']:,}",
        "nB": f"{counts['B']:,}",
        "nBpos": f"{counts['B_positive']:,}",
        "nBneg": f"{counts['B_negative']:,}",
        "nBdup": f"{notes['B_duplicate_text_rows']:,}",
        "mojibakepct": f"{100 * share:.1f}",
        "resamples": f"{metrics['resamples']:,}",
        "nstrategies": str(len(Strategy)),
        "cpuname": environment["cpu"],
        "pyversion": environment["python"],
        "shaA": next(iter(sources["A"]["files"].values()))[:16],
    }
    values.update({k: f"{v:,}" for k, v in a_misses(manifest).items()})
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    footprint = json.loads((RESULTS / "footprint.json").read_text())["methods"]
    manifest = json.loads(MANIFEST.read_text())
    environment = json.loads((RESULTS / "environment.json").read_text())
    methods = metrics["methods"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    style()
    figure_results(methods)
    outputs = {
        "table_survey.tex": survey_table(methods),
        "table_tweets.tex": tweets_table(methods),
        "table_footprint.tex": footprint_table(footprint),
        "numbers.tex": macros(metrics, footprint, manifest, environment),
    }
    for name, text in outputs.items():
        (TABLES / name).write_text(text)
    print(f"figures: wrote {FIGURES} and {TABLES}")


if __name__ == "__main__":
    main()
