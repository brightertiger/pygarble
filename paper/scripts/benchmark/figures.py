"""Figure, LaTeX tables and number macros for the preprint.

Reads `results/metrics.json`, `results/footprint.json`, the A predictions
and the manifest; writes a PDF to `paper/preprint/figures/` and LaTeX
fragments to `paper/preprint/tables/`. Nothing in the paper is typed by
hand. Run from the repository root:

    python -m paper.scripts.benchmark.figures
"""

import json
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
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
    "english": "pygarble english (default)",
    "english_fusion": r"pygarble english\_fusion",
    "gibberish_detector": "gibberish-detector",
    "lingua": "lingua (not English)",
    "english_tuned": "pygarble english, tuned",
}
PLAIN = {
    "english": "pygarble english (default)",
    "english_fusion": "pygarble english_fusion",
    "gibberish_detector": "gibberish-detector",
    "lingua": "lingua (not English)",
    "english_tuned": "pygarble english, tuned\n(post hoc)",
}
# Categorical slots 1 and 2 of the validated default palette, for the two
# B metrics; the A panel uses ink so colour is not needed to read it.
PRECISION = "#2a78d6"
RECALL = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
METRICS = ("A_capture", "B_precision", "B_recall", "B_fpr")
# Methods whose joint misses on A are counted for the label-noise note.
NOISE_PANEL = UNTUNED


def save(fig: Any, name: str) -> None:
    """No creation date, so a rebuild gives the same bytes."""
    fig.savefig(FIGURES / name, metadata={"CreationDate": None})
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
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
    names = UNTUNED + TUNED
    ys = positions()
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for y, name in zip(ys, names):
        point = methods[name]["A_capture"]
        left.barh(y, point["value"], height=0.6, color="#86b6ef")
        interval(left, point["value"], point["ci"], y, INK)
        for key, colour, marker, dy in (
            ("B_precision", PRECISION, "o", 0.14),
            ("B_recall", RECALL, "s", -0.14),
        ):
            p = methods[name][key]
            right.scatter(
                p["value"], y + dy, s=20, marker=marker, color=colour, zorder=3
            )
            interval(right, p["value"], p["ci"], y + dy, colour)
    left.set_xlabel("A: capture rate on survey junk")
    right.set_xlabel("B: precision and recall")
    for ax in (left, right):
        ax.set_xlim(0, 1.0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    left.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    right.set_xlim(0.6, 1.0)
    right.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0], [".6", ".7", ".8", ".9", "1"])
    left.set_yticks(ys, [PLAIN[n] for n in names])
    left.axhline(-3.8, color=MUTED, linewidth=0.5, linestyle=":")
    right.axhline(-3.8, color=MUTED, linewidth=0.5, linestyle=":")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=PRECISION,
            label="precision",
        ),
        plt.Line2D(
            [], [], marker="s", linestyle="", color=RECALL, label="recall"
        ),
    ]
    right.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.29, right=0.98, top=0.9, bottom=0.18, wspace=0.12
    )
    save(fig, "fig1_results.pdf")


def number(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def with_ci(point: Dict[str, Any]) -> str:
    if point["value"] is None:
        return "n/a"
    low, high = point["ci"]
    return f"{point['value']:.3f} {{\\scriptsize [{low:.3f}, {high:.3f}]}}"


def results_table(methods: Dict[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{p{4.6cm}llll}",
        r"\toprule",
        r"Method & A capture & B precision & B recall & B FPR \\",
        r"\midrule",
    ]
    for name in UNTUNED:
        cells = [with_ci(methods[name][k]) for k in METRICS]
        lines.append(LABELS[name] + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Post hoc, domain-specific tuning: "
        r"english without mojibake and control characters}} \\",
    ]
    for name in TUNED:
        cells = [with_ci(methods[name][k]) for k in METRICS]
        lines.append(LABELS[name] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def footprint_table(footprint: Dict[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & ms per 1,000 texts & Install (MB) \\",
        r"\midrule",
    ]
    for name in UNTUNED + TUNED:
        if name in TUNED:
            lines.append(r"\midrule")
        row = footprint[name]
        lines.append(
            f"{LABELS[name]} & {row['ms_per_1000']:.1f} & "
            f"{row['install_bytes'] / 1e6:.2f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
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
        "table1_results.tex": results_table(methods),
        "table2_footprint.tex": footprint_table(footprint),
        "numbers.tex": macros(metrics, footprint, manifest, environment),
    }
    for name, text in outputs.items():
        (TABLES / name).write_text(text)
    print(f"figures: wrote {FIGURES} and {TABLES}")


if __name__ == "__main__":
    main()
