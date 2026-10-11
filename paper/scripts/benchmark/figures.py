"""Figures, LaTeX tables and number macros for the preprint.

Reads `results/metrics.json`, `results/footprint.json` and the manifest;
writes PDFs to `paper/preprint/figures/` and LaTeX fragments to
`paper/preprint/tables/`. Nothing in the paper is typed by hand. Run from
the repository root:

    python -m paper.scripts.benchmark.figures
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import MANIFEST, REPOSITORY  # noqa: E402
from .metrics import PREDICTIONS, RESULTS, read_predictions  # noqa: E402

PREPRINT = REPOSITORY / "paper" / "preprint"
FIGURES = PREPRINT / "figures"
TABLES = PREPRINT / "tables"

PROFILES = (
    "english_fusion",
    "english",
    "english_extended",
    "legacy",
    "llm_output",
    "corruption",
    "spoofing",
)
VOTING = (
    "english_vote_any",
    "english_vote_majority",
    "english_vote_all",
    "english_vote_average",
    "english_vote_weighted",
    "english_vote_fisher",
)
BASELINES = (
    "gibberish_detector",
    "nostril",
    "gibberish_chat_detector",
    "lingua",
    "gopher_rules",
    "c4_rules",
)
GROUPS = (
    ("pygarble profiles", PROFILES),
    ("Voting modes on english", VOTING),
    ("Baselines", BASELINES),
)
LABELS = {
    "english_vote_any": "english, any (default)",
    "english_vote_majority": "english, majority",
    "english_vote_all": "english, all",
    "english_vote_average": "english, average",
    "english_vote_weighted": "english, weighted",
    "english_vote_fisher": "english, fisher",
    "gibberish_detector": "gibberish-detector",
    "gibberish_chat_detector": "gibberish-chat-detector",
    "nostril": "nostril",
    "lingua": "lingua (not English)",
    "gopher_rules": "Gopher rules",
    "c4_rules": "C4 rules",
}
# Categorical slots 1 and 2 of the validated default palette; voting modes
# take a lighter step of slot 1. Baselines also get a hatch so the
# distinction survives greyscale printing.
COLOURS = {
    "pygarble profiles": "#2a78d6",
    "Voting modes on english": "#86b6ef",
    "Baselines": "#eb6834",
}
HATCH = {"Baselines": "////"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
METRICS = ("A_capture", "B_precision", "B_recall", "B_fpr")
# Methods whose joint misses on A are counted for the label-noise note:
# the four English profiles and the three non-rule baselines.
NOISE_PANEL = (
    "english_fusion",
    "english",
    "english_extended",
    "legacy",
    "nostril",
    "gibberish_detector",
    "lingua",
)


def save(fig: Any, name: str) -> None:
    """No creation date, so a rebuild gives the same bytes."""
    fig.savefig(FIGURES / name, metadata={"CreationDate": None})
    plt.close(fig)


def label(name: str) -> str:
    return LABELS.get(name, name)


def tex(text: str) -> str:
    return text.replace("_", r"\_")


def group_of(name: str) -> str:
    return next(g for g, names in GROUPS if name in names)


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


def ordered(methods: Dict[str, Any], key: str) -> List[Tuple[str, str]]:
    """(group, name) rows: groups in order, each sorted by `key` descending."""
    rows: List[Tuple[str, str]] = []
    for group, names in GROUPS:
        chosen = sorted(
            names,
            key=lambda n: -(methods[n][key]["value"] or 0.0),
        )
        rows += [(group, n) for n in chosen]
    return rows


def bars(
    ax: Any,
    methods: Dict[str, Any],
    rows: Sequence[Tuple[str, str]],
    metric: str,
    positions: Sequence[float],
) -> None:
    for y, (group, name) in zip(positions, rows):
        point = methods[name][metric]
        if not point["value"]:
            text = "n/a" if point["value"] is None else "0"
            ax.text(0.01, y, text, va="center", fontsize=7, color=MUTED)
            continue
        ax.barh(
            y,
            point["value"],
            height=0.7,
            color=COLOURS[group],
            hatch=HATCH.get(group),
            edgecolor="white",
            linewidth=0.5,
        )
        low, high = point["ci"]
        ax.errorbar(
            point["value"],
            y,
            xerr=[[point["value"] - low], [high - point["value"]]],
            color=INK,
            linewidth=0.8,
            capsize=1.5,
        )
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def layout(rows: Sequence[Tuple[str, str]]) -> List[float]:
    """y positions with a gap between groups, top to bottom."""
    positions, y, previous = [], 0.0, None
    for group, _ in rows:
        if previous is not None and group != previous:
            y += 0.8
        positions.append(-y)
        y += 1.0
        previous = group
    return positions


def legend(fig: Any) -> None:
    handles = [
        matplotlib.patches.Patch(
            facecolor=COLOURS[g],
            hatch=HATCH.get(g),
            edgecolor="white",
            label=g,
        )
        for g, _ in GROUPS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )


def figure_capture(methods: Dict[str, Any]) -> None:
    rows = ordered(methods, "A_capture")
    positions = layout(rows)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    bars(ax, methods, rows, "A_capture", positions)
    ax.set_yticks(positions, [label(n) for _, n in rows])
    ax.set_xlabel("Capture rate on A (share of junk rows flagged)")
    fig.subplots_adjust(left=0.36, right=0.97, top=0.98, bottom=0.17)
    legend(fig)
    save(fig, "fig1_capture_a.pdf")


def figure_precision_recall(methods: Dict[str, Any]) -> None:
    rows = ordered(methods, "B_recall")
    positions = layout(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.2), sharey=True)
    for ax, metric, title in zip(
        axes, ("B_precision", "B_recall"), ("Precision on B", "Recall on B")
    ):
        bars(ax, methods, rows, metric, positions)
        ax.set_xlabel(title)
    axes[0].set_yticks(positions, [label(n) for _, n in rows])
    fig.subplots_adjust(
        left=0.27, right=0.98, top=0.98, bottom=0.17, wspace=0.12
    )
    legend(fig)
    save(fig, "fig2_precision_recall_b.pdf")


# Label offsets in points for the cost figure, chosen once by eye so that
# labels do not overlap; methods not listed are labelled to the right.
OFFSETS = {
    "english": (4, -9),
    "english_fusion": (4, 3),
    "legacy": (-5, 2),
    "corruption": (-5, 3),
    "gopher_rules": (-5, -3),
    "lingua": (-5, -10),
}
ALIGN_LEFT = {"legacy", "corruption", "gopher_rules", "lingua"}


def figure_cost(methods: Dict[str, Any], footprint: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for group, names in GROUPS:
        for name in names:
            ms = footprint[name]["ms_per_1000"]
            capture = methods[name]["A_capture"]["value"]
            ax.scatter(
                ms,
                capture,
                s=22,
                marker="s" if group == "Baselines" else "o",
                color=COLOURS[group],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            if group == "Voting modes on english":
                continue
            dx, dy = OFFSETS.get(name, (4, 2))
            ax.annotate(
                label(name),
                (ms, capture),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.5,
                color=INK,
                ha="right" if name in ALIGN_LEFT else "left",
            )
    ax.set_xscale("log")
    ax.set_xlim(3, 3000)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Milliseconds per 1,000 texts (log scale, one thread)")
    ax.set_ylabel("Capture rate on A")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="s" if g == "Baselines" else "o",
            linestyle="",
            color=COLOURS[g],
            label=g + (" (unlabelled)" if g.startswith("Voting") else ""),
        )
        for g, _ in GROUPS
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.14)
    save(fig, "fig3_cost.pdf")


def number(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def with_ci(point: Dict[str, Any]) -> str:
    if point["value"] is None:
        return "n/a"
    low, high = point["ci"]
    return f"{point['value']:.3f} {{\\scriptsize [{low:.3f}, {high:.3f}]}}"


def results_table(
    methods: Dict[str, Any], groups: Sequence[Tuple[str, Sequence[str]]]
) -> str:
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Method & A capture & B precision & B recall & B FPR \\",
        r"\midrule",
    ]
    for i, (group, names) in enumerate(groups):
        if len(groups) > 1:
            if i:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{group}}}}} \\")
        for name in names:
            cells = [with_ci(methods[name][k]) for k in METRICS]
            lines.append(tex(label(name)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def footprint_table(footprint: Dict[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & ms per 1,000 texts & Import (s) & Install (MB) \\",
        r"\midrule",
    ]
    for i, (group, names) in enumerate(GROUPS):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{group}}}}} \\")
        for name in names:
            row = footprint[name]
            lines.append(
                f"{tex(label(name))} & {row['ms_per_1000']:.1f} & "
                f"{row['import_s']:.3f} & {row['install_bytes'] / 1e6:.2f}"
                r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def strategy_table(methods: Dict[str, Any], footprint: Dict[str, Any]) -> str:
    names = sorted(
        (n for n, m in methods.items() if m["family"] == "strategy"),
        key=lambda n: -methods[n]["A_capture"]["value"],
    )
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Strategy & A capture & B precision & B recall & B FPR "
        r"& ms per 1,000 \\",
        r"\midrule",
    ]
    for name in names:
        cells = [number(methods[name][k]["value"]) for k in METRICS]
        ms = f"{footprint[name]['ms_per_1000']:.1f}"
        short = tex(name.replace("strategy_", ""))
        lines.append(f"{short} & " + " & ".join(cells) + f" & {ms} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def macro_name(name: str) -> str:
    return name.replace("_", "")


def a_misses(manifest: Dict) -> Dict[str, int]:
    flags = {
        name: read_predictions(
            PREDICTIONS / "A" / f"{name}.csv", manifest["A_ids"]
        )[1]
        for name in NOISE_PANEL
    }
    flagged_by_any = sum(flags.values())
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
        "nstrategies": str(
            sum(m["family"] == "strategy" for m in metrics["methods"].values())
        ),
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
    figure_capture(methods)
    figure_precision_recall(methods)
    figure_cost(methods, footprint)
    outputs = {
        "table1_results.tex": results_table(methods, GROUPS),
        "table2_footprint.tex": footprint_table(footprint),
        "tableA_strategies.tex": strategy_table(methods, footprint),
        "numbers.tex": macros(metrics, footprint, manifest, environment),
    }
    for name, text in outputs.items():
        (TABLES / name).write_text(text)
    print(f"figures: wrote {FIGURES} and {TABLES}")


if __name__ == "__main__":
    main()
