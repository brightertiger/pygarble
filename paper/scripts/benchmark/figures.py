"""Figure, LaTeX tables and number macros for the preprint.

Reads `results/metrics.json`, `results/footprint.json`, the A predictions
the manifest and the tuning study in `results/tuning/`; writes the results
figure (PDF, and a 300 dpi PNG for checking) to `paper/preprint/figures/`
and the LaTeX fragments `table_survey.tex`, `table_tweets.tex`,
`table_footprint.tex`, `table_tuning.tex`, `table_weights.tex` and
`numbers.tex` to `paper/preprint/tables/`. Nothing in the paper is typed by
hand. Run from the repository root:

    python -m paper.scripts.benchmark.figures
"""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
# fontTools reports cmr10's 1990s timestamps as suspicious; they are fine.
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402

from pygarble.gibberish.registry import Strategy  # noqa: E402

from . import tune  # noqa: E402
from .data import MANIFEST, REPOSITORY  # noqa: E402
from .metrics import PREDICTIONS, RESULTS, read_predictions  # noqa: E402

PREPRINT = REPOSITORY / "paper" / "preprint"
FIGURES = PREPRINT / "figures"
TABLES = PREPRINT / "tables"

# Table and figure rows, top to bottom.
METHODS = ("english", "english_fusion", "gibberish_detector", "lingua")
LABELS = {
    "english": r"pygarble (\texttt{english})",
    "english_fusion": r"pygarble (\texttt{english\_fusion})",
    "gibberish_detector": "gibberish-detector",
    "lingua": "lingua",
}
# Typewriter marks the pygarble profiles, as in the tables.
PLAIN = {
    "english": r"pygarble ($\mathtt{english}$)",
    "english_fusion": r"pygarble ($\mathtt{english\_fusion}$)",
    "gibberish_detector": "gibberish-detector",
    "lingua": "lingua",
}
# Runtime dependencies, read from the installed distributions: none of the
# three declares a requirement. gibberish-detector ships no model and the
# benchmark trains one from big.txt; lingua's models sit inside its wheel.
DEPENDENCIES = {
    "english": "none",
    "english_fusion": "none",
    "gibberish_detector": r"none$^a$",
    "lingua": r"none$^b$",
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
NOISE_PANEL = METHODS
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
    """y positions, top to bottom."""
    return [0.0, -1.0, -2.0, -3.0]


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
    names = METHODS
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
        ax.set_ylim(-3.5, 0.5)
        ax.set_yticks(ys, [PLAIN[n] for n in names])
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", pad=1.5)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
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
    # Sits in the empty band left of the points, clear of the data.
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
        left=0.5, right=0.97, top=0.985, bottom=0.13, hspace=0.5
    )
    save(fig, "fig1_results")


def number(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def ranks(methods: Dict[str, Any], key: str) -> Dict[str, str]:
    """'*' for the best and '**' for the second best value of one metric
    among the rows (ties share a rank). Lowest is best for FPR, highest
    for everything else."""
    shown = {n: round(methods[n][key]["value"], 3) for n in METHODS}
    return marks_for(shown, key)


def marks_for(shown: Dict[str, float], key: str) -> Dict[str, str]:
    distinct = sorted(set(shown.values()), reverse=key != "B_fpr")
    marks = {distinct[0]: "*"}
    if len(distinct) > 1:
        marks[distinct[1]] = "**"
    return {n: marks.get(v, "") for n, v in shown.items()}


def bracket(point: Dict[str, Any]) -> str:
    low, high = point["ci"]
    return f"[{low:.3f}, {high:.3f}]"


def stack(lines: List[str], align: str = "c", pos: str = "t") -> str:
    """Several lines in one cell, with only the packages already loaded."""
    body = r" \\ ".join(lines)
    spec = f"[{pos}]{{@{{}}{align}@{{}}}}"
    return rf"\begin{{tabular}}{spec}{body}\end{{tabular}}"


def rows(
    cells: Any, names: Sequence[str] = METHODS, labels: Any = None
) -> List[str]:
    """Body rows in table order."""
    labels = labels or LABELS
    lines: List[str] = []
    for name in names:
        lines.append(labels[name] + " & " + " & ".join(cells(name)) + r" \\")
    return lines


# Method names on two lines, matching the two-line cells of Table 3.
STACKED = {
    "english": stack(["pygarble", r"(\texttt{english})"], "l"),
    "english_fusion": stack(["pygarble", r"(\texttt{english\_fusion})"], "l"),
    "gibberish_detector": "gibberish-detector",
    "lingua": "lingua",
}


def survey_table(methods: Dict[str, Any]) -> str:
    """Capture rate on Survey Junk, every row of which is junk."""
    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Method & Capture rate & 95\% CI \\",
        r"\midrule",
    ]
    marks = ranks(methods, "A_capture")
    lines += rows(
        lambda name: [
            number(methods[name]["A_capture"]["value"]) + marks.get(name, ""),
            bracket(methods[name]["A_capture"]),
        ]
    )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def tweets_table(methods: Dict[str, Any]) -> str:
    """Precision, recall and FPR on Tweets vs. Random Strings, each with its
    95% CI on a second, smaller line."""

    marks = {
        k: ranks(methods, k) for k in ("B_precision", "B_recall", "B_fpr")
    }

    def cells(name: str) -> List[str]:
        return [
            stack(
                [
                    number(methods[name][key]["value"])
                    + marks[key].get(name, ""),
                    r"{\scriptsize " + bracket(methods[name][key]) + "}",
                ]
            )
            for key in ("B_precision", "B_recall", "B_fpr")
        ]

    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\begin{tabular}{@{}l"
        r"@{\hspace{6pt}}c@{\hspace{6pt}}c@{\hspace{6pt}}c@{}}",
        r"\toprule",
        r"Method & Precision & Recall & FPR \\",
        r"\midrule",
    ]
    body = rows(cells, labels=STACKED)
    # A little air between the two-line rows.
    lines += [line + r"[2pt]" for line in body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def footprint_table(footprint: Dict[str, Any]) -> str:
    """Speed, import time, install size and dependencies per method."""

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
        r"\begin{tabular}{@{}l"
        r"@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r"
        r"@{\hspace{8pt}}l@{}}",
        r"\toprule",
        "Method & "
        + " & ".join(
            [
                "ms/1k",
                stack(["Import", "(s)"], "r", "b"),
                stack(["Install", "(MB)"], "r", "b"),
                "Deps",
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
        "Amedianchars": f"{notes['A_median_chars']:g}",
        "Asingletokenpct": f"{100 * notes['A_single_token_fraction']:.1f}",
        "nBlanguages": spelled(notes["B_languages"]),
    }
    values.update({k: f"{v:,}" for k, v in a_misses(manifest).items()})
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


TUNING_ROWS = ("english", "english_fusion", "tuned")
TUNING_LABELS = {
    "english": r"pygarble (\texttt{english})",
    "english_fusion": r"pygarble (\texttt{english\_fusion})",
    "tuned": "pygarble (tuned weights)",
}


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven")


def spelled(count: int) -> str:
    return WORDS[count] if count < len(WORDS) else str(count)


def tex_name(name: str) -> str:
    """Typewriter name that may break after an underscore in prose."""
    return r"\texttt{" + name.replace("_", r"\_\allowbreak{}") + "}"


def tuning_table(tuning: Dict[str, Any]) -> str:
    """Held-out metrics: the primary draw with intervals, then the mean
    and standard deviation over all draws."""
    primary = tuning["primary"]["methods"]
    spread = tuning["mean_sd"]
    seed = tuning["primary_seed"]
    draws = spelled(len(tuning["seeds"]))

    def block_marks(values: Dict[str, Dict[str, float]]) -> Dict:
        return {
            key: marks_for(
                {n: round(values[n][key], 3) for n in TUNING_ROWS}, key
            )
            for key in METRICS
        }

    first = block_marks(
        {n: {k: primary[n][k]["value"] for k in METRICS} for n in primary}
    )
    second = block_marks(
        {n: {k: spread[n][k]["mean"] for k in METRICS} for n in spread}
    )

    def point(name: str) -> List[str]:
        return [
            stack(
                [
                    number(primary[name][key]["value"]) + first[key][name],
                    r"{\scriptsize " + bracket(primary[name][key]) + "}",
                ]
            )
            for key in METRICS
        ]

    def mean_sd(name: str) -> List[str]:
        return [
            f"{spread[name][key]['mean']:.3f}{second[key][name]} $\\pm$ "
            f"{spread[name][key]['sd']:.3f}"
            for key in METRICS
        ]

    def body(cells: Any) -> List[str]:
        return [
            TUNING_LABELS[n] + " & " + " & ".join(cells(n)) + r" \\"
            for n in TUNING_ROWS
        ]

    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r" & Survey Junk & \multicolumn{3}{c}{Tweets vs.\ Random Strings} \\",
        r"\cmidrule(lr){2-2}\cmidrule(l){3-5}",
        r"Method & Capture & Precision & Recall & FPR \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Primary draw (seed "
        + str(seed)
        + r"), with 95\,\% bootstrap interval}} \\",
    ]
    lines += [line + r"[2pt]" for line in body(point)]
    lines += [
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Mean $\pm$ standard deviation over "
        + draws
        + r" draws}} \\",
    ]
    lines += body(mean_sd)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def by_mean_weight(weights: Dict[str, Any]) -> List[str]:
    return sorted(
        weights["mean_weight"],
        key=lambda n: (-weights["mean_weight"][n], n),
    )


def primary_draw(weights: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return next(d for d in weights["draws"] if d["seed"] == seed)


def weights_table(weights: Dict[str, Any], seed: int) -> str:
    """Every strategy's mean weight over the draws and its weight in the
    primary draw, sorted by mean weight."""
    primary = primary_draw(weights, seed)["weights"]
    lines = [
        "% Generated by paper/scripts/benchmark/figures.py; do not edit.",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Strategy & Mean weight & Seed " + str(seed) + r" \\",
        r"\midrule",
    ]
    lines += [
        f"{tex_name(n)} & {weights['mean_weight'][n]:.2f} & "
        f"{primary[n]:.2f} \\\\"
        for n in by_mean_weight(weights)
    ]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def listing(items: List[str]) -> str:
    if len(items) < 3:
        return " and ".join(items)
    # A serial comma keeps a final item that itself contains "and" apart.
    last = (", and " if " and " in items[-1] else " and ") + items[-1]
    return ", ".join(items[:-1]) + last


def weight_groups(
    order: List[str], mean: Dict[str, float], count: int
) -> List[str]:
    """The first `count` strategies of `order`, extended so that no group
    of strategies tied at the same two-decimal mean weight is cut, each
    tied group written once with "each"."""
    groups: List[List[str]] = []
    for name in order:
        if groups and f"{mean[groups[-1][0]]:.2f}" == f"{mean[name]:.2f}":
            groups[-1].append(name)
        elif sum(map(len, groups)) >= count:
            break
        else:
            groups.append([name])
    return [
        listing([tex_name(n) for n in group])
        + f" ({mean[group[0]]:.2f}{' each' if len(group) > 1 else ''})"
        for group in groups
    ]


ENGLISH_MEMBERS = {
    "englishmarkovpct": ("markov_chain",),
    "englishcontrolpct": ("control_characters",),
    "englishmojibakepct": ("mojibake",),
    "englishencodingpct": ("mojibake", "control_characters"),
    "englishrestpct": (
        "markov_chain",
        "log_likelihood_ratio",
        "word_anomaly",
        "keyboard_adjacency",
    ),
}


def member_macros(manifest: Dict) -> Dict[str, str]:
    """Share of B tweets flagged by members of the english profile, alone
    or in groups, from the cached strategy scores (a member flags at a
    score of 0.5, the profile's threshold, when it applies)."""
    names = tune.strategy_names()
    scores, applicable = tune.read_scores(manifest, names)["B"]
    tweets = [manifest["B_labels"][i] == 0 for i in manifest["B_ids"]]
    flags = (scores >= 0.5) & applicable
    out = {}
    for macro, members in ENGLISH_MEMBERS.items():
        columns = [names.index(m) for m in members]
        flagged = flags[:, columns].any(axis=1)[tweets]
        out[macro] = f"{100 * flagged.mean():.1f}"
    return out


def tuning_macros(
    tuning: Dict[str, Any], weights: Dict[str, Any], splits: Dict
) -> str:
    """\\tres{method}{metric} (primary draw), \\trespct, \\tresci,
    \\tmean, \\tmeanpct, \\tsd, \\tsdpct and the tuning setup."""
    lines = [r"\makeatletter"]

    def define(name: str, value: str) -> None:
        lines.append(rf"\expandafter\def\csname {name}\endcsname{{{value}}}")

    for name in TUNING_ROWS:
        for key in METRICS:
            point = tuning["primary"]["methods"][name][key]
            spread = tuning["mean_sd"][name][key]
            base = f"t@{name}@{key}"
            define(base, number(point["value"]))
            define(f"{base}@pct", f"{100 * point['value']:.1f}")
            low, high = point["ci"]
            define(f"{base}@ci", f"{low:.3f}--{high:.3f}")
            define(f"{base}@mean", f"{spread['mean']:.3f}")
            define(f"{base}@meanpct", f"{100 * spread['mean']:.1f}")
            define(f"{base}@sd", f"{spread['sd']:.3f}")
            define(f"{base}@sdpct", f"{100 * spread['sd']:.1f}")
    lines += [
        r"\newcommand{\tres}[2]{\csname t@#1@#2\endcsname}",
        r"\newcommand{\trespct}[2]{\csname t@#1@#2@pct\endcsname}",
        r"\newcommand{\tresci}[2]{\csname t@#1@#2@ci\endcsname}",
        r"\newcommand{\tmean}[2]{\csname t@#1@#2@mean\endcsname}",
        r"\newcommand{\tmeanpct}[2]{\csname t@#1@#2@meanpct\endcsname}",
        r"\newcommand{\tsd}[2]{\csname t@#1@#2@sd\endcsname}",
        r"\newcommand{\tsdpct}[2]{\csname t@#1@#2@sdpct\endcsname}",
        r"\makeatother",
    ]
    seed = tuning["primary_seed"]
    primary = primary_draw(weights, seed)
    draws = weights["draws"]
    members = [sum(w > 0 for w in d["weights"].values()) for d in draws]
    thresholds = [d["threshold"] for d in draws]
    f1s = [d["tuning_f1"] for d in tuning["per_seed"]]
    order = by_mean_weight(weights)
    mean = weights["mean_weight"]
    sample = splits["tuning_ids"][str(seed)]
    rows = tuning["primary"]["rows"]
    worst = min(
        tuning["per_seed"], key=lambda d: d["methods"]["tuned"]["B_recall"]
    )["methods"]["tuned"]
    # The bottom groups are found from the smallest weight up, then listed
    # from the larger weight down like the top ones.
    bottom = weight_groups(
        sorted(order, key=lambda n: (mean[n], n)), mean, 3
    )[::-1]
    values = {
        "tuningdraws": spelled(len(draws)),
        "tuningtrials": f"{weights['trials']:,}",
        "tuningpct": f"{100 * tuning['fraction_tuning']:.0f}",
        "heldoutpct": f"{100 * (1 - tuning['fraction_tuning']):.0f}",
        "tuningseed": str(seed),
        "tuningseedlast": str(draws[-1]["seed"]),
        "tuningweightstep": f"{weights['weight_step']:.2f}",
        "tuninglow": f"{weights['threshold_range'][0]:.2f}",
        "tuninghigh": f"{weights['threshold_range'][1]:.2f}",
        "tuningthreshold": f"{primary['threshold']:.2f}",
        "tuningthresholdmin": f"{min(thresholds):.2f}",
        "tuningthresholdmax": f"{max(thresholds):.2f}",
        "tuningmembers": str(sum(w > 0 for w in primary["weights"].values())),
        "tuningmembersmin": str(min(members)),
        "tuningmembersmax": str(max(members)),
        "tuningfonemin": f"{min(f1s):.3f}",
        "tuningfonemax": f"{max(f1s):.3f}",
        "tuningseconds": f"{weights['seconds_total'] / len(draws):.0f}",
        "tuningtotal": f"{weights['seconds_total']:.0f}",
        "tuningscoring": f"{weights['scoring_seconds']:.0f}",
        "optunaversion": weights["optuna"],
        "nTuneA": f"{len(sample['A']):,}",
        "nTuneB": f"{len(sample['B']):,}",
        "nTune": f"{len(sample['A']) + len(sample['B']):,}",
        "tuningbestmin": f"{min(d['best_trial'] for d in draws):,}",
        "tuningbestmax": f"{max(d['best_trial'] for d in draws):,}",
        "nHeldA": f"{rows['A']:,}",
        "nHeldB": f"{rows['B']:,}",
        "nHeldBpos": f"{rows['B_positive']:,}",
        "nHeldBneg": f"{rows['B_negative']:,}",
        "tuningtop": listing(weight_groups(order, mean, 4)),
        "tuningbottom": listing(bottom),
        "tuningworstrecall": f"{100 * worst['B_recall']:.1f}",
        "tuningworstcapture": f"{100 * worst['A_capture']:.1f}",
        "tuningalloff": listing(
            [
                tex_name(n)
                for n in order
                if all(d["weights"][n] == 0 for d in draws)
            ]
        ),
        "tuningoff": listing(
            [tex_name(n) for n, w in primary["weights"].items() if w == 0]
        ),
    }
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    footprint = json.loads((RESULTS / "footprint.json").read_text())["methods"]
    manifest = json.loads(MANIFEST.read_text())
    environment = json.loads((RESULTS / "environment.json").read_text())
    methods = metrics["methods"]
    tuning_dir = RESULTS / "tuning"
    tuning = json.loads((tuning_dir / "tuning.json").read_text())
    weights = json.loads((tuning_dir / "weights.json").read_text())
    splits = json.loads((tuning_dir / "splits.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    style()
    figure_results(methods)
    outputs = {
        "table_survey.tex": survey_table(methods),
        "table_tweets.tex": tweets_table(methods),
        "table_footprint.tex": footprint_table(footprint),
        "table_tuning.tex": tuning_table(tuning),
        "table_weights.tex": weights_table(weights, tuning["primary_seed"]),
        "numbers.tex": macros(metrics, footprint, manifest, environment)
        + tuning_macros(tuning, weights, splits)
        + "".join(
            rf"\newcommand{{\{k}}}{{{v}}}" + "\n"
            for k, v in member_macros(manifest).items()
        ),
    }
    for name, text in outputs.items():
        (TABLES / name).write_text(text)
    print(f"figures: wrote {FIGURES} and {TABLES}")


if __name__ == "__main__":
    main()
