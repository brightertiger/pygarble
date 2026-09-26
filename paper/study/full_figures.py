"""Publication figures for the complete corpus; reads saved outcomes only."""

import gzip
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .chunk_metrics import summarize as chunk_summary  # noqa: E402
from .data import ROOT  # noqa: E402
from .full_paper import CONTROL_NAMES, NAMES  # noqa: E402


def save(fig: object, name: str) -> None:
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / (name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(directory / (name + ".png"), dpi=160)
    plt.close(fig)


def confusion_figure(summary: dict) -> None:
    rows = {r["method"]: r for r in chunk_summary(summary)}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for ax, method in zip(axes, ("english", "word_lookup", "hf_strict")):
        row = rows[method]
        counts = [[row["tp"], row["fn"]], [row["fp"], row["tn"]]]
        fractions = [[v / sum(line) for v in line] for line in counts]
        ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        for y, line in enumerate(counts):
            for x, count in enumerate(line):
                ax.text(
                    x,
                    y,
                    "{:,}\n({:.1%})".format(count, fractions[y][x]),
                    ha="center",
                    va="center",
                    color="white" if fractions[y][x] > 0.55 else "black",
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Gibberish", "Meaningful"],
            yticks=[0, 1],
            yticklabels=["Gibberish", "Meaningful"],
            xlabel="Predicted class",
            ylabel="Inherited source class",
            title="{}\nAccuracy {:.2%}".format(NAMES[method], row["accuracy"]),
        )
    save(fig, "chunk-confusions")


def main() -> None:
    summary = json.loads((ROOT / "full-results/summary.json").read_text())
    docs = json.loads(
        gzip.decompress(
            (ROOT / "full-results/document-results.json.gz").read_bytes()
        )
    )
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    confusion_figure(summary)
    primary = {r["method"]: r for r in summary["primary"]}
    ys = list(range(len(NAMES)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for y, method in zip(ys, NAMES):
        row = primary[method]
        lo, hi = row["recall_wilson95_conditional"]
        point = row["document_recall"]
        axes[0].errorbar(
            point * 100,
            y,
            xerr=[[max(0, point - lo) * 100], [max(0, hi - point) * 100]],
            fmt="o",
            color="#245b8c",
            capsize=4,
        )
    axes[0].set(
        yticks=ys,
        yticklabels=list(NAMES.values()),
        xlim=(-3, 105),
        xlabel="Positive document recall (%)",
        title="38 documents; majority-chunk rule",
    )
    for offset, (member, label) in enumerate(CONTROL_NAMES.items()):
        lookup = {r["method"]: r for r in docs if r["member"] == member}
        axes[1].scatter(
            [lookup[m]["flagged_fraction"] * 100 for m in NAMES],
            [y + (offset - 1.5) * 0.13 for y in ys],
            marker=["o", "s", "D", "^"][offset],
            label=label,
            s=30,
        )
    axes[1].set(
        yticks=ys,
        yticklabels=list(NAMES.values()),
        xlim=(-3, 105),
        xlabel="False-flagged chunks within source (%)",
        title="All 5,200 English control chunks",
    )
    axes[1].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    save(fig, "full-comparison")

    languages = sorted({r["language"] for r in summary["by_language"]})
    lookup = {(r["method"], r["language"]): r for r in summary["by_language"]}
    matrix = [
        [lookup[(m, language)]["macro_fpr"] * 100 for m in NAMES]
        for language in languages
    ]
    fig, ax = plt.subplots(figsize=(8.5, 10), layout="constrained")
    chart = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(
                x,
                y,
                "{:.0f}".format(value),
                ha="center",
                va="center",
                color="white" if value > 65 else "black",
                fontsize=7,
            )
    ax.set(
        xticks=range(len(NAMES)),
        xticklabels=list(NAMES.values()),
        yticks=range(len(languages)),
        yticklabels=[
            "{} ({})".format(lang, lookup[("english", lang)]["documents"])
            for lang in languages
        ],
        title="Meaningful controls: mean within-document false-flag rate (%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    fig.colorbar(
        chart, ax=ax, label="Macro document chunk FPR (%)", shrink=0.6
    )
    save(fig, "full-languages")


if __name__ == "__main__":
    main()
