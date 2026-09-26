"""Render publication figures from saved results; no detector calls."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import ROOT

ORDER = [
    "english",
    "english_extended",
    "legacy",
    "word_lookup",
    "entropy_based",
]
LABELS = ["English", "Extended", "Legacy", "Word lookup", "Entropy"]


def main() -> None:
    summary = json.loads((ROOT / "results/summary.json").read_text())
    runtime = json.loads((ROOT / "results/runtime.json").read_text())
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    defaults = {
        row["method"]: row
        for row in summary
        if row["experiment"] == "default" and row["view"] == 400
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ys = list(range(len(ORDER)))
    for y, method in zip(ys, ORDER):
        row = defaults[method]
        lo, hi = row["recall_wilson95_conditional"]
        axes[0].errorbar(
            row["recall"] * 100,
            y,
            xerr=[
                [max(0, row["recall"] - lo) * 100],
                [max(0, hi - row["recall"]) * 100],
            ],
            fmt="o",
            color="#245b8c",
            capsize=4,
        )
    axes[0].set(
        yticks=ys,
        yticklabels=LABELS,
        xlim=(-3, 105),
        xlabel="Detected published gibberish (%)",
        title="38 document prefixes; conditional Wilson 95%",
    )
    axes[0].invert_yaxis()
    for offset, (source, marker) in enumerate(
        [("kjv", "o"), ("net", "s"), ("secreta", "^"), ("wiki", "D")]
    ):
        axes[1].scatter(
            [
                defaults[m]["by_negative_document"][source]["fpr"] * 100
                for m in ORDER
            ],
            [y + (offset - 1.5) * 0.13 for y in ys],
            marker=marker,
            label=source.upper(),
            s=35,
        )
    axes[1].set(
        yticks=ys,
        yticklabels=LABELS,
        xlim=(-0.3, 4.8),
        xlabel="False flags within control document (%)",
        title="100 blocks per document; no passage-binomial CI",
    )
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(
        directory / "default-results.pdf", metadata={"CreationDate": None}
    )
    fig.savefig(directory / "default-results.png", dpi=160)
    plt.close(fig)

    methods = ORDER + ["char_bigram", "char_trigram"]
    folds = ["bible", "secreta", "wiki"]
    lookup = {
        (r["method"], r["fold"]): r
        for r in summary
        if r["experiment"] == "calibrated"
    }
    matrix = [
        [lookup[(m, f)]["macro_document_fpr"] * 100 for f in folds]
        for m in methods
    ]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    chart = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(
                x,
                y,
                "{:.1f}%".format(value),
                ha="center",
                va="center",
                color="white" if value > 60 else "black",
            )
    ax.set(
        xticks=range(3),
        xticklabels=["Bible family", "Secreta", "Wiki"],
        yticks=range(len(methods)),
        yticklabels=LABELS + ["Bigram", "Trigram"],
        xlabel="Held-out control family",
        title=(
            "Calibration transfer: test false-positive rate\n"
            "Validation budget = 1%; only four control documents"
        ),
    )
    fig.colorbar(chart, ax=ax, label="Macro document false-positive rate (%)")
    fig.savefig(
        directory / "calibration-transfer.pdf", metadata={"CreationDate": None}
    )
    fig.savefig(directory / "calibration-transfer.png", dpi=160)
    plt.close(fig)

    speeds = {r["method"]: r for r in runtime}
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh(
        LABELS + ["Bigram", "Trigram"],
        [speeds[m]["warm_median_ms"] for m in methods],
        color="#245b8c",
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Median warm scoring time (ms / 400-character input)",
        title=(
            "Apple M2, Python 3.12; 76 inputs × 5 passes\n"
            "Full score path; excludes import and construction"
        ),
    )
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(directory / "runtime.pdf", metadata={"CreationDate": None})
    fig.savefig(directory / "runtime.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
