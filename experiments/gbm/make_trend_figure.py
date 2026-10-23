#!/usr/bin/env python3
"""Emit paper/fig_trend.pdf from experiments/gbm/slice_history.json.

One panel, four lines: how each real-CTIME metric moves as the slice grows.
Vector PDF, greyscale-safe (distinct markers and dash patterns), sized for a
single IEEE Access column.

    python3 experiments/gbm/make_trend_figure.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "gbm" / "slice_history.json"
DST = ROOT / "paper" / "fig_trend.pdf"

SERIES = [
    ("computeIso", "Compute isolation", "o", "-"),
    ("ariaAcc", "Four-way accuracy", "s", "--"),
    ("sensorIso", "Sensor isolation", "^", "-."),
    ("novelIso", "Novel isolation", "D", ":"),
]


def main() -> None:
    runs = json.loads(SRC.read_text())["runs"]
    x = [r["perClass"] for r in runs]

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    for key, label, marker, dash in SERIES:
        ax.plot(
            x,
            [100.0 * r[key] for r in runs],
            marker=marker,
            linestyle=dash,
            linewidth=1.1,
            markersize=4,
            color="black",
            markerfacecolor="white",
            label=label,
        )

    ax.set_xlabel("Triggers per class in the real CTIME slice")
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_ylim(-5, 105)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.07), frameon=False, fontsize=7, handlelength=2.6, labelspacing=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(DST, format="pdf", bbox_inches="tight")
    print(f"wrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
