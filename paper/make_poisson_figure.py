#!/usr/bin/env python3
"""Sensor isolation with and without the Poisson likelihood-ratio bit.

Reads experiments/gbm/poisson_filter.json. Writes paper/fig_poisson.pdf.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "experiments/gbm/poisson_filter.json").read_text())

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)
BLUE, ORANGE = "#0072BC", "#d9822b"

order = [
    ("Simulator", "simulator"),
    ("Draw 1", "draw1"),
    ("Draw 2", "draw2"),
    ("1 → 2", "oneToTwo"),
    ("2 → 1", "twoToOne"),
]
lookup = [100 * D[k]["headline"]["sensorIso"]["mean"] for _, k in order]
added = [100 * D[k]["orPoisson"]["sensorIso"]["mean"] for _, k in order]
lookup_e = [100 * D[k]["headline"]["sensorIso"]["std"] for _, k in order]
added_e = [100 * D[k]["orPoisson"]["sensorIso"]["std"] for _, k in order]

fig, ax = plt.subplots(figsize=(6.6, 3.5))
xx = np.arange(len(order))
w = 0.36
ax.bar(xx - w / 2, lookup, w, yerr=lookup_e, color=ORANGE, label="Lookup", capsize=2)
ax.bar(xx + w / 2, added, w, yerr=added_e, color=BLUE, label="Lookup + likelihood ratio", capsize=2)
ax.set_xticks(xx)
ax.set_xticklabels([lab for lab, _ in order])
ax.set_ylabel("sensor isolation (percent)")
ax.set_ylim(0, 118)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
out = ROOT / "paper" / "fig_poisson.pdf"
fig.savefig(out)
print(out)
