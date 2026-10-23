#!/usr/bin/env python3
"""Figures and numeric tables for the Heliyon manuscript.

Reads only JSON emitted by the evaluators. Rounding is half away from zero
at one decimal, matching JavaScript's toFixed, so a cell cannot print as
76.2 in one place and 76.3 in another.
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
R = json.loads((PAPER / "results.json").read_text())
E = json.loads((PAPER / "extra_results.json").read_text())
C = json.loads((ROOT / "experiments/gbm/last_eval.json").read_text())
MODES = json.loads((PAPER / "check_mode_comparison.json").read_text())

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
BLUE, GRAY, ORANGE, RED, GREEN = "#0072BC", "#6e6e6e", "#d9822b", "#c0392b", "#2e8b57"
FAM_COLOR = {"GRB": BLUE, "SGR": "#6b4c9a", "SFLARE": ORANGE, "TGF": RED}


def rnd(x: float) -> str:
    """One decimal, half away from zero."""
    scaled = 100.0 * float(x)
    sign = -1 if scaled < 0 else 1
    return f"{sign * math.floor(abs(scaled) * 10.0 + 0.5) / 10.0:.1f}"


def pm(block: dict) -> str:
    return f"{rnd(block['mean'])} $\\pm$ {rnd(block['std'])}"


def write(name: str, body: str) -> None:
    (PAPER / name).write_text(body)
    print("wrote", name)


def table(caption: str, label: str, colspec: str, header: str, rows: list[str], star: bool = False, fit: bool = False) -> str:
    env = "table*" if star else "table"
    open_fit = "\\resizebox{\\textwidth}{!}{%\n" if fit else ""
    close_fit = "}" if fit else ""
    lines = [
        f"\\begin{{{env}}}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        "\\small",
        open_fit + f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *[r + " \\\\" for r in rows],
        "\\bottomrule",
        "\\end{tabular}" + close_fit,
        f"\\end{{{env}}}",
        "",
    ]
    return "\n".join(lines)


# ---------- tables that are easier to keep generated ----------
seed_rows = []
for s in R["perSeed"]:
    seed_rows.append(
        f"{s['seed']} & {rnd(s['aria'])} & {rnd(s['range'])} & {rnd(s['novelAsAria'])} & "
        f"{rnd(s['computeAsAria'])} & {rnd(s['sensorAsAria'])} & {rnd(s['computeFlaggedEnergy'])}"
    )
write(
    "tab_seeds.tex",
    table(
        "Per-seed simulator results (percent), evaluation seed 123. ARIA and Range are four-way accuracy. Nov., Comp., and Sens.\\ are ARIA isolation of novel, compute, and sensor trials. E/MSP is the share of compute trials flagged by energy or maximum-softmax rejection.",
        "tab:seeds",
        "ccccccc",
        "Seed & ARIA & Range & Nov. & Comp. & Sens. & E/MSP",
        seed_rows,
    ),
)

order = ["known-clean", "novel-clean", "sensor", "compute"]
short = ["known", "novel", "sensor", "compute"]
conf = R["seed7ConfusionAria"]
conf_rows = []
for t, lab in zip(order, short):
    cells = " & ".join(str(conf[t][p]) for p in order)
    conf_rows.append(f"{lab} & {cells}")
write(
    "tab_conf.tex",
    table(
        "ARIA confusion counts for training seed 7 in the simulator (36 trials in the novel row and in each compute half; 108 known-clean trials; 72 sensor trials).",
        "tab:conf",
        "lcccc",
        "True $\\backslash$ predicted & known & novel & sensor & compute",
        conf_rows,
    ),
)

real_conf = E["realConfusionSeed7"]
rc_rows = []
for t, lab in zip(order, short):
    cells = " & ".join(str(real_conf[t][p]) for p in order)
    rc_rows.append(f"{lab} & {cells}")
write(
    "tab_realconf.tex",
    table(
        "ARIA calls on the public CTIME slice at training seed 7 and evaluation seed 123, using the published injection budget (the first 40 known cubes). All 25 held-out TGF cubes are called known.",
        "tab:realconf",
        "lcccc",
        "True $\\backslash$ predicted & known & novel & sensor & compute",
        rc_rows,
    ),
)

sweep_rows = []
for row in E["morphSweep"]:
    sweep_rows.append(
        f"{row['margin']:g} & {rnd(row['ariaAcc'])} & {rnd(row['novelIso'])} & "
        f"{rnd(row['knownAsNovel'])} & {rnd(row['novelAsKnown'])}"
    )
write(
    "tab_sweep.tex",
    table(
        "Morphology-threshold sweep on the public CTIME slice (training seed 7, published injection budget). The margin multiplies the 95th percentile of clean known-class morphology distances. The shipped rule uses margin 1.25. Lowering it does not recover TGF as novel without calling a large share of known triggers novel.",
        "tab:sweep",
        "ccccc",
        "Margin & Acc. & Novel & Known$\\to$novel & Novel$\\to$known",
        sweep_rows,
    ),
)

fam_rows = []
for fam, block in E["morphology"]["summary"].items():
    fam_rows.append(
        f"{fam} & {block['n']} & {block['mean']:.2f} & {block['min']:.2f} & {block['max']:.2f} & {block['aboveThreshold']}"
    )
write(
    "tab_morph.tex",
    table(
        "Morphology distance to the nearest known-class centroid on clean public CTIME cubes (training seed 7). The shipped threshold is "
        + f"{E['morphology']['morphThreshold']:.2f}. TGF is closer to the known centroids than GRB is. Above: cubes whose distance exceeds that threshold.",
        "tab:morph",
        "lrrrrr",
        "Family & $n$ & Mean & Min & Max & Above",
        fam_rows,
    ),
)

def sens_row(label: str, block: dict) -> str:
    return (
        f"{label} & {pm(block['ariaAcc'])} & {pm(block['computeIso'])} & "
        f"{pm(block['sensorIso'])} & {pm(block['novelAsCompute'])} & {pm(block['novelIso'])}"
    )


# realCritical numbers are rates already averaged
rc = E["realCriticalCap40"]
ra = E["realCriticalAllKnown"]
rmild = E["realMildCap40"]
cv = E["cv"]["cap40"]["metrics"]
ab = E["split"]["aToB"]["metrics"]
ba = E["split"]["bToA"]["metrics"]
proto_rows = [
    sens_row("In-sample, first 40 known cubes", rc),
    sens_row("In-sample, every known cube", ra),
    sens_row("Mild flips, first 40 known cubes", rmild),
    sens_row("5$\\times$5 cross-validation", cv),
    sens_row("Half-split, A$\\to$B", ab),
    sens_row("Half-split, B$\\to$A", ba),
]
write(
    "tab_protocols.tex",
    table(
        "Public CTIME under protocols that the headline table does not show (percent, mean $\\pm$ standard deviation). Cross-validation spread is over 25 overlapping folds. Half-split standard deviations are across training seeds on one fixed partition, which is why they are zero. Neither protocol is a new draw from the catalog.",
        "tab:protocols",
        "lccccc",
        "Protocol & Acc. & Compute & Sensor & Nov.$\\to$comp. & Novel",
        proto_rows,
        star=True,
        fit=True,
    ),
)

def fam_line(title: str, block: dict) -> str:
    cells = []
    for fam in ["GRB", "SGR", "SFLARE"]:
        if fam not in block:
            cells.append("--")
        else:
            cells.append(f"{rnd(block[fam]['rate'])} ({int(block[fam]['n'])})")
    return f"{title} & " + " & ".join(cells)

write(
    "tab_family.tex",
    table(
        "Sensor isolation by trigger family on public CTIME (percent, pooled over five training seeds). The count in parentheses is the number of injected cubes of that family. The published budget never injects a spike into a solar flare.",
        "tab:family",
        "lccc",
        "Budget & GRB & SGR & SFLARE",
        [
            fam_line("First 40 known cubes", E["sensorByFamily"]["cap40"]),
            fam_line("Every known cube", E["sensorByFamily"]["allKnown"]),
        ],
    ),
)

sp_h = E["simulatorSpike"]["headline"]
sp_o = E["simulatorSpike"]["orSpike"]
rp = E["realOrSpikeCap40"]
spike_rows = [
    f"Simulator, headline & {pm(sp_h['ariaAcc'])} & {pm(sp_h['sensorIso'])} & {pm(sp_h['knownAsSensor'])} & {pm(sp_h['novelAsSensor'])}",
    f"Simulator, spike bit added & {pm(sp_o['ariaAcc'])} & {pm(sp_o['sensorIso'])} & {pm(sp_o['knownAsSensor'])} & {pm(sp_o['novelAsSensor'])}",
    f"CTIME, headline & {pm(rc['ariaAcc'])} & {pm(rc['sensorIso'])} & {pm(rc['knownAsSensor'])} & {pm(rc['novelAsSensor'])}",
    f"CTIME, spike bit added & {pm(rp['ariaAcc'])} & {pm(rp['sensorIso'])} & {pm(rp['knownAsSensor'])} & {pm(rp['novelAsSensor'])}",
]
write(
    "tab_spike.tex",
    table(
        "Short-window spike score, thresholded on clean cubes only and OR-ed into the sensor bit (percent). On the simulator the bit raises sensor isolation. On the published CTIME budget it does not.",
        "tab:spike",
        "lcccc",
        "Rule & Acc. & Sensor & Known$\\to$sensor & Novel$\\to$sensor",
        spike_rows,
        star=True,
        fit=True,
    ),
)

# real slice table from this run, calibrated fp32
trend_rows = []
for row in MODES["calibrated"]:
    if row["prec"] != "fp32":
        continue
    trend_rows.append(
        f"{row['perClass']} & {rnd(row['ariaAcc']['mean'])} & {rnd(row['computeIso']['mean'])} & "
        f"{rnd(row['sensorIso']['mean'])} & {rnd(row['novelAsCompute']['mean'])} & {rnd(row['novelIso']['mean'])}"
    )
write(
    "tab_trend.tex",
    table(
        "Calibrated fp32 ARIA on public CTIME as the slice grows (percent, mean over five training seeds). The 8- and 12-cube rows are prefixes of the 25-cube slice.",
        "tab:trend",
        "cccccc",
        "Per class & Acc. & Compute & Sensor & Novel$\\to$comp. & Novel",
        trend_rows,
    ),
)

# ---------- figures ----------
def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PAPER / name)
    plt.close(fig)
    print("wrote", name)


# sim vs real
labels = ["Four-way\naccuracy", "Novel\nisolation", "Compute\nisolation", "Sensor\nisolation"]
sim = [R["fourWay"]["aria"], R["isolation"]["novel"], R["isolation"]["compute"], R["isolation"]["sensor"]]
real_m = [C["ariaAcc"], C["novelIso"], C["computeIso"], C["sensorIso"]]
real_s = [C["std"]["ariaAcc"], C["std"]["novelIso"], C["std"]["computeIso"], C["std"]["sensorIso"]]
x = np.arange(4)
w = 0.36
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.bar(x - w / 2, [100 * s["mean"] for s in sim], w, yerr=[100 * s["std"] for s in sim], color=BLUE, label="Simulator", capsize=3)
ax.bar(x + w / 2, [100 * m for m in real_m], w, yerr=[100 * s for s in real_s], color=ORANGE, label="Real CTIME", capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("percent")
ax.set_ylim(0, 118)
ax.legend(frameon=False, ncol=2)
save(fig, "fig_sim_vs_real.pdf")

# ablation
A = R["ablations"]
names = ["Full ARIA", "No checksum", "Checksum only", "Physics only", "Morphology only"]
keys = ["full", "no-checksum", "checksum", "physics", "morph"]
m = [100 * A[k]["mean"] for k in keys]
s = [100 * A[k]["std"] for k in keys]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
y = np.arange(len(names))[::-1]
ax.barh(y, m, xerr=s, color=[BLUE] + [GRAY] * 4, capsize=3)
for yi, mi in zip(y, m):
    ax.text(mi + 1.5, yi, f"{mi:.1f}", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlim(0, 110)
ax.set_xlabel("four-way accuracy (%)")
save(fig, "fig_ablation.pdf")

# confusion
cm = np.array([[conf[t][p] for p in order] for t in order], dtype=float)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
norm = cm / cm.sum(1, keepdims=True)
ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.55 else "black")
ax.set_xticks(range(4))
ax.set_xticklabels(short)
ax.set_yticks(range(4))
ax.set_yticklabels(short)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
save(fig, "fig_confusion.pdf")

# seeds
seeds = [str(p["seed"]) for p in R["perSeed"]]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
xs = np.arange(len(seeds))
ax.plot(xs, [100 * p["aria"] for p in R["perSeed"]], "o-", color=BLUE, label="ARIA")
ax.plot(xs, [100 * p["range"] for p in R["perSeed"]], "s--", color=GRAY, label="Range check")
ax.set_xticks(xs)
ax.set_xticklabels(seeds)
ax.set_xlabel("training seed")
ax.set_ylabel("four-way accuracy (%)")
ax.set_ylim(30, 100)
ax.legend(frameon=False)
save(fig, "fig_seeds.pdf")

# trend from this run
runs = [row for row in MODES["calibrated"] if row["prec"] == "fp32"]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
series = [
    ("computeIso", "Compute isolation", "o", "-"),
    ("ariaAcc", "Four-way accuracy", "s", "--"),
    ("sensorIso", "Sensor isolation", "^", "-."),
    ("novelIso", "Novel isolation", "D", ":"),
    ("novelAsCompute", "Novel named compute", "v", ":"),
]
for key, lab, mk, ls in series:
    ax.plot([r["perClass"] for r in runs], [100 * r[key]["mean"] for r in runs], marker=mk, ls=ls, label=lab)
ax.set_xlabel("triggers per class")
ax.set_ylabel("percent")
ax.set_xticks([8, 12, 25])
ax.set_ylim(-5, 110)
ax.legend(frameon=False, fontsize=8, ncol=2)
save(fig, "fig_trend.pdf")

# morphology
pts = E["morphPoints"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
data = [[p["morph"] for p in pts if p["family"] == f] for f in ["GRB", "SGR", "SFLARE", "TGF"]]
bp = axes[0].boxplot(data, tick_labels=["GRB", "SGR", "SFLARE", "TGF"], patch_artist=True)
for patch, fam in zip(bp["boxes"], ["GRB", "SGR", "SFLARE", "TGF"]):
    patch.set_facecolor(FAM_COLOR[fam])
    patch.set_alpha(0.7)
axes[0].axhline(E["morphology"]["morphThreshold"], color="k", ls="--", lw=0.8, label="shipped threshold")
axes[0].set_ylabel("distance to nearest known centroid")
axes[0].legend(frameon=False, fontsize=8)
for fam in ["GRB", "SGR", "SFLARE", "TGF"]:
    xs_ = [p["duration"] for p in pts if p["family"] == fam]
    ys = [p["fwhm"] for p in pts if p["family"] == fam]
    axes[1].scatter(xs_, ys, s=22, color=FAM_COLOR[fam], label=fam, alpha=0.85)
axes[1].set_xlabel("duration fraction")
axes[1].set_ylabel("FWHM fraction")
axes[1].legend(frameon=False, fontsize=8, ncol=2)
save(fig, "fig_morph.pdf")

# sweep
fig, ax = plt.subplots(figsize=(6.2, 3.4))
margins = [r["margin"] for r in E["morphSweep"]]
ax.plot(margins, [100 * r["ariaAcc"] for r in E["morphSweep"]], "o-", color=BLUE, label="Four-way accuracy")
ax.plot(margins, [100 * r["novelIso"] for r in E["morphSweep"]], "D-", color=RED, label="Novel isolation")
ax.plot(margins, [100 * r["knownAsNovel"] for r in E["morphSweep"]], "s--", color=GRAY, label="Known called novel")
ax.axvline(1.25, color="k", ls=":", lw=0.8)
ax.set_xlabel("margin on the clean 95th percentile")
ax.set_ylabel("percent")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig_sweep.pdf")

# protocols
fig, ax = plt.subplots(figsize=(6.6, 3.6))
labs = ["In-sample\n(cap 40)", "In-sample\n(all known)", "Cross-\nvalidation", "Split\nA to B", "Split\nB to A"]
blocks = [rc, ra, cv, ab, ba]
xx = np.arange(len(labs))
w = 0.25
for i, (key, lab, col) in enumerate(
    [("ariaAcc", "Accuracy", BLUE), ("computeIso", "Compute", GREEN), ("sensorIso", "Sensor", ORANGE)]
):
    ax.bar(
        xx + (i - 1) * w,
        [100 * b[key]["mean"] for b in blocks],
        w,
        yerr=[100 * b[key]["std"] for b in blocks],
        color=col,
        label=lab,
        capsize=2,
    )
ax.set_xticks(xx)
ax.set_xticklabels(labs)
ax.set_ylabel("percent")
ax.set_ylim(0, 120)
ax.legend(frameon=False, ncol=3)
save(fig, "fig_protocols.pdf")

# spike
fig, ax = plt.subplots(figsize=(6.4, 3.4))
groups = [
    ("Sim.\nheadline", sp_h),
    ("Sim.\n+ spike", sp_o),
    ("CTIME\nheadline", rc),
    ("CTIME\n+ spike", rp),
]
xx = np.arange(len(groups))
w = 0.36
ax.bar(xx - w / 2, [100 * g[1]["sensorIso"]["mean"] for g in groups], w, color=ORANGE, label="Sensor isolation")
ax.bar(xx + w / 2, [100 * g[1]["ariaAcc"]["mean"] for g in groups], w, color=BLUE, label="Four-way accuracy")
ax.set_xticks(xx)
ax.set_xticklabels([g[0] for g in groups])
ax.set_ylim(0, 110)
ax.set_ylabel("percent")
ax.legend(frameon=False)
save(fig, "fig_spike.pdf")

# compound
fig, ax = plt.subplots(figsize=(5.4, 3.2))
comp = E["compound"]
vals = [100 * comp[k]["mean"] for k in ["both", "arithOnly", "physOnly", "neither"]]
errs = [100 * comp[k]["std"] for k in ["both", "arithOnly", "physOnly", "neither"]]
ax.bar(["Both\nsignatures", "Checksum\nonly", "Physics\nonly", "Neither"], vals, yerr=errs, color=[BLUE, GRAY, ORANGE, GREEN], capsize=3)
ax.set_ylabel("share of compound trials (%)")
ax.set_ylim(0, 110)
save(fig, "fig_compound.pdf")

print("assets ok")
