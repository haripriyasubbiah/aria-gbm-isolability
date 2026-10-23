"""Result figures for the ARIA paper. Reads real values only:
  results.json            (simulator, from `npm run test:paper`)
  last_eval.json          (real CTIME, from `npm run test:gbm`)
Usage: python make_result_figures.py <results.json> <last_eval.json> <outdir>
The seed-7 confusion matrix is read from results.json (seed7ConfusionAria).
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res, ctime, out = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(res)); C = json.load(open(ctime))
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, GRAY, ORANGE = "#0072BC", "#9a9a9a", "#d9822b"
pc = lambda x: 100 * x

# Fig A: ARIA simulator vs real CTIME
labels = ["Four-way\naccuracy", "Novel\nisolation", "Compute\nisolation", "Sensor\nisolation"]
sim = [R["fourWay"]["aria"], R["isolation"]["novel"], R["isolation"]["compute"], R["isolation"]["sensor"]]
real_m = [C["ariaAcc"], C["novelIso"], C["computeIso"], C["sensorIso"]]
real_s = [C["std"]["ariaAcc"], C["std"]["novelIso"], C["std"]["computeIso"], C["std"]["sensorIso"]]
x = np.arange(4); w = 0.36
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.bar(x - w/2, [pc(s["mean"]) for s in sim], w, yerr=[pc(s["std"]) for s in sim], color=BLUE, label="Simulator", capsize=2)
ax.bar(x + w/2, [pc(m) for m in real_m], w, yerr=[pc(s) for s in real_s], color=ORANGE, label="Real CTIME", capsize=2)
for i, m in enumerate(real_m): ax.text(i + w/2 + 0.04, pc(m) + 2, f"{pc(m):.1f}", ha="left", fontsize=6.5)
for i, s in enumerate(sim): ax.text(i - w/2 - 0.04, pc(s["mean"]) + 2, f"{pc(s['mean']):.1f}", ha="right", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("percent"); ax.set_ylim(0, 115)
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=7, bbox_to_anchor=(1, 1.12))
fig.tight_layout(); fig.savefig(f"{out}/fig_sim_vs_real.pdf"); plt.close(fig)

# Fig B: ablation
A = R["ablations"]
names = ["Full ARIA", "No checksum", "Checksum only", "Physics only", "Morphology only"]
keys = ["full", "no-checksum", "checksum", "physics", "morph"]
m = [pc(A[k]["mean"]) for k in keys]; s = [pc(A[k]["std"]) for k in keys]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
y = np.arange(len(names))[::-1]
ax.barh(y, m, xerr=s, color=[BLUE] + [GRAY] * 4, capsize=2)
for yi, mi in zip(y, m): ax.text(mi + 2.5, yi, f"{mi:.1f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 105); ax.set_xlabel("four-way accuracy (%)")
fig.tight_layout(); fig.savefig(f"{out}/fig_ablation.pdf"); plt.close(fig)

# Fig C: seed-7 confusion heatmap (ARIA)
C7 = R["seed7ConfusionAria"]
cm = np.array([[C7[t][p] for p in ["known-clean", "novel-clean", "sensor", "compute"]] for t in ["known-clean", "novel-clean", "sensor", "compute"]])
cls = ["known", "novel", "sensor", "compute"]
fig, ax = plt.subplots(figsize=(3.0, 2.7))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
ax.set_xticks(range(4)); ax.set_xticklabels(cls); ax.set_yticks(range(4)); ax.set_yticklabels(cls)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(); fig.savefig(f"{out}/fig_confusion.pdf"); plt.close(fig)

# Fig D: per-seed accuracy, ARIA vs range-check
seeds = [str(p["seed"]) for p in R["perSeed"]]
fig, ax = plt.subplots(figsize=(3.4, 2.0))
xs = np.arange(len(seeds))
ax.plot(xs, [pc(p["aria"]) for p in R["perSeed"]], "o-", color=BLUE, label="ARIA")
ax.plot(xs, [pc(p["range"]) for p in R["perSeed"]], "s--", color=GRAY, label="Range-check")
ax.set_xticks(xs); ax.set_xticklabels(seeds); ax.set_xlabel("training seed"); ax.set_ylabel("four-way acc. (%)")
ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.5, 0.68), ncol=2)
fig.tight_layout(); fig.savefig(f"{out}/fig_seeds.pdf"); plt.close(fig)
print("ok")
