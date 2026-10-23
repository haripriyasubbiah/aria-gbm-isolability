"""Checksum figures and tables for the ARIA paper. Reads real values only:
  check_figures.json            (npx tsx src/lib/aria/check_figures.ts)
  results_bounded.json          (ARIA_CHECK=bounded npx tsx src/lib/aria/paper_eval.ts)
  check_mode_comparison.json    (npx tsx src/lib/aria/real_compare.ts)
Writes fig_check_scatter.pdf, fig_check_bits.pdf, fig_check_roc.pdf, fig_check_prec.pdf and checkmodes_tables.tex.
Usage: python make_check_figures.py <paper_dir>
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = pathlib.Path(sys.argv[1])
F = json.load(open(P / "check_figures.json"))
B = json.load(open(P / "results_bounded.json"))
Rm = json.load(open(P / "check_mode_comparison.json"))
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, GRAY, ORANGE, RED, GREEN = "#0072BC", "#9a9a9a", "#d9822b", "#c0392b", "#2e8b57"
T = F["trials"]; FLOOR = 1e-9

# ---------- Fig 1: per-trial checksum statistic by true cause ----------
cats = [("known-clean", "-", "known", GRAY), ("novel-clean", "-", "TGF", RED),
        ("sensor", "-", "sensor", GRAY), ("compute", "mild", "mild", ORANGE), ("compute", "critical", "crit.", ORANGE)]
panels = [("calibrated", "fp32", "(a) calibrated, fp32"), ("bounded", "fp32", "(b) bounded, fp32"),
          ("calibrated", "fp16", "(c) calibrated, fp16"), ("bounded", "fp16", "(d) bounded, fp16")]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
rng = np.random.default_rng(0)
for ax, (mode, prec, title) in zip(axes, panels):
    for i, (truth, sev, lab, col) in enumerate(cats):
        v = np.array([max(t["exec"], FLOOR) for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == truth and t["severity"] == sev])
        v = np.minimum(v, 1e6 if mode == "bounded" else 1.0)
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=4, color=col, alpha=0.55, linewidths=0)
    if mode == "bounded":
        ax.axhline(1.0, color="k", lw=0.9, ls="--"); ax.text(-0.45, 2.0, "threshold = 1", ha="left", fontsize=6)
        ax.set_ylim(1e-6, 1e7); ax.set_ylabel("residual / bound" if ax is axes[1] else "")
    else:
        th = [t["threshold"] for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == "known-clean"]
        for u in sorted(set(th)): ax.axhline(u, color="k", lw=0.5, ls=":", alpha=0.8)
        ax.set_ylim(1e-9, 3); ax.set_ylabel("normalised residual" if ax is axes[0] else "")
    ax.set_yscale("log"); ax.set_xticks(range(5)); ax.set_xticklabels([c[2] for c in cats], fontsize=5.5, rotation=40, ha="right")
    ax.set_title(title, fontsize=7)
fig.tight_layout(); fig.savefig(P / "fig_check_scatter.pdf"); plt.close(fig)

# ---------- Fig 2: detection vs flipped bit ----------
fig, ax = plt.subplots(figsize=(3.4, 2.75))
ax.axvspan(-0.5, 22.5, color="#f1f1f1"); ax.axvspan(22.5, 30.5, color="#fde9d3"); ax.axvspan(30.5, 31.5, color="#e6e6fa")
for x0, lab in [(11, "mantissa"), (26.5, "exponent"), (31, "")]: ax.text(x0, 112, lab, ha="center", fontsize=6)
sty = {("calibrated", "fp32"): (BLUE, "-", "o"), ("bounded", "fp32"): (BLUE, "--", "s"),
       ("calibrated", "fp16"): (ORANGE, "-", "o"), ("bounded", "fp16"): (ORANGE, "--", "s")}
for (mode, prec), (c, ls, mk) in sty.items():
    s = sorted([r for r in F["sweep"] if r["mode"] == mode and r["prec"] == prec], key=lambda r: r["bit"])
    ax.plot([r["bit"] for r in s], [100 * r["rate"] for r in s], color=c, ls=ls, marker=mk, ms=2.2, lw=1, label=f"{mode}, {prec}")
ax.fill_between([17.5, 23.5], 0, 4, color=GREEN, alpha=0.85, lw=0); ax.text(20.5, 6, "mild", ha="center", fontsize=6, color=GREEN)
ax.fill_between([27.5, 30.5], 0, 4, color=RED, alpha=0.85, lw=0); ax.text(29, 6, "crit.", ha="center", fontsize=6, color=RED)
ax.set_xlim(-0.5, 31.5); ax.set_ylim(0, 118); ax.set_xlabel("flipped bit of the fp32 weight"); ax.set_ylabel("detected as compute (%)")
ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
fig.tight_layout(); fig.savefig(P / "fig_check_bits.pdf"); plt.close(fig)

# ---------- Fig 3: ROC, compute faults vs held-out TGF ----------
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for (mode, prec), (c, ls, mk) in sty.items():
    pos = [t["exec"] for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == "compute"]
    neg = [t["exec"] for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == "novel-clean"]
    ths = np.unique(np.concatenate([pos, neg, [0, np.inf]]))[::-1]
    fpr = [np.mean(np.asarray(neg) > t) for t in ths]; tpr = [np.mean(np.asarray(pos) > t) for t in ths]
    ax.plot(fpr, tpr, color=c, ls=ls, lw=1, label=f"{mode}, {prec} (AUC {auc(pos, neg):.3f})")
    # operating point actually used: per-seed threshold
    op = [(t["exec"] > t["threshold"]) for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == "novel-clean"]
    oq = [(t["exec"] > t["threshold"]) for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == "compute"]
    ax.scatter([np.mean(op)], [np.mean(oq)], color=c, marker=mk, s=18, zorder=5, edgecolor="k", linewidths=0.5)
ax.set_xlim(-0.01, 0.4); ax.set_ylim(0, 1.03); ax.set_xlabel("held-out TGF named compute (rate)"); ax.set_ylabel("compute faults detected (rate)")
ax.legend(frameon=False, fontsize=5.8, loc="lower right")
fig.tight_layout(); fig.savefig(P / "fig_check_roc.pdf"); plt.close(fig)

# ---------- Fig 4: precision sensitivity, calibrated vs bounded ----------
M = B["modes"]; precs = ["fp64", "fp32", "fp16"]
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.2))
for ax, key, ttl in [(axs[0], "compute", "compute faults detected (%)"), (axs[1], "novelAsCompute", "TGF named compute (%)")]:
    x = np.arange(3); w = 0.38
    for k, (mode, col) in enumerate([("calibrated", BLUE), ("bounded", ORANGE)]):
        m = [100 * M[mode][p][key]["mean"] for p in precs]; s = [100 * M[mode][p][key]["std"] for p in precs]
        ax.bar(x + (k - 0.5) * w, m, w, yerr=s, color=col, capsize=1.5, label=mode, error_kw={"lw": 0.7})
    ax.set_xticks(x); ax.set_xticklabels(precs); ax.set_title(ttl, fontsize=6.5)
axs[0].set_ylim(0, 115); axs[1].set_ylim(0, 16); axs[1].legend(frameon=False, fontsize=6, loc="upper center", ncol=2, handlelength=1, columnspacing=0.8)
fig.tight_layout(); fig.savefig(P / "fig_check_prec.pdf"); plt.close(fig)

# ---------- Tables (generated, so no number is typed by hand) ----------
def one(x):
    scaled = 100.0 * float(x)
    sign = -1 if scaled < 0 else 1
    return f"{sign * __import__('math').floor(abs(scaled) * 10.0 + 0.5) / 10.0:.1f}"
pm = lambda d: f"{one(d['mean'])} $\\pm$ {one(d['std'])}"
L = []
L.append("% Auto-generated by make_check_figures.py. Do not edit by hand.")
L.append("\\begin{table*}[!t]")
L.append("\\caption{Calibrated and rounding-error-bounded checksum in the simulator (\\%, mean $\\pm$ std over five training seeds, same trials in every row). Comp.\\ (all): compute trials, mild and critical flips together. Comp.\\ (mild): the $36$ mild-flip trials per seed (bits 18--23). The calibrated binary32 row is the headline rule in Table~\\ref{tab:main}.}")
L.append("\\label{tab:modes}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llccccc}\n\\toprule")
L.append("Checksum & Prec. & Acc. & Nov.$\\to$comp. & Comp.\\ (all) & Comp.\\ (mild) & Known$\\to$comp. \\\\\n\\midrule")
for mode in ["calibrated", "bounded"]:
    for p in precs:
        d = M[mode][p]
        L.append(f"{mode} & {p} & {pm(d['aria'])} & {pm(d['novelAsCompute'])} & {pm(d['compute'])} & {pm(d['computeMild'])} & {pm(d['knownAsCompute'])} \\\\")
    if mode == "calibrated": L.append("\\midrule")
L.append("\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
L.append("\\begin{table}[!t]")
L.append("\\caption{Calibrated and bounded checksum on real CTIME (\\%, mean over five training seeds; the across-seed spread is $0.0$ in every cell, as explained in Section~\\ref{sec:real}). Triggers per class; compute faults are critical flips only. Nov.$\\to$comp.: held-out TGF named compute.}")
L.append("\\label{tab:realmodes}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llccccc}\n\\toprule")
L.append("Checksum & Prec. & n/cl. & Acc. & Comp. & Sens. & Nov.$\\to$comp. \\\\\n\\midrule")
L[-1] = L[-1].replace("lcccccc", "lccccc")
for mode in ["calibrated", "bounded"]:
    for prec in ["fp32", "fp16"]:
        for r in [r for r in Rm[mode] if r["prec"] == prec]:
            f = lambda k, r=r: one(r[k]["mean"])
            L.append(f"{mode} & {prec} & {r['perClass']} & {f('ariaAcc')} & {f('computeIso')} & {f('sensorIso')} & {f('novelAsCompute')} \\\\")
    if mode == "calibrated": L.append("\\midrule")
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(P / "checkmodes_tables.tex").write_text("\n".join(L))

# ---------- Macros for prose and abstract ----------
BC = json.load(open(P / "bound_check.json"))
def rocauc(mode, prec):
    pos = [t["exec"] for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == "compute"]
    neg = [t["exec"] for t in T if t["mode"] == mode and t["prec"] == prec and t["truth"] == "novel-clean"]
    return auc(pos, neg)
real = lambda mode, prec, n, k: next(r for r in Rm[mode] if r["prec"] == prec and r["perClass"] == n)[k]["mean"]
X = ["% Auto-generated by make_check_figures.py. Do not edit by hand."]
def mac(name, val): X.append(f"\\newcommand{{\\{name}}}{{{val}}}")
for mode, tag in [("calibrated", "Cal"), ("bounded", "Bnd")]:
    for prec, ptag in [("fp32", "ThirtyTwo"), ("fp16", "Sixteen")]:
        d = M[mode][prec]
        mac(f"{tag}Acc{ptag}", pm(d["aria"])); mac(f"{tag}NovComp{ptag}", pm(d["novelAsCompute"]))
        mac(f"{tag}Comp{ptag}", pm(d["compute"])); mac(f"{tag}Mild{ptag}", pm(d["computeMild"]))
        mac(f"{tag}Auc{ptag}", f"{rocauc(mode, prec):.3f}")
mac("BoundWorstRatio", f"{max(BC['worstCleanRatio'].values()):.2f}")
mac("BoundInputs", f"{BC['inputsPerPrecision']:,}".replace(",", "{,}"))
mac("BoundMaxSigma", str(BC["maxScaleSigma"]))
mac("RealBndSensEight", f"{100*real('bounded','fp32',8,'sensorIso'):.1f}"); mac("RealCalSensEight", f"{100*real('calibrated','fp32',8,'sensorIso'):.1f}")
mac("RealCalNovCompTwelve", f"{100*real('calibrated','fp32',12,'novelAsCompute'):.1f}")
for n, nt in [(8, "Eight"), (12, "Twelve"), (25, "TwentyFive")]:
    mac(f"RealBndCompSixteen{nt}", f"{100*real('bounded','fp16',n,'computeIso'):.1f}")
(P / "checkmodes_macros.tex").write_text("\n".join(X) + "\n")
print("ok")
