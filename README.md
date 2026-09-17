# ARIA — Fault or discovery

A working slice of a **methodological** answer to the Fermi-GBM “fault or discovery” problem: not another fused OOD+fault classifier, but a **structured residual with isolable 0/1 signatures**.

Public data: Fermi GBM trigger-type counts from HEASARC `fermigtrig` (12,981 triggers on 17 Sep 2026). Lightcubes in the demo are physics-faithful simulations of 8 NaI-like detectors with a cosine angular response, Poisson counts, and class morphologies (GRB / SGR / solar flare / held-out TGF). The residual algebra does not change if you later swap in real TTE.

## The actual gap (from the literature, not from combining papers)

Four lines of work each solve a piece and **confound the rest**:

| Work | What it does | Why it does not isolate discovery |
|---|---|---|
| [Gavarini et al., IOLTS 2022](https://doi.org/10.1109/IOLTS56730.2022.9897805) | Uses open-set scores (MLS, energy, ODIN) **as fault detectors** | A held-out transient looks like a fault. That is the failure mode this project exists to prevent. |
| [ProGIP, ACM TECS 2025](https://doi.org/10.1145/3761796) | Range checks on GIP gradients/OOD scores; reports ID / OOD / fault | Closest prior art. Still magnitude space. ReAct (NeurIPS 2021) already showed OOD produces extreme activations — the same symptom as high-order bit-flips. |
| [Dr. DNA, ASPLOS 2024](https://doi.org/10.1145/3620666.3651349) | SDC signatures in neuron-activation histograms | OOD moves those histograms too. |
| [Gertler & Singer, Automatica 1990](https://doi.org/10.1016/0005-1098(90)90133-3) | Isolability via structured residuals | Dictionary is sensor / process / actuator. No symbol for “new astrophysical class” or “the diagnostic NN is corrupted.” |
| [John et al., arXiv:2510.22412](https://arxiv.org/abs/2510.22412) | Fermi-GBM four-class CNN-RNN + outlier flag | No execution-fault model. Another compact classifier is incremental. |

The gap that is still **methodological**, not a mashup: construct residuals that are **selectively blind** so the incidence pattern names the cause. IEEE Access does not require a new sequential test; it requires a distinct, technically sound method. The Access manuscript is `paper/ieeeaccess.tex`.

## The method: ARIA

**Attribution by Residual-Invariance Analysis**

1. `r_phys` — leave-one-detector-out cosine vs a median lightcurve of the other bright detectors. Blind to neural weights. Fires on single-detector spikes (proxy for GBM phosphorescent / local-particle events). Silent for a point source of any class, including held-out TGF.
2. `r_morph` — distance of that **shared** profile to known-class centroids. Blind to a corrupted head. Fires on TGF-like morphology.
3. `U_det` — physics residual collapses when the worst detector is dropped.
4. `U_arith` — stored row-sum checksum of the linear head vs the live matmul ([FT-CNN / ABFT](https://doi.org/10.1109/TPDS.2020.2991788)). Blind to input novelty.

Decision is a lookup, not a trained 6-way head:

```
if checksum fails     -> compute  (recompute)
else if detector parity fails -> sensor   (isolate-channel)
else if morphology far    -> novel    (preserve)
else                      -> known    (accept)
```

That is the contribution: **isolability over a new fault dictionary**, not “late-fusion of existing scores.”

## What this slice deliberately does not do

The original DA1 asked for six states, five actions, six fault domains, eight baselines, five seeds, conformal mixtures. That is a paper series. This repo keeps RQ1/RQ3 in executable form: can physics + checksum separate clean novelty from sensor and shared compute faults, and do OOD/range baselines fail to name the cause?

## Run

```bash
npm install
npm run test:aria     # single-seed isolability check (no GPU)
npm run test:paper    # five-seed tables → paper/results.json and paper/results.tex
npm run dev           # http://127.0.0.1:43173
```

IEEE Access checklist: `paper/ACCESS_CHECKLIST.md`. Compile `paper/ieeeaccess.tex` on Overleaf with IEEEtran (or the official Access class).

- `/` literature gap
- `/method` residual algebra
- `/lab` generate one event
- `/experiment` Monte Carlo confusion vs energy/MSP and range-check

## Honest limits

- Simulated lightcubes, not downloaded TTE/CSPEC (catalog counts are real).
- Linear head, not the John et al. Conv-RNN. The checksum is on this head; the physics residual never needs it.
- Single-cause isolability. Sensor+compute together is unisolable here and left unlabeled.
- Software bit-flips are radiation-**like**, not a beam test.
- Sensor spikes are a proxy for catalog `LOCLPAR` / `DISTPAR` / phosphorescent NaI events, not a claim that we classified those catalog rows.
