# What to say tomorrow

One page for the novelty grilling. Do not mention SPRT, e-processes, or SIT.

## The sentence

We do not detect that something is wrong. We build a **structured residual whose 0/1 pattern names the cause**: compute fault, sensor/detector fault, novel astrophysical class, or known class.

Existing OOD and silent-corruption detectors live in the **same activation-magnitude space**, so a new sky class and a bit-flip look the same. Classical isolability (Gertler) has **no symbol for “this is a new class.”** That dictionary is the contribution.

## Closest papers (she will pick one)

| If she says | You say |
|---|---|
| Gavarini (IOLTS 2022) uses OOD as a fault detector | Yes. That is the failure mode. A held-out TGF would be called a fault. We refuse fused scores. |
| ProGIP already does ID / OOD / fault | Closest paper. Still range checks on GIP/OOD scores. ReAct already showed OOD produces extreme activations — same symptom as high-order bit-flips. Range is not identifiability, and it does not use instrument physics. |
| Dr. DNA / BLINK look at activations | They flag that an inference is corrupt. They do not name *discovery* as a non-fault cause. |
| Gertler already has structured residuals | Dictionary is sensor / process / actuator. No “new astrophysical class,” no “the diagnostic NN is corrupted.” |
| FT-CNN already has checksums | We **use** an ABFT checksum as one residual. That is not the contribution. The contribution is the isolability table that includes discovery. |
| John et al. already classify Fermi-GBM | Four-class CNN-RNN plus outlier flag. No execution-fault model. Another compact classifier is incremental. |

## What is new vs what is not

**New (defend this):** the isolability dictionary `{compute, sensor, novel, known}` with residuals that are *selectively blind* — physics residual is blind to neural weights; checksum is blind to the sky; morphology is computed on the shared multi-detector profile, not on a possibly-corrupt head.

**Not new (do not claim):** checksums, OOD scores, SPRT, e-processes, Fermi classification, bit-flip injection.

## Public data + runnable results

- Catalog: HEASARC `fermigtrig` (public). Counts used here: GRB 4373, SFLARE 2800, TGF 1658, SGR 684 (12,979 triggers).
- Demo events: physics-faithful simulations of 8 NaI-like detectors (cosine angular response, Poisson counts, catalog morphologies). Swap-in of real TTE does not change the residual algebra.
- Reproduce in one command, no GPU:

```bash
npm install
npm run test:aria
```

Expected (~1 s): ARIA 4-way accuracy **0.892**, range-check **0.497**. Novel events: ARIA 36/36 correct, never called compute. Energy/MSP flags **100%** of novel as “anomaly” and cannot say why.

UI: `npm run dev` then `/experiment` (confusion vs baselines) and `/lab` (one event).

## If she asks “why not the ECG fault-vs-drift paper?”

That paper is SPRT on two activation features. BLINK already uses BatchNorm for faults; Hindy already does sequential cause diagnosis for shift. The intersection is a workshop paper, not a new object. This Fermi dictionary is the object that is not already implied by those works.
