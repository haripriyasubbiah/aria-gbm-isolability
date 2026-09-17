# Real Fermi-GBM kill experiment

Public CTIME (NaI) from HEASARC, then the same ARIA residuals as the simulator.

## What to download

You do **not** pick files by hand. `prepare.py` pulls the catalog and the FITS files.

Per trigger it fetches the latest

`glg_ctime_n0…nb_bnYYMMDDfff_vXX.pha`

from

`https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/20YY/bnYYMMDDfff/current/`

`--per-class 12` → 48 triggers × up to 12 NaI files ≈ **250–400 MB**.  
`--per-class 25` is the paper-sized slice (~0.8 GB).

Do not download TTE for this first kill test. CTIME is enough to see whether a real multi-detector TGF is still rank-1 (not a checksum failure).

## What to install (once)

```bash
python3 -m pip install --user astropy numpy
```

Node deps are already in the repo (`npm install` if needed).

## What to run

From the repo root:

```bash
# 1. Catalog + download + bin 8×64 cubes  (10–30 min depending on HEASARC)
python3 experiments/gbm/prepare.py --per-class 12

# 2. Train linear head on real GRB/SGR/SFLARE, hold out TGF, inject sensor/compute
npm run test:gbm
```

Resume-safe: already-downloaded `.pha` files in `experiments/gbm/raw/` are skipped.

## How to read the result

`npm run test:gbm` must print something like:

- `novelAsCompute` near **0** → ARIA still works: a real held-out TGF is not a bit-flip.
- `novelIso` high → morphology still sees TGF as new.
- `sensorIso` / `computeIso` → injected faults on real arrays, not on the simulator.

## First run (8 triggers × 4 classes, this environment)

```
ariaAcc 0.788   rangeAcc 0.40
novelAsCompute 0     ← kill test passed (TGF ≠ bit-flip)
novelIso 0           ← CTIME is too coarse to name TGF as novel
computeIso 0.92      sensorIso 0.75
```

Morphology needs millisecond TTE for the discovery box. Checksum vs sky already holds on public CTIME.

## Files written

| Path | What |
|---|---|
| `experiments/gbm/triggers.csv` | HEASARC names + types |
| `experiments/gbm/raw/` | FITS (gitignored, large) |
| `experiments/gbm/cubes.jsonl` | 8-detector × 64-bin cubes |
