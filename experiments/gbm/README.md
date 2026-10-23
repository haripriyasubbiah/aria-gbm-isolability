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

## You can skip Python for the first kill test

`experiments/gbm/cubes.jsonl` after `prepare.py --per-class 12` is 48 cubes (12 GRB / 12 SGR / 12 SFLARE / 12 TGF). From the repo root:

```bash
npm run test:gbm
```

That is the command that printed `novelAsCompute 0` on the laptop. Ignore `npm audit` noise.

Python is only needed to **re-download** or enlarge the slice (`--per-class 12` or `25`).

On Windows, do **not** use `python3`. That name is often a Microsoft Store alias and prints `Python was not found`. Use:

```powershell
py -3 --version
py -3 -m pip install --user astropy numpy
py -3 experiments/gbm/prepare.py --per-class 12
```

If `py -3 --version` also fails, install Python from https://www.python.org/downloads/windows/, tick **Add python.exe to PATH**, then disable the `python.exe` / `python3.exe` App execution aliases (Settings → Apps → Advanced app settings).

On macOS/Linux:

```bash
python3 -m pip install --user astropy numpy
python3 experiments/gbm/prepare.py --per-class 12
```

Resume-safe: already-downloaded `.pha` files in `experiments/gbm/raw/` are skipped.

## How to read the result

`npm run test:gbm` must print something like:

- `novelAsCompute` near **0** → ARIA still works: a real held-out TGF is not a bit-flip.
- `novelIso` high → morphology still sees TGF as new.
- `sensorIso` / `computeIso` → injected faults on real arrays, not on the simulator.

## Laptop run (`--per-class 12`, 12 triggers × 4 classes)

```
ariaAcc 0.758   rangeAcc 0.40
novelAsCompute 0     ← kill test passed (TGF ≠ bit-flip)
novelIso 0           ← CTIME is too coarse to name TGF as novel
computeIso 1.00      sensorIso 0.583
```

Morphology needs millisecond TTE for the discovery box. Checksum vs sky already holds on public CTIME. Sensor isolation is weaker than on the 32-cube starter slice; report 58.3%, do not round it up.

## Files written

| Path | What |
|---|---|
| `experiments/gbm/triggers.csv` | HEASARC names + types |
| `experiments/gbm/raw/` | FITS (gitignored, large) |
| `experiments/gbm/cubes.jsonl` | 8-detector × 64-bin cubes |
| `experiments/gbm/last_eval.json` | `npm run test:gbm` numbers |
| `paper/gbm_real.tex` | LaTeX macros from that eval |
