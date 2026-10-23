#!/usr/bin/env python3
"""Bin public Fermi-GBM TTE at 1 ms for the GRB and TGF rows in triggers.csv.

The window is TRIGTIME-0.5 s to TRIGTIME+1.5 s (2000 bins). That is long enough
for a sub-millisecond flash to occupy a few bins and for a short burst to occupy
many. Raw FITS land in experiments/gbm/raw/ (gitignored). The binned cubes are
experiments/gbm/tte_cubes.jsonl.
"""
from __future__ import annotations

import csv
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits

import prepare as P

DT = 0.001
LO = -0.5
HI = 1.5
N_BINS = int(round((HI - LO) / DT))
FAMILIES = {"GRB", "SGR", "SFLARE", "TGF"}


def list_tte(name: str) -> dict[str, str]:
    year = P.year_of(name)
    url = f"{P.FTP}/{year}/{name}/current/"
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode("latin1", "replace")
    except Exception:
        return {}
    files = re.findall(r'href="(glg_tte_(n[0-9ab]+)_%s_v(\d+)\.fit)"' % name, html)
    best: dict[str, tuple[int, str]] = {}
    for fname, det, ver in files:
        v = int(ver)
        if det not in best or v > best[det][0]:
            best[det] = (v, f"{url}{fname}")
    return {det: loc for det, (_, loc) in best.items()}


def bin_tte(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    with fits.open(path) as hdul:
        trig = float(hdul[0].header["TRIGTIME"])
        ev = hdul["EVENTS"].data
        if ev is None or len(ev) < 20:
            return None
        t = np.asarray(ev["TIME"], dtype=np.float64)
        pha = np.asarray(ev["PHA"], dtype=np.int16)
        eb = hdul["EBOUNDS"].data
        channels = np.asarray(eb["CHANNEL"], dtype=np.int16)
        emax = np.asarray(eb["E_MAX"], dtype=np.float64)
    soft_of = np.zeros(int(channels.max()) + 1, dtype=bool)
    for ch, em in zip(channels, emax):
        if 0 <= int(ch) < soft_of.size and em <= 50.0:
            soft_of[int(ch)] = True
    lo = trig + LO
    hi = trig + HI
    m = (t >= lo) & (t < hi)
    if int(m.sum()) < 20:
        return None
    t = t[m]
    pha = pha[m]
    idx = np.floor((t - lo) / DT).astype(np.int32)
    idx = np.clip(idx, 0, N_BINS - 1)
    counts = np.bincount(idx, minlength=N_BINS).astype(np.float64)
    pha_ok = (pha >= 0) & (pha < soft_of.size)
    soft_hits = pha_ok & soft_of[np.clip(pha, 0, soft_of.size - 1)]
    soft = np.bincount(idx[soft_hits], minlength=N_BINS).astype(np.float64)
    hard = counts - soft
    return counts, soft, hard


def cube_for(name: str, family: str) -> dict | None:
    listing = list_tte(name)
    if len(listing) < P.N_DET:
        return None
    series = {}
    for det, url in listing.items():
        dest = P.RAW / name / Path(url).name
        if not P.download(url, dest):
            continue
        try:
            binned = bin_tte(dest)
        except Exception as e:
            print(f"bin fail {name} {det}: {e}", flush=True)
            continue
        if binned is None:
            continue
        series[det] = binned
    if len(series) < P.N_DET:
        return None
    ranked = sorted(series.items(), key=lambda kv: float(kv[1][0].sum()), reverse=True)
    pick = ranked[: P.N_DET]
    return {
        "name": name,
        "family": family,
        "detectors": [kv[0] for kv in pick],
        "dt": DT,
        "window": [LO, HI],
        "counts": [kv[1][0].tolist() for kv in pick],
        "soft": [kv[1][1].tolist() for kv in pick],
        "hard": [kv[1][2].tolist() for kv in pick],
    }


def main() -> None:
    rows = [r for r in csv.DictReader((P.OUT / "triggers.csv").open()) if r["family"] in FAMILIES]
    print(f"tte targets {len(rows)}", flush=True)
    got: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(cube_for, r["name"], r["family"]): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                cube = fut.result()
            except Exception as e:
                print(f"FAIL {r['name']}: {e}", flush=True)
                continue
            if cube is None:
                print(f"skip {r['name']} {r['family']}", flush=True)
                continue
            got.append(cube)
            print(f"ok   {r['name']} {r['family']} [{len(got)}/{len(rows)}]", flush=True)
    got.sort(key=lambda c: (c["family"], c["name"]))
    out = P.OUT / "tte_cubes.jsonl"
    with out.open("w") as f:
        for c in got:
            f.write(json.dumps(c) + "\n")
    by: dict[str, int] = {}
    for c in got:
        by[c["family"]] = by.get(c["family"], 0) + 1
    print(json.dumps({"n": len(got), "by": by, "bins": N_BINS, "dt": DT}))


if __name__ == "__main__":
    main()
