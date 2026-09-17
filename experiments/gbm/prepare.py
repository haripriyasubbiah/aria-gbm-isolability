#!/usr/bin/env python3
"""Download public Fermi-GBM CTIME (NaI) and bin 8×64 cubes for ARIA.

Usage (from repo root):

  python3 experiments/gbm/prepare.py --per-class 12

Writes:
  experiments/gbm/triggers.csv
  experiments/gbm/raw/...   (FITS, gitignored)
  experiments/gbm/cubes.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "gbm"
RAW = OUT / "raw"
NAI = [f"n{i}" for i in range(10)] + ["na", "nb"]
TYPES = ["GRB", "SGR", "SFLARE", "TGF"]
N_TIME = 64
N_DET = 8
TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
FTP = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers"


def year_of(name: str) -> str:
    yy = int(name[2:4])
    return f"20{yy:02d}"


def decode_votable_strings(xml: bytes) -> list[tuple[str, str]]:
    text = xml.decode("latin1")
    m = re.search(r'<STREAM encoding="base64">\s*(.*?)\s*</STREAM>', text, re.S)
    if not m:
        raise RuntimeError("TAP returned no binary table")
    raw = __import__("base64").b64decode(re.sub(r"\s+", "", m.group(1)))
    off = 0
    rows = []
    while off + 8 <= len(raw):
        n1 = struct.unpack(">I", raw[off : off + 4])[0]
        off += 4
        if off + n1 + 4 > len(raw):
            break
        a = raw[off : off + n1].decode()
        off += n1
        n2 = struct.unpack(">I", raw[off : off + 4])[0]
        off += 4
        if off + n2 > len(raw):
            break
        b = raw[off : off + n2].decode()
        off += n2
        rows.append((a, b))
    return rows


def fetch_catalog(per_class: int) -> list[dict]:
    rows = []
    for kind in TYPES:
        q = (
            "SELECT TOP %d trigger_name, trigger_type FROM fermigtrig "
            "WHERE trigger_type='%s'" % (per_class * 4, kind)
        )
        url = TAP + "?" + urllib.parse.urlencode(
            {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": q}
        )
        xml = urllib.request.urlopen(url, timeout=90).read()
        parsed = decode_votable_strings(xml)
        seen = []
        for name, typ in parsed:
            if typ != kind:
                continue
            if not name.startswith("bn"):
                continue
            seen.append({"name": name, "family": kind})
            if len(seen) >= per_class:
                break
        if len(seen) < per_class:
            raise RuntimeError(f"only {len(seen)} {kind} triggers from TAP")
        rows.extend(seen)
    return rows


def list_ctime_files(name: str) -> dict[str, str]:
    year = year_of(name)
    url = f"{FTP}/{year}/{name}/current/"
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode("latin1", "replace")
    except Exception:
        return {}
    files = re.findall(r'href="(glg_ctime_(n[0-9ab]+)_%s_v(\d+)\.pha)"' % name, html)
    best: dict[str, tuple[int, str]] = {}
    for fname, det, ver in files:
        v = int(ver)
        if det not in best or v > best[det][0]:
            best[det] = (v, f"{url}{fname}")
    return {det: loc for det, (_, loc) in best.items()}


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(dest)
        return True
    except Exception:
        if tmp.exists():
            tmp.unlink()
        return False


def bin_detector(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    with fits.open(path) as hdul:
        trig = float(hdul[0].header["TRIGTIME"])
        spec = hdul["SPECTRUM"].data
        t = spec["TIME"]
        counts = spec["COUNTS"].astype(np.float64)
        quality = spec["QUALITY"]
        e = hdul["EBOUNDS"].data
        soft_ch = e["E_MAX"] <= 50.0
        hard_ch = ~soft_ch
    good = quality == 0
    t = t[good]
    counts = counts[good]
    lo, hi = trig - 6.0, trig + 10.0
    m = (t >= lo) & (t < hi)
    if m.sum() < N_TIME:
        return None
    t = t[m]
    counts = counts[m]
    tot = counts.sum(axis=1)
    peak = int(np.argmax(tot))
    start = max(0, min(peak - N_TIME // 3, len(tot) - N_TIME))
    sl = slice(start, start + N_TIME)
    c = counts[sl]
    if c.shape[0] != N_TIME:
        return None
    allc = c.sum(axis=1)
    soft = c[:, soft_ch].sum(axis=1) if soft_ch.any() else allc * 0.5
    hard = c[:, hard_ch].sum(axis=1) if hard_ch.any() else allc * 0.5
    return allc, soft, hard


def cube_for_trigger(name: str, family: str) -> dict | None:
    listing = list_ctime_files(name)
    if len(listing) < N_DET:
        return None
    series = {}
    for det, url in listing.items():
        dest = RAW / name / Path(url).name
        if not download(url, dest):
            continue
        binned = bin_detector(dest)
        if binned is None:
            continue
        series[det] = binned
    if len(series) < N_DET:
        return None
    ranked = sorted(series.items(), key=lambda kv: float(kv[1][0].sum()), reverse=True)
    pick = ranked[:N_DET]
    counts = [kv[1][0].tolist() for kv in pick]
    soft = [kv[1][1].tolist() for kv in pick]
    hard = [kv[1][2].tolist() for kv in pick]
    return {
        "name": name,
        "family": family,
        "detectors": [kv[0] for kv in pick],
        "counts": counts,
        "soft": soft,
        "hard": hard,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    catalog = fetch_catalog(args.per_class)
    with (OUT / "triggers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "family"])
        w.writeheader()
        w.writerows(catalog)
    print(f"catalog {len(catalog)} triggers", flush=True)

    cubes = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cube_for_trigger, row["name"], row["family"]): row for row in catalog}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                cube = fut.result()
            except Exception as e:
                print(f"FAIL {row['name']}: {e}", flush=True)
                continue
            if cube is None:
                print(f"skip {row['name']} ({row['family']})", flush=True)
            else:
                cubes.append(cube)
                print(f"ok   {row['name']} {row['family']} dets={cube['detectors']}", flush=True)
            if i % 5 == 0:
                time.sleep(0.2)

    by = {}
    for c in cubes:
        by.setdefault(c["family"], 0)
        by[c["family"]] += 1
    with (OUT / "cubes.jsonl").open("w") as f:
        for c in cubes:
            f.write(json.dumps(c) + "\n")
    print(json.dumps({"n": len(cubes), "by": by, "path": str(OUT / "cubes.jsonl")}))


if __name__ == "__main__":
    main()
