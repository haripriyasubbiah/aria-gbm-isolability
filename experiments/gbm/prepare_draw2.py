#!/usr/bin/env python3
"""Second, DISJOINT draw of public Fermi-GBM CTIME triggers for ARIA.

prepare.py takes the first N triggers the TAP service returns, so rerunning it
with a bigger --per-class only re-includes the same events. This script instead
(1) pulls a large candidate pool per class, (2) removes every trigger already in
experiments/gbm/triggers.csv, (3) shuffles the rest with --draw-seed, and
(4) keeps downloading until it has --per-class usable cubes per class.

Usage (from repo root):
  python3 experiments/gbm/prepare_draw2.py --per-class 25 --draw-seed 2
  python3 experiments/gbm/prepare_draw2.py --per-class 25 --draw-seed 3 --tag draw3 \
      --exclude experiments/gbm/triggers.csv experiments/gbm/triggers_draw2.csv

Writes experiments/gbm/triggers_<tag>.csv and cubes_<tag>.jsonl (default tag draw2).
Raw FITS go to experiments/gbm/raw/ (resume-safe, shared with prepare.py).
Reuses prepare.py's binning code unchanged, so cubes are comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare as P  # noqa: E402


def candidate_pool(kind: str, top: int) -> list[str]:
    q = f"SELECT TOP {top} trigger_name, trigger_type FROM fermigtrig WHERE trigger_type='{kind}'"
    url = P.TAP + "?" + urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": q}
    )
    xml = urllib.request.urlopen(url, timeout=120).read()
    rows = P.decode_votable_strings(xml)
    names = sorted({n for n, t in rows if t == kind and n.startswith("bn")})
    return names


def read_names(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        with open(p, newline="") as f:
            out |= {r["name"] for r in csv.DictReader(f)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--draw-seed", type=int, default=2)
    ap.add_argument("--tag", default="draw2")
    ap.add_argument("--pool", type=int, default=3000, help="TAP candidates per class")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--exclude", nargs="*", default=[str(P.OUT / "triggers.csv")])
    args = ap.parse_args()

    P.RAW.mkdir(parents=True, exist_ok=True)
    banned = read_names(args.exclude)
    print(f"excluding {len(banned)} already-used triggers", flush=True)
    rng = random.Random(args.draw_seed)

    kept_cubes, kept_rows = [], []
    for kind in P.TYPES:
        pool = [n for n in candidate_pool(kind, args.pool) if n not in banned]
        rng.shuffle(pool)
        print(f"{kind}: {len(pool)} unused candidates", flush=True)
        got: list[dict] = []
        i = 0
        # Process in chunks so we stop as soon as the quota is met.
        while len(got) < args.per_class and i < len(pool):
            chunk = pool[i : i + max(args.workers * 2, args.per_class - len(got))]
            i += len(chunk)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(P.cube_for_trigger, n, kind): n for n in chunk}
                for fut in as_completed(futs):
                    n = futs[fut]
                    try:
                        cube = fut.result()
                    except Exception as e:  # network hiccup: skip, keep going
                        print(f"FAIL {n}: {e}", flush=True)
                        continue
                    if cube is None:
                        print(f"skip {n} ({kind})", flush=True)
                        continue
                    if len(got) < args.per_class:
                        got.append(cube)
                        print(f"ok   {n} {kind} [{len(got)}/{args.per_class}]", flush=True)
        if len(got) < args.per_class:
            print(f"WARNING: only {len(got)} usable {kind} cubes (pool exhausted)", flush=True)
        kept_cubes += got
        kept_rows += [{"name": c["name"], "family": kind} for c in got]

    with (P.OUT / f"triggers_{args.tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "family"])
        w.writeheader()
        w.writerows(kept_rows)
    with (P.OUT / f"cubes_{args.tag}.jsonl").open("w") as f:
        for c in kept_cubes:
            f.write(json.dumps(c) + "\n")
    by: dict[str, int] = {}
    for c in kept_cubes:
        by[c["family"]] = by.get(c["family"], 0) + 1
    print(json.dumps({"n": len(kept_cubes), "by": by, "tag": args.tag}))


if __name__ == "__main__":
    main()
