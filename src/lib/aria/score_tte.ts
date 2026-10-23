/**
 * Morphology-only novel test on 1 ms TTE cubes.
 *
 * Known class is GRB only (the bursts the flashes were sitting inside).
 * Distance is Euclidean in the GRB-standardized 8-feature morphology vector.
 * Each GRB is scored against the centroid of the other GRBs. The threshold is
 * the 95th percentile of those leave-one-out distances, times 1.25, the same
 * margin the lookup uses. A TGF is scored against the centroid of every GRB.
 *
 *   npx tsx src/lib/aria/score_tte.ts experiments/gbm/tte_cubes_grb_tgf.jsonl
 */
import { readFileSync, writeFileSync } from "node:fs";
import { morphologyFeatures } from "./residuals";

interface Cube {
  name: string;
  family: string;
  counts: number[][];
  soft: number[][];
  hard: number[][];
  dt?: number;
  window?: number[];
}

const path = process.argv[2] ?? "experiments/gbm/tte_cubes_grb_tgf.jsonl";
const rawCubes: Cube[] = readFileSync(path, "utf8")
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as Cube);

/** Pre-trigger median, bins covering [-0.5 s, -0.1 s). The shared morphology code estimates background from the first and last 8 bins, which is right for a 64-bin CTIME cube and wrong for a 2 s TTE cube. After this subtraction the trailing 8 bins are set to the residual background so that estimator does not treat the burst tail as background. */
function preSubtract(c: Cube): Cube {
  const n = c.counts[0]?.length ?? 0;
  const edge = Math.min(400, Math.floor(n / 5));
  const fix = (grid: number[][]): number[][] =>
    grid.map((row) => {
      const pre = row.slice(0, edge).slice().sort((a, b) => a - b);
      const bg = pre[Math.floor(pre.length / 2)] ?? 0;
      const out = row.map((v) => v - bg);
      for (let i = 0; i < 8 && i < n; i++) out[n - 1 - i] = 0;
      return out;
    });
  return { ...c, counts: fix(c.counts), soft: fix(c.soft), hard: fix(c.hard) };
}

const cubes = rawCubes.map(preSubtract);

function summedFwhmMs(c: Cube): number {
  const n = c.counts[0]?.length ?? 1;
  const dt = c.dt ?? 1;
  const prof = Array(n).fill(0);
  for (const row of c.counts) for (let t = 0; t < n; t++) prof[t] += Math.max(0, row[t] ?? 0);
  const peak = Math.max(...prof, 1e-9);
  const thr = 0.5 * peak;
  let left = 0;
  let right = n - 1;
  while (left < n && prof[left] < thr) left += 1;
  while (right > left && prof[right] < thr) right -= 1;
  return (right - left + 1) * dt * 1000;
}

function vec(c: Cube): { name: string; family: string; x: number[]; durationMs: number; fwhmMs: number; summedFwhmMs: number; hardness: number } {
  const m = morphologyFeatures(c.counts, c.soft, c.hard);
  const n = c.counts[0]?.length ?? 1;
  const dt = c.dt ?? 1;
  const spanMs = n * dt * 1000;
  return {
    name: c.name,
    family: c.family,
    x: m.vector,
    durationMs: m.features.duration * spanMs,
    fwhmMs: m.features.fwhm * spanMs,
    summedFwhmMs: summedFwhmMs(c),
    hardness: m.features.hardness,
  };
}

function percentile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))));
  return s[i]!;
}

function dist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i]! - b[i]!) ** 2;
  return Math.sqrt(s);
}

const rows = cubes.map(vec);
const grb = rows.filter((r) => r.family === "GRB");
const tgf = rows.filter((r) => r.family === "TGF");
if (grb.length < 5 || tgf.length < 5) {
  console.error(`Need GRB and TGF cubes (got ${grb.length} GRB, ${tgf.length} TGF).`);
  process.exit(1);
}

const d = grb[0]!.x.length;
const mu = Array(d).fill(0);
const sd = Array(d).fill(0);
for (const g of grb) for (let j = 0; j < d; j++) mu[j] += g.x[j]!;
for (let j = 0; j < d; j++) mu[j] /= grb.length;
for (const g of grb) for (let j = 0; j < d; j++) sd[j] += (g.x[j]! - mu[j]!) ** 2;
for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j] / grb.length) || 1;
const z = (x: number[]) => x.map((v, j) => (v - mu[j]!) / sd[j]!);
const zg = grb.map((g) => z(g.x));
const zt = tgf.map((g) => z(g.x));

function centroid(rowsZ: number[][], skip: number): number[] {
  const c = Array(d).fill(0);
  let n = 0;
  for (let i = 0; i < rowsZ.length; i++) {
    if (i === skip) continue;
    n += 1;
    for (let j = 0; j < d; j++) c[j] += rowsZ[i]![j]!;
  }
  return c.map((v) => v / n);
}

const loo = zg.map((row, i) => dist(row, centroid(zg, i)));
const full = centroid(zg, -1);
const tgfD = zt.map((row) => dist(row, full));
const threshold = percentile(loo, 0.95) * 1.25;
const med = (xs: number[]) => percentile(xs, 0.5);

const out = {
  path,
  nGrb: grb.length,
  nTgf: tgf.length,
  dt: cubes[0]?.dt ?? null,
  window: cubes[0]?.window ?? null,
  background: "per-detector median of [-0.5 s, -0.1 s), then the shared 8-bin estimator sees a zero trailing edge",
  threshold,
  grbLoo: {
    median: med(loo),
    p95: percentile(loo, 0.95),
    above: loo.filter((v) => v > threshold).length,
  },
  tgf: {
    median: med(tgfD),
    min: Math.min(...tgfD),
    max: Math.max(...tgfD),
    above: tgfD.filter((v) => v > threshold).length,
  },
  durationMs: {
    grbMedian: med(grb.map((g) => g.durationMs)),
    tgfMedian: med(tgf.map((g) => g.durationMs)),
  },
  fwhmMs: {
    grbMedian: med(grb.map((g) => g.fwhmMs)),
    tgfMedian: med(tgf.map((g) => g.fwhmMs)),
    summedGrbMedian: med(grb.map((g) => g.summedFwhmMs)),
    summedTgfMedian: med(tgf.map((g) => g.summedFwhmMs)),
  },
  hardness: {
    grbMedian: med(grb.map((g) => g.hardness)),
    tgfMedian: med(tgf.map((g) => g.hardness)),
  },
  grb: grb.map((g, i) => ({ name: g.name, distance: loo[i], durationMs: g.durationMs, fwhmMs: g.fwhmMs, summedFwhmMs: g.summedFwhmMs })),
  flashes: tgf.map((g, i) => ({
    name: g.name,
    distance: tgfD[i],
    durationMs: g.durationMs,
    fwhmMs: g.fwhmMs,
    summedFwhmMs: g.summedFwhmMs,
    above: tgfD[i]! > threshold,
  })),
};

writeFileSync("experiments/gbm/tte_morph.json", JSON.stringify(out, null, 2));
console.log(JSON.stringify({
  nGrb: out.nGrb,
  nTgf: out.nTgf,
  threshold: out.threshold,
  grbLoo: out.grbLoo,
  tgf: out.tgf,
  durationMs: out.durationMs,
  fwhmMs: out.fwhmMs,
}, null, 2));
