/**
 * Repeated stratified k-fold cross-validation on real CTIME cubes.
 * Every cube is scored out-of-fold, spread comes from folds x repeats (real
 * uncertainty, unlike training-seed spread which is 0 on real cubes), and
 * sensor/bitflip injections cover ALL known held-out cubes (incl. SFLARE).
 *
 *   npx tsx src/lib/aria/real_eval_cv.ts experiments/gbm/cubes.jsonl experiments/gbm/cubes_draw2.jsonl
 *
 * Positional args: one or more cube files (pooled). Flags:
 *   --folds 5  --repeats 5  --train-seed 7  --eval-seed 123
 *   --fault-cap 40   (legacy protocol: only first 40 known cubes get injections)
 *   --out experiments/gbm/last_cv.json
 */
import { writeFileSync } from "node:fs";
import { evaluateReal, loadCubes, trainOnCubes, type RealCube } from "./real_data";

const argv = process.argv.slice(2);
const flag = (n: string) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 ? argv[i + 1] : undefined;
};
const files = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
if (!files.length) files.push("experiments/gbm/cubes.jsonl");
const K = Number(flag("folds") ?? 5);
const R = Number(flag("repeats") ?? 5);
const trainSeed = Number(flag("train-seed") ?? 7);
const evalSeed = Number(flag("eval-seed") ?? 123);
const faultCap = flag("fault-cap") ? Number(flag("fault-cap")) : Infinity;
const outPath = flag("out") ?? "experiments/gbm/last_cv.json";

const seen = new Set<string>();
const cubes: RealCube[] = [];
for (const f of files) for (const c of loadCubes(f)) if (!seen.has(c.name)) (seen.add(c.name), cubes.push(c));

function mulberry(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle<T>(xs: T[], rnd: () => number) {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

const METRICS = ["ariaAcc", "rangeAcc", "novelIso", "novelAsCompute", "computeIso", "sensorIso", "energyNovel", "energyCompute"] as const;
const fams = [...new Set(cubes.map((c) => c.family))];
const runs: Record<string, number>[] = [];

for (let r = 0; r < R; r++) {
  const rnd = mulberry(1000 + r);
  const folds: RealCube[][] = Array.from({ length: K }, () => []);
  for (const fam of fams) shuffle(cubes.filter((c) => c.family === fam), rnd).forEach((c, i) => folds[i % K].push(c));
  for (let k = 0; k < K; k++) {
    const test = folds[k];
    const train = folds.flatMap((f, i) => (i === k ? [] : f));
    const res = evaluateReal(test, trainOnCubes(train, trainSeed), evalSeed, faultCap) as Record<string, unknown>;
    runs.push(Object.fromEntries(METRICS.map((m) => [m, Number(res[m])])));
  }
}
const stat = (xs: number[]) => {
  const mu = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - mu) ** 2, 0) / Math.max(xs.length - 1, 1));
  const s = xs.slice().sort((a, b) => a - b);
  return { mean: mu, std: sd, p2_5: s[Math.floor(0.025 * (s.length - 1))], p97_5: s[Math.ceil(0.975 * (s.length - 1))] };
};
const summary = Object.fromEntries(METRICS.map((m) => [m, stat(runs.map((x) => x[m]))]));
const out = {
  files, nCubes: cubes.length,
  byFamily: Object.fromEntries(fams.map((f) => [f, cubes.filter((c) => c.family === f).length])),
  folds: K, repeats: R, trainSeed, evalSeed, faultCap: Number.isFinite(faultCap) ? faultCap : "all",
  note: "mean/std/percentiles are over folds x repeats; folds overlap across repeats so std is a descriptive spread, not an independent-sample SE",
  summary,
};
writeFileSync(outPath, JSON.stringify(out, null, 2));
const pct = (x: number) => (100 * x).toFixed(1);
console.log(`cubes=${out.nCubes} ${JSON.stringify(out.byFamily)} folds=${K} repeats=${R} faultCap=${out.faultCap}`);
for (const m of METRICS) { const s = summary[m]; console.log(m.padEnd(15), `${pct(s.mean)} +/- ${pct(s.std)}   [${pct(s.p2_5)}, ${pct(s.p97_5)}]`); }
