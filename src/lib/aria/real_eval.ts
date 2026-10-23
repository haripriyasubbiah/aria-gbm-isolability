/**
 * Kill experiment: ARIA on public Fermi-GBM CTIME cubes.
 *
 *   python3 experiments/gbm/prepare.py --per-class 25
 *   npm run test:gbm
 *
 * Runs the same five training seeds as the simulator tables (paper_eval.ts) so
 * the real-data table can report mean +/- std instead of a single run. The
 * evaluation seed is fixed, exactly as in the simulator protocol.
 *
 * Optional overrides:
 *   npm run test:gbm -- --seeds 7,19,31 --eval-seed 123
 */
import { writeFileSync } from "node:fs";
import { evaluateReal, loadCubes, trainOnCubes } from "./real_data";

const DEFAULT_SEEDS = [7, 19, 31, 43, 61];
const DEFAULT_EVAL_SEED = 123;

const argv = process.argv.slice(2);
function flag(name: string): string | undefined {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
}

const path = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--"))) ?? "experiments/gbm/cubes.jsonl";
const seeds = (flag("seeds") ?? DEFAULT_SEEDS.join(",")).split(",").map((s) => Number(s.trim()));
const evalSeed = Number(flag("eval-seed") ?? DEFAULT_EVAL_SEED);

const cubes = loadCubes(path);
if (cubes.length < 8) {
  console.error(`Need cubes at ${path}. Run: python3 experiments/gbm/prepare.py --per-class 25`);
  process.exit(1);
}

const METRICS = [
  "ariaAcc",
  "rangeAcc",
  "novelIso",
  "novelAsCompute",
  "computeIso",
  "sensorIso",
  "energyNovel",
  "energyCompute",
] as const;

function meanStd(xs: number[]): { mean: number; std: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
  const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(xs.length, 1);
  return { mean, std: Math.sqrt(v) };
}

const perSeed = seeds.map((seed) => {
  const system = trainOnCubes(cubes, seed);
  const r = evaluateReal(cubes, system, evalSeed) as Record<string, unknown>;
  return { seed, ...r } as Record<string, unknown> & { seed: number };
});

const first = perSeed[0];
const mean: Record<string, number> = {};
const std: Record<string, number> = {};
for (const m of METRICS) {
  const { mean: mu, std: sd } = meanStd(perSeed.map((p) => Number(p[m])));
  mean[m] = mu;
  std[m] = sd;
}

const result = {
  nCubes: first.nCubes,
  byFamily: first.byFamily,
  nTrials: first.nTrials,
  seeds,
  evalSeed,
  // Top-level metric values are across-seed means so downstream consumers
  // (experiments/gbm/make_realdata_tex.py) keep working; spreads are in std.
  ...mean,
  std,
  perSeed,
};

console.log(JSON.stringify({ nCubes: result.nCubes, nTrials: result.nTrials, seeds, evalSeed, mean, std }, null, 2));
writeFileSync("experiments/gbm/last_eval.json", JSON.stringify(result, null, 2));

const worstNovelAsCompute = Math.max(...perSeed.map((p) => Number(p.novelAsCompute)));
if (worstNovelAsCompute > 0.15) {
  console.error("FAIL: novelty is leaking into compute on real CTIME in at least one seed.");
  process.exit(2);
}
console.log(`\nOK - held-out TGF is not named compute in any of ${seeds.length} seeds.`);
if (mean.novelIso < 0.5) {
  console.log("NOTE - morphology did not isolate TGF as novel on CTIME (64 ms-class bins). Next: TTE at ~1 ms if you need the discovery box.");
}
