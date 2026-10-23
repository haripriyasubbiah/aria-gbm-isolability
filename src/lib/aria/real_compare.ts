/**
 * Calibrated vs rounding-error-bounded checksum on real CTIME cubes, at 8 / 12 / 25 per class,
 * 5 training seeds each, evaluation seed 123 (same protocol as real_eval.ts). Writes
 * experiments/gbm/check_mode_comparison.json; does not touch last_eval.json.
 *
 *   npx tsx src/lib/aria/real_compare.ts
 */
import { writeFileSync } from "node:fs";
import { setCheckMode } from "./classifier";
import { evaluateReal, loadCubes, trainOnCubes, type RealCube } from "./real_data";
import type { Precision, TransientFamily } from "./types";

const SEEDS = [7, 19, 31, 43, 61];
const FAMILIES: TransientFamily[] = ["GRB", "SGR", "SFLARE", "TGF"];
const all = loadCubes("experiments/gbm/cubes.jsonl");
const slice = (n: number): RealCube[] => FAMILIES.flatMap((f) => all.filter((c) => c.family === f).slice(0, n));

const meanStd = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return { mean: m, std: Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length) };
};
const KEYS = ["ariaAcc", "computeIso", "sensorIso", "novelIso", "novelAsCompute"] as const;

const out: Record<string, unknown[]> = {};
for (const mode of ["calibrated", "bounded"] as const) {
  setCheckMode(mode);
  out[mode] = [];
  for (const prec of ["fp32", "fp16"] as Precision[]) {
    for (const n of [8, 12, 25]) {
      const cubes = slice(n);
      const per = SEEDS.map((sd) => evaluateReal(cubes, trainOnCubes(cubes, sd, prec), 123) as unknown as Record<string, number>);
      const row: Record<string, unknown> = { prec, perClass: n, nCubes: cubes.length };
      for (const k of KEYS) row[k] = meanStd(per.map((p) => p[k]!));
      // per-seed TGF->compute so the "5 of 5 seeds clean" claim is checkable
      row.novelAsComputePerSeed = per.map((p) => p.novelAsCompute);
      (out[mode] as unknown[]).push(row);
    }
  }
}
writeFileSync("experiments/gbm/check_mode_comparison.json", JSON.stringify(out, null, 2));
const f = (x: { mean: number; std: number }) => `${(100 * x.mean).toFixed(1)}±${(100 * x.std).toFixed(1)}`;
for (const mode of ["calibrated", "bounded"]) {
  console.log(`\n${mode}`);
  console.log("prec  n/class  acc          computeIso   sensorIso    novelIso     novel->compute");
  for (const r of out[mode] as Record<string, any>[]) {
    console.log(`${r.prec}  ${String(r.perClass).padEnd(7)}  ${f(r.ariaAcc).padEnd(12)} ${f(r.computeIso).padEnd(12)} ${f(r.sensorIso).padEnd(12)} ${f(r.novelIso).padEnd(12)} ${f(r.novelAsCompute)}`);
  }
}
