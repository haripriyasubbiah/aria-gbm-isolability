/**
 * Soundness check for the rounding-error-bounded checksum.
 * On fault-free inputs the ratio |live - stored| / bound must never exceed 1,
 * for any input, including ones far outside the calibration range.
 *
 *   npx tsx src/lib/aria/verify_bound.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { executionResidual, setCheckMode } from "./classifier";
import { trainSystem } from "./experiment";
import { Rng } from "./rng";
import type { Precision } from "./types";

setCheckMode("bounded");
const rng = new Rng(2024);
let failed = false;
const summary: Record<string, number> = {};
const N_INPUTS = 40000;
for (const prec of ["fp64", "fp32", "fp16"] as Precision[]) {
  const sys = trainSystem(7, prec);
  let worst = 0;
  const byScale: Record<string, number> = {};
  const N = N_INPUTS;
  for (let n = 0; n < N; n++) {
    const scale = [0.01, 1, 3, 10, 100, 1000][n % 6]!;
    // random raw features in standardised units, then mapped back to raw so executionResidual sees a normal input
    const xRaw = sys.head.featureMean.map((m, j) => m + sys.head.featureStd[j]! * rng.gaussian(0, scale));
    const ratio = executionResidual(sys.head, xRaw, prec);
    worst = Math.max(worst, ratio);
    const k = `x${scale}`;
    byScale[k] = Math.max(byScale[k] ?? 0, ratio);
  }
  console.log(prec.padEnd(5), "max clean ratio =", worst.toExponential(2), " by input scale:", Object.fromEntries(Object.entries(byScale).map(([k, v]) => [k, +v.toExponential(2)])));
  summary[prec] = worst;
  if (worst > 1) failed = true;
}
const paperDir = join(dirname(fileURLToPath(import.meta.url)), "../../../paper");
mkdirSync(paperDir, { recursive: true });
writeFileSync(join(paperDir, "bound_check.json"), JSON.stringify({ inputsPerPrecision: N_INPUTS, maxScaleSigma: 1000, worstCleanRatio: summary }, null, 2));
console.log(failed ? "FAIL: bound violated on clean input" : "OK: bound holds (ratio <= 1) on all clean inputs");
process.exit(failed ? 1 : 0);
