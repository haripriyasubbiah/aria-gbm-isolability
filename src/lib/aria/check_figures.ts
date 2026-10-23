/**
 * Data behind the checksum figures in the paper. Real per-trial values, no fitting:
 *   1. per-trial checksum statistic for each true cause (calibrated residual and bounded ratio)
 *   2. detection rate of a single flipped bit at every position 0..31
 * Same five training seeds and evaluation seed as paper_eval.ts. Writes paper/check_figures.json.
 *
 *   npx tsx src/lib/aria/check_figures.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { executionResidual, flipWeight, setCheckMode } from "./classifier";
import { runTrials, trainSystem } from "./experiment";
import { generateEvent } from "./physics";
import { morphologyFeatures } from "./residuals";
import { Rng } from "./rng";
import type { CheckMode, Precision } from "./types";

const SEEDS = [7, 19, 31, 43, 61];
const PER_CELL = 36;
const MODES: CheckMode[] = ["calibrated", "bounded"];
const PRECS: Precision[] = ["fp32", "fp16"];

const trials: { mode: string; prec: string; seed: number; truth: string; severity: string; exec: number; threshold: number }[] = [];
for (const mode of MODES) {
  setCheckMode(mode);
  for (const prec of PRECS) {
    for (const seed of SEEDS) {
      const sys = trainSystem(seed, prec);
      const rows = runTrials(sys, 123, PER_CELL);
      let computeSeen = 0;
      for (const r of rows) {
        let severity = "-";
        if (r.truth === "compute") severity = computeSeen++ < PER_CELL ? "mild" : "critical";
        if (r.truth === "compound") continue;
        trials.push({ mode, prec, seed, truth: r.truth, severity, exec: r.residuals.exec, threshold: sys.thresholds.exec });
      }
    }
  }
}

// Bit-position sweep: one flipped bit in one weight of the clean head, clean known-class event, N draws per bit.
const N_PER_BIT = 200;
const sweep: { mode: string; prec: string; bit: number; rate: number }[] = [];
for (const mode of MODES) {
  setCheckMode(mode);
  for (const prec of PRECS) {
    const hits = new Array(32).fill(0);
    let n = 0;
    for (const seed of SEEDS) {
      const sys = trainSystem(seed, prec);
      const rng = new Rng(seed * 31 + 5);
      const nW = sys.head.weights.length * sys.head.weights[0]!.length;
      for (let k = 0; k < N_PER_BIT / SEEDS.length; k++) {
        const fam = rng.pick(["GRB", "SGR", "SFLARE"] as const);
        const ev = generateEvent({ family: fam, fault: "none", seed: rng.int(1, 1e9) });
        const x = morphologyFeatures(ev.counts, ev.soft, ev.hard).vector;
        const idx = rng.int(0, nW - 1);
        for (let bit = 0; bit < 32; bit++) {
          const live = flipWeight(sys.head, idx, bit);
          if (executionResidual(live, x, prec, mode) > sys.thresholds.exec) hits[bit]++;
        }
        n++;
      }
    }
    for (let bit = 0; bit < 32; bit++) sweep.push({ mode, prec, bit, rate: hits[bit] / n });
  }
}

const paperDir = join(dirname(fileURLToPath(import.meta.url)), "../../../paper");
mkdirSync(paperDir, { recursive: true });
writeFileSync(join(paperDir, "check_figures.json"), JSON.stringify({ seeds: SEEDS, perCell: PER_CELL, nPerBit: N_PER_BIT, trials, sweep }, null, 2));
console.log(`wrote paper/check_figures.json: ${trials.length} trials, ${sweep.length} sweep points`);
