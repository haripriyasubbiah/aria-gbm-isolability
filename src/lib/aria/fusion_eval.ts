/**
 * Lookup versus a supervised fusion of the same four residuals.
 *
 * The fusion is a class-weighted softmax regression on
 * (r_phys, r_morph, r_exec, dropGain). It is trained on a disjoint simulator
 * stream that carries cause labels, including held-out TGF as "novel". The
 * lookup never sees those labels. Both are scored on the published evaluation
 * stream (seed 123, 36 trials per cell) and on the same 80 compound trials.
 *
 *   npx tsx src/lib/aria/fusion_eval.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { attribute } from "./aria";
import { generateEvent } from "./physics";
import { runTrials, specGrid, trainSystem, type TrainedSystem } from "./experiment";
import type { CauseLabel, EventSpec } from "./types";

const SEEDS = [7, 19, 31, 43, 61] as const;
const EVAL_SEED = 123;
const PER_CELL = 36;
const LABELS: CauseLabel[] = ["known-clean", "novel-clean", "sensor", "compute"];

interface Sample {
  x: number[];
  y: number;
}

interface Fusion {
  w: number[][];
  b: number[];
  mu: number[];
  sd: number[];
}

function feats(a: ReturnType<typeof attribute>): number[] {
  // log10 on the checksum only. The raw residual spans ~1e-8 to 1, so a linear
  // softmax standardizes away the separation a threshold can see.
  return [a.residuals.physics, a.residuals.morph, Math.log10(a.residuals.exec + 1e-12), a.residuals.dropGain];
}

function samplesFrom(system: TrainedSystem, seed: number, perCell: number): Sample[] {
  const out: Sample[] = [];
  for (const spec of specGrid(seed, perCell)) {
    const ev = generateEvent(spec);
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    const y = LABELS.indexOf(ev.truth as CauseLabel);
    if (y < 0) continue;
    out.push({ x: feats(a), y });
  }
  return out;
}

function fitFusion(samples: Sample[], l2 = 0.05, steps = 500, lr = 0.2): Fusion {
  const n = samples.length;
  const d = samples[0]!.x.length;
  const k = LABELS.length;
  const mu = Array(d).fill(0);
  const sd = Array(d).fill(0);
  for (const s of samples) for (let j = 0; j < d; j++) mu[j] += s.x[j]!;
  for (let j = 0; j < d; j++) mu[j] /= n;
  for (const s of samples) for (let j = 0; j < d; j++) sd[j] += (s.x[j]! - mu[j]!) ** 2;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j] / n) || 1;
  const z = samples.map((s) => s.x.map((v, j) => (v - mu[j]!) / sd[j]!));
  const counts = Array(k).fill(0);
  for (const s of samples) counts[s.y]! += 1;
  const classWeight = counts.map((c) => n / (k * Math.max(c, 1)));
  const w = Array.from({ length: k }, () => Array(d).fill(0));
  const b = Array(k).fill(0);
  for (let step = 0; step < steps; step++) {
    const gW = w.map((row) => row.map(() => 0));
    const gB = Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const logits = b.map((bi, c) => bi + w[c]!.reduce((s, wij, j) => s + wij * z[i]![j]!, 0));
      const m = Math.max(...logits);
      const ex = logits.map((v) => Math.exp(v - m));
      const sum = ex.reduce((p, q) => p + q, 0);
      const prob = ex.map((v) => v / sum);
      const cw = classWeight[samples[i]!.y]!;
      for (let c = 0; c < k; c++) {
        const diff = ((prob[c]! - (c === samples[i]!.y ? 1 : 0)) * cw) / n;
        gB[c] += diff;
        for (let j = 0; j < d; j++) gW[c]![j] += diff * z[i]![j]!;
      }
    }
    for (let c = 0; c < k; c++) {
      for (let j = 0; j < d; j++) {
        gW[c]![j] += l2 * w[c]![j]!;
        w[c]![j] -= lr * gW[c]![j]!;
      }
      b[c] -= lr * gB[c]!;
    }
  }
  return { w, b, mu, sd };
}

function predict(model: Fusion, x: number[]): CauseLabel {
  const z = x.map((v, j) => (v - model.mu[j]!) / model.sd[j]!);
  let best = 0;
  let bestV = -Infinity;
  for (let c = 0; c < LABELS.length; c++) {
    const v = model.b[c]! + model.w[c]!.reduce((s, wij, j) => s + wij * z[j]!, 0);
    if (v > bestV) {
      bestV = v;
      best = c;
    }
  }
  return LABELS[best]!;
}

function rate(rows: { truth: string; pred: string }[], truth: string, pred: string): number {
  const xs = rows.filter((r) => r.truth === truth);
  return xs.filter((r) => r.pred === pred).length / Math.max(xs.length, 1);
}

function meanStd(xs: number[]): { mean: number; std: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const std = Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length);
  return { mean, std };
}

function compoundRows(system: TrainedSystem, model: Fusion, n = 80) {
  const rows: { lookup: CauseLabel; fusion: CauseLabel }[] = [];
  for (let i = 0; i < n; i++) {
    const spec: EventSpec = {
      family: "GRB",
      fault: "sensor-and-compute",
      severity: i % 2 === 0 ? "mild" : "critical",
      seed: 9000 + i * 17,
    };
    const ev = generateEvent(spec);
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    rows.push({ lookup: a.cause, fusion: predict(model, feats(a)) });
  }
  const share = (key: "lookup" | "fusion", label: string) => rows.filter((r) => r[key] === label).length / n;
  return {
    n,
    lookupCompute: share("lookup", "compute"),
    lookupNovel: share("lookup", "novel-clean"),
    fusionCompute: share("fusion", "compute"),
    fusionNovel: share("fusion", "novel-clean"),
    fusionSensor: share("fusion", "sensor"),
    fusionKnown: share("fusion", "known-clean"),
  };
}

function main() {
  const perSeed = [];
  for (const trainSeed of SEEDS) {
    const system = trainSystem(trainSeed);
    const model = fitFusion(samplesFrom(system, trainSeed + 8000, PER_CELL));
    const evalRows = runTrials(system, EVAL_SEED, PER_CELL).map((r) => ({ truth: r.truth, aria: r.aria }));
    const fusionRows = samplesFrom(system, EVAL_SEED, PER_CELL).map((s) => ({
      truth: LABELS[s.y]!,
      pred: predict(model, s.x),
    }));
    const four = (rows: { truth: string; pred: string }[]) =>
      rows.filter((r) => r.pred === r.truth).length / rows.length;
    const trainRows = samplesFrom(system, trainSeed + 8000, PER_CELL).map((s) => ({
      truth: LABELS[s.y]!,
      pred: predict(model, s.x),
    }));
    perSeed.push({
      seed: trainSeed,
      lookupAcc: four(evalRows.map((r) => ({ truth: r.truth, pred: r.aria }))),
      fusionAcc: four(fusionRows),
      fusionTrainCompute: rate(trainRows, "compute", "compute"),
      fusionComputeAsSensor: rate(fusionRows, "compute", "sensor"),
      fusionComputeAsKnown: rate(fusionRows, "compute", "known-clean"),
      lookupNovel: rate(evalRows.map((r) => ({ truth: r.truth, pred: r.aria })), "novel-clean", "novel-clean"),
      fusionNovel: rate(fusionRows, "novel-clean", "novel-clean"),
      lookupNovelAsCompute: rate(evalRows.map((r) => ({ truth: r.truth, pred: r.aria })), "novel-clean", "compute"),
      fusionNovelAsCompute: rate(fusionRows, "novel-clean", "compute"),
      lookupComputeAsNovel: rate(evalRows.map((r) => ({ truth: r.truth, pred: r.aria })), "compute", "novel-clean"),
      fusionComputeAsNovel: rate(fusionRows, "compute", "novel-clean"),
      lookupCompute: rate(evalRows.map((r) => ({ truth: r.truth, pred: r.aria })), "compute", "compute"),
      fusionCompute: rate(fusionRows, "compute", "compute"),
      lookupSensor: rate(evalRows.map((r) => ({ truth: r.truth, pred: r.aria })), "sensor", "sensor"),
      fusionSensor: rate(fusionRows, "sensor", "sensor"),
      compound: compoundRows(system, model),
    });
    console.log(JSON.stringify(perSeed[perSeed.length - 1]));
  }
  const keys = [
    "lookupAcc",
    "fusionAcc",
    "lookupNovel",
    "fusionNovel",
    "lookupNovelAsCompute",
    "fusionNovelAsCompute",
    "lookupComputeAsNovel",
    "fusionComputeAsNovel",
    "lookupCompute",
    "fusionCompute",
    "lookupSensor",
    "fusionSensor",
    "fusionTrainCompute",
    "fusionComputeAsSensor",
    "fusionComputeAsKnown",
  ] as const;
  const summary = Object.fromEntries(keys.map((k) => [k, meanStd(perSeed.map((s) => s[k] as number))]));
  const compoundKeys = ["lookupCompute", "lookupNovel", "fusionCompute", "fusionNovel", "fusionSensor", "fusionKnown"] as const;
  const compound = Object.fromEntries(
    compoundKeys.map((k) => [k, meanStd(perSeed.map((s) => s.compound[k]))]),
  );
  const out = {
    note: "Fusion is a class-weighted softmax on (r_phys, r_morph, log10(r_exec), dropGain), trained on a disjoint labeled stream (seed+8000) that includes TGF labeled novel. Lookup thresholds use clean known-class residuals only. Both are scored on eval seed 123. The log is a monotone rescaling so the linear model can see a residual that otherwise spans eight decades.",
    summary,
    compound,
    perSeed,
  };
  mkdirSync("experiments/gbm", { recursive: true });
  writeFileSync("experiments/gbm/fusion.json", JSON.stringify(out, null, 2));
  console.log(JSON.stringify({ summary, compound }, null, 2));
}

main();
