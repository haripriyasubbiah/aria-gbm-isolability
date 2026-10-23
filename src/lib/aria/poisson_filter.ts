/**
 * Maximized-amplitude Poisson likelihood ratio for a single-detector excess.
 *
 * Perera, Zackay, and Venumadhav (ApJS 2025) fold a bank of spectral templates
 * through the GBM detector response and score a fixed-amplitude Poisson matched
 * filter. These cubes have no response matrices. The bit here is the likelihood
 * ratio with the excess amplitude maximized, on the soft and hard counts already
 * stored, two time bins, one detector. A flat fixed-amplitude template on bins
 * with similar background reduces to a Gaussian excess; that is the short-window
 * score already reported, and it is not repeated here.
 *
 *   npx tsx src/lib/aria/poisson_filter.ts
 *
 * Writes experiments/gbm/poisson_filter.json.
 * The lookup is unchanged. The bit is OR-ed in only when the lookup does not
 * already say compute. The threshold is the 96th percentile of clean known
 * scores on the training draw, times 1.25, and the drop in score when the
 * winning detector is removed must exceed 0.28.
 */
import { writeFileSync } from "node:fs";
import { attribute } from "./aria";
import { trainSystem } from "./experiment";
import { generateEvent } from "./physics";
import { cubeToEvent, loadCubes, trainOnCubes, type RealCube } from "./real_data";
import { Rng } from "./rng";
import type { EventCube, EventSpec } from "./types";

const SEEDS = [7, 19, 31, 43, 61] as const;
const EVAL_SEED = 123;
const DROP = 0.28;
const B_FLOOR = 0.5;

function percentile(xs: number[], q: number): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))]!;
}

function meanStd(xs: number[]): { mean: number; std: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
  const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(xs.length, 1);
  return { mean, std: Math.sqrt(v) };
}

/** Upper median of the first and last eight bins, same index as subtractBackground, floored so log(d/b) is defined. */
function edgeBackground(row: number[]): number {
  const edge = row.slice(0, 8).concat(row.slice(-8));
  const bg = edge.sort((a, b) => a - b)[Math.floor(edge.length / 2)] ?? 0;
  return Math.max(bg, B_FLOOR);
}

/** Poisson LLR for a non-negative excess, amplitude maximized. Zero when the bin is not above background. */
export function poissonBinLLR(d: number, b: number): number {
  if (!(d > b) || !(b > 0) || !Number.isFinite(d)) return 0;
  return d * Math.log(d / b) - (d - b);
}

export function poissonSpike(soft: number[][], hard: number[][]): { score: number; dropGain: number; det: number } {
  const n = soft.length;
  const m = soft[0]?.length ?? 0;
  const scores = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const bs = edgeBackground(soft[i]!);
    const bh = edgeBackground(hard[i]!);
    let best = 0;
    for (let t = 0; t < m - 1; t++) {
      const s =
        poissonBinLLR(soft[i]![t]!, bs) +
        poissonBinLLR(soft[i]![t + 1]!, bs) +
        poissonBinLLR(hard[i]![t]!, bh) +
        poissonBinLLR(hard[i]![t + 1]!, bh);
      if (s > best) best = s;
    }
    scores[i] = best;
  }
  let det = 0;
  for (let i = 1; i < n; i++) if (scores[i]! > scores[det]!) det = i;
  const second = Math.max(0, ...scores.filter((_, i) => i !== det));
  const best = scores[det]!;
  return { score: best, dropGain: (best - second) / (best + 1e-9), det };
}

function checkIdentity(): { llr: number; det: number; dropGain: number } {
  const expected = 20 * Math.log(2) - 10;
  const got = poissonBinLLR(20, 10);
  if (Math.abs(got - expected) > 1e-9) throw new Error(`LLR identity ${got} != ${expected}`);
  if (poissonBinLLR(10, 10) !== 0 || poissonBinLLR(5, 10) !== 0) throw new Error("LLR should be zero at or below background");
  const soft = Array.from({ length: 8 }, () => Array.from({ length: 64 }, () => 10));
  const hard = Array.from({ length: 8 }, () => Array.from({ length: 64 }, () => 8));
  soft[3]![20] = 100;
  soft[3]![21] = 100;
  const sp = poissonSpike(soft, hard);
  if (sp.det !== 3 || sp.dropGain < 0.9 || !(sp.score > 50)) {
    throw new Error(`synthetic spike not recovered ${JSON.stringify(sp)}`);
  }
  return { llr: got, det: sp.det, dropGain: sp.dropGain };
}

type Tagged = {
  truth: string;
  pred: string;
  score: number;
  dropGain: number;
  fires: boolean;
  scoreOnly: boolean;
  dropOnly: boolean;
  detMatch: boolean | null;
};

function rate(rows: Tagged[], truth: string, pred: string): number {
  const xs = rows.filter((r) => r.truth === truth);
  return xs.filter((r) => r.pred === pred).length / Math.max(xs.length, 1);
}

function fourAcc(rows: Tagged[]): number {
  return rows.filter((r) => r.pred === r.truth).length / Math.max(rows.length, 1);
}

function pack(rows: Tagged[]) {
  return {
    ariaAcc: fourAcc(rows),
    novelIso: rate(rows, "novel-clean", "novel-clean"),
    novelAsCompute: rate(rows, "novel-clean", "compute"),
    novelAsSensor: rate(rows, "novel-clean", "sensor"),
    computeIso: rate(rows, "compute", "compute"),
    computeAsSensor: rate(rows, "compute", "sensor"),
    sensorIso: rate(rows, "sensor", "sensor"),
    knownAsSensor: rate(rows, "known-clean", "sensor"),
    n: rows.length,
  };
}

function orPoisson(rows: Tagged[]): Tagged[] {
  return rows.map((r) => {
    if (r.pred === "compute") return r;
    if (r.fires && r.pred !== "sensor") return { ...r, pred: "sensor" };
    return r;
  });
}

function avgPacks(packs: ReturnType<typeof pack>[]) {
  const keys = Object.keys(packs[0]!).filter((k) => k !== "n") as (keyof ReturnType<typeof pack>)[];
  const out: Record<string, { mean: number; std: number }> = {};
  for (const k of keys) out[k] = meanStd(packs.map((p) => Number(p[k])));
  return out;
}

function frac(rows: Tagged[], pred: (r: Tagged) => boolean, truth?: string): number {
  const xs = truth ? rows.filter((r) => r.truth === truth) : rows;
  return xs.filter(pred).length / Math.max(xs.length, 1);
}

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : 0.5 * (s[mid - 1]! + s[mid]!);
}

function tag(ev: EventCube, pred: string, threshold: number): Tagged {
  const sp = poissonSpike(ev.soft, ev.hard);
  const scoreOnly = sp.score > threshold;
  const dropOnly = sp.dropGain > DROP;
  return {
    truth: ev.truth,
    pred,
    score: sp.score,
    dropGain: sp.dropGain,
    fires: scoreOnly && dropOnly,
    scoreOnly,
    dropOnly,
    detMatch: ev.sensorDetector === null ? null : sp.det === ev.sensorDetector,
  };
}

function summarize(per: Tagged[][]) {
  const head = per.map(pack);
  const ored = per.map((rows) => pack(orPoisson(rows)));
  const bitOf = (truth: string, pred: (r: Tagged) => boolean) =>
    meanStd(per.map((rows) => frac(rows, pred, truth)));
  const sensorRows = (rows: Tagged[]) => rows.filter((r) => r.truth === "sensor");
  return {
    headline: avgPacks(head),
    orPoisson: avgPacks(ored),
    threshold: meanStd(per.map(() => 0)),
    bit: {
      sensor: bitOf("sensor", (r) => r.fires),
      known: bitOf("known-clean", (r) => r.fires),
      novel: bitOf("novel-clean", (r) => r.fires),
      compute: bitOf("compute", (r) => r.fires),
      sensorScoreOnly: bitOf("sensor", (r) => r.scoreOnly),
      sensorDropOnly: bitOf("sensor", (r) => r.dropOnly),
      knownScoreOnly: bitOf("known-clean", (r) => r.scoreOnly),
      knownDropOnly: bitOf("known-clean", (r) => r.dropOnly),
    },
    detMatch: meanStd(
      per.map((rows) => {
        const xs = sensorRows(rows).filter((r) => r.detMatch !== null);
        return xs.filter((r) => r.detMatch).length / Math.max(xs.length, 1);
      }),
    ),
    detMatchWhenFires: meanStd(
      per.map((rows) => {
        const xs = sensorRows(rows).filter((r) => r.fires && r.detMatch !== null);
        return xs.filter((r) => r.detMatch).length / Math.max(xs.length, 1);
      }),
    ),
    medianScore: {
      known: meanStd(per.map((rows) => median(rows.filter((r) => r.truth === "known-clean").map((r) => r.score)))),
      sensor: meanStd(per.map((rows) => median(rows.filter((r) => r.truth === "sensor").map((r) => r.score)))),
      novel: meanStd(per.map((rows) => median(rows.filter((r) => r.truth === "novel-clean").map((r) => r.score)))),
    },
  };
}

function simulatorOne(seed: number): { rows: Tagged[]; threshold: number } {
  const system = trainSystem(seed);
  const rng = new Rng(50 + seed);
  const clean: number[] = [];
  for (let i = 0; i < 120; i++) {
    const ev = generateEvent({
      family: (["GRB", "SGR", "SFLARE"] as const)[i % 3]!,
      fault: "none",
      seed: rng.int(1, 1e9),
    });
    clean.push(poissonSpike(ev.soft, ev.hard).score);
  }
  const threshold = percentile(clean, 0.96) * 1.25;
  const grid = new Rng(EVAL_SEED);
  const specs: Omit<EventSpec, "seed">[] = [
    { family: "GRB", fault: "none" },
    { family: "SGR", fault: "none" },
    { family: "SFLARE", fault: "none" },
    { family: "TGF", fault: "none" },
    { family: "GRB", fault: "sensor-spike" },
    { family: "SGR", fault: "sensor-spike" },
    { family: "GRB", fault: "shared-bitflip", severity: "mild" },
    { family: "GRB", fault: "shared-bitflip", severity: "critical" },
  ];
  const rows: Tagged[] = [];
  for (const spec of specs) {
    for (let i = 0; i < 36; i++) {
      const ev = generateEvent({ ...spec, seed: grid.int(1, 1e9) });
      const a = attribute(ev, system.head, system.thresholds, system.prec);
      rows.push(tag(ev, a.cause, threshold));
    }
  }
  return { rows, threshold };
}

function realOne(train: RealCube[], test: RealCube[], seed: number, faultCap: number): { rows: Tagged[]; threshold: number } {
  const system = trainOnCubes(train, seed);
  const knownTrain = train.filter((c) => c.family !== "TGF");
  const clean: number[] = [];
  for (const c of knownTrain) {
    const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: 1 });
    clean.push(poissonSpike(ev.soft, ev.hard).score);
  }
  const threshold = percentile(clean, 0.96) * 1.25;
  const rng = new Rng(EVAL_SEED);
  const known = test.filter((c) => c.family !== "TGF");
  const novel = test.filter((c) => c.family === "TGF");
  const rows: Tagged[] = [];
  const push = (cube: RealCube, fault: EventSpec["fault"], severity?: EventSpec["severity"]) => {
    const ev = cubeToEvent(cube, { family: cube.family, fault, severity, seed: rng.int(1, 1e9) });
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    rows.push(tag(ev, a.cause, threshold));
  };
  for (const c of known) push(c, "none");
  for (const c of novel) push(c, "none");
  const injected = known.slice(0, Math.min(known.length, faultCap));
  for (const c of injected) push(c, "sensor-spike");
  for (const c of injected) push(c, "shared-bitflip", "critical");
  return { rows, threshold };
}

function close(got: number, expected: number, tol = 0.012): boolean {
  return Math.abs(got - expected) <= tol;
}

function main() {
  const identity = checkIdentity();
  const draw1 = loadCubes("experiments/gbm/cubes.jsonl");
  const draw2 = loadCubes("experiments/gbm/cubes_draw2.jsonl");
  const overlap = draw2.filter((c) => draw1.some((t) => t.name === c.name));
  if (overlap.length) throw new Error(`draw overlap ${overlap.length}`);

  console.log("simulator ...");
  const simRuns = SEEDS.map((s) => simulatorOne(s));
  console.log("draw 1 in-sample ...");
  const d1 = SEEDS.map((s) => realOne(draw1, draw1, s, 40));
  console.log("draw 2 in-sample ...");
  const d2 = SEEDS.map((s) => realOne(draw2, draw2, s, 40));
  console.log("draw 1 -> draw 2 ...");
  const oneToTwo = SEEDS.map((s) => realOne(draw1, draw2, s, 40));
  console.log("draw 2 -> draw 1 ...");
  const twoToOne = SEEDS.map((s) => realOne(draw2, draw1, s, 40));

  const block = (runs: { rows: Tagged[]; threshold: number }[]) => {
    const summary = summarize(runs.map((r) => r.rows));
    summary.threshold = meanStd(runs.map((r) => r.threshold));
    return summary;
  };

  const simulator = block(simRuns);
  const draw1In = block(d1);
  const draw2In = block(d2);
  const held12 = block(oneToTwo);
  const held21 = block(twoToOne);

  const sanity = {
    simulatorSensor: simulator.headline.sensorIso!.mean,
    draw1Sensor: draw1In.headline.sensorIso!.mean,
    draw2Sensor: draw2In.headline.sensorIso!.mean,
    oneToTwoSensor: held12.headline.sensorIso!.mean,
    twoToOneSensor: held21.headline.sensorIso!.mean,
    expected: { simulator: 0.775, draw1: 0.6, draw2: 0.25, oneToTwo: 0.65, twoToOne: 0.075 },
  };
  const ok =
    close(sanity.simulatorSensor, 0.775) &&
    close(sanity.draw1Sensor, 0.6) &&
    close(sanity.draw2Sensor, 0.25) &&
    close(sanity.oneToTwoSensor, 0.65) &&
    close(sanity.twoToOneSensor, 0.075);
  if (!ok) {
    console.error("SANITY FAIL", sanity);
    process.exit(1);
  }

  const out = {
    statistic:
      "Maximized-amplitude Poisson LLR on two adjacent bins of soft and of hard, per detector. Background is the upper median of the first and last eight bins, floored at 0.5. No detector response and no spectral template.",
    cite: "Perera, Zackay, Venumadhav, ApJS 281 (2025) 4, for the Poisson matched-filter idea. Amplitude is maximized because a glitch amplitude is not a GRB detection limit.",
    dropGain: DROP,
    margin: 1.25,
    percentile: 0.96,
    evalSeed: EVAL_SEED,
    seeds: SEEDS,
    identity,
    sanity,
    simulator,
    draw1: draw1In,
    draw2: draw2In,
    oneToTwo: held12,
    twoToOne: held21,
  };
  writeFileSync("experiments/gbm/poisson_filter.json", JSON.stringify(out, null, 2));

  const pct = (x: { mean: number; std: number }) => `${(100 * x.mean).toFixed(1)} ± ${(100 * x.std).toFixed(1)}`;
  const line = (name: string, s: typeof simulator) =>
    [
      name,
      `sensor ${pct(s.headline.sensorIso!)} -> ${pct(s.orPoisson.sensorIso!)}`,
      `acc ${pct(s.headline.ariaAcc!)} -> ${pct(s.orPoisson.ariaAcc!)}`,
      `known→s ${pct(s.headline.knownAsSensor!)} -> ${pct(s.orPoisson.knownAsSensor!)}`,
      `novel→s ${pct(s.headline.novelAsSensor!)} -> ${pct(s.orPoisson.novelAsSensor!)}`,
      `compute ${pct(s.headline.computeIso!)} -> ${pct(s.orPoisson.computeIso!)}`,
      `bit sensor/known/novel/compute ${pct(s.bit.sensor)} / ${pct(s.bit.known)} / ${pct(s.bit.novel)} / ${pct(s.bit.compute)}`,
      `sensor score-only ${pct(s.bit.sensorScoreOnly)} drop-only ${pct(s.bit.sensorDropOnly)}`,
      `known score-only ${pct(s.bit.knownScoreOnly)} drop-only ${pct(s.bit.knownDropOnly)}`,
      `det match ${pct(s.detMatch)} when fires ${pct(s.detMatchWhenFires)}`,
      `median score known ${s.medianScore.known.mean.toFixed(1)} sensor ${s.medianScore.sensor.mean.toFixed(1)} novel ${s.medianScore.novel.mean.toFixed(1)}`,
      `threshold ${s.threshold.mean.toFixed(2)}`,
    ].join("\n  ");
  console.log(line("simulator", simulator));
  console.log(line("draw1", draw1In));
  console.log(line("draw2", draw2In));
  console.log(line("1to2", held12));
  console.log(line("2to1", held21));
  console.log("sanity ok; wrote experiments/gbm/poisson_filter.json");
}

main();
