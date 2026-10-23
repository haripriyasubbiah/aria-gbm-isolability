/**
 * Does the isolate-channel action name the detector that received the spike?
 *
 * The published injector puts the spike on the dimmest detector that is not
 * the brightest. The same index on the clean cube, with no spike, is the
 * baseline: if the residual already points there, a match is not localization.
 *
 * A second probe, not used in the headline tables, draws the detector uniformly
 * from all eight and uses the same spike amplitude.
 *
 *   npx tsx src/lib/aria/channel_id.ts
 *
 * Writes experiments/gbm/channel_id.json. Does not overwrite other result files.
 */
import { writeFileSync } from "node:fs";
import { attribute } from "./aria";
import { specGrid, trainSystem } from "./experiment";
import { generateEvent } from "./physics";
import { cubeToEvent, loadCubes, trainOnCubes, type RealCube } from "./real_data";
import { Rng } from "./rng";
import { N_DET, N_TIME, type EventCube, type EventSpec, type TransientFamily } from "./types";

const SEEDS = [7, 19, 31, 43, 61] as const;
const EVAL_SEED = 123;
const PER_CELL = 36;

function meanStd(xs: number[]): { mean: number; std: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
  const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(xs.length, 1);
  return { mean, std: Math.sqrt(v) };
}

function dimmest(counts: number[][]): number {
  const totals = counts.map((row) => row.reduce((a, b) => a + b, 0));
  const brightest = totals.reduce((best, v, i) => (v > totals[best]! ? i : best), 0);
  const dim = totals
    .map((v, i) => [v, i] as const)
    .filter(([, i]) => i !== brightest)
    .sort((a, b) => a[0] - b[0]);
  return dim[0]?.[1] ?? 0;
}

function cloneGrid(g: number[][]): number[][] {
  return g.map((row) => row.slice());
}

/** Same spike recipe as the published injector, on a chosen detector. */
function spikeOn(event: EventCube, det: number, rng: Rng): EventCube {
  const counts = cloneGrid(event.counts);
  const soft = cloneGrid(event.soft);
  const peak = Math.max(...counts.flat(), 20);
  const t0 = Math.min(N_TIME - 3, Math.max(8, rng.int(10, 48)));
  const width = rng.int(2, 3);
  const spike = peak * rng.uniform(1.6, 3.2);
  for (let t = t0; t < Math.min(N_TIME, t0 + width); t++) {
    soft[det]![t] += spike;
    counts[det]![t] += spike;
  }
  return { ...event, counts, soft, sensorDetector: det };
}

interface Hit {
  named: boolean;
  called: boolean;
  family: string;
}

function pack(hits: Hit[]) {
  const n = hits.length;
  const called = hits.filter((h) => h.called);
  const rate = (xs: Hit[]) => xs.filter((h) => h.named).length / Math.max(xs.length, 1);
  const byFamily: Record<string, { n: number; named: number; called: number; namedIfCalled: number }> = {};
  for (const h of hits) {
    const b = (byFamily[h.family] ??= { n: 0, named: 0, called: 0, namedIfCalled: 0 });
    b.n += 1;
    if (h.named) b.named += 1;
    if (h.called) {
      b.called += 1;
      if (h.named) b.namedIfCalled += 1;
    }
  }
  return {
    n,
    nCalled: called.length,
    callRate: called.length / Math.max(n, 1),
    namedRate: rate(hits),
    namedIfCalled: rate(called),
    byFamily,
  };
}

function simPublished(seed: number): { spiked: Hit[]; baseline: Hit[] } {
  const system = trainSystem(seed);
  const spiked: Hit[] = [];
  const baseline: Hit[] = [];
  for (const spec of specGrid(EVAL_SEED, PER_CELL)) {
    if (spec.fault !== "sensor-spike") continue;
    const ev = generateEvent(spec);
    const clean = generateEvent({ ...spec, fault: "none" });
    const target = ev.sensorDetector;
    if (target === null) continue;
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    const b = attribute(clean, system.head, system.thresholds, system.prec);
    spiked.push({ named: a.worstDetector === target, called: a.cause === "sensor", family: spec.family });
    baseline.push({ named: b.worstDetector === target, called: b.cause === "sensor", family: spec.family });
  }
  return { spiked, baseline };
}

function simRandom(seed: number): Hit[] {
  const system = trainSystem(seed);
  const rng = new Rng(EVAL_SEED + 5000);
  const hits: Hit[] = [];
  for (const spec of specGrid(EVAL_SEED, PER_CELL)) {
    if (spec.fault !== "sensor-spike") continue;
    const clean = generateEvent({ ...spec, fault: "none" });
    const det = rng.int(0, N_DET - 1);
    const ev = spikeOn(clean, det, rng);
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    hits.push({ named: a.worstDetector === det, called: a.cause === "sensor", family: spec.family });
  }
  return hits;
}

function ctimeRows(cubes: RealCube[], seed: number, faultCap: number): { spiked: Hit[]; baseline: Hit[]; random: Hit[] } {
  const system = trainOnCubes(cubes, seed);
  const rng = new Rng(EVAL_SEED);
  const known = cubes.filter((c) => c.family !== "TGF");
  const novel = cubes.filter((c) => c.family === "TGF");
  const spiked: Hit[] = [];
  const baseline: Hit[] = [];
  // Match evaluateReal's seed consumption so these spikes are the published ones.
  for (const c of known) rng.int(1, 1e9);
  for (const c of novel) rng.int(1, 1e9);
  const injected = known.slice(0, Math.min(known.length, faultCap));
  for (const c of injected) {
    const spec: EventSpec = { family: c.family as TransientFamily, fault: "sensor-spike", seed: rng.int(1, 1e9) };
    const ev = cubeToEvent(c, spec);
    const clean = cubeToEvent(c, { ...spec, fault: "none" });
    const target = ev.sensorDetector;
    if (target === null) continue;
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    const b = attribute(clean, system.head, system.thresholds, system.prec);
    spiked.push({ named: a.worstDetector === target, called: a.cause === "sensor", family: c.family });
    baseline.push({ named: b.worstDetector === target, called: b.cause === "sensor", family: c.family });
  }
  const rr = new Rng(EVAL_SEED + 5000);
  const random: Hit[] = [];
  for (const c of injected) {
    const clean = cubeToEvent(c, { family: c.family as TransientFamily, fault: "none", seed: rr.int(1, 1e9) });
    const det = rr.int(0, N_DET - 1);
    const ev = spikeOn(clean, det, rr);
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    random.push({ named: a.worstDetector === det, called: a.cause === "sensor", family: c.family });
  }
  return { spiked, baseline, random };
}

function across(packs: ReturnType<typeof pack>[], key: "namedRate" | "namedIfCalled" | "callRate") {
  return meanStd(packs.map((p) => p[key]));
}

function main() {
  const simS = SEEDS.map((s) => simPublished(s));
  const simR = SEEDS.map((s) => pack(simRandom(s)));
  const draw1 = loadCubes("experiments/gbm/cubes.jsonl");
  const draw2 = loadCubes("experiments/gbm/cubes_draw2.jsonl");
  const ctime = (cubes: RealCube[], cap: number) => SEEDS.map((s) => ctimeRows(cubes, s, cap));

  const blocks = {
    simulatorPublished: simS.map((s) => pack(s.spiked)),
    simulatorCleanBaseline: simS.map((s) => pack(s.baseline)),
    simulatorRandomDetector: simR,
    ctimeCap40: ctime(draw1, 40).map((s) => pack(s.spiked)),
    ctimeCap40Baseline: ctime(draw1, 40).map((s) => pack(s.baseline)),
    ctimeCap40Random: ctime(draw1, 40).map((s) => pack(s.random)),
    ctimeAll: ctime(draw1, Number.POSITIVE_INFINITY).map((s) => pack(s.spiked)),
    ctimeAllBaseline: ctime(draw1, Number.POSITIVE_INFINITY).map((s) => pack(s.baseline)),
    ctimeAllRandom: ctime(draw1, Number.POSITIVE_INFINITY).map((s) => pack(s.random)),
    draw2Cap40: ctime(draw2, 40).map((s) => pack(s.spiked)),
    draw2Cap40Baseline: ctime(draw2, 40).map((s) => pack(s.baseline)),
    draw2All: ctime(draw2, Number.POSITIVE_INFINITY).map((s) => pack(s.spiked)),
    draw2AllRandom: ctime(draw2, Number.POSITIVE_INFINITY).map((s) => pack(s.random)),
  };

  const out = {
    note: "namedRate is the fraction of spiked trials whose worst detector is the injected detector, whether or not the lookup says sensor. namedIfCalled is that fraction among trials the lookup calls sensor. The clean baseline is the same index with no spike. Random draws the detector uniformly from eight. Chance for a uniform guess is 1/8. Seeds 7, 19, 31, 43, 61. Evaluation seed 123. Simulator per-cell count 36, matching paper_eval.ts.",
    nDet: N_DET,
    seeds: SEEDS,
    summary: Object.fromEntries(
      Object.entries(blocks).map(([k, packs]) => [
        k,
        {
          nPerSeed: packs[0]?.n ?? 0,
          namedRate: across(packs, "namedRate"),
          namedIfCalled: across(packs, "namedIfCalled"),
          callRate: across(packs, "callRate"),
          perSeed: packs,
        },
      ]),
    ),
  };
  writeFileSync("experiments/gbm/channel_id.json", JSON.stringify(out, null, 2));
  const brief: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(out.summary)) {
    const s = v as { nPerSeed: number; namedRate: { mean: number; std: number }; namedIfCalled: { mean: number; std: number }; callRate: { mean: number; std: number } };
    brief[k] = {
      n: s.nPerSeed,
      named: `${(100 * s.namedRate.mean).toFixed(1)} ± ${(100 * s.namedRate.std).toFixed(1)}`,
      ifCalled: `${(100 * s.namedIfCalled.mean).toFixed(1)} ± ${(100 * s.namedIfCalled.std).toFixed(1)}`,
      called: `${(100 * s.callRate.mean).toFixed(1)} ± ${(100 * s.callRate.std).toFixed(1)}`,
    };
  }
  console.log(JSON.stringify(brief, null, 2));
}

main();
