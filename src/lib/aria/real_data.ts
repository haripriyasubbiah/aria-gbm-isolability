import { readFileSync } from "node:fs";
import { attribute } from "./aria";
import { energyReject, rangeCheck } from "./baselines";
import { trainHead } from "./classifier";
import type { TrainedSystem } from "./experiment";
import { morphologyFeatures } from "./residuals";
import { Rng } from "./rng";
import {
  KNOWN_CLASSES,
  N_DET,
  N_TIME,
  type EventCube,
  type EventSpec,
  type FaultKind,
  type KnownClass,
  type TransientFamily,
} from "./types";

export interface RealCube {
  name: string;
  family: TransientFamily;
  detectors: string[];
  counts: number[][];
  soft: number[][];
  hard: number[][];
}

export function loadCubes(path: string): RealCube[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as RealCube)
    .filter((c) => c.counts.length === N_DET && c.counts[0]?.length === N_TIME);
}

function cloneGrid(g: number[][]): number[][] {
  return g.map((row) => row.slice());
}

function injectSensor(counts: number[][], soft: number[][], rng: Rng): { counts: number[][]; soft: number[][]; det: number } {
  const totals = counts.map((row) => row.reduce((a, b) => a + b, 0));
  const brightest = totals.reduce((best, v, i) => (v > totals[best]! ? i : best), 0);
  const dim = totals
    .map((v, i) => [v, i] as const)
    .filter(([, i]) => i !== brightest)
    .sort((a, b) => a[0] - b[0]);
  const det = dim[0]?.[1] ?? 0;
  const peak = Math.max(...counts.flat(), 20);
  const t0 = Math.min(N_TIME - 3, Math.max(8, rng.int(10, 48)));
  const spike = peak * rng.uniform(1.6, 3.2);
  const c = cloneGrid(counts);
  const s = cloneGrid(soft);
  for (let t = t0; t < Math.min(N_TIME, t0 + 2); t++) {
    c[det]![t] += spike;
    s[det]![t] += spike;
  }
  return { counts: c, soft: s, det };
}

export function cubeToEvent(cube: RealCube, spec: EventSpec): EventCube {
  const rng = new Rng(spec.seed);
  let counts = cloneGrid(cube.counts);
  let soft = cloneGrid(cube.soft);
  const hard = cloneGrid(cube.hard);
  let sensorDetector: number | null = null;
  if (spec.fault === "sensor-spike" || spec.fault === "sensor-and-compute") {
    const inj = injectSensor(counts, soft, rng);
    counts = inj.counts;
    soft = inj.soft;
    sensorDetector = inj.det;
  }
  const truth =
    spec.fault === "sensor-and-compute"
      ? "compound"
      : spec.fault === "sensor-spike"
        ? "sensor"
        : spec.fault === "shared-bitflip"
          ? "compute"
          : cube.family === "TGF"
            ? "novel-clean"
            : "known-clean";
  return {
    spec: { ...spec, family: cube.family },
    truth,
    counts,
    soft,
    hard,
    source: { azDeg: 0, elDeg: 40 },
    sensorDetector,
    flippedBit: null,
    flippedWeightIndex: null,
  };
}

function percentile(xs: number[], q: number): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))));
  return s[i]!;
}

export function trainOnCubes(cubes: RealCube[], seed = 7): TrainedSystem {
  const known = cubes.filter((c) => (KNOWN_CLASSES as readonly string[]).includes(c.family));
  const rng = new Rng(seed);
  const trainSet = known.map((c) => {
    const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: rng.int(1, 1e9) });
    return { x: morphologyFeatures(ev.counts, ev.soft, ev.hard).vector, y: c.family as KnownClass };
  });
  const head = trainHead(trainSet);
  const phys: number[] = [];
  const morph: number[] = [];
  const energy: number[] = [];
  const msp: number[] = [];
  const absLogit: number[] = [];
  const execClean: number[] = [];
  const open = { physics: 1, morph: 1, exec: 1, dropGain: 1, energyHigh: 1e9, energyLow: -1e9, mspLow: 0, logitRange: 1e9 };
  for (const c of known) {
    const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: rng.int(1, 1e9) });
    const a = attribute(ev, head, open);
    phys.push(a.residuals.physics);
    morph.push(a.residuals.morph);
    energy.push(a.residuals.energy);
    msp.push(a.residuals.msp);
    absLogit.push(a.residuals.maxAbsLogit);
    execClean.push(a.residuals.exec);
  }
  const execFault: number[] = [];
  for (const c of known) {
    const ev = cubeToEvent(c, { family: c.family, fault: "shared-bitflip", severity: "critical", seed: rng.int(1, 1e9) });
    const a = attribute(ev, head, { ...open, exec: 0 });
    execFault.push(a.residuals.exec);
  }
  const thresholds = {
    physics: Math.min(0.62, Math.max(0.38, percentile(phys, 0.96) * 1.25)),
    morph: percentile(morph, 0.95) * 1.25,
    exec: Math.max(5e-4, (percentile(execClean, 0.995) + percentile(execFault, 0.08)) / 2),
    dropGain: 0.28,
    energyHigh: percentile(energy, 0.97),
    energyLow: percentile(energy, 0.03),
    mspLow: Math.min(0.55, percentile(msp, 0.08)),
    logitRange: percentile(absLogit, 0.995) * 1.4,
  };
  return { head, thresholds };
}

export function evaluateReal(cubes: RealCube[], system: TrainedSystem, seed = 123) {
  const rng = new Rng(seed);
  const known = cubes.filter((c) => c.family !== "TGF");
  const novel = cubes.filter((c) => c.family === "TGF");
  const rows: { truth: string; aria: string; energy: string; range: string; name: string }[] = [];

  const push = (cube: RealCube, fault: FaultKind, severity?: EventSpec["severity"]) => {
    const ev = cubeToEvent(cube, { family: cube.family, fault, severity, seed: rng.int(1, 1e9) });
    const a = attribute(ev, system.head, system.thresholds);
    rows.push({
      truth: ev.truth,
      aria: a.cause,
      energy: energyReject(a.residuals, system.thresholds).cause,
      range: rangeCheck(a.residuals, system.thresholds).cause,
      name: cube.name,
    });
  };

  for (const c of known) push(c, "none");
  for (const c of novel) push(c, "none");
  for (const c of known.slice(0, Math.min(known.length, 40))) push(c, "sensor-spike");
  for (const c of known.slice(0, Math.min(known.length, 40))) push(c, "shared-bitflip", "critical");

  const of = (t: string) => rows.filter((r) => r.truth === t);
  const acc = (t: string, pred: string) => {
    const xs = of(t);
    return xs.filter((r) => r.aria === pred).length / Math.max(xs.length, 1);
  };
  const four = rows.filter((r) => r.truth !== "compound");
  const ariaAcc = four.filter((r) => r.aria === r.truth).length / Math.max(four.length, 1);
  const rangeAcc = four.filter((r) => r.range === r.truth).length / Math.max(four.length, 1);
  return {
    nCubes: cubes.length,
    byFamily: Object.fromEntries(
      (["GRB", "SGR", "SFLARE", "TGF"] as TransientFamily[]).map((f) => [f, cubes.filter((c) => c.family === f).length]),
    ),
    nTrials: four.length,
    ariaAcc,
    rangeAcc,
    novelIso: acc("novel-clean", "novel-clean"),
    novelAsCompute: acc("novel-clean", "compute"),
    computeIso: acc("compute", "compute"),
    sensorIso: acc("sensor", "sensor"),
    energyNovel: of("novel-clean").filter((r) => r.energy === "anomaly").length / Math.max(of("novel-clean").length, 1),
    energyCompute: of("compute").filter((r) => r.energy === "anomaly").length / Math.max(of("compute").length, 1),
  };
}
