import { attribute } from "./aria";
import { energyReject, rangeCheck } from "./baselines";
import { CHECK_MODE, trainHead, UNIT_ROUNDOFF } from "./classifier";
import { generateEvent } from "./physics";
import { morphologyFeatures } from "./residuals";
import {
  KNOWN_CLASSES,
  type CauseLabel,
  type EventSpec,
  type GroundTruth,
  type KnownClass,
  type LinearHead,
  type Precision,
  type Signature,
  type Thresholds,
  type TransientFamily,
} from "./types";
import { Rng } from "./rng";

/** Arithmetic precision of the emulated inference. Fixed to fp32 before any run, to match the fp32 bit-flip injection; fp64/fp16 are sensitivity checks. */
export const DEFAULT_PREC: Precision = (process.env.ARIA_PREC as Precision | undefined) ?? "fp32";

export interface TrainedSystem {
  head: LinearHead;
  thresholds: Thresholds;
  prec: Precision;
}

export interface TrialRow {
  truth: GroundTruth;
  family: TransientFamily;
  aria: CauseLabel;
  energy: ReturnType<typeof energyReject>["cause"];
  range: ReturnType<typeof rangeCheck>["cause"];
  signature: Signature;
  residuals: {
    physics: number;
    morph: number;
    exec: number;
    msp: number;
    energy: number;
    maxAbsLogit: number;
  };
}

function percentile(xs: number[], q: number): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))));
  return s[i]!;
}

export function makeTrainingSet(seed: number, perClass = 140): { x: number[]; y: KnownClass }[] {
  const rng = new Rng(seed);
  const out: { x: number[]; y: KnownClass }[] = [];
  for (const y of KNOWN_CLASSES) {
    for (let i = 0; i < perClass; i++) {
      const ev = generateEvent({ family: y, fault: "none", seed: rng.int(1, 1e9) });
      out.push({ x: morphologyFeatures(ev.counts, ev.soft, ev.hard).vector, y });
    }
  }
  return out;
}

export function calibrate(head: LinearHead, seed: number, prec: Precision = DEFAULT_PREC): Thresholds {
  const rng = new Rng(seed);
  const phys: number[] = [];
  const morph: number[] = [];
  const energy: number[] = [];
  const msp: number[] = [];
  const absLogit: number[] = [];
  const execClean: number[] = [];

  for (let i = 0; i < 180; i++) {
    const family = rng.pick(KNOWN_CLASSES);
    const ev = generateEvent({ family, fault: "none", seed: rng.int(1, 1e9) });
    const a = attribute(ev, head, {
      physics: 1,
      morph: 1,
      exec: 1,
      dropGain: 1,
      energyHigh: 1e9,
      energyLow: -1e9,
      mspLow: 0,
      logitRange: 1e9,
    }, prec);
    phys.push(a.residuals.physics);
    morph.push(a.residuals.morph);
    energy.push(a.residuals.energy);
    msp.push(a.residuals.msp);
    absLogit.push(a.residuals.maxAbsLogit);
    execClean.push(a.residuals.exec);
  }

  return {
    physics: Math.min(0.62, Math.max(0.38, percentile(phys, 0.96) * 1.25)),
    morph: percentile(morph, 0.95) * 1.25,
    // Clean-only: no injected faults are used. Same rule as physics/morph (percentile x 1.25), floored at the unit roundoff of the arithmetic.
    // bounded: the residual is already divided by its rounding-error bound, so the threshold is 1 and nothing is calibrated.
    exec: CHECK_MODE === "bounded" ? 1 : Math.max(percentile(execClean, 0.995) * 1.25, UNIT_ROUNDOFF[prec]),
    dropGain: 0.28,
    energyHigh: percentile(energy, 0.97),
    energyLow: percentile(energy, 0.03),
    mspLow: Math.min(0.55, percentile(msp, 0.08)),
    logitRange: percentile(absLogit, 0.995) * 1.4,
  };
}

export function trainSystem(seed = 7, prec: Precision = DEFAULT_PREC): TrainedSystem {
  const head = trainHead(makeTrainingSet(seed));
  const thresholds = calibrate(head, seed + 99, prec);
  return { head, thresholds, prec };
}

let cached: TrainedSystem | undefined;
export function getSystem(): TrainedSystem {
  cached ??= trainSystem(7);
  return cached;
}

export function* specGrid(seed: number, perCell: number): Generator<EventSpec> {
  const rng = new Rng(seed);
  const cells: { family: TransientFamily; fault: EventSpec["fault"]; severity?: EventSpec["severity"]; truth: GroundTruth }[] = [
    { family: "GRB", fault: "none", truth: "known-clean" },
    { family: "SGR", fault: "none", truth: "known-clean" },
    { family: "SFLARE", fault: "none", truth: "known-clean" },
    { family: "TGF", fault: "none", truth: "novel-clean" },
    { family: "GRB", fault: "sensor-spike", truth: "sensor" },
    { family: "SGR", fault: "sensor-spike", truth: "sensor" },
    { family: "GRB", fault: "shared-bitflip", severity: "mild", truth: "compute" },
    { family: "GRB", fault: "shared-bitflip", severity: "critical", truth: "compute" },
  ];
  for (const cell of cells) {
    for (let i = 0; i < perCell; i++) {
      yield {
        family: cell.family,
        fault: cell.fault,
        severity: cell.severity,
        seed: rng.int(1, 1e9),
      };
    }
  }
}

export function runTrials(system: TrainedSystem, seed = 123, perCell = 40): TrialRow[] {
  const rows: TrialRow[] = [];
  for (const spec of specGrid(seed, perCell)) {
    const ev = generateEvent(spec);
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    rows.push({
      truth: ev.truth,
      family: spec.family,
      aria: a.cause,
      energy: energyReject(a.residuals, system.thresholds).cause,
      range: rangeCheck(a.residuals, system.thresholds).cause,
      signature: a.signature,
      residuals: {
        physics: a.residuals.physics,
        morph: a.residuals.morph,
        exec: a.residuals.exec,
        msp: a.residuals.msp,
        energy: a.residuals.energy,
        maxAbsLogit: a.residuals.maxAbsLogit,
      },
    });
  }
  return rows;
}

export const CAUSES: CauseLabel[] = ["known-clean", "novel-clean", "sensor", "compute"];

export function confusion(rows: TrialRow[], key: "aria" | "range"): Record<string, Record<string, number>> {
  const labels = [...CAUSES, "anomaly"];
  const m: Record<string, Record<string, number>> = {};
  for (const t of CAUSES) {
    m[t] = {};
    for (const p of labels) m[t]![p] = 0;
  }
  for (const row of rows) {
    if (row.truth === "compound") continue;
    const pred = key === "aria" ? row.aria : row.range;
    m[row.truth]![pred] = (m[row.truth]![pred] ?? 0) + 1;
  }
  return m;
}

export function energySplit(rows: TrialRow[]): { novelFlagged: number; computeFlagged: number; knownFlagged: number } {
  const by = (t: GroundTruth) => rows.filter((r) => r.truth === t);
  const rate = (t: GroundTruth) => {
    const xs = by(t);
    return xs.filter((r) => r.energy === "anomaly").length / Math.max(xs.length, 1);
  };
  return {
    novelFlagged: rate("novel-clean"),
    computeFlagged: rate("compute"),
    knownFlagged: rate("known-clean"),
  };
}

export function accuracy(rows: TrialRow[], key: "aria" | "range"): number {
  const ok = rows.filter((r) => (key === "aria" ? r.aria : r.range) === r.truth).length;
  return ok / Math.max(rows.length, 1);
}
