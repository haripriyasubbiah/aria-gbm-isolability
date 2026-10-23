/**
 * Analyses that the October draft listed and did not run:
 *   - stratified cross-validation on the committed CTIME slice
 *   - a disjoint half-split of that slice (both directions)
 *   - where held-out TGFs actually land
 *   - morphology distances, and a clean-only threshold sweep
 *   - mild versus critical bit-flips on real cubes
 *   - injection on every known cube, not only the first 40
 *   - compound trials whose two signatures both fire
 *   - a short-window spike score, thresholded on clean cubes only
 *
 *   npx tsx src/lib/aria/extra_eval.ts
 *
 * Writes paper/extra_results.json and paper/extra_macros.tex.
 * Does not overwrite results.json or last_eval.json.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { attribute, decide, decideMultiple } from "./aria";
import { trainSystem } from "./experiment";
import { generateEvent, subtractBackground } from "./physics";
import { evaluateReal, loadCubes, trainOnCubes, cubeToEvent, type RealCube } from "./real_data";
import { morphologyFeatures } from "./residuals";
import { Rng } from "./rng";
import { KNOWN_CLASSES, type CauseLabel, type EventCube, type EventSpec, type TransientFamily } from "./types";

const SEEDS = [7, 19, 31, 43, 61] as const;
const EVAL_SEED = 123;
const FAMILIES: TransientFamily[] = ["GRB", "SGR", "SFLARE", "TGF"];
const CAUSES: CauseLabel[] = ["known-clean", "novel-clean", "sensor", "compute"];

function meanStd(xs: number[]): { mean: number; std: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
  const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(xs.length, 1);
  return { mean, std: Math.sqrt(v) };
}

function mulberry(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(xs: T[], rnd: () => number): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j]!, a[i]!];
  }
  return a;
}

function percentile(xs: number[], q: number): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))]!;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : 0.5 * (s[mid - 1]! + s[mid]!);
}

/** Two-bin excess of the worst detector against the median of the others. */
function shortSpike(counts: number[][]): { score: number; dropGain: number } {
  const X = subtractBackground(counts);
  const n = X.length;
  const m = X[0]?.length ?? 0;
  const scores = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let best = 0;
    for (let t = 0; t < m - 1; t++) {
      const others: number[] = [];
      for (let d = 0; d < n; d++) {
        if (d === i) continue;
        others.push(0.5 * (X[d]![t]! + X[d]![t + 1]!));
      }
      const med = median(others);
      const excess = 0.5 * (X[i]![t]! + X[i]![t + 1]!) - med;
      best = Math.max(best, excess / (Math.sqrt(Math.max(med, 0)) + 1));
    }
    scores[i] = best;
  }
  let det = 0;
  for (let i = 1; i < n; i++) if (scores[i]! > scores[det]!) det = i;
  const second = Math.max(...scores.filter((_, i) => i !== det), 0);
  return { score: scores[det]!, dropGain: (scores[det]! - second) / (scores[det]! + 1e-9) };
}

function rate(rows: { truth: string; pred: string }[], truth: string, pred: string): number {
  const xs = rows.filter((r) => r.truth === truth);
  return xs.filter((r) => r.pred === pred).length / Math.max(xs.length, 1);
}

function fourAcc(rows: { truth: string; pred: string }[]): number {
  const xs = rows.filter((r) => r.truth !== "compound");
  return xs.filter((r) => r.pred === r.truth).length / Math.max(xs.length, 1);
}

type Tagged = { truth: string; pred: string; spike: boolean; name?: string };

function scoreReal(cubes: RealCube[], seed: number, faultCap: number, severity: "mild" | "critical"): { rows: Tagged[]; tSpike: number } {
  const system = trainOnCubes(cubes, seed);
  const known = cubes.filter((c) => c.family !== "TGF");
  const novel = cubes.filter((c) => c.family === "TGF");
  // Spike threshold uses its own stream. The evaluation stream matches evaluateReal
  // (Rng(EVAL_SEED), known, novel, then the first faultCap injections) so the
  // headline cells are the published protocol, not a second draw.
  const cal = new Rng(seed + 10007);
  const cleanSpikes: number[] = [];
  for (const c of known) {
    const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: cal.int(1, 1e9) });
    cleanSpikes.push(shortSpike(ev.counts).score);
  }
  const tSpike = percentile(cleanSpikes, 0.96) * 1.25;
  const rng = new Rng(EVAL_SEED);
  const rows: Tagged[] = [];
  const push = (cube: RealCube, fault: EventSpec["fault"], sev?: EventSpec["severity"]) => {
    const ev = cubeToEvent(cube, { family: cube.family, fault, severity: sev, seed: rng.int(1, 1e9) });
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    const sp = shortSpike(ev.counts);
    rows.push({
      truth: ev.truth,
      pred: a.cause,
      spike: sp.score > tSpike && sp.dropGain > 0.28,
      name: cube.name,
    });
  };
  for (const c of known) push(c, "none");
  for (const c of novel) push(c, "none");
  const injected = known.slice(0, Math.min(known.length, faultCap));
  for (const c of injected) push(c, "sensor-spike");
  for (const c of injected) push(c, "shared-bitflip", severity);
  return { rows, tSpike };
}

function orSpike(rows: Tagged[]): Tagged[] {
  return rows.map((r) => {
    if (r.pred === "compute") return r;
    if (r.spike && r.pred !== "sensor") return { ...r, pred: "sensor" };
    return r;
  });
}

function pack(rows: Tagged[]) {
  return {
    ariaAcc: fourAcc(rows),
    novelIso: rate(rows, "novel-clean", "novel-clean"),
    novelAsCompute: rate(rows, "novel-clean", "compute"),
    novelAsKnown: rate(rows, "novel-clean", "known-clean"),
    novelAsSensor: rate(rows, "novel-clean", "sensor"),
    computeIso: rate(rows, "compute", "compute"),
    sensorIso: rate(rows, "sensor", "sensor"),
    knownAsSensor: rate(rows, "known-clean", "sensor"),
    knownAsNovel: rate(rows, "known-clean", "novel-clean"),
    n: rows.length,
  };
}

function avg(packs: ReturnType<typeof pack>[]) {
  const keys = Object.keys(packs[0]!).filter((k) => k !== "n") as (keyof ReturnType<typeof pack>)[];
  const out: Record<string, { mean: number; std: number }> = {};
  for (const k of keys) out[k] = meanStd(packs.map((p) => Number(p[k])));
  return out;
}

function confusionOf(rows: Tagged[]) {
  const m: Record<string, Record<string, number>> = {};
  for (const t of CAUSES) {
    m[t] = {};
    for (const p of CAUSES) m[t]![p] = 0;
  }
  for (const r of rows) {
    if (!m[r.truth] || m[r.truth]![r.pred] === undefined) continue;
    m[r.truth]![r.pred]! += 1;
  }
  return m;
}

function stratifiedHalves(cubes: RealCube[], salt: number): [RealCube[], RealCube[]] {
  const rnd = mulberry(salt);
  const a: RealCube[] = [];
  const b: RealCube[] = [];
  for (const fam of FAMILIES) {
    const xs = shuffle(cubes.filter((c) => c.family === fam), rnd);
    const mid = Math.floor(xs.length / 2);
    a.push(...xs.slice(0, mid));
    b.push(...xs.slice(mid));
  }
  return [a, b];
}

function evalSplit(train: RealCube[], test: RealCube[]) {
  const rowsBySeed = SEEDS.map((seed) => {
    const system = trainOnCubes(train, seed);
    const rng = new Rng(EVAL_SEED);
    const known = test.filter((c) => c.family !== "TGF");
    const novel = test.filter((c) => c.family === "TGF");
    const rows: Tagged[] = [];
    const push = (cube: RealCube, fault: EventSpec["fault"], sev?: EventSpec["severity"]) => {
      const ev = cubeToEvent(cube, { family: cube.family, fault, severity: sev, seed: rng.int(1, 1e9) });
      const a = attribute(ev, system.head, system.thresholds, system.prec);
      rows.push({ truth: ev.truth, pred: a.cause, spike: false, name: cube.name });
    };
    for (const c of known) push(c, "none");
    for (const c of novel) push(c, "none");
    const injected = known.slice(0, Math.min(known.length, 40));
    for (const c of injected) push(c, "sensor-spike");
    for (const c of injected) push(c, "shared-bitflip", "critical");
    return pack(rows);
  });
  return {
    nTrain: train.length,
    nEval: test.length,
    byFamilyEval: Object.fromEntries(FAMILIES.map((f) => [f, test.filter((c) => c.family === f).length])),
    metrics: avg(rowsBySeed),
  };
}

function cv(cubes: RealCube[], folds: number, repeats: number, faultCap: number) {
  const runs: ReturnType<typeof pack>[] = [];
  for (let r = 0; r < repeats; r++) {
    const rnd = mulberry(1000 + r);
    const parts: RealCube[][] = Array.from({ length: folds }, () => []);
    for (const fam of FAMILIES) {
      shuffle(cubes.filter((c) => c.family === fam), rnd).forEach((c, i) => parts[i % folds]!.push(c));
    }
    for (let k = 0; k < folds; k++) {
      const test = parts[k]!;
      const train = parts.flatMap((f, i) => (i === k ? [] : f));
      const system = trainOnCubes(train, 7);
      const rng = new Rng(EVAL_SEED);
      const known = test.filter((c) => c.family !== "TGF");
      const novel = test.filter((c) => c.family === "TGF");
      const rows: Tagged[] = [];
      const push = (cube: RealCube, fault: EventSpec["fault"], sev?: EventSpec["severity"]) => {
        const ev = cubeToEvent(cube, { family: cube.family, fault, severity: sev, seed: rng.int(1, 1e9) });
        const a = attribute(ev, system.head, system.thresholds, system.prec);
        rows.push({ truth: ev.truth, pred: a.cause, spike: false });
      };
      for (const c of known) push(c, "none");
      for (const c of novel) push(c, "none");
      const injected = known.slice(0, Math.min(known.length, faultCap));
      for (const c of injected) push(c, "sensor-spike");
      for (const c of injected) push(c, "shared-bitflip", "critical");
      runs.push(pack(rows));
    }
  }
  return {
    folds,
    repeats,
    faultCap: Number.isFinite(faultCap) ? faultCap : "all",
    nRuns: runs.length,
    metrics: avg(runs),
    note: "Spread is over overlapping folds and repeats. It describes this slice. It is not a standard error for a new catalog draw.",
  };
}

function morphology(cubes: RealCube[]) {
  const system = trainOnCubes(cubes, 7);
  const byFam: Record<string, number[]> = {};
  const pred: Record<string, Record<string, number>> = {};
  for (const fam of FAMILIES) pred[fam] = Object.fromEntries(CAUSES.map((c) => [c, 0]));
  const points: { family: string; morph: number; duration: number; fwhm: number; hardness: number }[] = [];
  for (const c of cubes) {
    const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: 1 });
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    const feat = morphologyFeatures(ev.counts, ev.soft, ev.hard).features;
    (byFam[c.family] ??= []).push(a.residuals.morph);
    pred[c.family]![a.cause]! += 1;
    points.push({ family: c.family, morph: a.residuals.morph, duration: feat.duration, fwhm: feat.fwhm, hardness: feat.hardness });
  }
  const summary = Object.fromEntries(
    FAMILIES.map((f) => {
      const xs = byFam[f] ?? [];
      return [f, { ...meanStd(xs), min: Math.min(...xs), max: Math.max(...xs), n: xs.length, aboveThreshold: xs.filter((x) => x > system.thresholds.morph).length }];
    }),
  );
  return { morphThreshold: system.thresholds.morph, summary, calls: pred, points };
}

function morphSweep(cubes: RealCube[]) {
  const base = trainOnCubes(cubes, 7);
  const known = cubes.filter((c) => (KNOWN_CLASSES as readonly string[]).includes(c.family));
  const rng = new Rng(11);
  const cleanMorph: number[] = [];
  for (const c of known) {
    const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: rng.int(1, 1e9) });
    cleanMorph.push(attribute(ev, base.head, base.thresholds, base.prec).residuals.morph);
  }
  const p95 = percentile(cleanMorph, 0.95);
  const margins = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4];
  return margins.map((margin) => {
    const thresholds = { ...base.thresholds, morph: p95 * margin };
    const { rows } = (() => {
      const rng2 = new Rng(EVAL_SEED);
      const novel = cubes.filter((c) => c.family === "TGF");
      const tagged: Tagged[] = [];
      const push = (cube: RealCube, fault: EventSpec["fault"], sev?: EventSpec["severity"]) => {
        const ev = cubeToEvent(cube, { family: cube.family, fault, severity: sev, seed: rng2.int(1, 1e9) });
        const a = attribute(ev, base.head, thresholds, base.prec);
        tagged.push({ truth: ev.truth, pred: a.cause, spike: false });
      };
      for (const c of known) push(c, "none");
      for (const c of novel) push(c, "none");
      for (const c of known.slice(0, 40)) push(c, "sensor-spike");
      for (const c of known.slice(0, 40)) push(c, "shared-bitflip", "critical");
      return { rows: tagged };
    })();
    return { margin, threshold: p95 * margin, ...pack(rows) };
  });
}

function compoundSignatures() {
  const per = SEEDS.map((seed) => {
    const system = trainSystem(seed);
    let both = 0;
    let arithOnly = 0;
    let physOnly = 0;
    let neither = 0;
    const labels: Record<string, number> = { compute: 0, sensor: 0, "novel-clean": 0, "known-clean": 0, multiple: 0 };
    const n = 80;
    for (let i = 0; i < n; i++) {
      const ev = generateEvent({
        family: "GRB",
        fault: "sensor-and-compute",
        severity: i % 2 === 0 ? "mild" : "critical",
        seed: 9000 + seed * 100 + i * 17,
      });
      const a = attribute(ev, system.head, system.thresholds, system.prec);
      const sensorish = a.signature.U_det || a.signature.S_phys;
      if (a.signature.U_arith && sensorish) both += 1;
      else if (a.signature.U_arith) arithOnly += 1;
      else if (sensorish) physOnly += 1;
      else neither += 1;
      labels[decide(a.signature)]! += 1;
      labels[decideMultiple(a.signature)] = (labels[decideMultiple(a.signature)] ?? 0);
      // count multiple separately without double-counting decide()
    }
    // recount multiple cleanly
    let multiple = 0;
    for (let i = 0; i < n; i++) {
      const ev = generateEvent({
        family: "GRB",
        fault: "sensor-and-compute",
        severity: i % 2 === 0 ? "mild" : "critical",
        seed: 9000 + seed * 100 + i * 17,
      });
      const a = attribute(ev, system.head, system.thresholds, system.prec);
      if (decideMultiple(a.signature) === "multiple") multiple += 1;
    }
    return { both: both / n, arithOnly: arithOnly / n, physOnly: physOnly / n, neither: neither / n, multiple: multiple / n, namedCompute: labels.compute! / n };
  });
  return {
    nPerSeed: 80,
    both: meanStd(per.map((p) => p.both)),
    arithOnly: meanStd(per.map((p) => p.arithOnly)),
    physOnly: meanStd(per.map((p) => p.physOnly)),
    neither: meanStd(per.map((p) => p.neither)),
    multiple: meanStd(per.map((p) => p.multiple)),
    namedCompute: meanStd(per.map((p) => p.namedCompute)),
  };
}

function simulatorSpike() {
  const perHead: ReturnType<typeof pack>[] = [];
  const perOr: ReturnType<typeof pack>[] = [];
  for (const seed of SEEDS) {
    const system = trainSystem(seed);
    const rng = new Rng(50 + seed);
    const clean: number[] = [];
    for (let i = 0; i < 120; i++) {
      const ev = generateEvent({ family: (["GRB", "SGR", "SFLARE"] as const)[i % 3]!, fault: "none", seed: rng.int(1, 1e9) });
      clean.push(shortSpike(ev.counts).score);
    }
    const tSpike = percentile(clean, 0.96) * 1.25;
    const rows: Tagged[] = [];
    const cells: EventSpec[] = [];
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
    for (const spec of specs) {
      for (let i = 0; i < 36; i++) cells.push({ ...spec, seed: grid.int(1, 1e9) });
    }
    for (const spec of cells) {
      const ev: EventCube = generateEvent(spec);
      const a = attribute(ev, system.head, system.thresholds, system.prec);
      const sp = shortSpike(ev.counts);
      rows.push({ truth: ev.truth, pred: a.cause, spike: sp.score > tSpike && sp.dropGain > 0.28 });
    }
    perHead.push(pack(rows));
    perOr.push(pack(orSpike(rows)));
  }
  return { headline: avg(perHead), orSpike: avg(perOr) };
}

function main() {
  const cubes = loadCubes("experiments/gbm/cubes.jsonl");
  const known = cubes.filter((c) => c.family !== "TGF");
  const first40 = known.slice(0, 40);
  const familyOfCap = Object.fromEntries(FAMILIES.map((f) => [f, first40.filter((c) => c.family === f).length]));

  console.log("real critical / mild / all-cube ...");
  const critical = SEEDS.map((s) => pack(scoreReal(cubes, s, 40, "critical").rows));
  const mild = SEEDS.map((s) => pack(scoreReal(cubes, s, 40, "mild").rows));
  const allCube = SEEDS.map((s) => pack(scoreReal(cubes, s, Number.POSITIVE_INFINITY, "critical").rows));
  const withSpike = SEEDS.map((s) => pack(orSpike(scoreReal(cubes, s, 40, "critical").rows)));

  console.log("confusion, morphology, sweep ...");
  const confSeed = scoreReal(cubes, 7, 40, "critical").rows;
  const morph = morphology(cubes);
  const sweep = morphSweep(cubes);

  console.log("split and cv ...");
  const [halfA, halfB] = stratifiedHalves(cubes, 20261007);
  const overlap = halfA.filter((c) => halfB.some((t) => t.name === c.name)).length;
  if (overlap) throw new Error(`split overlap ${overlap}`);
  const split = {
    salt: 20261007,
    note: "Two halves of the same committed 100-cube slice. Not a second HEASARC draw.",
    aToB: evalSplit(halfA, halfB),
    bToA: evalSplit(halfB, halfA),
  };
  const cross = {
    cap40: cv(cubes, 5, 5, 40),
    all: cv(cubes, 5, 5, Number.POSITIVE_INFINITY),
  };

  console.log("compound and simulator spike ...");
  const compound = compoundSignatures();
  const simSpike = simulatorSpike();
  const perFamily = (faultCap: number) => {
    const acc: Record<string, { n: number; hit: number }> = {};
    for (const fam of FAMILIES) acc[fam] = { n: 0, hit: 0 };
    for (const seed of SEEDS) {
      for (const row of scoreReal(cubes, seed, faultCap, "critical").rows) {
        if (row.truth !== "sensor" || !row.name) continue;
        const fam = cubes.find((c) => c.name === row.name)?.family;
        if (!fam) continue;
        acc[fam]!.n += 1;
        if (row.pred === "sensor") acc[fam]!.hit += 1;
      }
    }
    return Object.fromEntries(
      FAMILIES.filter((f) => acc[f]!.n > 0).map((f) => [
        f,
        { n: acc[f]!.n / SEEDS.length, rate: acc[f]!.hit / acc[f]!.n },
      ]),
    );
  };
  const sensorByFamily = { cap40: perFamily(40), allKnown: perFamily(Number.POSITIVE_INFINITY) };

  const out = {
    nCubes: cubes.length,
    injectionCap40ByFamily: familyOfCap,
    realCriticalCap40: avg(critical),
    realMildCap40: avg(mild),
    realCriticalAllKnown: avg(allCube),
    realOrSpikeCap40: avg(withSpike),
    realConfusionSeed7: confusionOf(confSeed),
    morphology: { morphThreshold: morph.morphThreshold, summary: morph.summary, calls: morph.calls },
    morphPoints: morph.points,
    morphSweep: sweep,
    split,
    cv: cross,
    compound,
    simulatorSpike: simSpike,
    sensorByFamily,
  };

  const paperDir = join(dirname(fileURLToPath(import.meta.url)), "../../../paper");
  mkdirSync(paperDir, { recursive: true });
  writeFileSync(join(paperDir, "extra_results.json"), JSON.stringify(out, null, 2));

  const pct = (x: { mean: number; std: number }) => `${(100 * x.mean).toFixed(1)}\\pm${(100 * x.std).toFixed(1)}`;
  const one = (x: { mean: number; std: number }) => (100 * x.mean).toFixed(1);
  const lines = [
    "% Auto-generated by src/lib/aria/extra_eval.ts. Do not edit by hand.",
    `\\newcommand{\\CvAcc}{${pct(cross.cap40.metrics.ariaAcc!)}}`,
    `\\newcommand{\\CvNovel}{${pct(cross.cap40.metrics.novelIso!)}}`,
    `\\newcommand{\\CvNovelComp}{${pct(cross.cap40.metrics.novelAsCompute!)}}`,
    `\\newcommand{\\CvComp}{${pct(cross.cap40.metrics.computeIso!)}}`,
    `\\newcommand{\\CvSens}{${pct(cross.cap40.metrics.sensorIso!)}}`,
    `\\newcommand{\\CvAllAcc}{${pct(cross.all.metrics.ariaAcc!)}}`,
    `\\newcommand{\\CvAllSens}{${pct(cross.all.metrics.sensorIso!)}}`,
    `\\newcommand{\\CvAllComp}{${pct(cross.all.metrics.computeIso!)}}`,
    `\\newcommand{\\SplitABAcc}{${pct(split.aToB.metrics.ariaAcc!)}}`,
    `\\newcommand{\\SplitABSens}{${pct(split.aToB.metrics.sensorIso!)}}`,
    `\\newcommand{\\SplitABComp}{${pct(split.aToB.metrics.computeIso!)}}`,
    `\\newcommand{\\SplitABNovC}{${pct(split.aToB.metrics.novelAsCompute!)}}`,
    `\\newcommand{\\SplitBAAcc}{${pct(split.bToA.metrics.ariaAcc!)}}`,
    `\\newcommand{\\SplitBASens}{${pct(split.bToA.metrics.sensorIso!)}}`,
    `\\newcommand{\\SplitBAComp}{${pct(split.bToA.metrics.computeIso!)}}`,
    `\\newcommand{\\SplitBANovC}{${pct(split.bToA.metrics.novelAsCompute!)}}`,
    `\\newcommand{\\MildComp}{${pct(mild[0] ? avg(mild).computeIso! : { mean: 0, std: 0 })}}`,
    `\\newcommand{\\AllSens}{${pct(avg(allCube).sensorIso!)}}`,
    `\\newcommand{\\AllAcc}{${pct(avg(allCube).ariaAcc!)}}`,
    `\\newcommand{\\AllComp}{${pct(avg(allCube).computeIso!)}}`,
    `\\newcommand{\\BothSig}{${pct(compound.both)}}`,
    `\\newcommand{\\ArithOnly}{${pct(compound.arithOnly)}}`,
    `\\newcommand{\\MultiRate}{${pct(compound.multiple)}}`,
    `\\newcommand{\\SimSpikeSens}{${pct(simSpike.orSpike.sensorIso!)}}`,
    `\\newcommand{\\SimHeadSens}{${pct(simSpike.headline.sensorIso!)}}`,
    `\\newcommand{\\SimSpikeKnown}{${pct(simSpike.orSpike.knownAsSensor!)}}`,
    `\\newcommand{\\SimHeadKnown}{${pct(simSpike.headline.knownAsSensor!)}}`,
    `\\newcommand{\\SimSpikeAcc}{${pct(simSpike.orSpike.ariaAcc!)}}`,
    `\\newcommand{\\RealSpikeSens}{${pct(avg(withSpike).sensorIso!)}}`,
    `\\newcommand{\\RealSpikeKnown}{${pct(avg(withSpike).knownAsSensor!)}}`,
    `\\newcommand{\\RealSpikeAcc}{${pct(avg(withSpike).ariaAcc!)}}`,
    `\\newcommand{\\RealSpikeNovS}{${pct(avg(withSpike).novelAsSensor!)}}`,
    `\\newcommand{\\TgfAsKnown}{${one(avg(critical).novelAsKnown!)}}`,
    "",
  ];
  writeFileSync(join(paperDir, "extra_macros.tex"), lines.join("\n"));
  console.log(lines.filter((l) => l.startsWith("\\newcommand")).join("\n"));
  console.log("wrote paper/extra_results.json");
}

main();
