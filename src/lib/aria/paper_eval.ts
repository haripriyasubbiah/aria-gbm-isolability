/**
 * Paper tables for the IEEE Access manuscript.
 * Reproduces every numeric claim from a fixed seed list. No GPU.
 *
 *   npm run test:paper
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { attribute, decideAblation, type AblationMode } from "./aria";
import { CHECK_MODE, setCheckMode } from "./classifier";
import { accuracy, confusion, energySplit, runTrials, trainSystem, type TrialRow } from "./experiment";
import { generateEvent } from "./physics";
import { CATALOG_COUNTS, type CauseLabel } from "./types";

const SEEDS = [7, 19, 31, 43, 61] as const;
const EVAL_SEED = 123;
const PER_CELL = 36;
const ABLATIONS: AblationMode[] = ["full", "checksum", "physics", "morph", "no-checksum"];

function meanStd(xs: number[]): { mean: number; std: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
  const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(xs.length, 1);
  return { mean, std: Math.sqrt(v) };
}

function rate(rows: TrialRow[], truth: TrialRow["truth"], pred: CauseLabel): number {
  const xs = rows.filter((r) => r.truth === truth);
  return xs.filter((r) => r.aria === pred).length / Math.max(xs.length, 1);
}

function ablationAccuracy(rows: TrialRow[], mode: AblationMode): number {
  const xs = rows.filter((r) => r.truth !== "compound");
  const ok = xs.filter((r) => decideAblation(r.signature, mode) === r.truth).length;
  return ok / Math.max(xs.length, 1);
}

function compoundStudy(system: ReturnType<typeof trainSystem>, n = 80) {
  const counts: Record<CauseLabel, number> = {
    "known-clean": 0,
    "novel-clean": 0,
    sensor: 0,
    compute: 0,
  };
  for (let i = 0; i < n; i++) {
    const ev = generateEvent({
      family: "GRB",
      fault: "sensor-and-compute",
      severity: i % 2 === 0 ? "mild" : "critical",
      seed: 9000 + i * 17,
    });
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    counts[a.cause]++;
  }
  return { n, ...counts, namedCompute: counts.compute / n };
}

function main() {
  const perSeed: {
    seed: number;
    aria: number;
    range: number;
    novelAsAria: number;
    novelAsCompute: number;
    computeAsAria: number;
    sensorAsAria: number;
    novelFlaggedEnergy: number;
    computeFlaggedEnergy: number;
    ablations: Record<AblationMode, number>;
  }[] = [];

  let lastRows: TrialRow[] = [];
  let seed7Rows: TrialRow[] = [];
  let lastSystem = trainSystem(SEEDS[0]);

  for (const trainSeed of SEEDS) {
    const system = trainSystem(trainSeed);
    const rows = runTrials(system, EVAL_SEED, PER_CELL);
    lastRows = rows;
    if (trainSeed === 7) seed7Rows = rows;
    lastSystem = system;
    perSeed.push({
      seed: trainSeed,
      aria: accuracy(rows, "aria"),
      range: accuracy(rows, "range"),
      novelAsAria: rate(rows, "novel-clean", "novel-clean"),
      novelAsCompute: rate(rows, "novel-clean", "compute"),
      computeAsAria: rate(rows, "compute", "compute"),
      sensorAsAria: rate(rows, "sensor", "sensor"),
      novelFlaggedEnergy: energySplit(rows).novelFlagged,
      computeFlaggedEnergy: energySplit(rows).computeFlagged,
      ablations: Object.fromEntries(ABLATIONS.map((m) => [m, ablationAccuracy(rows, m)])) as Record<
        AblationMode,
        number
      >,
    });
  }

  const aria = meanStd(perSeed.map((s) => s.aria));
  const range = meanStd(perSeed.map((s) => s.range));
  const novel = meanStd(perSeed.map((s) => s.novelAsAria));
  const compute = meanStd(perSeed.map((s) => s.computeAsAria));
  const sensor = meanStd(perSeed.map((s) => s.sensorAsAria));
  const ablationMeans = Object.fromEntries(
    ABLATIONS.map((m) => [m, meanStd(perSeed.map((s) => s.ablations[m]))]),
  ) as Record<AblationMode, { mean: number; std: number }>;
  const novelCompute = meanStd(perSeed.map((s) => s.novelAsCompute));
  const compound = compoundStudy(lastSystem);
  // Sensitivity of the checksum rule to inference precision and to the decision mode (same seeds, same events).
  const headlineMode = CHECK_MODE;
  const precision: Record<string, unknown> = {};
  const modes: Record<string, Record<string, unknown>> = {};
  for (const mode of ["calibrated", "bounded"] as const) {
    setCheckMode(mode);
    modes[mode] = {};
    for (const prec of ["fp64", "fp32", "fp16"] as const) {
      const per = SEEDS.map((sd) => {
        const r = runTrials(trainSystem(sd, prec), EVAL_SEED, PER_CELL);
        const mildRows = r.filter((x) => x.truth === "compute").slice(0, PER_CELL);
        return {
          aria: accuracy(r, "aria"),
          novel: rate(r, "novel-clean", "novel-clean"),
          novelAsCompute: rate(r, "novel-clean", "compute"),
          compute: rate(r, "compute", "compute"),
          computeMild: mildRows.filter((x) => x.aria === "compute").length / Math.max(mildRows.length, 1),
          knownAsCompute: rate(r, "known-clean", "compute"),
          sensor: rate(r, "sensor", "sensor"),
        };
      });
      const keys = ["aria", "novel", "novelAsCompute", "compute", "computeMild", "knownAsCompute", "sensor"] as const;
      modes[mode]![prec] = Object.fromEntries(keys.map((k) => [k, meanStd(per.map((x) => x[k]))]));
    }
  }
  setCheckMode(headlineMode);
  Object.assign(precision, modes[headlineMode]);
  const split = energySplit(lastRows);

  const out = {
    catalog: CATALOG_COUNTS,
    protocol: {
      trainSeeds: [...SEEDS],
      evalSeed: EVAL_SEED,
      perCell: PER_CELL,
      note: "Lightcubes are physics-faithful simulations of 8 NaI-like detectors. Catalog counts are live HEASARC fermigtrig totals.",
    },
    fourWay: { aria, range },
    isolation: { novel, compute, sensor, novelAsCompute: novelCompute },
    precision,
    checkMode: headlineMode,
    modes,
    execPrecision: lastSystem.prec,
    energyConfound: {
      novelFlagged: meanStd(perSeed.map((s) => s.novelFlaggedEnergy)),
      computeFlagged: meanStd(perSeed.map((s) => s.computeFlaggedEnergy)),
      last: split,
    },
    ablations: ablationMeans,
    compound,
    seed7ConfusionAria: confusion(seed7Rows, "aria"),
    lastConfusionAria: confusion(lastRows, "aria"),
    lastConfusionRange: confusion(lastRows, "range"),
    lastThresholds: lastSystem.thresholds,
    perSeed,
  };

  const paperDir = join(dirname(fileURLToPath(import.meta.url)), "../../../paper");
  mkdirSync(paperDir, { recursive: true });
  // Shipped (calibrated) results keep their names; the bounded variant never overwrites them.
  const suffix = headlineMode === "bounded" ? "_bounded" : "";
  writeFileSync(join(paperDir, `results${suffix}.json`), JSON.stringify(out, null, 2));

  const pct = (x: { mean: number; std: number }) => `${(100 * x.mean).toFixed(1)}\\pm${(100 * x.std).toFixed(1)}`;
  writeFileSync(
    join(paperDir, `results${suffix}.tex`),
    [
      `% Auto-generated by src/lib/aria/paper_eval.ts. Do not edit by hand.`,
      `\\newcommand{\\AriaAcc}{${pct(aria)}}`,
      `\\newcommand{\\RangeAcc}{${pct(range)}}`,
      `\\newcommand{\\NovelIso}{${pct(novel)}}`,
      `\\newcommand{\\ComputeIso}{${pct(compute)}}`,
      `\\newcommand{\\NovelAsCompute}{${pct(novelCompute)}}`,
      `\\newcommand{\\ExecPrec}{${lastSystem.prec}}`,
      `\\newcommand{\\SensorIso}{${pct(sensor)}}`,
      `\\newcommand{\\AblFull}{${pct(ablationMeans.full)}}`,
      `\\newcommand{\\AblChecksum}{${pct(ablationMeans.checksum)}}`,
      `\\newcommand{\\AblPhysics}{${pct(ablationMeans.physics)}}`,
      `\\newcommand{\\AblMorph}{${pct(ablationMeans.morph)}}`,
      `\\newcommand{\\AblNoChecksum}{${pct(ablationMeans["no-checksum"])}}`,
      `\\newcommand{\\CompoundCompute}{${(100 * compound.namedCompute).toFixed(1)}}`,
      `\\newcommand{\\CatalogTotal}{${CATALOG_COUNTS.total}}`,
      `\\newcommand{\\CatalogGRB}{${CATALOG_COUNTS.GRB}}`,
      `\\newcommand{\\CatalogTGF}{${CATALOG_COUNTS.TGF}}`,
      `\\newcommand{\\CatalogSGR}{${CATALOG_COUNTS.SGR}}`,
      `\\newcommand{\\CatalogSflare}{${CATALOG_COUNTS.SFLARE}}`,
      "",
    ].join("\n"),
  );

  console.log(
    JSON.stringify(
      { fourWay: out.fourWay, isolation: out.isolation, ablations: ablationMeans, compound, split },
      null,
      2,
    ),
  );
  console.log(`\nWrote paper/results${suffix}.json and paper/results${suffix}.tex  (checksum mode: ${headlineMode})`);
}

main();
