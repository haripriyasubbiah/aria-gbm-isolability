/**
 * Held-out real-data check: train on one trigger draw, evaluate on a DISJOINT draw.
 *
 *   npm run test:gbm:heldout -- experiments/gbm/cubes.jsonl experiments/gbm/cubes_draw2.jsonl
 *
 * Positional args: <train cubes> <eval cubes> (defaults as above). Aborts if the
 * two files share any trigger name, so known-clean cells are never scored on
 * training data. Also reports the same-draw (in-sample) number for comparison.
 * Writes experiments/gbm/last_heldout.json.
 *
 * Optional: --seeds 7,19,31,43,61 --eval-seed 123 --out <path>
 */
import { writeFileSync } from "node:fs";
import { evaluateReal, loadCubes, trainOnCubes } from "./real_data";

const argv = process.argv.slice(2);
function flag(name: string): string | undefined {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
}
const pos = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
const trainPath = pos[0] ?? "experiments/gbm/cubes.jsonl";
const evalPath = pos[1] ?? "experiments/gbm/cubes_draw2.jsonl";
const seeds = (flag("seeds") ?? "7,19,31,43,61").split(",").map((s) => Number(s.trim()));
const evalSeed = Number(flag("eval-seed") ?? 123);
const outPath = flag("out") ?? "experiments/gbm/last_heldout.json";

const train = loadCubes(trainPath);
const test = loadCubes(evalPath);
if (train.length < 8 || test.length < 8) {
  console.error(`Need cubes in both files (train ${train.length}, eval ${test.length}).`);
  process.exit(1);
}
const overlap = test.filter((c) => train.some((t) => t.name === c.name)).map((c) => c.name);
if (overlap.length) {
  console.error(`FAIL: ${overlap.length} triggers appear in both files, e.g. ${overlap.slice(0, 3).join(", ")}`);
  process.exit(3);
}

const METRICS = ["ariaAcc", "rangeAcc", "novelIso", "novelAsCompute", "computeIso", "sensorIso", "knownAsSensor", "energyNovel", "energyCompute"] as const;
const meanStd = (xs: number[]) => {
  const mu = xs.reduce((a, b) => a + b, 0) / xs.length;
  return { mean: mu, std: Math.sqrt(xs.reduce((s, x) => s + (x - mu) ** 2, 0) / xs.length) };
};

const rows = seeds.map((seed) => {
  const system = trainOnCubes(train, seed);
  return {
    seed,
    heldout: evaluateReal(test, system, evalSeed) as Record<string, unknown>,
    inSample: evaluateReal(train, system, evalSeed) as Record<string, unknown>,
  };
});
const summarize = (key: "heldout" | "inSample") => {
  const mean: Record<string, number> = {};
  const std: Record<string, number> = {};
  for (const m of METRICS) {
    const s = meanStd(rows.map((r) => Number(r[key][m])));
    mean[m] = s.mean;
    std[m] = s.std;
  }
  return { mean, std };
};
const out = {
  trainPath, evalPath, seeds, evalSeed,
  nTrain: train.length, nEval: test.length,
  nTrialsEval: rows[0].heldout.nTrials,
  heldout: summarize("heldout"),
  inSample: summarize("inSample"),
  perSeed: rows,
};
writeFileSync(outPath, JSON.stringify(out, null, 2));
console.log(JSON.stringify({ nTrain: out.nTrain, nEval: out.nEval, heldout: out.heldout, inSample: out.inSample }, null, 2));
