/**
 * Kill experiment: ARIA on public Fermi-GBM CTIME cubes.
 *
 *   python3 experiments/gbm/prepare.py --per-class 12
 *   npm run test:gbm
 */
import { writeFileSync } from "node:fs";
import { evaluateReal, loadCubes, trainOnCubes } from "./real_data";

const path = process.argv[2] ?? "experiments/gbm/cubes.jsonl";
const cubes = loadCubes(path);
if (cubes.length < 8) {
  console.error(`Need cubes at ${path}. Run: python3 experiments/gbm/prepare.py --per-class 12`);
  process.exit(1);
}
const system = trainOnCubes(cubes, 7);
const result = evaluateReal(cubes, system, 123);
console.log(JSON.stringify(result, null, 2));
writeFileSync("experiments/gbm/last_eval.json", JSON.stringify(result, null, 2));
if (result.novelAsCompute > 0.15) {
  console.error("FAIL: novelty is leaking into compute on real CTIME.");
  process.exit(2);
}
console.log("\nOK — held-out TGF is not being named compute on this real-CTIME slice.");
if (result.novelIso < 0.5) {
  console.log("NOTE — morphology did not isolate TGF as novel on CTIME (64 ms-class bins). Next: TTE at ~1 ms if you need the discovery box.");
}
