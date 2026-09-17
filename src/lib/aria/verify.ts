import { accuracy, energySplit, runTrials, trainSystem } from "./experiment";
import { CAUSES } from "./experiment";
import { confusion } from "./experiment";

function fmt(n: number): string {
  return n.toFixed(3);
}

function printMatrix(title: string, m: Record<string, Record<string, number>>) {
  const preds = [...CAUSES, "anomaly"];
  console.log(`\n${title}`);
  console.log(["true\\pred", ...preds].map((s) => s.padEnd(14)).join(""));
  for (const t of CAUSES) {
    const row = [t, ...preds.map((p) => String(m[t]?.[p] ?? 0))];
    console.log(row.map((s) => s.padEnd(14)).join(""));
  }
}

const system = trainSystem(7);
const rows = runTrials(system, 123, 36);
const ariaAcc = accuracy(rows, "aria");
const rangeAcc = accuracy(rows, "range");
const split = energySplit(rows);

printMatrix("ARIA", confusion(rows, "aria"));
printMatrix("Range-check baseline", confusion(rows, "range"));

console.log("\nEnergy/MSP reject rate (Gavarini-style, cannot name the cause)");
console.log(split);
console.log("\nAccuracy  ARIA", fmt(ariaAcc), "  range-check", fmt(rangeAcc));
console.log("thresholds", system.thresholds);

const novelAsCompute =
  rows.filter((r) => r.truth === "novel-clean" && r.range === "compute").length /
  Math.max(rows.filter((r) => r.truth === "novel-clean").length, 1);
const computeAsNovel =
  rows.filter((r) => r.truth === "compute" && (r.range === "novel-clean" || r.energy === "anomaly")).length /
  Math.max(rows.filter((r) => r.truth === "compute").length, 1);
const ariaNovel =
  rows.filter((r) => r.truth === "novel-clean" && r.aria === "novel-clean").length /
  Math.max(rows.filter((r) => r.truth === "novel-clean").length, 1);
const ariaCompute =
  rows.filter((r) => r.truth === "compute" && r.aria === "compute").length /
  Math.max(rows.filter((r) => r.truth === "compute").length, 1);

console.log({ novelAsCompute, computeAsNovel, ariaNovel, ariaCompute });

if (ariaAcc < 0.82) {
  console.error("ARIA accuracy too low");
  process.exit(1);
}
if (split.novelFlagged < 0.5 || split.computeFlagged < 0.5) {
  console.warn("Energy confound is weaker than expected; still useful if both rates are non-trivial.");
}
if (ariaNovel < 0.7 || ariaCompute < 0.75) {
  console.error("ARIA failed to isolate novelty or compute");
  process.exit(1);
}
console.log("\nOK — isolability holds on this seed.");
