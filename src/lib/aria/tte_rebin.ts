/**
 * Same 1 ms TTE cubes, summed into coarser bins, then the full lookup.
 * No pre-trigger hack: background is the usual first and last 8 bins.
 *
 *   npx tsx src/lib/aria/tte_rebin.ts
 */
import { readFileSync, writeFileSync } from "node:fs";
import { attribute } from "./aria";
import { cubeToEvent, trainOnCubes, type RealCube } from "./real_data";

function rebinGrid(grid: number[][], k: number): number[][] {
  return grid.map((row) => {
    const n = Math.floor(row.length / k);
    const out = Array(n).fill(0);
    for (let i = 0; i < n * k; i++) out[Math.floor(i / k)]! += row[i]!;
    return out;
  });
}

function rebin(c: RealCube, k: number): RealCube {
  return {
    ...c,
    counts: rebinGrid(c.counts, k),
    soft: rebinGrid(c.soft, k),
    hard: rebinGrid(c.hard, k),
  };
}

const raw: RealCube[] = readFileSync("experiments/gbm/tte_cubes.jsonl", "utf8")
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as RealCube);

const widths = [8, 16, 50];
const seeds = [7, 19, 31, 43, 61];
const out: Record<string, unknown>[] = [];

for (const ms of widths) {
  const cubes = raw.map((c) => rebin(c, ms));
  const perSeed = seeds.map((seed) => {
    const system = trainOnCubes(cubes, seed);
    const calls: Record<string, Record<string, number>> = {};
    const phys: Record<string, number[]> = {};
    for (const c of cubes) {
      const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: 1 });
      const a = attribute(ev, system.head, system.thresholds, system.prec);
      calls[c.family] ??= {};
      calls[c.family][a.cause] = (calls[c.family][a.cause] ?? 0) + 1;
      (phys[c.family] ??= []).push(a.residuals.physics);
    }
    const physMed: Record<string, number> = {};
    for (const [fam, xs] of Object.entries(phys)) {
      const s = xs.slice().sort((a, b) => a - b);
      physMed[fam] = s[Math.floor(s.length / 2)]!;
    }
    return { seed, physicsThreshold: system.thresholds.physics, morphThreshold: system.thresholds.morph, physMed, calls };
  });
  out.push({ ms, bins: cubes[0]!.counts[0]!.length, perSeed });
  console.log(JSON.stringify({ ms, bins: cubes[0]!.counts[0]!.length, seed7: perSeed[0] }));
}

writeFileSync("experiments/gbm/tte_rebin.json", JSON.stringify(out, null, 2));
