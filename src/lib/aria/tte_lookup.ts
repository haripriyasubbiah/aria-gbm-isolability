/**
 * Full ARIA lookup on 1 ms TTE cubes. Trains on GRB, SGR, and SFLARE.
 * TGF is held out. No injected faults: this asks whether a flash is called
 * novel or compute, and whether the known classes stay known.
 *
 *   npx tsx src/lib/aria/tte_lookup.ts
 */
import { readFileSync, writeFileSync } from "node:fs";
import { attribute } from "./aria";
import { cubeToEvent, trainOnCubes, type RealCube } from "./real_data";

function preSubtract(c: RealCube): RealCube {
  const n = c.counts[0]?.length ?? 0;
  const edge = Math.min(400, Math.floor(n / 5));
  const fix = (grid: number[][]) =>
    grid.map((row) => {
      const pre = row.slice(0, edge).slice().sort((a, b) => a - b);
      const bg = pre[Math.floor(pre.length / 2)] ?? 0;
      const out = row.map((v) => v - bg);
      for (let i = 0; i < 8 && i < n; i++) out[n - 1 - i] = 0;
      return out;
    });
  return { ...c, counts: fix(c.counts), soft: fix(c.soft), hard: fix(c.hard) };
}

const raw: RealCube[] = readFileSync("experiments/gbm/tte_cubes.jsonl", "utf8")
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line) as RealCube);
const cubes = raw.map(preSubtract);
const seeds = [7, 19, 31, 43, 61];

  const perSeed = seeds.map((seed) => {
  const system = trainOnCubes(cubes, seed);
  const calls: Record<string, Record<string, number>> = {};
  const phys: Record<string, number[]> = {};
  const morph: Record<string, number[]> = {};
  const exec: Record<string, number[]> = {};
  for (const c of cubes) {
    const ev = cubeToEvent(c, { family: c.family, fault: "none", seed: 1 });
    const a = attribute(ev, system.head, system.thresholds, system.prec);
    calls[c.family] ??= {};
    calls[c.family][a.cause] = (calls[c.family][a.cause] ?? 0) + 1;
    (phys[c.family] ??= []).push(a.residuals.physics);
    (morph[c.family] ??= []).push(a.residuals.morph);
    (exec[c.family] ??= []).push(a.residuals.exec);
  }
  const median = (xs: number[]) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;
  const above = (xs: number[], t: number) => xs.filter((v) => v > t).length;
  const physMed: Record<string, number> = {};
  const morphMed: Record<string, number> = {};
  const execMed: Record<string, number> = {};
  const physAbove: Record<string, number> = {};
  const morphAbove: Record<string, number> = {};
  for (const fam of Object.keys(phys)) {
    physMed[fam] = median(phys[fam]!);
    morphMed[fam] = median(morph[fam]!);
    execMed[fam] = median(exec[fam]!);
    physAbove[fam] = above(phys[fam]!, system.thresholds.physics);
    morphAbove[fam] = above(morph[fam]!, system.thresholds.morph);
  }
  return {
    seed,
    physicsThreshold: system.thresholds.physics,
    morphThreshold: system.thresholds.morph,
    execThreshold: system.thresholds.exec,
    physMed,
    morphMed,
    execMed,
    physAbove,
    morphAbove,
    calls,
  };
});

writeFileSync("experiments/gbm/tte_lookup.json", JSON.stringify(perSeed, null, 2));
console.log(JSON.stringify(perSeed, null, 2));
