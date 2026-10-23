/**
 * Checksum on a small convolutional head. This is not the convolutional-recurrent
 * network of John et al. It is four filters of width 5 over the eight detector
 * traces, then a linear readout to the three known classes.
 *
 * The conv checksum is the same linearity identity as the linear head:
 * summing the filter outputs equals convolving with the summed filter.
 * A flip in a kernel breaks it for every input. A flip in the readout does not,
 * and the readout's own row-sum checksum does not see a kernel flip.
 *
 *   npx tsx src/lib/aria/conv_checksum.ts
 *
 * Writes experiments/gbm/conv_checksum.json.
 */
import { writeFileSync } from "node:fs";
import { bitFlipF32, Rng } from "./rng";
import { generateEvent, subtractBackground } from "./physics";
import { cubeToEvent, loadCubes, type RealCube } from "./real_data";
import { KNOWN_CLASSES, N_DET, N_TIME, type EventCube, type KnownClass, type TransientFamily } from "./types";

const N_OUT = 4;
const KERNEL = 5;
const T_OUT = N_TIME - KERNEL + 1;
const N_CLASS = KNOWN_CLASSES.length;
const SEEDS = [7, 19, 31, 43, 61] as const;
const EVAL_SEED = 123;
const PER_CELL = 36;

const r32 = Math.fround;

interface ConvHead {
  kernel: number[][][];
  convBias: number[];
  kernelSum: number[][];
  biasSum: number;
  readoutW: number[][];
  readoutB: number[];
  readoutChecksum: number[];
  mean: number;
  std: number;
}

function cloneKernel(k: number[][][]): number[][][] {
  return k.map((out) => out.map((row) => row.slice()));
}

function standardize(counts: number[][], mean: number, std: number): number[][] {
  return counts.map((row) => row.map((v) => (v - mean) / std));
}

function convAt(kernel: number[][][], bias: number[], x: number[][], use32: boolean): number[][] {
  const y = Array.from({ length: N_OUT }, () => Array(T_OUT).fill(0));
  const add = (a: number, b: number) => (use32 ? r32(a + b) : a + b);
  const mul = (a: number, b: number) => (use32 ? r32(r32(a) * r32(b)) : a * b);
  for (let c = 0; c < N_OUT; c++) {
    for (let t = 0; t < T_OUT; t++) {
      let s = use32 ? r32(bias[c]!) : bias[c]!;
      for (let i = 0; i < N_DET; i++) {
        const row = kernel[c]![i]!;
        const trace = x[i]!;
        for (let k = 0; k < KERNEL; k++) s = add(s, mul(row[k]!, trace[t + k]!));
      }
      y[c]![t] = s;
    }
  }
  return y;
}

/** Convolution with the stored summed filter. One output trace. */
function convSummed(head: ConvHead, x: number[][], use32: boolean): number[] {
  const add = (a: number, b: number) => (use32 ? r32(a + b) : a + b);
  const mul = (a: number, b: number) => (use32 ? r32(r32(a) * r32(b)) : a * b);
  const out = Array(T_OUT).fill(0);
  for (let t = 0; t < T_OUT; t++) {
    let s = use32 ? r32(head.biasSum) : head.biasSum;
    for (let i = 0; i < N_DET; i++) {
      const row = head.kernelSum[i]!;
      const trace = x[i]!;
      for (let k = 0; k < KERNEL; k++) s = add(s, mul(row[k]!, trace[t + k]!));
    }
    out[t] = s;
  }
  return out;
}

function pool(y: number[][]): number[] {
  return y.map((row) => row.reduce((a, b) => a + b, 0) / T_OUT);
}

function logitsOf(head: ConvHead, pooled: number[]): number[] {
  return head.readoutW.map((row, m) => {
    let s = head.readoutB[m]!;
    for (let c = 0; c < N_OUT; c++) s += row[c]! * pooled[c]!;
    return s;
  });
}

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

function predict(head: ConvHead, counts: number[][]): KnownClass {
  const x = standardize(counts, head.mean, head.std);
  const y = convAt(head.kernel, head.convBias, x, false);
  const probs = softmax(logitsOf(head, pool(y)));
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
  return KNOWN_CLASSES[best]!;
}

/** Normalized |sum of live filter outputs − stored summed filter|. */
export function convResidual(head: ConvHead, counts: number[][], use32 = true): number {
  const x = standardize(counts, head.mean, head.std).map((row) => (use32 ? row.map(r32) : row));
  const y = convAt(head.kernel, head.convBias, x, use32);
  const stored = convSummed(head, x, use32);
  let num = 0;
  let den = 1;
  for (let t = 0; t < T_OUT; t++) {
    let live = 0;
    for (let c = 0; c < N_OUT; c++) live = use32 ? r32(live + y[c]![t]!) : live + y[c]![t]!;
    const diff = Math.abs(live - stored[t]!);
    num += diff;
    den += Math.abs(live) + Math.abs(stored[t]!);
  }
  if (!Number.isFinite(num) || !Number.isFinite(den)) return 1;
  return num / den;
}

/** Row-sum checksum of the readout, on whatever pooled vector the live conv produced. */
export function readoutResidual(head: ConvHead, counts: number[][], use32 = true): number {
  const x = standardize(counts, head.mean, head.std);
  const pooled = pool(convAt(head.kernel, head.convBias, x, false)).map((v) => (use32 ? r32(v) : v));
  const add = (a: number, b: number) => (use32 ? r32(a + b) : a + b);
  const mul = (a: number, b: number) => (use32 ? r32(r32(a) * r32(b)) : a * b);
  let live = 0;
  for (let m = 0; m < N_CLASS; m++) {
    let z = use32 ? r32(head.readoutB[m]!) : head.readoutB[m]!;
    const row = head.readoutW[m]!;
    for (let c = 0; c < N_OUT; c++) z = add(z, mul(row[c]!, pooled[c]!));
    live = add(live, z);
  }
  let stored = 0;
  for (let c = 0; c < N_OUT; c++) stored = add(stored, mul(head.readoutChecksum[c]!, pooled[c]!));
  let bsum = 0;
  for (const b of head.readoutB) bsum = add(bsum, use32 ? r32(b) : b);
  stored = add(stored, bsum);
  if (!Number.isFinite(live) || !Number.isFinite(stored)) return 1;
  return Math.abs(live - stored) / (Math.abs(live) + Math.abs(stored) + 1);
}

function storeChecksums(head: ConvHead): ConvHead {
  const kernelSum = Array.from({ length: N_DET }, () => Array(KERNEL).fill(0));
  for (let c = 0; c < N_OUT; c++) {
    for (let i = 0; i < N_DET; i++) {
      for (let k = 0; k < KERNEL; k++) kernelSum[i]![k] += head.kernel[c]![i]![k]!;
    }
  }
  const readoutChecksum = Array(N_OUT).fill(0);
  for (let m = 0; m < N_CLASS; m++) {
    for (let c = 0; c < N_OUT; c++) readoutChecksum[c] += head.readoutW[m]![c]!;
  }
  return {
    ...head,
    kernelSum,
    biasSum: head.convBias.reduce((a, b) => a + b, 0),
    readoutChecksum,
  };
}

function flipConv(head: ConvHead, index: number, bit: number): ConvHead {
  const kernel = cloneKernel(head.kernel);
  const nPer = N_DET * KERNEL;
  const c = Math.floor(index / nPer);
  const rest = index % nPer;
  const i = Math.floor(rest / KERNEL);
  const k = rest % KERNEL;
  const next = bitFlipF32(kernel[c]![i]![k]!, bit);
  kernel[c]![i]![k] = Number.isFinite(next) ? next : kernel[c]![i]![k]! * 16;
  return { ...head, kernel };
}

function flipReadout(head: ConvHead, index: number, bit: number): ConvHead {
  const readoutW = head.readoutW.map((row) => row.slice());
  const m = Math.floor(index / N_OUT);
  const c = index % N_OUT;
  const next = bitFlipF32(readoutW[m]![c]!, bit);
  readoutW[m]![c] = Number.isFinite(next) ? next : readoutW[m]![c]! * 16;
  return { ...head, readoutW };
}

function nConvWeights(): number {
  return N_OUT * N_DET * KERNEL;
}

function faultIndex(seed: number, n: number): { index: number; bitCritical: number; bitMild: number } {
  return {
    index: Math.abs(seed * 17 + 9) % n,
    bitCritical: 28 + (Math.abs(seed) % 3),
    bitMild: 18 + (Math.abs(seed) % 6),
  };
}

export function trainConv(samples: { counts: number[][]; y: KnownClass }[], seed: number): ConvHead {
  const rng = new Rng(seed);
  let mean = 0;
  let n = 0;
  for (const s of samples) for (const row of s.counts) for (const v of row) { mean += v; n += 1; }
  mean /= n;
  let varr = 0;
  for (const s of samples) for (const row of s.counts) for (const v of row) varr += (v - mean) ** 2;
  const std = Math.sqrt(varr / n) || 1;

  const kernel = Array.from({ length: N_OUT }, () =>
    Array.from({ length: N_DET }, () => Array.from({ length: KERNEL }, () => (rng.next() - 0.5) * 0.02)),
  );
  const convBias = Array(N_OUT).fill(0);
  const readoutW = Array.from({ length: N_CLASS }, () => Array.from({ length: N_OUT }, () => (rng.next() - 0.5) * 0.02));
  const readoutB = Array(N_CLASS).fill(0);

  const epochs = 50;
  const lr = 0.08;
  for (let ep = 0; ep < epochs; ep++) {
    const gK = kernel.map((o) => o.map((row) => row.map(() => 0)));
    const gCb = Array(N_OUT).fill(0);
    const gW = readoutW.map((row) => row.map(() => 0));
    const gB = Array(N_CLASS).fill(0);
    for (const s of samples) {
      const x = standardize(s.counts, mean, std);
      const y = convAt(kernel, convBias, x, false);
      const p = pool(y);
      const probs = softmax(logitsOf({ kernel, convBias, kernelSum: [], biasSum: 0, readoutW, readoutB, readoutChecksum: [], mean, std }, p));
      const yi = KNOWN_CLASSES.indexOf(s.y);
      const dLogit = probs.map((prob, m) => prob - (m === yi ? 1 : 0));
      const dPool = Array(N_OUT).fill(0);
      for (let m = 0; m < N_CLASS; m++) {
        gB[m] += dLogit[m]!;
        for (let c = 0; c < N_OUT; c++) {
          gW[m]![c] += dLogit[m]! * p[c]!;
          dPool[c] += dLogit[m]! * readoutW[m]![c]!;
        }
      }
      for (let c = 0; c < N_OUT; c++) {
        const dy = dPool[c]! / T_OUT;
        for (let t = 0; t < T_OUT; t++) {
          gCb[c] += dy;
          for (let i = 0; i < N_DET; i++) {
            for (let k = 0; k < KERNEL; k++) gK[c]![i]![k] += dy * x[i]![t + k]!;
          }
        }
      }
    }
    const scale = lr / samples.length;
    for (let c = 0; c < N_OUT; c++) {
      convBias[c] -= scale * gCb[c]!;
      for (let i = 0; i < N_DET; i++) for (let k = 0; k < KERNEL; k++) kernel[c]![i]![k] -= scale * gK[c]![i]![k]!;
    }
    for (let m = 0; m < N_CLASS; m++) {
      readoutB[m] -= scale * gB[m]!;
      for (let c = 0; c < N_OUT; c++) readoutW[m]![c] -= scale * gW[m]![c]!;
    }
  }
  return storeChecksums({ kernel, convBias, kernelSum: [], biasSum: 0, readoutW, readoutB, readoutChecksum: [], mean, std });
}

function percentile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))] ?? 0;
}

function countsOf(ev: EventCube): number[][] {
  return subtractBackground(ev.counts);
}

function meanStd(xs: number[]): { mean: number; std: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
  const v = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(xs.length, 1);
  return { mean, std: Math.sqrt(v) };
}

interface Rates {
  knownAcc: number;
  convOnCrit: number;
  convOnMild: number;
  convOnNovel: number;
  convOnKnown: number;
  convOnSensor: number;
  convOnReadoutFlip: number;
  readoutOnConvCrit: number;
  readoutOnConvCritNonfinite: number;
  readoutOnConvCritFinite: number;
  readoutIdentityBreak: number;
  readoutOnReadoutCrit: number;
  readoutOnNovel: number;
}

function rate(flags: boolean[]): number {
  return flags.filter(Boolean).length / Math.max(flags.length, 1);
}

function convNonfinite(head: ConvHead, counts: number[][]): boolean {
  const x = standardize(counts, head.mean, head.std);
  const y = convAt(head.kernel, head.convBias, x, false);
  return y.some((row) => row.some((v) => !Number.isFinite(v)));
}

function evalSim(seed: number): Rates {
  const rng = new Rng(seed);
  const samples: { counts: number[][]; y: KnownClass }[] = [];
  for (const y of KNOWN_CLASSES) {
    for (let i = 0; i < 80; i++) {
      const ev = generateEvent({ family: y, fault: "none", seed: rng.int(1, 1e9) });
      samples.push({ counts: countsOf(ev), y });
    }
  }
  const head = trainConv(samples, seed);
  const cal = new Rng(seed + 99);
  const convClean: number[] = [];
  const readClean: number[] = [];
  for (let i = 0; i < 120; i++) {
    const family = cal.pick(KNOWN_CLASSES);
    const ev = generateEvent({ family, fault: "none", seed: cal.int(1, 1e9) });
    const counts = countsOf(ev);
    convClean.push(convResidual(head, counts));
    readClean.push(readoutResidual(head, counts));
  }
  const convThr = Math.max(percentile(convClean, 0.995) * 1.25, 2 ** -24);
  const readThr = Math.max(percentile(readClean, 0.995) * 1.25, 2 ** -24);

  const evalRng = new Rng(EVAL_SEED);
  const knownFlags: boolean[] = [];
  const knownRight: boolean[] = [];
  const novelFlags: number[] = [];
  const sensorFlags: boolean[] = [];
  const crit: boolean[] = [];
  const mild: boolean[] = [];
  const convSeesReadout: boolean[] = [];
  const readSeesConv: boolean[] = [];
  const readSeesConvNonfinite: boolean[] = [];
  const readSeesConvFinite: boolean[] = [];
  const readIdentity: boolean[] = [];
  const readSeesRead: boolean[] = [];
  const readNovel: boolean[] = [];

  const cells: { family: TransientFamily; fault: "none" | "sensor-spike"; kind: string }[] = [
    { family: "GRB", fault: "none", kind: "known" },
    { family: "SGR", fault: "none", kind: "known" },
    { family: "SFLARE", fault: "none", kind: "known" },
    { family: "TGF", fault: "none", kind: "novel" },
    { family: "GRB", fault: "sensor-spike", kind: "sensor" },
    { family: "SGR", fault: "sensor-spike", kind: "sensor" },
  ];
  for (const cell of cells) {
    for (let i = 0; i < PER_CELL; i++) {
      const ev = generateEvent({ family: cell.family, fault: cell.fault, seed: evalRng.int(1, 1e9) });
      const counts = countsOf(ev);
      const fired = convResidual(head, counts) > convThr;
      if (cell.kind === "known") {
        knownFlags.push(fired);
        knownRight.push(predict(head, counts) === cell.family);
      } else if (cell.kind === "novel") {
        novelFlags.push(fired ? 1 : 0);
        readNovel.push(readoutResidual(head, counts) > readThr);
      } else sensorFlags.push(fired);
    }
  }
  for (let i = 0; i < PER_CELL; i++) {
    const ev = generateEvent({ family: "GRB", fault: "none", seed: evalRng.int(1, 1e9) });
    const counts = countsOf(ev);
    const loc = faultIndex(ev.spec.seed, nConvWeights());
    const critHead = flipConv(head, loc.index, loc.bitCritical);
    const mildHead = flipConv(head, loc.index, loc.bitMild);
    crit.push(convResidual(critHead, counts) > convThr);
    mild.push(convResidual(mildHead, counts) > convThr);
    const readFires = readoutResidual(critHead, counts) > readThr;
    const overflow = convNonfinite(critHead, counts);
    readSeesConv.push(readFires);
    readSeesConvNonfinite.push(readFires && overflow);
    readSeesConvFinite.push(readFires && !overflow);
    readIdentity.push(readoutResidual(critHead, counts, false) > 1e-6);
    const rloc = faultIndex(ev.spec.seed, N_CLASS * N_OUT);
    const readHead = flipReadout(head, rloc.index, rloc.bitCritical);
    convSeesReadout.push(convResidual(readHead, counts) > convThr);
    readSeesRead.push(readoutResidual(readHead, counts) > readThr);
  }
  return {
    knownAcc: rate(knownRight),
    convOnCrit: rate(crit),
    convOnMild: rate(mild),
    convOnNovel: novelFlags.reduce((a, b) => a + b, 0) / novelFlags.length,
    convOnKnown: rate(knownFlags),
    convOnSensor: rate(sensorFlags),
    convOnReadoutFlip: rate(convSeesReadout),
    readoutOnConvCrit: rate(readSeesConv),
    readoutOnConvCritNonfinite: rate(readSeesConvNonfinite),
    readoutOnConvCritFinite: rate(readSeesConvFinite),
    readoutIdentityBreak: rate(readIdentity),
    readoutOnReadoutCrit: rate(readSeesRead),
    readoutOnNovel: rate(readNovel),
  };
}

function evalReal(cubes: RealCube[], seed: number): Rates {
  const rng = new Rng(seed);
  const known = cubes.filter((c) => c.family !== "TGF");
  const novel = cubes.filter((c) => c.family === "TGF");
  const samples = known.map((c) => {
    const ev = cubeToEvent(c, { family: c.family as TransientFamily, fault: "none", seed: rng.int(1, 1e9) });
    return { counts: countsOf(ev), y: c.family as KnownClass };
  });
  const head = trainConv(samples, seed);
  const convClean = samples.map((s) => convResidual(head, s.counts));
  const readClean = samples.map((s) => readoutResidual(head, s.counts));
  const convThr = Math.max(percentile(convClean, 0.995) * 1.25, 2 ** -24);
  const readThr = Math.max(percentile(readClean, 0.995) * 1.25, 2 ** -24);
  const evalRng = new Rng(EVAL_SEED);
  const knownFlags: boolean[] = [];
  const knownRight: boolean[] = [];
  for (const c of known) {
    const ev = cubeToEvent(c, { family: c.family as TransientFamily, fault: "none", seed: evalRng.int(1, 1e9) });
    const counts = countsOf(ev);
    knownFlags.push(convResidual(head, counts) > convThr);
    knownRight.push(predict(head, counts) === c.family);
  }
  const novelFlags: boolean[] = [];
  const readNovel: boolean[] = [];
  for (const c of novel) {
    const ev = cubeToEvent(c, { family: "TGF", fault: "none", seed: evalRng.int(1, 1e9) });
    const counts = countsOf(ev);
    novelFlags.push(convResidual(head, counts) > convThr);
    readNovel.push(readoutResidual(head, counts) > readThr);
  }
  const injected = known.slice(0, 40);
  const sensorFlags: boolean[] = [];
  const crit: boolean[] = [];
  const mild: boolean[] = [];
  const convSeesReadout: boolean[] = [];
  const readSeesConv: boolean[] = [];
  const readSeesConvNonfinite: boolean[] = [];
  const readSeesConvFinite: boolean[] = [];
  const readIdentity: boolean[] = [];
  const readSeesRead: boolean[] = [];
  for (const c of injected) {
    const specSeed = evalRng.int(1, 1e9);
    const spike = cubeToEvent(c, { family: c.family as TransientFamily, fault: "sensor-spike", seed: specSeed });
    sensorFlags.push(convResidual(head, countsOf(spike)) > convThr);
    const clean = cubeToEvent(c, { family: c.family as TransientFamily, fault: "none", seed: specSeed });
    const counts = countsOf(clean);
    const loc = faultIndex(specSeed, nConvWeights());
    const critHead = flipConv(head, loc.index, loc.bitCritical);
    const mildHead = flipConv(head, loc.index, loc.bitMild);
    crit.push(convResidual(critHead, counts) > convThr);
    mild.push(convResidual(mildHead, counts) > convThr);
    const readFires = readoutResidual(critHead, counts) > readThr;
    const overflow = convNonfinite(critHead, counts);
    readSeesConv.push(readFires);
    readSeesConvNonfinite.push(readFires && overflow);
    readSeesConvFinite.push(readFires && !overflow);
    readIdentity.push(readoutResidual(critHead, counts, false) > 1e-6);
    const rloc = faultIndex(specSeed, N_CLASS * N_OUT);
    const readHead = flipReadout(head, rloc.index, rloc.bitCritical);
    convSeesReadout.push(convResidual(readHead, counts) > convThr);
    readSeesRead.push(readoutResidual(readHead, counts) > readThr);
  }
  return {
    knownAcc: rate(knownRight),
    convOnCrit: rate(crit),
    convOnMild: rate(mild),
    convOnNovel: rate(novelFlags),
    convOnKnown: rate(knownFlags),
    convOnSensor: rate(sensorFlags),
    convOnReadoutFlip: rate(convSeesReadout),
    readoutOnConvCrit: rate(readSeesConv),
    readoutOnConvCritNonfinite: rate(readSeesConvNonfinite),
    readoutOnConvCritFinite: rate(readSeesConvFinite),
    readoutIdentityBreak: rate(readIdentity),
    readoutOnReadoutCrit: rate(readSeesRead),
    readoutOnNovel: rate(readNovel),
  };
}

function identityCheck(): { fp64: number; flipped: number } {
  const rng = new Rng(1);
  const samples: { counts: number[][]; y: KnownClass }[] = [];
  for (const y of KNOWN_CLASSES) {
    for (let i = 0; i < 4; i++) {
      samples.push({ counts: countsOf(generateEvent({ family: y, fault: "none", seed: rng.int(1, 1e9) })), y });
    }
  }
  const head = trainConv(samples, 1);
  const counts = samples[0]!.counts;
  const clean = convResidual(head, counts, false);
  const bad = convResidual(flipConv(head, 3, 30), counts, false);
  return { fp64: clean, flipped: bad };
}

function pack(rows: Rates[]) {
  const keys = Object.keys(rows[0]!) as (keyof Rates)[];
  const out: Record<string, { mean: number; std: number }> = {};
  for (const k of keys) out[k] = meanStd(rows.map((r) => r[k]));
  return out;
}

function main() {
  const id = identityCheck();
  if (id.fp64 > 1e-8) {
    console.error("FAIL: clean conv checksum is not an identity in exact arithmetic", id);
    process.exit(1);
  }
  if (!(id.flipped > id.fp64 * 100)) {
    console.error("FAIL: a kernel bit-flip did not move the exact residual", id);
    process.exit(1);
  }
  console.log("identity", id);
  const sim = SEEDS.map((s) => {
    const row = evalSim(s);
    console.log("sim", s, row);
    return row;
  });
  const cubes = loadCubes("experiments/gbm/cubes.jsonl");
  const real = SEEDS.map((s) => {
    const row = evalReal(cubes, s);
    console.log("real", s, row);
    return row;
  });
  const out = {
    note: "Four filters, kernel width 5, eight detector inputs, linear readout to GRB/SGR/solar flare. Not the John et al. conv-recurrent network. Conv checksum: sum of filter outputs versus convolution with the stored summed filter, fp32. Readout checksum: row-sum of the linear layer. Thresholds are the 99.5th percentile of clean known residuals, times 1.25, floored at binary32 unit roundoff. Simulator: 80 training events per known class, 36 eval trials per cell, seeds 7, 19, 31, 43, 61, eval seed 123. Real CTIME: the 100-cube slice, faults on the first 40 known cubes.",
    identity: id,
    architecture: { inDetectors: N_DET, time: N_TIME, filters: N_OUT, kernel: KERNEL, classes: N_CLASS },
    simulator: { perSeed: sim, summary: pack(sim) },
    ctime: { perSeed: real, summary: pack(real) },
  };
  writeFileSync("experiments/gbm/conv_checksum.json", JSON.stringify(out, null, 2));
  const brief: Record<string, string> = {};
  for (const [name, block] of [["simulator", out.simulator.summary], ["ctime", out.ctime.summary]] as const) {
    for (const [k, v] of Object.entries(block)) brief[`${name}.${k}`] = `${(100 * v.mean).toFixed(1)} ± ${(100 * v.std).toFixed(1)}`;
  }
  console.log(JSON.stringify(brief, null, 2));
}

main();
