import { KNOWN_CLASSES, type CheckMode, type FeatureVector, type KnownClass, type LinearHead, type Precision } from "./types";
import { bitFlipF32 } from "./rng";

const N_CLASS = KNOWN_CLASSES.length;

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

export function standardize(x: FeatureVector, mean: number[], std: number[]): FeatureVector {
  return x.map((v, i) => (v - mean[i]!) / (std[i]! || 1));
}

export function logitsOf(head: LinearHead, x: FeatureVector): number[] {
  const z = head.weights.map((row, i) => {
    let s = head.bias[i]!;
    for (let j = 0; j < row.length; j++) s += row[j]! * x[j]!;
    return s;
  });
  return z;
}

export function predictProbs(head: LinearHead, xRaw: FeatureVector): {
  logits: number[];
  probs: number[];
  label: KnownClass;
  msp: number;
  energy: number;
  entropy: number;
} {
  const x = standardize(xRaw, head.featureMean, head.featureStd);
  const logits = logitsOf(head, x);
  const probs = softmax(logits);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
  const energy = Math.log(logits.reduce((s, z) => s + Math.exp(z - Math.max(...logits)), 0)) + Math.max(...logits);
  const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
  return {
    logits,
    probs,
    label: KNOWN_CLASSES[best]!,
    msp: probs[best]!,
    energy,
    entropy,
  };
}

/** Unit roundoff of each emulated precision. */
export const UNIT_ROUNDOFF: Record<Precision, number> = { fp64: 2 ** -53, fp32: 2 ** -24, fp16: 2 ** -11 };

/** Round-to-nearest emulation of IEEE binary16 (normals, subnormals, overflow to +-Inf). */
export function roundHalf(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const ax = Math.abs(x);
  if (ax >= 65520) return x > 0 ? Infinity : -Infinity;
  if (ax < 2 ** -25) return 0;
  const e = Math.max(Math.floor(Math.log2(ax)), -14);
  const step = 2 ** (e - 10);
  return Math.sign(x) * Math.round(ax / step) * step;
}

/** Checksum decision mode. Default "calibrated" (the shipped rule); "bounded" is the rounding-error-bounded residual. Env ARIA_CHECK or setCheckMode(). */
export let CHECK_MODE: CheckMode = (process.env.ARIA_CHECK as CheckMode | undefined) ?? "calibrated";
export function setCheckMode(m: CheckMode): void {
  CHECK_MODE = m;
}

/** Largest absolute rounding error of one operation that lands in the subnormal range (half the smallest subnormal spacing). */
const UNDERFLOW_ABS: Record<Precision, number> = { fp64: 0, fp32: 2 ** -150, fp16: 2 ** -25 };

/** Longest chain of rounded operations any term of either dot product passes through (round W or c, multiply, <=9 adds in a row, <=3 adds across rows) with margin for the fp64 reference checksum. */
const N_OPS = 20;
/** Upper bound on the number of operations that can each lose an absolute UNDERFLOW_ABS. */
const N_UNDERFLOW_OPS = 128;

function gamma(n: number, u: number): number {
  return (n * u) / (1 - n * u);
}

/**
 * A-priori bound on |live - stored| when nothing is corrupted, from the standard
 * model fl(a op b) = (a op b)(1 + d), |d| <= u (Higham, ch. 3). Because the
 * standardised input x_r is shared by both evaluations, its rounding cancels.
 *
 *   B = gamma_N(u) * ( sum_j (a_j + |c_j|) |x_j| + sum|b| + |sum b| ) + eta
 *
 * where a_j = sum_i |W_ij|, c_j = sum_i W_ij and sum|b| are the *stored clean*
 * quantities. Using the live weights here would let a flipped high-order bit
 * inflate its own bound and hide itself.
 */
export function executionBound(head: LinearHead, xRaw: FeatureVector, prec: Precision): number {
  const x = standardize(xRaw, head.featureMean, head.featureStd);
  let s = head.absBias + Math.abs(head.bias.reduce((a, b) => a + b, 0));
  for (let j = 0; j < x.length; j++) s += (head.absChecksum[j]! + Math.abs(head.checksum[j]!)) * Math.abs(x[j]!);
  return gamma(N_OPS, UNIT_ROUNDOFF[prec]) * s + N_UNDERFLOW_OPS * UNDERFLOW_ABS[prec];
}

export function rounder(prec: Precision): (x: number) => number {
  if (prec === "fp32") return Math.fround;
  if (prec === "fp16") return roundHalf;
  return (x) => x;
}

/**
 * Algorithm-based checksum: 1ᵀ(Wx) vs (1ᵀW)x, using the stored clean checksum.
 * The live matvec and the stored-checksum dot product are both evaluated with
 * every operation rounded to `prec`, so the residual has the noise floor a
 * real fp32 / fp16 deployment would have. A non-finite live value (overflow
 * from a high-exponent flip) is reported as the maximum residual.
 */
export function executionResidual(head: LinearHead, xRaw: FeatureVector, prec: Precision = "fp64", mode: CheckMode = CHECK_MODE): number {
  const r = rounder(prec);
  const x = standardize(xRaw, head.featureMean, head.featureStd).map(r);
  let live = 0;
  for (let i = 0; i < head.weights.length; i++) {
    let zi = r(head.bias[i]!);
    const row = head.weights[i]!;
    for (let j = 0; j < row.length; j++) zi = r(zi + r(r(row[j]!) * x[j]!));
    live = r(live + zi);
  }
  let stored = 0;
  for (let j = 0; j < x.length; j++) stored = r(stored + r(r(head.checksum[j]!) * x[j]!));
  let bsum = 0;
  for (const b of head.bias) bsum = r(bsum + r(b));
  stored = r(stored + bsum);
  if (mode === "bounded") {
    if (!Number.isFinite(live) || !Number.isFinite(stored)) return 1e9;
    const diff = Math.abs(live - stored);
    const bound = executionBound(head, xRaw, prec);
    return bound > 0 ? diff / bound : diff > 0 ? 1e9 : 0;
  }
  if (!Number.isFinite(live) || !Number.isFinite(stored)) return 1;
  const mag = Math.abs(live) + Math.abs(stored) + 1;
  return Math.abs(live - stored) / mag;
}

export function nearestCentroidDistance(head: LinearHead, xRaw: FeatureVector): number {
  const x = standardize(xRaw, head.featureMean, head.featureStd);
  let best = Infinity;
  for (const cls of KNOWN_CLASSES) {
    const c = head.centroids[cls]!;
    let d = 0;
    for (let i = 0; i < x.length; i++) d += (x[i]! - c[i]!) ** 2;
    best = Math.min(best, Math.sqrt(d));
  }
  return best;
}

export function flipWeight(head: LinearHead, index: number, bit: number): LinearHead {
  const weights = head.weights.map((row) => row.slice());
  const nFeat = weights[0]!.length;
  const i = Math.floor(index / nFeat);
  const j = index % nFeat;
  const cur = weights[i]![j]!;
  const next = bitFlipF32(cur, bit);
  weights[i]![j] = Number.isFinite(next) ? next : cur * 16;
  return { ...head, weights };
}

export function trainHead(
  samples: { x: FeatureVector; y: KnownClass }[],
  epochs = 160,
  lr = 0.12,
): LinearHead {
  const nFeat = samples[0]!.x.length;
  const mean = Array(nFeat).fill(0);
  for (const s of samples) for (let j = 0; j < nFeat; j++) mean[j] += s.x[j]!;
  for (let j = 0; j < nFeat; j++) mean[j] /= samples.length;
  const std = Array(nFeat).fill(0);
  for (const s of samples) for (let j = 0; j < nFeat; j++) std[j] += (s.x[j]! - mean[j]!) ** 2;
  for (let j = 0; j < nFeat; j++) std[j] = Math.sqrt(std[j] / samples.length) || 1;

  const weights = Array.from({ length: N_CLASS }, () =>
    Array.from({ length: nFeat }, (_, j) => 0.01 * Math.sin(j + 1)),
  );
  const bias = Array(N_CLASS).fill(0);

  for (let ep = 0; ep < epochs; ep++) {
    const gW = weights.map((row) => row.map(() => 0));
    const gB = Array(N_CLASS).fill(0);
    for (const s of samples) {
      const x = standardize(s.x, mean, std);
      const logits = weights.map((row, i) => {
        let v = bias[i]!;
        for (let j = 0; j < nFeat; j++) v += row[j]! * x[j]!;
        return v;
      });
      const p = softmax(logits);
      const y = KNOWN_CLASSES.indexOf(s.y);
      for (let i = 0; i < N_CLASS; i++) {
        const err = p[i]! - (i === y ? 1 : 0);
        gB[i] += err;
        for (let j = 0; j < nFeat; j++) gW[i]![j] += err * x[j]!;
      }
    }
    const n = samples.length;
    for (let i = 0; i < N_CLASS; i++) {
      bias[i] -= (lr * gB[i]!) / n;
      for (let j = 0; j < nFeat; j++) weights[i]![j] -= (lr * gW[i]![j]!) / n;
    }
  }

  const checksum = Array(nFeat).fill(0);
  const absChecksum = Array(nFeat).fill(0);
  for (let i = 0; i < N_CLASS; i++) {
    for (let j = 0; j < nFeat; j++) {
      checksum[j] += weights[i]![j]!;
      absChecksum[j] += Math.abs(weights[i]![j]!);
    }
  }
  const absBias = bias.reduce((a, b) => a + Math.abs(b), 0);

  const grouped: Record<KnownClass, number[][]> = { GRB: [], SGR: [], SFLARE: [] };
  for (const s of samples) grouped[s.y].push(standardize(s.x, mean, std));
  const centroids = {} as Record<KnownClass, number[]>;
  for (const cls of KNOWN_CLASSES) {
    const rows = grouped[cls];
    const c = Array(nFeat).fill(0);
    for (const row of rows) for (let j = 0; j < nFeat; j++) c[j] += row[j]!;
    centroids[cls] = c.map((v) => v / Math.max(rows.length, 1));
  }

  return { weights, bias, checksum, absChecksum, absBias, featureMean: mean, featureStd: std, centroids };
}
