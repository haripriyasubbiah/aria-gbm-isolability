import { KNOWN_CLASSES, type FeatureVector, type KnownClass, type LinearHead } from "./types";
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

/** Algorithm-based checksum: 1ᵀ(Wx) vs (1ᵀW)x, using the stored clean checksum. */
export function executionResidual(head: LinearHead, xRaw: FeatureVector): number {
  const x = standardize(xRaw, head.featureMean, head.featureStd);
  const z = logitsOf(head, x);
  const live = z.reduce((s, v) => s + v, 0);
  let stored = 0;
  for (let j = 0; j < x.length; j++) stored += head.checksum[j]! * x[j]!;
  stored += head.bias.reduce((s, b) => s + b, 0);
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
  for (let i = 0; i < N_CLASS; i++) {
    for (let j = 0; j < nFeat; j++) checksum[j] += weights[i]![j]!;
  }

  const grouped: Record<KnownClass, number[][]> = { GRB: [], SGR: [], SFLARE: [] };
  for (const s of samples) grouped[s.y].push(standardize(s.x, mean, std));
  const centroids = {} as Record<KnownClass, number[]>;
  for (const cls of KNOWN_CLASSES) {
    const rows = grouped[cls];
    const c = Array(nFeat).fill(0);
    for (const row of rows) for (let j = 0; j < nFeat; j++) c[j] += row[j]!;
    centroids[cls] = c.map((v) => v / Math.max(rows.length, 1));
  }

  return { weights, bias, checksum, featureMean: mean, featureStd: std, centroids };
}
