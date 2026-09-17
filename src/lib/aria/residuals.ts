import type { FeatureVector, PhysicsFeatures, Rank1Fit } from "./types";
import { subtractBackground } from "./physics";

function hypot(xs: number[]): number {
  return Math.sqrt(xs.reduce((s, x) => s + x * x, 0));
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : 0.5 * (s[mid - 1]! + s[mid]!);
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

export function rank1(X: number[][], iters = 48): Rank1Fit {
  const n = X.length;
  const m = X[0]?.length ?? 0;
  let v = Array.from({ length: m }, () => 1 / Math.sqrt(m));
  let u = Array(n).fill(0);

  for (let t = 0; t < iters; t++) {
    for (let i = 0; i < n; i++) {
      let s = 0;
      const row = X[i]!;
      for (let j = 0; j < m; j++) s += row[j]! * v[j]!;
      u[i] = s;
    }
    const nu = hypot(u) || 1;
    u = u.map((x) => x / nu);
    const nextV = Array(m).fill(0);
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i]![j]! * u[i]!;
      nextV[j] = s;
    }
    const nv = hypot(nextV) || 1;
    v = nextV.map((x) => x / nv);
  }

  let sigma = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += X[i]![j]! * v[j]!;
    sigma += u[i]! * s;
  }

  let num = 0;
  let den = 0;
  const rowResidual = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let rnum = 0;
    let rden = 0;
    for (let j = 0; j < m; j++) {
      const rec = sigma * u[i]! * v[j]!;
      const e = X[i]![j]! - rec;
      rnum += e * e;
      rden += X[i]![j]! * X[i]![j]!;
      num += e * e;
      den += X[i]![j]! * X[i]![j]!;
    }
    rowResidual[i] = Math.sqrt(rnum) / (Math.sqrt(rden) + 1e-9);
  }
  let worstRow = 0;
  for (let i = 1; i < n; i++) if (rowResidual[i]! > rowResidual[worstRow]!) worstRow = i;

  return {
    u,
    v,
    sigma,
    frobeniusResidual: Math.sqrt(num) / (Math.sqrt(den) + 1e-9),
    rowResidual,
    worstRow,
  };
}

export function dropRow(X: number[][], row: number): number[][] {
  return X.filter((_, i) => i !== row);
}

function medianProfile(X: number[][]): number[] {
  const m = X[0]?.length ?? 0;
  const profile = Array(m).fill(0);
  for (let t = 0; t < m; t++) profile[t] = median(X.map((row) => row[t]!));
  return profile;
}

/**
 * Robust multi-detector parity: the shared lightcurve is the per-bin median,
 * so a single loud spike cannot become the principal component (as SVD would allow).
 * Inconsistency of detector i is 1 − cosine(row_i, median profile).
 */
export function detectorParity(X: number[][]): {
  profile: number[];
  inconsistency: number[];
  worst: number;
  physics: number;
  dropGain: number;
  amplitudes: number[];
} {
  const energy = X.map((row) => row.reduce((s, v) => s + Math.abs(v), 0));
  const maxE = Math.max(...energy, 1e-9);
  const bright = energy
    .map((e, i) => (e >= 0.28 * maxE ? i : -1))
    .filter((i) => i >= 0);
  const profileIdx = bright.length >= 2 ? bright : energy
    .map((e, i) => [e, i] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, 2)
    .map(([, i]) => i);
  const profile = medianProfile(profileIdx.map((i) => X[i]!));
  const physicsScores = X.map((_, i) => {
    if (!profileIdx.includes(i)) return -1;
    const others = profileIdx.filter((j) => j !== i);
    if (others.length === 0) return 0;
    const loo = others.length === 1 ? X[others[0]!]! : medianProfile(others.map((j) => X[j]!));
    return 1 - Math.max(0, cosine(X[i]!, loo));
  });
  let worst = profileIdx[0] ?? 0;
  for (const i of profileIdx) {
    if (physicsScores[i]! > physicsScores[worst]!) worst = i;
  }
  const physics = Math.max(0, physicsScores[worst]!);
  const reducedIdx = profileIdx.filter((i) => i !== worst);
  const max2 =
    reducedIdx.length <= 1
      ? 0
      : Math.max(...reducedIdx.map((i) => {
          const others = reducedIdx.filter((j) => j !== i);
          const loo = others.length === 1 ? X[others[0]!]! : medianProfile(others.map((j) => X[j]!));
          return 1 - Math.max(0, cosine(X[i]!, loo));
        }));
  const dropGain = (physics - max2) / (physics + 1e-9);
  const inconsistency = physicsScores.map((s) => Math.max(0, s));
  const amplitudes = X.map((row) => {
    let s = 0;
    for (let t = 0; t < row.length; t++) s += row[t]! * profile[t]!;
    return s;
  });
  return {
    profile,
    inconsistency,
    worst,
    physics,
    dropGain,
    amplitudes,
  };
}

function fwhm(profile: number[]): number {
  const peak = Math.max(...profile, 1e-9);
  const thr = 0.5 * peak;
  let left = 0;
  let right = profile.length - 1;
  while (left < profile.length && profile[left]! < thr) left += 1;
  while (right > left && profile[right]! < thr) right -= 1;
  return (right - left + 1) / profile.length;
}

function durationFrac(profile: number[]): number {
  const peak = Math.max(...profile, 1e-9);
  const thr = 0.1 * peak;
  return profile.filter((x) => x >= thr).length / profile.length;
}

export function morphologyFeatures(
  netCounts: number[][],
  soft: number[][],
  hard: number[][],
): {
  features: PhysicsFeatures;
  vector: FeatureVector;
  fit: Rank1Fit;
  dropGain: number;
  structured: number;
  inconsistency: number[];
} {
  const X = subtractBackground(netCounts);
  const parity = detectorParity(X);
  const fit = rank1(X);
  const profile = parity.profile.map((x) => Math.max(0, x));
  const peakIdx = profile.reduce((best, x, i) => (x > profile[best]! ? i : best), 0);
  const deriv = profile.slice(1).map((x, i) => x - profile[i]!);
  const meanP = profile.reduce((s, x) => s + x, 0) / Math.max(profile.length, 1);
  const varP = profile.reduce((s, x) => s + (x - meanP) ** 2, 0) / Math.max(profile.length, 1);
  const ac =
    profile.slice(1).reduce((s, x, i) => s + x * profile[i]!, 0) /
    (profile.reduce((s, x) => s + x * x, 0) + 1e-9);

  const nDet = netCounts.length;
  const nTime = netCounts[0]?.length ?? 0;
  let softSum = 0;
  let hardSum = 0;
  for (let d = 0; d < nDet; d++) {
    if (d === parity.worst && parity.physics > 0.35) continue;
    for (let t = 0; t < nTime; t++) {
      softSum += soft[d]![t]!;
      hardSum += hard[d]![t]!;
    }
  }

  const amps = parity.amplitudes.map(Math.abs);
  const usable = amps.filter((_, i) => !(i === parity.worst && parity.physics > 0.35));
  const meanA = usable.reduce((s, x) => s + x, 0) / Math.max(usable.length, 1) + 1e-9;
  const maxA = Math.max(...usable, 0);

  let total = 0;
  for (let d = 0; d < nDet; d++) {
    if (d === parity.worst && parity.physics > 0.35) continue;
    for (const v of netCounts[d]!) total += v;
  }

  const features: PhysicsFeatures = {
    duration: durationFrac(profile),
    peakTime: peakIdx / Math.max(profile.length, 1),
    fwhm: fwhm(profile),
    variability: Math.sqrt(deriv.reduce((s, x) => s + x * x, 0) / Math.max(deriv.length, 1)) / (Math.sqrt(varP) + 1e-9),
    hardness: hardSum / (softSum + 1e-9),
    logCounts: Math.log1p(total),
    concentration: maxA / meanA,
    smoothness: ac,
  };

  const vector: FeatureVector = [
    features.duration,
    features.peakTime,
    features.fwhm,
    features.variability,
    features.hardness,
    features.logCounts,
    features.concentration,
    features.smoothness,
  ];

  return {
    features,
    vector,
    fit: { ...fit, worstRow: parity.worst, rowResidual: parity.inconsistency },
    dropGain: parity.dropGain,
    structured: parity.physics,
    inconsistency: parity.inconsistency,
  };
}

export const FEATURE_NAMES = [
  "duration",
  "peakTime",
  "fwhm",
  "variability",
  "hardness",
  "logCounts",
  "concentration",
  "smoothness",
] as const;
