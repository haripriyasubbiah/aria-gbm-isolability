import { N_DET, N_TIME, type EventCube, type EventSpec, type GroundTruth, type SkySource } from "./types";
import { Rng } from "./rng";

/** Simplified NaI-like placements around the spacecraft (azimuth, elevation in degrees). */
const DETECTORS: SkySource[] = [
  { azDeg: 0, elDeg: 25 },
  { azDeg: 45, elDeg: 20 },
  { azDeg: 90, elDeg: 30 },
  { azDeg: 135, elDeg: 18 },
  { azDeg: 180, elDeg: 28 },
  { azDeg: 225, elDeg: 16 },
  { azDeg: 270, elDeg: 32 },
  { azDeg: 315, elDeg: 22 },
];

function toRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

function angularSeparation(a: SkySource, b: SkySource): number {
  const az1 = toRad(a.azDeg);
  const az2 = toRad(b.azDeg);
  const el1 = toRad(a.elDeg);
  const el2 = toRad(b.elDeg);
  const cos =
    Math.sin(el1) * Math.sin(el2) +
    Math.cos(el1) * Math.cos(el2) * Math.cos(az1 - az2);
  return Math.acos(Math.min(1, Math.max(-1, cos)));
}

/** Cosine-like effective area used by GBM NaI detectors, with a small isotropic floor. */
export function detectorResponse(source: SkySource, det: SkySource): number {
  const mu = Math.cos(angularSeparation(source, det));
  return Math.max(0.06, mu);
}

function zeros(n: number, m: number): number[][] {
  return Array.from({ length: n }, () => Array(m).fill(0));
}

function fred(t: number, t0: number, rise: number, decay: number): number {
  if (t < t0) return Math.exp((t - t0) / Math.max(rise, 0.25));
  return Math.exp(-(t - t0) / Math.max(decay, 0.25));
}

function gaussianPulse(t: number, t0: number, width: number): number {
  const z = (t - t0) / Math.max(width, 0.4);
  return Math.exp(-0.5 * z * z);
}

function familyTemplate(family: EventSpec["family"], rng: Rng): number[] {
  const pulse = Array(N_TIME).fill(0);
  if (family === "GRB") {
    const nPeaks = rng.next() < 0.35 ? 2 : 1;
    for (let p = 0; p < nPeaks; p++) {
      const t0 = rng.uniform(12, 40);
      const rise = rng.uniform(1.8, 6);
      const decay = rng.uniform(8, 22);
      const amp = rng.uniform(0.55, 1);
      for (let t = 0; t < N_TIME; t++) pulse[t] += amp * fred(t, t0, rise, decay);
    }
  } else if (family === "SGR") {
    const t0 = rng.uniform(22, 42);
    const rise = rng.uniform(0.6, 1.6);
    const decay = rng.uniform(2.0, 4.5);
    for (let t = 0; t < N_TIME; t++) pulse[t] = fred(t, t0, rise, decay);
  } else if (family === "SFLARE") {
    const t0 = rng.uniform(18, 36);
    const width = rng.uniform(14, 24);
    for (let t = 0; t < N_TIME; t++) pulse[t] = gaussianPulse(t, t0, width);
  } else {
    const t0 = rng.int(18, 46);
    for (let t = 0; t < N_TIME; t++) {
      pulse[t] = Math.abs(t - t0) <= 1 ? (t === t0 ? 1 : 0.35) : 0;
    }
  }
  const peak = Math.max(...pulse, 1e-9);
  return pulse.map((x) => x / peak);
}

function familyAmplitude(family: EventSpec["family"], rng: Rng): { amp: number; hardness: number } {
  switch (family) {
    case "GRB":
      return { amp: rng.uniform(70, 280), hardness: rng.uniform(0.45, 1.15) };
    case "SGR":
      return { amp: rng.uniform(90, 240), hardness: rng.uniform(0.12, 0.38) };
    case "SFLARE":
      return { amp: rng.uniform(80, 220), hardness: rng.uniform(0.04, 0.18) };
    case "TGF":
      return { amp: rng.uniform(90, 260), hardness: rng.uniform(2.2, 6.5) };
  }
}

function truthOf(spec: EventSpec): GroundTruth {
  if (spec.fault === "sensor-and-compute") return "compound";
  if (spec.fault === "sensor-spike") return "sensor";
  if (spec.fault === "shared-bitflip") return "compute";
  return spec.family === "TGF" ? "novel-clean" : "known-clean";
}

export function generateEvent(spec: EventSpec): EventCube {
  const rng = new Rng(spec.seed);
  const source: SkySource = {
    azDeg: rng.uniform(0, 360),
    elDeg: rng.uniform(5, 70),
  };
  const pulse = familyTemplate(spec.family, rng);
  const { amp, hardness } = familyAmplitude(spec.family, rng);
  const soft = zeros(N_DET, N_TIME);
  const hard = zeros(N_DET, N_TIME);
  const counts = zeros(N_DET, N_TIME);

  for (let d = 0; d < N_DET; d++) {
    const resp = detectorResponse(source, DETECTORS[d]!);
    const bgSoft = rng.uniform(6, 14);
    const bgHard = rng.uniform(3, 8);
    for (let t = 0; t < N_TIME; t++) {
      const s = rng.poisson(bgSoft + resp * amp * pulse[t]!);
      const h = rng.poisson(bgHard + resp * amp * hardness * pulse[t]!);
      soft[d]![t] = s;
      hard[d]![t] = h;
      counts[d]![t] = s + h;
    }
  }

  let sensorDetector: number | null = null;
  if (spec.fault === "sensor-spike" || spec.fault === "sensor-and-compute") {
    const totals = counts.map((row) => row.reduce((a, b) => a + b, 0));
    const brightest = totals.reduce((best, v, i) => (v > totals[best]! ? i : best), 0);
    const dim = totals
      .map((v, i) => [v, i] as const)
      .filter(([, i]) => i !== brightest)
      .sort((a, b) => a[0] - b[0]);
    sensorDetector = dim[0]?.[1] ?? 0;
    const peak = Math.max(...counts.flat(), 20);
    const t0 = rng.int(10, 48);
    const width = rng.int(2, 3);
    const spike = peak * rng.uniform(1.6, 3.2);
    for (let t = t0; t < Math.min(N_TIME, t0 + width); t++) {
      soft[sensorDetector]![t] += spike;
      counts[sensorDetector]![t] += spike;
    }
  }

  return {
    spec,
    truth: truthOf(spec),
    counts,
    soft,
    hard,
    source,
    sensorDetector,
    flippedBit: null,
    flippedWeightIndex: null,
  };
}

export function subtractBackground(X: number[][]): number[][] {
  return X.map((row) => {
    const pre = row.slice(0, 8);
    const post = row.slice(-8);
    const bg =
      [...pre, ...post].sort((a, b) => a - b)[Math.floor((pre.length + post.length) / 2)] ?? 0;
    return row.map((v) => v - bg);
  });
}
