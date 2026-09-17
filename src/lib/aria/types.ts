export const N_DET = 8;
export const N_TIME = 64;
export const KNOWN_CLASSES = ["GRB", "SGR", "SFLARE"] as const;
export const ALL_FAMILIES = ["GRB", "SGR", "SFLARE", "TGF"] as const;

export type KnownClass = (typeof KNOWN_CLASSES)[number];
export type TransientFamily = (typeof ALL_FAMILIES)[number];

export type GroundTruth =
  | "known-clean"
  | "novel-clean"
  | "sensor"
  | "compute"
  | "compound";

export type CauseLabel = Exclude<GroundTruth, "compound">;

export type FaultKind = "none" | "sensor-spike" | "shared-bitflip" | "sensor-and-compute";

export type BitSeverity = "mild" | "critical";

export interface SkySource {
  azDeg: number;
  elDeg: number;
}

export interface EventSpec {
  family: TransientFamily;
  fault: FaultKind;
  severity?: BitSeverity;
  seed: number;
}

export interface EventCube {
  spec: EventSpec;
  truth: GroundTruth;
  /** Background-subtracted counts, detectors × time, summed over energy. */
  counts: number[][];
  /** Soft (low-energy) and hard (high-energy) totals per time bin, shared across detectors after summing. */
  soft: number[][];
  hard: number[][];
  source: SkySource;
  sensorDetector: number | null;
  flippedBit: number | null;
  flippedWeightIndex: number | null;
}

export interface Rank1Fit {
  u: number[];
  v: number[];
  sigma: number;
  frobeniusResidual: number;
  rowResidual: number[];
  worstRow: number;
}

export interface PhysicsFeatures {
  duration: number;
  peakTime: number;
  fwhm: number;
  variability: number;
  hardness: number;
  logCounts: number;
  concentration: number;
  smoothness: number;
}

export type FeatureVector = number[];

export interface LinearHead {
  weights: number[][];
  bias: number[];
  /** Row-sum checksum of the clean weights, stored at "compile" time. */
  checksum: number[];
  featureMean: number[];
  featureStd: number[];
  centroids: Record<KnownClass, number[]>;
}

export interface Thresholds {
  physics: number;
  morph: number;
  exec: number;
  dropGain: number;
  energyHigh: number;
  energyLow: number;
  mspLow: number;
  logitRange: number;
}

export interface Residuals {
  physics: number;
  morph: number;
  exec: number;
  dropGain: number;
  worstDetector: number;
  energy: number;
  msp: number;
  maxAbsLogit: number;
  entropy: number;
}

export interface Signature {
  S_phys: boolean;
  S_morph: boolean;
  U_det: boolean;
  U_arith: boolean;
}

export interface Attribution {
  cause: CauseLabel;
  signature: Signature;
  residuals: Residuals;
  predictedClass: KnownClass | "unknown";
  action: "accept" | "preserve" | "isolate-channel" | "recompute";
  flippedBit: number | null;
  worstDetector: number;
  sensorDetector: number | null;
}

export interface BaselineCall {
  method: "energy-reject" | "range-check" | "aria";
  cause: CauseLabel | "anomaly";
}

export const CATALOG_COUNTS = {
  GRB: 4375,
  SFLARE: 2800,
  TGF: 1658,
  SGR: 684,
  LOCLPAR: 2019,
  DISTPAR: 148,
  TRANSNT: 448,
  UNCERT: 844,
  other: 5,
  total: 12981,
} as const;
