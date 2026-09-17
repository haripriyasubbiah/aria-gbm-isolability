import type { BaselineCall, CauseLabel, Residuals, Thresholds } from "./types";

/**
 * Gavarini-style OSR-as-fault-detector: unusual softmax energy/MSP is treated
 * as a fault. Novel clean events and compute faults both look "unusual".
 */
export function energyReject(r: Residuals, t: Thresholds): BaselineCall {
  const unusual = r.msp < t.mspLow || r.energy > t.energyHigh || r.energy < t.energyLow;
  if (!unusual) return { method: "energy-reject", cause: "known-clean" };
  return { method: "energy-reject", cause: "anomaly" };
}

/**
 * ProGIP-style range check on logits, with leftover unusual scores called novelty.
 * Critical bit-flips explode logits; mild flips and some bright novel events do not
 * sit in disjoint ranges.
 */
export function rangeCheck(r: Residuals, t: Thresholds): BaselineCall {
  if (r.maxAbsLogit > t.logitRange) return { method: "range-check", cause: "compute" };
  const unusual = r.msp < t.mspLow || r.energy > t.energyHigh || r.energy < t.energyLow;
  if (unusual) return { method: "range-check", cause: "novel-clean" };
  return { method: "range-check", cause: "known-clean" };
}

export function collapseAnomaly(call: BaselineCall, prefer: CauseLabel): CauseLabel {
  if (call.cause === "anomaly") return prefer;
  return call.cause;
}
