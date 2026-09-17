import type { Attribution, CauseLabel, EventCube, LinearHead, Residuals, Signature, Thresholds } from "./types";
import { morphologyFeatures } from "./residuals";
import { executionResidual, flipWeight, nearestCentroidDistance, predictProbs } from "./classifier";

export function applyComputeFault(event: EventCube, head: LinearHead): { head: LinearHead; event: EventCube } {
  if (event.spec.fault !== "shared-bitflip" && event.spec.fault !== "sensor-and-compute") {
    return { head, event };
  }
  const nFeat = head.weights[0]!.length;
  const nW = head.weights.length * nFeat;
  const index = Math.abs(event.spec.seed * 17 + 9) % nW;
  const bit = event.spec.severity === "mild" ? 18 + (Math.abs(event.spec.seed) % 6) : 28 + (Math.abs(event.spec.seed) % 3);
  return {
    head: flipWeight(head, index, bit),
    event: { ...event, flippedBit: bit, flippedWeightIndex: index },
  };
}

export function measure(event: EventCube, cleanHead: LinearHead): { residuals: Residuals; predictedClass: Attribution["predictedClass"]; liveHead: LinearHead; event: EventCube } {
  const { head: liveHead, event: ev } = applyComputeFault(event, cleanHead);
  const morph = morphologyFeatures(ev.counts, ev.soft, ev.hard);
  const pred = predictProbs(liveHead, morph.vector);
  const residuals: Residuals = {
    physics: morph.structured,
    morph: nearestCentroidDistance(cleanHead, morph.vector),
    exec: executionResidual(liveHead, morph.vector),
    dropGain: morph.dropGain,
    worstDetector: morph.fit.worstRow,
    energy: pred.energy,
    msp: pred.msp,
    maxAbsLogit: Math.max(...pred.logits.map(Math.abs)),
    entropy: pred.entropy,
  };
  return { residuals, predictedClass: pred.msp < 0.45 ? "unknown" : pred.label, liveHead, event: ev };
}

export function signatureOf(r: Residuals, t: Thresholds): Signature {
  return {
    S_phys: r.physics > t.physics,
    S_morph: r.morph > t.morph,
    U_det: r.physics > t.physics && (r.dropGain > t.dropGain || r.physics > t.physics * 1.15),
    U_arith: r.exec > t.exec,
  };
}

/**
 * Isolable decision rule. Not a learned fusion head.
 *
 * Signature columns (ideal, single-cause):
 *   known-clean  : 0000
 *   novel-clean  : 0100  (morphology of the shared rank-1 profile)
 *   sensor       : 1x10  (physics residual + leave-one-detector drop)
 *   compute      : xx01  (arithmetic checksum of the live head)
 */
export function decide(sig: Signature): CauseLabel {
  if (sig.U_arith) return "compute";
  if (sig.U_det || sig.S_phys) return "sensor";
  if (sig.S_morph) return "novel-clean";
  return "known-clean";
}

/** Ablations used only in the paper evaluation. Production ARIA is `decide`. */
export type AblationMode = "full" | "checksum" | "physics" | "morph" | "no-checksum";

export function decideAblation(sig: Signature, mode: AblationMode): CauseLabel {
  if (mode === "full") return decide(sig);
  if (mode === "checksum") return sig.U_arith ? "compute" : "known-clean";
  if (mode === "physics") return sig.U_det || sig.S_phys ? "sensor" : "known-clean";
  if (mode === "morph") return sig.S_morph ? "novel-clean" : "known-clean";
  if (sig.U_det || sig.S_phys) return "sensor";
  if (sig.S_morph) return "novel-clean";
  return "known-clean";
}

export function actionFor(cause: CauseLabel): Attribution["action"] {
  switch (cause) {
    case "known-clean":
      return "accept";
    case "novel-clean":
      return "preserve";
    case "sensor":
      return "isolate-channel";
    case "compute":
      return "recompute";
  }
}

export function attribute(event: EventCube, head: LinearHead, thresholds: Thresholds): Attribution {
  const { residuals, predictedClass, event: ev } = measure(event, head);
  const signature = signatureOf(residuals, thresholds);
  const cause = decide(signature);
  return {
    cause,
    signature,
    residuals,
    predictedClass,
    action: actionFor(cause),
    flippedBit: ev.flippedBit,
    worstDetector: residuals.worstDetector,
    sensorDetector: ev.sensorDetector,
  };
}

export const SIGNATURE_MATRIX: { cause: CauseLabel; pattern: string; meaning: string }[] = [
  { cause: "known-clean", pattern: "0 0 0 0", meaning: "Rank-1 physics, known morphology, checksum holds." },
  { cause: "novel-clean", pattern: "0 1 0 0", meaning: "Still a point source; the shared lightcurve is unlike known families." },
  { cause: "sensor", pattern: "1 · 1 0", meaning: "One detector is not on the rank-1 sheet; dropping it restores consistency." },
  { cause: "compute", pattern: "· · 0 1", meaning: "Input physics is intact; the live matmul disagrees with its stored checksum." },
];
