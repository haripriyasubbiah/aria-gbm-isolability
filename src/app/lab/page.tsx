"use client";

import { useMemo, useState } from "react";
import {
  attribute,
  energyReject,
  generateEvent,
  getSystem,
  rangeCheck,
  type FaultKind,
  type TransientFamily,
} from "@/lib/aria";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DetectorGrid } from "@/components/detector-grid";
import { CauseBadge, SignatureBits } from "@/components/signature";

const FAMILIES: TransientFamily[] = ["GRB", "SGR", "SFLARE", "TGF"];
const FAULTS: { id: FaultKind; label: string }[] = [
  { id: "none", label: "clean execution" },
  { id: "sensor-spike", label: "single-detector spike" },
  { id: "shared-bitflip", label: "weight bit-flip" },
  { id: "sensor-and-compute", label: "spike + bit-flip (unisolable)" },
];

export default function LabPage() {
  const [family, setFamily] = useState<TransientFamily>("GRB");
  const [fault, setFault] = useState<FaultKind>("none");
  const [severity, setSeverity] = useState<"mild" | "critical">("critical");
  const [seed, setSeed] = useState(2026);
  const system = useMemo(() => getSystem(), []);

  const event = useMemo(
    () => generateEvent({ family, fault, severity, seed }),
    [family, fault, severity, seed],
  );
  const attr = useMemo(() => attribute(event, system.head, system.thresholds), [event, system]);
  const energy = energyReject(attr.residuals, system.thresholds);
  const range = rangeCheck(attr.residuals, system.thresholds);

  return (
    <div className="space-y-6">
      <div>
        <p className="font-mono text-[11px] tracking-[0.25em] text-primary uppercase">Event lab</p>
        <h1 className="text-3xl font-medium tracking-tight">One observation, three answers</h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Generate a Fermi-GBM-like cube, then compare ARIA’s signature with an energy/MSP rejector
          (Gavarini-style) and a logit range check (ProGIP-style). Held-out family is TGF.
        </p>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-4 pt-1">
          <div className="flex flex-wrap gap-2">
            {FAMILIES.map((f) => (
              <Button key={f} size="sm" variant={family === f ? "default" : "outline"} onClick={() => setFamily(f)}>
                {f}
                {f === "TGF" ? " (novel)" : ""}
              </Button>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {FAULTS.map((f) => (
              <Button key={f.id} size="sm" variant={fault === f.id ? "default" : "outline"} onClick={() => setFault(f.id)}>
                {f.label}
              </Button>
            ))}
            {fault === "shared-bitflip" ? (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => setSeverity((s) => (s === "mild" ? "critical" : "mild"))}
              >
                bit: {severity}
              </Button>
            ) : null}
            <Button size="sm" variant="ghost" onClick={() => setSeed((s) => s + 1)}>
              resample #{seed}
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <CardHeader>
            <CardTitle>Counts</CardTitle>
          </CardHeader>
          <CardContent>
            <DetectorGrid counts={event.counts} highlight={attr.sensorDetector ?? attr.worstDetector} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>ARIA signature</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-muted-foreground">truth</span>
              <CauseBadge cause={event.truth} />
              <span className="text-xs text-muted-foreground">ARIA</span>
              <CauseBadge cause={attr.cause} />
            </div>
            <SignatureBits signature={attr.signature} />
            <p className="text-xs text-muted-foreground">
              action: <span className="font-mono text-foreground">{attr.action}</span>
              {attr.flippedBit != null ? ` · flipped bit ${attr.flippedBit}` : ""}
              {event.sensorDetector != null ? ` · spike on n${event.sensorDetector}` : ""}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-xs tracking-widest uppercase">Residuals</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 font-mono text-xs">
            <div>r_phys {attr.residuals.physics.toFixed(3)}</div>
            <div>r_morph {attr.residuals.morph.toFixed(3)}</div>
            <div>r_exec {attr.residuals.exec.toExponential(2)}</div>
            <div>drop {attr.residuals.dropGain.toFixed(3)}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-xs tracking-widest uppercase">Energy / MSP reject</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <CauseBadge cause={energy.cause} />
            <p className="text-xs text-muted-foreground">
              MSP {attr.residuals.msp.toFixed(2)} · energy {attr.residuals.energy.toExponential(2)}. This
              method has no name for the cause.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-xs tracking-widest uppercase">Logit range check</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <CauseBadge cause={range.cause} />
            <p className="text-xs text-muted-foreground">
              max |logit| {attr.residuals.maxAbsLogit.toExponential(2)}. Critical exponent flips explode;
              mild flips and novel hardness often do not.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
