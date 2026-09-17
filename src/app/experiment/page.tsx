"use client";

import { useMemo, useState } from "react";
import {
  CAUSES,
  accuracy,
  confusion,
  energySplit,
  getSystem,
  runTrials,
} from "@/lib/aria";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const PRED = [...CAUSES, "anomaly"] as const;

function Matrix({
  title,
  matrix,
}: {
  title: string;
  matrix: Record<string, Record<string, number>>;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-left font-mono text-[11px]">
          <thead>
            <tr className="text-muted-foreground">
              <th className="py-1 pr-2">true \\ pred</th>
              {PRED.map((p) => (
                <th key={p} className="px-1 py-1">
                  {p.replace("-clean", "")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CAUSES.map((t) => (
              <tr key={t} className="border-t border-white/10">
                <td className="py-1 pr-2 text-muted-foreground">{t}</td>
                {PRED.map((p) => {
                  const v = matrix[t]?.[p] ?? 0;
                  const rowSum = PRED.reduce((s, k) => s + (matrix[t]?.[k] ?? 0), 0);
                  const hit = p === t;
                  return (
                    <td
                      key={p}
                      className={`px-1 py-1 tabular-nums ${hit ? "text-primary" : ""}`}
                    >
                      {v}
                      <span className="text-muted-foreground">
                        {rowSum ? ` ${(100 * v / rowSum).toFixed(0)}%` : ""}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

export default function ExperimentPage() {
  const [n, setN] = useState(28);
  const [seed, setSeed] = useState(123);
  const system = useMemo(() => getSystem(), []);
  const rows = useMemo(() => runTrials(system, seed, n), [system, seed, n]);
  const aria = confusion(rows, "aria");
  const range = confusion(rows, "range");
  const split = energySplit(rows);

  return (
    <div className="space-y-6">
      <div>
        <p className="font-mono text-[11px] tracking-[0.25em] text-primary uppercase">
          Isolability trial
        </p>
        <h1 className="text-3xl font-medium tracking-tight">Does the signature stay unique?</h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Known classes GRB / SGR / SFLARE, held-out TGF, single-detector spikes, and mild/critical
          bit-flips. Thresholds are fit on known-clean plus injected faults only — not on TGF.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={() => setN((x) => (x === 28 ? 40 : 28))}>
          {n} per cell
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            setSeed((s) => s + 7);
          }}
        >
          new seed {seed}
        </Button>
        <span className="font-mono text-xs text-muted-foreground">
          ARIA acc {(100 * accuracy(rows, "aria")).toFixed(1)}% · range-check{" "}
          {(100 * accuracy(rows, "range")).toFixed(1)}%
        </span>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Matrix title="ARIA (4-way cause)" matrix={aria} />
        <Matrix title="Logit range check" matrix={range} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Energy / MSP as a fault detector — the confound</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            Fraction called “anomaly” by energy/MSP: novel{" "}
            <span className="font-mono text-foreground">{(100 * split.novelFlagged).toFixed(0)}%</span>
            , compute{" "}
            <span className="font-mono text-foreground">{(100 * split.computeFlagged).toFixed(0)}%</span>
            , known{" "}
            <span className="font-mono text-foreground">{(100 * split.knownFlagged).toFixed(0)}%</span>
            .
          </p>
          <p>
            If that reject is treated as a fault, every held-out transient is discarded. If it is
            treated as a discovery, silent bit-flips that barely move MSP are preserved. The score
            cannot name the cause. That is the Gavarini setting, and it is why a residual signature
            is the method rather than another threshold on the same logits.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
