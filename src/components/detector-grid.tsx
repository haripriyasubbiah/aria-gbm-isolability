import { N_DET, N_TIME } from "@/lib/aria";

function color(v: number, max: number): string {
  const t = Math.max(0, Math.min(1, v / (max || 1)));
  const r = Math.round(8 + t * 232);
  const g = Math.round(18 + t * 210);
  const b = Math.round(28 + t * 80);
  return `rgb(${r},${g},${b})`;
}

export function DetectorGrid({
  counts,
  highlight,
}: {
  counts: number[][];
  highlight?: number | null;
}) {
  const max = Math.max(...counts.flat(), 1);
  return (
    <div className="overflow-x-auto">
      <div
        className="grid gap-px"
        style={{ gridTemplateColumns: `auto repeat(${N_TIME}, minmax(3px, 1fr))` }}
      >
        <div />
        {Array.from({ length: N_TIME }, (_, t) => (
          <div key={t} className="h-3" />
        ))}
        {Array.from({ length: N_DET }, (_, d) => (
          <div key={`r-${d}`} className="contents">
            <div
              className={`pr-2 text-right font-mono text-[10px] ${
                highlight === d ? "text-amber-400" : "text-muted-foreground"
              }`}
            >
              n{d}
            </div>
            {Array.from({ length: N_TIME }, (_, t) => (
              <div
                key={`${d}-${t}`}
                title={`det ${d}, t ${t}: ${counts[d]![t]!.toFixed(0)}`}
                className="h-3 min-w-[3px] rounded-[1px]"
                style={{ background: color(counts[d]![t]!, max) }}
              />
            ))}
          </div>
        ))}
      </div>
      <p className="mt-2 font-mono text-[10px] text-muted-foreground">
        8 NaI-like detectors × 64 time bins · brighter = more counts
      </p>
    </div>
  );
}
