import type { CauseLabel, Signature } from "@/lib/aria";
import { Badge } from "@/components/ui/badge";

const CAUSE_COPY: Record<CauseLabel, { label: string; tone: string }> = {
  "known-clean": { label: "known, clean", tone: "bg-slate-500/20 text-slate-200" },
  "novel-clean": { label: "novel, preserve", tone: "bg-emerald-500/20 text-emerald-300" },
  sensor: { label: "sensor / channel", tone: "bg-amber-500/20 text-amber-300" },
  compute: { label: "execution fault", tone: "bg-rose-500/20 text-rose-300" },
};

export function CauseBadge({ cause }: { cause: CauseLabel | "anomaly" | "compound" }) {
  if (cause === "anomaly") {
    return <Badge className="bg-violet-500/20 text-violet-200">anomaly (unnamed)</Badge>;
  }
  if (cause === "compound") {
    return <Badge className="bg-orange-500/20 text-orange-200">compound (unisolable)</Badge>;
  }
  const c = CAUSE_COPY[cause];
  return <Badge className={c.tone}>{c.label}</Badge>;
}

export function SignatureBits({ signature }: { signature: Signature }) {
  const bits = [
    { k: "S_phys", v: signature.S_phys, hint: "detector parity" },
    { k: "S_morph", v: signature.S_morph, hint: "shared-profile novelty" },
    { k: "U_det", v: signature.U_det, hint: "leave-one-detector unstable" },
    { k: "U_arith", v: signature.U_arith, hint: "checksum disagrees" },
  ];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {bits.map((b) => (
        <div
          key={b.k}
          className={`rounded-lg border px-2 py-2 font-mono text-[11px] ${
            b.v ? "border-primary/40 bg-primary/10 text-primary" : "border-white/10 text-muted-foreground"
          }`}
        >
          <div className="text-[10px] tracking-wide uppercase">{b.k}</div>
          <div className="text-base">{b.v ? "1" : "0"}</div>
          <div className="text-[10px] opacity-80">{b.hint}</div>
        </div>
      ))}
    </div>
  );
}
