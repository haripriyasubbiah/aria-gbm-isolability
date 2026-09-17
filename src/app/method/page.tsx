import { SIGNATURE_MATRIX } from "@/lib/aria";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function MethodPage() {
  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <p className="font-mono text-[11px] tracking-[0.25em] text-primary uppercase">ARIA</p>
        <h1 className="text-3xl font-medium tracking-tight">
          Attribution by Residual-Invariance Analysis
        </h1>
        <p className="max-w-2xl text-muted-foreground">
          The contribution is a structured residual with a unique incidence pattern — Gertler
          isolability — over a fault dictionary that classical FDI never had: open-set scientific
          novelty and neural execution corruption. Scores are not concatenated. They are constructed
          to be selectively blind.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Four residuals, four invariances</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm leading-relaxed">
          <p>
            <span className="font-mono text-primary">r_phys</span> — leave-one-detector-out cosine
            against a median lightcurve formed from the other bright detectors. A true point source
            (GRB, SGR, flare, or an unknown family) stays rank-consistent. A single-detector spike
            does not. Computed from counts. Blind to the neural weights.
          </p>
          <p>
            <span className="font-mono text-primary">r_morph</span> — distance of that shared
            profile’s features (duration, hardness, variability) to known-class centroids. Blind to
            a corrupted classifier. High for a held-out TGF-like pulse that is still a point source.
          </p>
          <p>
            <span className="font-mono text-primary">U_det</span> — the physics residual collapses
            when the worst detector is dropped. Localizes the channel.
          </p>
          <p>
            <span className="font-mono text-primary">U_arith</span> — stored row-sum checksum of the
            linear head versus the live matmul (ABFT). Independent of whether the input is novel.
            This is why a bit-flip cannot hide inside an OOD score.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Isolability table (single cause)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">Cause</th>
                  <th className="py-2 pr-3">S_phys S_morph U_det U_arith</th>
                  <th className="py-2">Why that column is unique</th>
                </tr>
              </thead>
              <tbody>
                {SIGNATURE_MATRIX.map((row) => (
                  <tr key={row.cause} className="border-t border-white/10">
                    <td className="py-2 pr-3 text-foreground">{row.cause}</td>
                    <td className="py-2 pr-3 text-primary">{row.pattern}</td>
                    <td className="py-2 font-sans text-muted-foreground">{row.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Decision is a lookup, not a trained 6-way head: checksum first, then detector parity,
            then morphology, else known. Compound sensor+compute is unisolable by design and is
            out of scope for this slice.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>What this is not</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>Not ProGIP. Range checks on logits still live in activation-magnitude space.</p>
          <p>Not Gavarini. Open-set scores used as fault detectors will call every held-out TGF a fault.</p>
          <p>Not Dr. DNA. Marginal neuron histograms move for OOD and for SDC.</p>
          <p>Not another Fermi classifier. John et al. already classify GRB/TGF/SGR/flare and flag outliers.</p>
          <p>
            The public-data claim: Fermi-GBM trigger types and the NaI cosine-response geometry are
            real. Lightcurves here are physics-faithful simulations of that geometry, so the
            isolability test can run without a TTE download. Swap in HEASARC TTE without changing
            the residual algebra.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
