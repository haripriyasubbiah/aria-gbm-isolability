import Link from "next/link";
import { CATALOG_COUNTS, CITATIONS, GAP_CLAIMS } from "@/lib/aria";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function HomePage() {
  return (
    <div className="space-y-10">
      <section className="space-y-4">
        <p className="font-mono text-[11px] tracking-[0.25em] text-primary uppercase">
          Method, not a mashup
        </p>
        <h1 className="max-w-3xl text-3xl font-medium tracking-tight sm:text-4xl">
          A surprising onboard inference is not an out-of-distribution score. It is a cause.
        </h1>
        <p className="max-w-2xl text-muted-foreground">
          Fermi-GBM class counts below are from the public HEASARC trigger catalog
          (queried {CATALOG_COUNTS.total.toLocaleString()} triggers). The algorithm does not
          concatenate OOD, SelfChecker, and bit-flip monitors. It builds four residuals with
          disjoint sensitivity, then reads their 0/1 signature.
        </p>
        <div className="flex flex-wrap gap-2">
          <Link href="/lab">
            <Button>Open the event lab</Button>
          </Link>
          <Link href="/method">
            <Button variant="outline">Read the isolability rule</Button>
          </Link>
        </div>
      </section>

      <section className="grid gap-3 sm:grid-cols-4">
        {(
          [
            ["GRB", CATALOG_COUNTS.GRB, "known class"],
            ["SFLARE", CATALOG_COUNTS.SFLARE, "known class"],
            ["TGF", CATALOG_COUNTS.TGF, "held-out novelty"],
            ["SGR", CATALOG_COUNTS.SGR, "known class"],
          ] as const
        ).map(([name, n, role]) => (
          <Card key={name}>
            <CardHeader>
              <CardTitle className="font-mono text-xs tracking-widest uppercase">{name}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl tabular-nums">{n.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground">{role}</div>
            </CardContent>
          </Card>
        ))}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">What the literature actually leaves open</h2>
        <div className="grid gap-3 md:grid-cols-2">
          {GAP_CLAIMS.map((claim) => (
            <Card key={claim}>
              <CardContent className="pt-1 text-sm leading-relaxed text-muted-foreground">
                {claim}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Closest papers — read as limitations, not decorations</h2>
        <ul className="space-y-2 text-sm">
          {CITATIONS.slice(0, 8).map((c) => (
            <li key={c.id} className="border-b border-white/5 pb-2">
              <a href={c.href} className="text-primary hover:underline" target="_blank" rel="noreferrer">
                {c.authors} ({c.year}). {c.title}
              </a>
              <span className="text-muted-foreground"> — {c.venue}. {c.role}</span>
            </li>
          ))}
        </ul>
        <p className="text-xs text-muted-foreground">
          Local particles (LOCLPAR {CATALOG_COUNTS.LOCLPAR.toLocaleString()}, DISTPAR{" "}
          {CATALOG_COUNTS.DISTPAR}) are real single-detector artifacts in the same catalog. The
          sensor-spike model is a software proxy for that class, not a radiation-beam test.
        </p>
      </section>
    </div>
  );
}
