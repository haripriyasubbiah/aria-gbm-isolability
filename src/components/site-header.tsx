"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Gap" },
  { href: "/method", label: "Algorithm" },
  { href: "/lab", label: "Event lab" },
  { href: "/experiment", label: "Isolability trial" },
];

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-[oklch(0.16_0.02_248_/0.92)] backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <Link href="/" className="flex items-baseline gap-2">
          <span className="font-mono text-[11px] tracking-[0.22em] text-primary uppercase">
            ARIA
          </span>
          <span className="text-sm text-muted-foreground">Fault or discovery</span>
        </Link>
        <nav className="flex flex-wrap gap-1">
          {LINKS.map((l) => {
            const active = path === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  "rounded-md px-2.5 py-1 font-mono text-[12px] tracking-wide uppercase",
                  active
                    ? "bg-primary/15 text-primary"
                    : "text-muted-foreground hover:bg-white/5 hover:text-foreground",
                )}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
