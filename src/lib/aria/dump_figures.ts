/**
 * Emit TikZ scatter figures from one training seed.
 *   npx tsx src/lib/aria/dump_figures.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { runTrials, trainSystem } from "./experiment";
import type { CauseLabel } from "./types";

const CAUSES: CauseLabel[] = ["known-clean", "novel-clean", "sensor", "compute"];
const MARK: Record<CauseLabel, string> = {
  "known-clean": "mark=*, mark size=1.1pt",
  "novel-clean": "mark=triangle*, mark size=1.3pt",
  sensor: "mark=square*, mark size=1.2pt",
  compute: "mark=diamond*, mark size=1.3pt",
};
const LEGEND: Record<CauseLabel, string> = {
  "known-clean": "known-clean",
  "novel-clean": "novel-clean",
  sensor: "sensor",
  compute: "compute",
};

function clip(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function coords(
  rows: { truth: string; residuals: { physics: number; exec: number; energy: number; msp: number } }[],
  cause: CauseLabel,
  xy: (r: (typeof rows)[number]) => [number, number],
  cap: number,
): string {
  const pts = rows
    .filter((r) => r.truth === cause)
    .slice(0, cap)
    .map((r) => {
      const [x, y] = xy(r);
      return `(${x.toFixed(4)},${y.toFixed(4)})`;
    });
  return pts.join(" ");
}

function main() {
  const system = trainSystem(7);
  const rows = runTrials(system, 123, 36);
  const paperDir = join(dirname(fileURLToPath(import.meta.url)), "../../../paper");
  mkdirSync(paperDir, { recursive: true });

  const physExecPlots = CAUSES.map((c) => {
    const body = coords(rows, c, (r) => [clip(r.residuals.physics, 0, 1), clip(Math.log10(r.residuals.exec + 1e-8), -8, 0)], 28);
    return `\\addplot[${MARK[c]}] coordinates { ${body} };\\addlegendentry{${LEGEND[c]}}`;
  }).join("\n");

  const energyPlots = CAUSES.map((c) => {
    const body = coords(rows, c, (r) => [clip(r.residuals.msp, 0, 1), clip(r.residuals.energy, -8, 12)], 28);
    return `\\addplot[${MARK[c]}] coordinates { ${body} };\\addlegendentry{${LEGEND[c]}}`;
  }).join("\n");

  const residualTex = `\\documentclass[tikz,border=6pt]{standalone}
\\usepackage{times}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
  width=8.6cm, height=6.2cm,
  xlabel={physics residual $r_{\\mathrm{phys}}$},
  ylabel={$\\log_{10}(r_{\\mathrm{exec}}+10^{-8})$},
  xmin=-0.02, xmax=1.02,
  ymin=-8.2, ymax=0.2,
  legend style={at={(0.02,0.98)}, anchor=north west, font=\\scriptsize, draw=none, fill=white, fill opacity=0.85},
  grid=major, major grid style={dotted, gray!45},
  tick label style={font=\\small}, label style={font=\\small},
]
${physExecPlots}
\\end{axis}
\\end{tikzpicture}
\\end{document}
`;

  const energyTex = `\\documentclass[tikz,border=6pt]{standalone}
\\usepackage{times}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
  width=8.6cm, height=6.2cm,
  xlabel={maximum softmax probability},
  ylabel={softmax energy},
  xmin=-0.02, xmax=1.02,
  legend style={at={(0.98,0.98)}, anchor=north east, font=\\scriptsize, draw=none, fill=white, fill opacity=0.85},
  grid=major, major grid style={dotted, gray!45},
  tick label style={font=\\small}, label style={font=\\small},
]
${energyPlots}
\\end{axis}
\\end{tikzpicture}
\\end{document}
`;

  writeFileSync(join(paperDir, "fig_residuals.tex"), residualTex);
  writeFileSync(join(paperDir, "fig_energy.tex"), energyTex);
  console.log("Wrote paper/fig_residuals.tex and paper/fig_energy.tex from seed 7 / eval 123");
}

main();
