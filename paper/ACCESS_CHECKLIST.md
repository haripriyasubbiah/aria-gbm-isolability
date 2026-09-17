# IEEE Access submission checklist (course manuscript)

Official list: https://ieeeaccess.ieee.org/authors/submission-guidelines/

This repo is prepared as an IEEE Access **research article**. Filling the template does not guarantee acceptance. It removes the bounce-to-draft reasons we control.

| # | Requirement | Status in this repo |
|---|---|---|
| 1–2 | Double-column IEEE Access / IEEEtran template; Word or LaTeX + matching PDF | `paper/ieeeaccess.tex` (IEEEtran). On Overleaf, paste into the official Access kit if mam requires `ieeeaccess.cls`. |
| 3 | Author list complete; no anonymous “et al.” in the byline | Rithik, 24BCE5415, VIT Chennai. Add co-authors before a real IEEE upload. |
| 4–7 | Figures/tables readable in print; captions; IEEE-style references | Tables 1–5 in the tex; numbered `\bibitem`s. |
| 8 | Not submitted elsewhere at the same time | Course file only until you decide. |
| 9 | Original writing, not a cut of another paper | Isolability dictionary claim; no copied ECG SPRT text. |
| 10 | Conclusions match the data | Abstract reports $89.3\pm0.3\%$, sensor $77.8\%$, compound unisolable $98.8\%$ named compute. No $100\%$ four-way claim. |
| 11 | Keywords (3–8) | Six keywords in the tex. |
| 12 | Abstract $\le$ 250 words | One paragraph; states that we do **not** propose a new sequential test. |

## Reviewer bar we wrote to

IEEE Access reviewers are **not** required to demand high novelty. They are required to demand: distinct from prior work, technically sound, enough experiment detail, conclusions supported, English, references, in scope.

We made the paper distinct by naming ProGIP and Gavarini and saying what they do not do. We made it sound by printing ablations and the double-fault failure.

## Reproduce every number in the paper

```bash
npm install
npm run test:paper    # writes paper/results.json and paper/results.tex
npm run test:aria     # single-seed confusion (Table 3)
```

Catalog totals were refreshed from HEASARC `fermigtrig` on 17 Sep 2026 (12981 triggers).

## What this still is not

- Not downloaded Fermi TTE. The paper says so in the title footnote and the limitations section.
- Not a new SPRT / e-process.
- Not a guaranteed IEEE accept. A reviewer can still say “simulator only.” The next empirical upgrade is public CTIME/TTE under the same residuals.
