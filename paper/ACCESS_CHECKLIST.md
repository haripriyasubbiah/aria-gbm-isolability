# Course checklist (mam PDF, IEEE double-column)

This zip is a **course manuscript** in plain `IEEEtran`, not an IEEE Author Portal or Elsevier Editorial Manager upload. Journal pick and publisher checklists: `paper/JOURNAL.md`.

| Need | Status |
|---|---|
| Double-column IEEE look | **OK** (`IEEEtran`, no Access logos) |
| Author S. Haripriya, 24BCE5415 | **OK** in byline, header, biography (Her) |
| Runnable code + public data | **OK** GitHub cubes + `npm run test:paper` / `test:gbm` / `test:figures` |
| Tables (each cited in the text) | **8** — related-work comparison, dictionary, catalog, simulator, confusion, ablation, real CTIME, slice trend |
| Figures (graphical results) | **9** — method lookup, detector cubes, residual scatter, energy/MSP confound, isolation bars, confusion heatmap, ablation bars, sim vs real, CTIME trend |
| Stepwise algorithm | **OK** Algorithm 1 (ARIA lookup) |
| References | **~45** (OOD, SDC, ABFT, FDI, GBM catalogs) |
| Honest limits | **OK** — TGF never named compute; TGF not isolated as novel on CTIME; sensor 60% real vs 77.8% sim |
| AI disclosure | **Omitted on purpose.** This PDF is for mam. Add IEEE/Elsevier AI statements only if submitting. |
| ORCID / ScholarOne / opposed reviewers | **N/A** for course |

Recommended journal if an APC is later paid: **Elsevier Array** (gold OA, IF 4.5, Q1, GPOA ~$450 for India-only authors). IEEE Access remains the IEEE-brand backup ($2160, no geo-price).

Mam line: ARIA is an isolability **method** (selectively blind residuals + lookup), not a new sequential test and not a mashup of OOD + SDC scores.
