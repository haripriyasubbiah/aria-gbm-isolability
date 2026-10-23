# Journal shortlist and submission checklists

Mam asked for venues that are **Q1 or Q2**, **Impact Factor < 5**, **fully open access**, **lower cost**, and **not predatory**. The Elsevier Journal Finder query used the paper abstract (gold OA + subscription journals, recommend-stem). Metrics below are the values shown on the publisher pages in September 2026. Clarivate JCR numbers move every June; **re-check IF on ScienceDirect / IEEE Access bibliometrics the week you submit**.

India-only authorship (VIT Chennai) is treated as **World Bank lower-middle income**. Elsevier’s Geographical Pricing for Open Access (GPOA) then bills **20% of the list APC** on participating gold-OA journals. That is the cost lever. IEEE Access does **not** use GPOA.

## Pick (in this order)

| Rank | Journal | Why it fits | Skip if |
|---|---|---|---|
| **1** | **Array** (Elsevier, ISSN 2590-0056) | Gold OA, ScienceDirect IF **4.5**, CiteScore **10.3**, JCR Q1 in CS Theory & Methods (ESCI). Broad CS including AI/ML, signal processing, scientific computing. Abstract-shaped methods paper is in scope. **GPOA: list $2,250 → ~$450 for India-only authors.** DOAJ listed. | Mam insists on an IEEE logo. |
| **2** | **Systems and Soft Computing** (Elsevier, ISSN 2772-9419) | Gold OA, IF **3.6**, JCR **Q2** (AI; interdisciplinary applications). Companion to *Applied Soft Computing*. Neural networks and real-world systems are in the aims. **GPOA: list ~$2,280 → ~$456.** | Mam wants Q1 specifically. |
| **3** | **IEEE Access** | Gold OA, official IF **4.2**, JCR **Q2** (Electrical & Electronic Engineering). Mam already knows the format. APC **$2,160** (IEEE member discounts 5–20%; **no India geo-price**). | Cost is the constraint. |
| **Backup** | **Intelligent Systems with Applications (ISWA)** | Best topical fit. Gold OA, ScienceDirect currently shows IF **4.3**; some 2026 third-party tables show **5.5**. Scopus Q1. GPOA list **$2,300 → ~$460**. | If mam’s library catalog lists IF ≥ 5, it fails the brief. Confirm on ScienceDirect the day you submit. |

**Do not use for this brief**

- *Results in Engineering* — Elsevier gold OA but IF ~7.9 (fails IF < 5).
- *Machine Learning with Applications* — excellent scope and GPOA, but ScienceDirect/JCR listings disagree (4.9 vs 6.1). If the live JCR is 6.1 it fails IF < 5.
- *e-Prime* — cheap (list ~$1,400, GPOA ~$280) and Scopus Q1, **no official JIF** on ScienceDirect. Mam asked for IF < 5, which implies a journal that *has* an IF.
- *Measurement: Sensors* — gold OA, GPOA, Scopus Q1, **no JIF** on the journal homepage.
- *SoftwareX* — decent Elsevier gold OA (IF 2.4, Q2, $1,560) but it is an **original software publication**. The methods paper would have to be recast around the GitHub artifact. List APC is also **higher than Array-with-GPOA**.
- MDPI *Sensors* and similar high-volume MDPI titles — not on this shortlist (predatory-adjacent reputation; mam asked for decent venues).

**Recommendation for the course draft:** keep writing in IEEE double-column for mam, and treat **Array** as the Elsevier target if she later pays an APC. Array is the only venue that is simultaneously Elsevier, fully OA, Q1, IF < 5, DOAJ, and ~$450 under GPOA.

## Checklists walked before locking the draft

### Array / ISWA / Systems and Soft Computing (shared Elsevier gold-OA checklist)

Walked from Elsevier author policies and the *Machine Learning with Applications* guide (same Elsevier Editorial Manager skeleton Array uses). Confirm the live [Array guide for authors](https://www.sciencedirect.com/journal/array/publish/guide-for-authors) on submission day.

| Item | What they want | This draft |
|---|---|---|
| Article type | Full-length research article | Yes |
| Originality | Not published / not under review elsewhere | Course manuscript only until mam picks a journal |
| Abstract | Concise; no citations | ≤250 words, no cites |
| Keywords | 3–10 | Present |
| **Highlights** | 3–5 bullets, **≤85 characters each** including spaces | `paper/highlights.txt` |
| **Graphical abstract** | One figure that can stand alone | Fig. 1 (method lookup) |
| Numbered sections | IMRaD: intro, related, method, experiments, results, limits, conclusion | Yes |
| Figures | Editable or high-res; **every figure cited in the text** | 7 figures, each cited |
| Tables | Caption above; **every table cited in the text** | 8 tables, each cited as “Table N” |
| References | Every in-text cite in the list and vice versa; no retracted items | Expanded list (~45) |
| Data | Public repository encouraged | HEASARC + GitHub cubes |
| Code | Not mandatory; we still ship it | `npm run test:paper` / `test:gbm` |
| CRediT | Author contribution statement | Single-author CRediT in the Elsevier tex |
| Competing interests | Declaration required | None |
| Funding | Statement required | No specific grant |
| Generative AI | Disclose only if used in manuscript preparation | **Omitted in the mam PDF.** Add Elsevier’s AI paragraph only if she submits. |
| Corresponding author | One person, ORCID in Editorial Manager | S. Haripriya; create ORCID before a real submit |
| APC | Author-pays gold OA; GPOA applied at checkout | India-only → ~20% of list on GPOA titles |
| Simultaneous submit | Forbidden | Submit to **one** journal |

Elsevier LaTeX: `elsarticle` is accepted. CAS templates are optional. Upload `.tex` + PDFs of figures, not a single compiled blob as the only source.

### IEEE Access (if mam stays with IEEE)

Walked from the live [IEEE Access Submission Guidelines](https://ieeeaccess.ieee.org/guide-for-authors/submission-guidelines/).

| Item | What they want | This draft |
|---|---|---|
| Template | Official IEEE Access two-column template; Word/LaTeX **and** matching PDF, <40 MB | Course PDF uses plain `IEEEtran` (mam asked to drop Access logos). Swap `ieeeaccess.cls` only for a real portal upload |
| Manuscript type | Research Article | Yes |
| Keywords | 3–10 at submission | Yes |
| ORCID | Submitting author, public profile | Create before portal upload |
| Short biography | Required for every author | Present |
| Grammar | Poor grammar is desk-rejected | Edited |
| AI text | Disclose in acknowledgements; cite the system | **Omitted for mam.** Add if submitting to Access |
| Page count | No hard limit; **keep under 20 pages** | Target 8–10 |
| References | Accurate, no retracted, no padding | Expanded, all cited |
| Duplicate submit | Forbidden | One journal |
| Lena image | Forbidden | Not used |
| APC | $2,160 + tax; no GPOA | IEEE member discount only |

### What the draft still is

This repository is a **course manuscript** (Embedded Systems DA1). It is formatted so mam can read it, and so a later Array or IEEE Access upload does not require rewriting the science. Do not upload to two journals at once. Do not pay an APC until mam names the venue.
