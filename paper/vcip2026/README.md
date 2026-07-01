# VCIP 2026 submission — Safe Adaptive Quantization (SAQ)

Built on the project's best result (the Safe-RDO gate,
`experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`).

## Build
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Produces `main.pdf` (5 pages: 4 content + 1 references-only, IEEE two-column
conference format, as required by the VCIP 2026 CFP).

## Files
- `main.tex` — paper source (anonymized for double-blind review).
- `refs.bib` — references.
- `figures/` — all figures (PDF).
- `make_rd_curves.py` — Fig. 2 (rate–perception curves) from the stored
  evaluation CSVs in `experiments/real_codec/`.
- `make_qualitative.py` — Fig. 3 (Original / GLC / SAQ reconstruction).
- `make_gate_analysis.py` — Fig. 4 (gate map) + the gate-vs-statistics
  correlations reported in Sec. IV-D, computed on the Kodak set.

## Before submission
- This version is **anonymized**. Add author names/affiliations only for the
  camera-ready (the submission must stay double-blind).
- All headline numbers come from `experiments/real_codec/*safe_rdo_gate_from_sb03_2000*`.
