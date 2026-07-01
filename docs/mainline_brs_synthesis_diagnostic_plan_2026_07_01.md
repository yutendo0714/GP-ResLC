# Mainline BRS / Omitted-Residual Synthesis Diagnostic Plan - 2026-07-01

This memo defines the next mechanism checks after the current CLIC2020 package
validation.  These are not rho/loss-weight sweeps.  They test whether the GLC
`y` residual stream contains symbols that can be removed and either left to the
prior mean or recovered by decoder-computable synthesis.

## Objective

Measure three things under the real serialized codec:

1. how much bpp is saved by omitting selected quantized residual symbols
   `y_q`;
2. how much DISTS/LPIPS/FID damage that omission causes;
3. whether deterministic no-side residual synthesis improves quality at the
   same bpp as the omission run.

This directly tests the GP-ResLC thesis:

```text
send only residual/control information that pretrained GLC cannot recover from
z_hat, q, context, and the generator.
```

## Anchor

Use the current SafeRDO fixed-z anchor:

```text
experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt
predictor_param_mode=stage_residual_entropy_quant_gate
predictor_delta_bound=0.3
z_entropy_mode=fixed
```

Use z entropy only after the mechanism behavior is understood.  Otherwise the
small zero-distortion z gain can mask the y-residual question.

## Fast Dataset Ladder

1. Kodak8 or Kodak24:
   quick mechanism check and visual inspection.
2. DIV2K validation:
   larger natural-image confirmation.
3. CLIC2020 test 428:
   only after a candidate is clearly promising.

## Diagnostic Matrix

The full matrix can be generated and run with:

```bash
.venv/bin/python scripts/run_brs_diagnostic_matrix.py \
  --input data_splits/eval/kodak24 \
  --out_root experiments/real_codec/kodak_brs_synthesis_diag_20260701 \
  --resume
```

Use `--dry_run` to inspect commands before launching, and `--only` to run a
subset of candidates.

### A. Hard residual omission

Omit quantized residual symbols, not the prior mean:

```bash
.venv/bin/python scripts/evaluate_real_codec.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --ckpt experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt \
  --input data_splits/eval/kodak24 \
  --out experiments/real_codec/kodak_stage3_yq_omit_zero \
  --q_indexes 0 1 2 3 \
  --predictor_param_mode stage_residual_entropy_quant_gate \
  --predictor_delta_bound 0.3 \
  --suppress_yq_stages 3 \
  --omitted_residual_mode zero \
  --z_entropy_mode fixed \
  --device cuda
```

Then repeat with:

- `--suppress_yq_stages 2 3`
- `--suppress_yq_stages 1 2 3`

Interpretation:

- If stage 3 omission saves meaningful bpp with small DISTS/LPIPS damage, BRS
  is viable.
- If stage 3 collapses quality, BRS must be block-level, selective, or paired
  with synthesis/control.

### B. Rho-threshold omission

Omit only regions that the current decoder-computable gate already treats as
safe:

```bash
--suppress_yq_stages 3 --suppress_rho_threshold 1.20
```

This is a bridge between the current rho lead and hard BRS.

### C. Same-bpp deterministic residual synthesis

Use the exact same omitted positions as A/B, but fill omitted quantized residuals
with decoder-computable pseudo-residuals:

```bash
--omitted_residual_mode hash_gaussian_clipped \
--omitted_residual_scale 0.25 \
--omitted_residual_clip 1.0
```

Also test a signed low-amplitude version:

```bash
--omitted_residual_mode hash_rademacher \
--omitted_residual_scale 0.25
```

All randomness is deterministic from tensor coordinates, q, and stage index, so
no image-specific seed or side information is hidden.

Interpretation:

- If synthesis improves DISTS/FID/LPIPS at identical bpp, the high-upside path
  is learned omitted-residual synthesis.
- If it improves FID but harms DISTS/LPIPS, synthesis needs a safe-to-synthesize
  mask or counted control.
- If it fails, focus next on residual entropy modeling rather than synthesis.

## Metrics

For Kodak:

```bash
.venv/bin/python scripts/evaluate_recon_grid.py \
  --orig data_splits/eval/kodak24 \
  --run Anchor=experiments/real_codec/kodak_stage_safe_rdo_current_fixed \
  --run Candidate=experiments/real_codec/kodak_stage3_yq_omit_zero \
  --q_indexes 0 1 2 3 \
  --patch 64 \
  --out_json experiments/real_codec/analysis/kodak_brs_diag_metrics.json \
  --out_csv experiments/real_codec/analysis/kodak_brs_diag_metrics.csv

.venv/bin/python scripts/summarize_bd_from_metrics_csv.py \
  --csv experiments/real_codec/analysis/kodak_brs_diag_metrics.csv \
  --anchor Anchor \
  --out_csv experiments/real_codec/analysis/kodak_brs_diag_bd.csv \
  --out_md experiments/real_codec/analysis/kodak_brs_diag_bd.md
```

## Promotion Decision

Promote only if the candidate either:

- beats the current SafeRDO anchor on real-codec BD-rate; or
- clearly exposes a mechanism that should be implemented as a full module
  (learned synthesis, learned BRS mask, or learned residual/control entropy).

Do not promote a result that is only a single-q win or only an estimated-bpp win.
