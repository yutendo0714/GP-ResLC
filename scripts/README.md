# Scripts Index

The scripts are kept at one level so old experiment commands remain reproducible.
Use this index instead of guessing from filenames.

Current research policy: use official pretrained GLC as the fixed base, stop
scratch as an active path, and prioritize full residual/control/entropy
mechanisms over rho-only sweeps. See `docs/research_priority.md`.

## Paper-Facing Real Codec

- `evaluate_real_codec.py`: primary real-codec evaluator for GLC and GP-ResLC
  pretrained extensions. Reports byte-backed bpp and encode/decode time.
- `build_clic2020_test_package.py`: builds the canonical CLIC2020 428-image
  combined symlink set and merged real-codec result trees.
- `compare_official_curves.py`: compares local curves against graph-extracted
  official GLC curves.
- `build_vcip_key_tables.py`: builds paper-facing tables from real-codec metrics.

## Pretrained-GLC Research Path

- `train_v1.py`: older prior-predictor-only path.
- `train_v2.py`: q-conditioned prior/gate path on frozen pretrained GLC.
- `test_v1.py`, `test_v2.py`: estimated/proxy-style reconstruction dumpers.
  Keep them for debugging, but paper claims should use `evaluate_real_codec.py`.
- `smoke_gp_reslc.py`: fast sanity checks for the pretrained overlay.

## GLC-Latent / Residual Research Path

- `train_glc_latent_residual.py`: high-upside residual branch that predicts
  frozen GLC/VQGAN latent content and sends only selected residual information.
- `evaluate_glc_latent_residual_fullres_realcodec.py`: full-resolution
  byte-backed evaluator for the GLC-latent residual branch.
- `evaluate_glc_latent_residual_realcodec.py`: older center/development
  real-codec bridge.
- `evaluate_glc_latent_residual.py`: deterministic center-crop/proxy evaluator.

## Scratch Research Path

- `train_scratch_stage_a.py`: semantic VQ autoencoder Stage A.
- `train_scratch_stage_a_adv.py`: adversarial Stage-A fine-tuning.
- `train_scratch_stage_b.py`: semantic-conditioned residual coding Stage B.
- `evaluate_scratch_stage_a.py`, `evaluate_scratch_stage_b.py`: deterministic
  scratch checkpoint evaluators.

## Analysis And Figures

- `analyze_gate_correlations.py`, `analyze_gate_maps.py`,
  `analyze_stage_quant_gate_sensitivity.py`: feature/gate diagnostics.
- `run_brs_diagnostic_matrix.py`: real-codec command matrix for hard
  residual omission and same-bpp omitted-residual synthesis diagnostics.
- `evaluate_recon_grid.py`, `summarize_bd_from_metrics_csv.py`,
  `summarize_matched_metric.py`: metric aggregation.
- `make_qualitative_grid.py`, `make_rho_overlay_grid.py`,
  `rank_qualitative_candidates.py`, `plot_metric_curves.py`: figures and
  qualitative evidence.
- `audit_glc_protocol.py`: protocol/data-count checks.

## Legacy Orchestration

- `eval_metrics.py`, `make_comparison.py`, `make_curves.py`, `run_ablation.py`:
  useful older orchestration utilities. Prefer the real-codec pipeline for new
  paper-facing comparisons.
