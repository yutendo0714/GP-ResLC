# Current VCIP Status for GP-ResLC

Last updated: 2026-06-20 JST

## Core claim

GP-ResLC targets ultra-low-bitrate perceptual image compression by not transmitting residual information that the decoder-side generator can recover from the already transmitted GLC semantic/hyper code. The current short-track implementation realizes this as a zero-extra-bit, decoder-recomputable quantization gate `rho(z_hat, q)` that coarsens latent residuals only where reconstruction is expected to be generator-predictable.

The short-track paper should be framed as R-P oriented. The full GP-ResLC goal remains R-D-P, but the current strongest evidence is DISTS/FID-oriented perceptual rate reduction under real serialized codec accounting.


## Real Codec Evaluation Update

Paper-facing rate evaluation should now use actual serialized bitstreams rather than estimated likelihood bpp. The real codec implementation is in `gp_reslc/real_codec.py`, with the runner `scripts/evaluate_real_codec.py` and protocol note `docs/real_codec_protocol.md`.

Real-codec evaluation is complete for Kodak, DIV2K validation, and the full CLIC2020 test set for GLC and the lead `rho1.16` checkpoint. CLIC2020 test is the 428-image professional+mobile union; it yields exactly 28,650 shifted 256-patches. The real decoder matches the previous forward reconstruction exactly in smoke checks (`max_abs=0.000e+00`), so the change affects rate/timing measurement, not reconstruction quality. The paper-facing package now uses these real-codec artifacts as source-of-truth.

Real-bpp headline: CLIC2020 test DISTS/FID BD-rate `-10.28% / -7.30%`, DIV2K `-10.79% / -5.61%`, Kodak `-4.47% / -1.70%`. Matched-DISTS bpp deltas are CLIC2020 test `-10.26%`, DIV2K `-10.27%`, and Kodak `-5.45%`. Across q0-q3, serialized bpp reductions are CLIC2020 `-11.36..-7.91%`, DIV2K `-10.39..-7.16%`, and Kodak `-9.52..-7.17%`. Because `z` and header bpp are identical, the saving is entirely in the arithmetic-coded `y` stream.

## Lead checkpoints

| role | checkpoint | W&B | use |
|---|---|---|---|
| Strong R-P lead | `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt` | `a2w5fjt4` | Main paper tables/figures |
| Balanced knob | `experiments/v2_gate_send_lR10_lp4_rho14_target112_send5_all_6k/v2_final.pt` | `gbmdxyr8` | Shows rate-perception controllability |

Both use:

- `rho_max=1.4`, `rho_min=1.0`, monotone residual suppression.
- frozen GLC and frozen prior predictor.
- always-on sendability teacher.
- no transmitted side map; `rho` is recomputed from `z_hat` and q at the decoder.

## Main quantitative status

BD-rate versus GLC real codec:

| dataset | run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | rho1.16 real | -10.28% | +0.19% | -0.98% | +0.38% | -7.30% | -7.10% |
| DIV2K validation | rho1.16 real | -10.79% | -0.54% | -1.49% | -0.17% | -5.61% | -6.50% |
| Kodak | rho1.16 real | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.14% |

Per-q real bpp reduction:

| dataset | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| CLIC2020 test | -11.36% | -10.32% | -8.74% | -7.91% |
| DIV2K validation | -10.39% | -9.29% | -8.15% | -7.16% |
| Kodak | -9.52% | -9.09% | -7.93% | -7.17% |

Best headline:

- CLIC2020 test (`/dpl/clic/professional/test` + `/dpl/clic/mobile/test`, 428 images, 28,650 shifted 256-patches): DISTS BD-rate -10.28%, FID BD-rate -7.30%, matched-DISTS bpp -10.26%, matched-FID bpp -6.02%.
- DIV2K validation (`/dpl/div2k`, 100 images, 6,573 shifted 256-patches): DISTS BD-rate -10.79%, FID BD-rate -5.61%, matched-DISTS bpp -10.27%.
- Kodak: DISTS BD-rate -4.47%, FID BD-rate -1.70%, matched-DISTS bpp -5.45%, matched-FID bpp -4.40%.
- LPIPS and KID are auxiliary: LPIPS is near-neutral to slightly worse at matched bpp on CLIC/DIV2K, and KID is patch-count-sensitive.

Legacy estimated-bpp CLIC professional/mobile validation results remain useful as development evidence and mechanism analysis, but they are no longer the VCIP package source-of-truth.

## Protocol Cleanup

Evaluator-side protocol mismatches have been cleaned up. `scripts/evaluate_recon_grid.py` now seeds via `init_func()` like official GLC `test_image.py`, and all metric CSVs record FID/KID patch counts. CLIC2020 test is now exact with respect to the reported GLC/HiFiC patch count: professional test plus mobile test yields 428 images and 28,650 shifted 256-patches. DIV2K is also exact: `/dpl/div2k` yields 6,573 shifted 256-patches. See `experiments/protocol_audit/glc_protocol_audit.md` and `docs/glc_eval_protocol_audit.md`.

## Official Paper-Curve Comparison

A secondary comparison against graph-extracted official GLC paper curves has been added in `experiments/paper_assets/official_curve_comparison/`. This comparison is useful for positioning against the paper figures, while the paired local real-codec anchor remains the cleanest controlled comparison. After switching CLIC to the full 428-image test set, local real-codec GLC matches the official graph curve closely, including FID.

Key external-positioning numbers versus official graph-extracted GLC: CLIC2020 gives DISTS/FID BD-rate `-9.07% / -6.10%`; DIV2K gives `-9.62% / -4.23%`; Kodak is essentially neutral on DISTS and has no official FID/KID plot. CLIC and DIV2K now both support the external-positioning claim; LPIPS remains auxiliary because it is near-neutral to slightly worse at matched quality.

## Mechanism evidence

q3 gate-correlation analysis for the lead `rho1.16` checkpoint:

| dataset | mean rho | rho std | corr(rho, base err) | corr(rho, ours err) | corr(rho, texture var) | corr(rho, gradient) |
|---|---:|---:|---:|---:|---:|---:|
| Kodak q3 | 1.1716 | 0.0308 | -0.234 | -0.235 | -0.243 | -0.213 |
| CLIC-prof q3 | 1.1815 | 0.0307 | -0.271 | -0.272 | -0.268 | -0.290 |
| CLIC-mobile q3 | 1.1804 | 0.0311 | -0.251 | -0.252 | -0.251 | -0.262 |

High-rho regions have roughly half the local error/gradient of low-rho regions. This directly supports the mechanism: GP-ResLC coarsens predictable residual regions and protects harder residual regions.

Files:

- `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_kodak_q3.json`
- `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_clic_q3.json`
- `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_maps_kodak_q3/`
- `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_maps_clic_q3/`
- `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_clic_mobile_q3.json`
- `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_maps_clic_mobile_q3/`

## Negative/secondary ablations

| ablation | result | decision |
|---|---|---|
| Alex LPIPS training loss | Comparable DISTS, does not fix pointwise LPIPS | Keep as ablation |
| texture-free teacher | Does not fix LPIPS; weakens DISTS/FID story | Do not lead |
| baseline L1/LPIPS distillation | q3 DISTS preserved, FID worsens; no LPIPS recovery | Code kept, not lead |
| edge guard teacher | Clean q3 result, but q0-q2 still weak | Appendix/control only |
| rho_target=1.20 | More bpp savings, over-coarsens q0-q2 | Upper knob only |
| V1 direct P_theta prior correction | Stable variants too weak; unconstrained variants degrade | Not short-track lead |

## Paper assets

- Paper draft: `docs/vcip_paper_draft.md`
- Method draft: `docs/vcip_method_draft.md`
- Submission outline: `docs/vcip_submission_outline.md`
- Key paper tables: `experiments/paper_assets/vcip_key_tables.md`
- VCIP real-codec package manifest: `experiments/paper_assets/vcip_real_codec_package.md`
- Key table builder: `scripts/build_vcip_key_tables.py`
- Real codec protocol: `docs/real_codec_protocol.md`
- Real codec merged BD summary: `experiments/paper_assets/real_codec_bd_rate_summary_all.csv`
- Real codec merged matched summary: `experiments/paper_assets/real_codec_matched_metric_bpp_summary_all.csv`
- Real codec merged metrics: `experiments/paper_assets/real_codec_metrics_all.csv`
- Official paper-curve comparison: `experiments/paper_assets/official_curve_comparison/`
- Protocol audit: `experiments/protocol_audit/glc_protocol_audit.md`, `docs/glc_eval_protocol_audit.md`
- CLIC2020 test real metrics: `experiments/real_codec/clic2020_test_real_metrics.csv`
- DIV2K real metrics: `experiments/real_codec/div2k_real_metrics.csv`
- Kodak real metrics: `experiments/real_codec/kodak_real_metrics.csv`
- CLIC2020 test real curves: `experiments/paper_assets/clic2020_test_real_curves/`
- DIV2K real curves: `experiments/paper_assets/div2k_real_curves/`
- Kodak real curves: `experiments/paper_assets/kodak_real_curves/`
- Real codec payload/recon roots: `experiments/real_codec/clic2020_test_glc/`, `experiments/real_codec/clic2020_test_gp_reslc_rho116/`, `experiments/real_codec/div2k_glc/`, `experiments/real_codec/div2k_gp_reslc_rho116/`, `experiments/real_codec/kodak_glc/`, `experiments/real_codec/kodak_gp_reslc_rho116/`
- Qualitative grids: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/qual_grid_clic_q3_top4.png`, `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/qual_grid_kodak_q3_top4.png`
- Rho overlay grids: `experiments/paper_assets/clic_q3_rho_overlay_top4.png`, `experiments/paper_assets/clic_mobile_q3_rho_overlay_top4.png`, `experiments/paper_assets/kodak_q3_rho_overlay_top4.png`
- Rho overlay script: `scripts/make_rho_overlay_grid.py`

## Recommended next work

1. Convert `docs/vcip_paper_draft.md` into the target VCIP LaTeX template using `experiments/paper_assets/vcip_key_tables.md` as the source-of-truth for copied numbers.
2. Finalize figure panels and captions from the real-codec DISTS/FID curves, rho overlays, and gate-correlation table.
3. Keep DISTS/FID/KID as primary R-P evidence and report LPIPS/PSNR/MS-SSIM honestly as auxiliary diagnostics.
4. Use official graph-extracted curves only as supplementary positioning; do not let CLIC official FID override the paired local real-codec result.
5. If adding another experiment, prioritize a stage-aware four-mask residual predictor only after the short-track paper package is stable. It is the more faithful full GP-ResLC path, but current evidence says it is not the short-track lead.
