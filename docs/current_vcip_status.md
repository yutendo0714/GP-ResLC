# Current VCIP Status for GP-ResLC

Last updated: 2026-06-21 JST

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

## Complete-Design Branch Status

The more faithful GP-ResLC branch is `stage_quant_gate`: a four-part-prior, decoder-recomputable quantization gate that reduces residual precision from already available context without transmitting a side map. It is conceptually closer to the original claim than the global rho1.16 shortcut, and the real codec path is exact (`max_abs=0.000e+00`).

Current CLIC2020 full-test status versus local GLC real codec: stage-quant quality gives DISTS BD-rate `-3.56%`, FID BD-rate `-1.81%`, and matched-DISTS bpp `-5.29%`. It is therefore real and on-axis, but it is still weaker than rho1.16 (`-10.28%` DISTS BD-rate, `-7.30%` FID BD-rate). On Kodak it slightly beats rho1.16 in DISTS BD-rate, but on CLIC2020 and DIV2K the shortcut remains the paper lead.

Recent escalation checks show that unfreezing GLC entropy/prior modules is not safe yet: several runs improved estimated/crop A/B likelihood but worsened serialized real-codec bpp. The near-term complete-design path should keep GLC frozen and tune stage-quant rho schedules/hinges. Scratch GP-ResLC is the longer-term path for training the semantic/predictable/residual decomposition jointly from the start.

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
5. If adding another experiment, prioritize fixed-GLC stage-quant refinements with intermediate rho targets and strong quality hinges. Avoid entropy unfreezing until a calibration strategy is in place. The scratch branch should be developed separately because current evidence says frozen GLC was not trained to expose a clean predictable/residual latent decomposition.


## Scratch Complete-Design Update

The scratch branch now has a working Stage-A VQ autoencoder with soft codebook entropy, dead-code restart, W&B logging, and configurable semantic-grid rate. The best 16x16 pilot so far is `scratch_stage_a_vq1024_b64_z128_softent_restart_2k` (W&B `2d7yi3uk`): it avoids VQ collapse and reaches validation hard perplexity `143.7`, but its fixed semantic cost is `0.03906` bpp and reconstruction quality is not yet competitive.

Important design correction: the full GP-ResLC scratch model should not rely on the 16x16 semantic stream as-is because it leaves no bitrate budget for residuals. The model now supports `num_down=5`, giving an 8x8 semantic stream at `0.00977` fixed bpp for 256 crops. This is better aligned with the VCIP claim: send a very cheap generator-conditioning code, then send only unpredictable residual.


## Scratch Stage-B Status

The scratch complete-design branch now has the first hard-quantized residual proof signal. Stage B freezes the 8x8 Stage-A semantic code, predicts `mu_theta(s)` at the decoder, and sends a narrow residual latent. The useful run is `scratch_stage_b_down5_r16_q1_lR0p1_pred001_3k` (W&B `8fgx365x`). On deterministic Kodak center crops, Stage-A base is LPIPS/DISTS `0.4578/0.4526` at semantic bpp `0.00977`; Stage-B final reaches LPIPS/DISTS `0.4348/0.4371` at total proxy bpp `0.0385` (`0.00977` semantic + `0.0287` residual).

This validates the original decomposition direction but is not yet a paper lead because absolute quality is far below GLC. The safety lead remains the real-codec pretrained GLC gate/rho branch; scratch is now a real high-upside branch rather than only a plan.


## Scratch Low-Rate Lead Update

The current scratch low-rate lead is now the r8 DISTS-heavy Stage-B sweep, not the newer relaxed-rate run. `scratch_stage_b_down5_r8_q1_lR0p5_d2_3k` gives the best low-rate point at total bpp `0.01775`, LPIPS `0.45258`, DISTS `0.43456`; `scratch_stage_b_down5_r8_q1_lR0p3_d2_3k` gives the best higher-quality scratch point at total bpp `0.02345`, LPIPS `0.44438`, DISTS `0.43024`. The DISTS-heavy Stage-A source plus `lambda_R=0.1` Stage-B run (`9gbu1r38`) improved its base but did not beat these points.


## Scratch Pareto Update

The scratch complete-design branch now has a stronger point from the attention-refined Stage-A plus low-rate Stage-B. `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d2_3k/stage_b_final.pt` reaches Kodak center-crop total bpp `0.01328`, LPIPS `0.43770`, DISTS `0.42446`. This improves over the previous scratch lead (`0.01775` bpp / DISTS `0.43456`) while using fewer bits.

This does not replace the pretrained real-codec GLC branch as the VCIP lead, because absolute perceptual quality is still far from GLC. It does, however, strengthen the method-faithful complete-design story and justifies continuing the scratch branch with a quality-side Stage-B sweep.


The current scratch curve has two useful points from the attention-refined Stage-A: `lambda_R=0.5` final at `0.01328` bpp / DISTS `0.42446`, and `lambda_R=0.3` final at `0.01588` bpp / DISTS `0.42396`. The second point improves DISTS only slightly, so future scratch work should change residual coding rather than only spending more residual bits.


Current scratch DISTS lead: `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` at Kodak center-crop bpp `0.01321`, LPIPS `0.43869`, DISTS `0.42313`. This supersedes the earlier `0.01328` bpp / DISTS `0.42446` point.


## Scratch Progressive Residual Update

A two-stage progressive residual bottleneck was added to test a more literal version of the GP-ResLC idea: send a coarse unpredictable residual first, then send a finer residual only where decoder-side context predicts it is needed. The best result so far does not replace the scratch lead. Non-gated progressive residual reaches Kodak center DISTS `0.41948`, but at bpp `0.01954`; gated progressive gives a useful lower-rate point at bpp `0.01299` / DISTS `0.42373`, but the hard gate closes stage 1 almost entirely. This means the current scratch lead remains the single-stage checkpoint at bpp `0.01321` / DISTS `0.42313`.

Research decision: the progressive branch is conceptually right but needs a stronger specialization mechanism. Next attempts should give the fine stage a separate correction decoder or train it with a stage-improvement hinge before rate pruning.


## Latest Scratch Progressive Result

The scratch complete-design branch now includes a progressive residual bottleneck with optional decoder-side stage gates, fine correction decoders, stage-improvement hinge loss, and extra-stage-only warmup. The strongest scratch low-rate point is still the single-stage checkpoint (`0.01321` bpp / Kodak-center DISTS `0.42313`). The newest scratch quality-side point is `scratch_stage_b_progressive2_finedec_stage1warm_gateft_from_attnA_r8_q1q05_lR0p6_si5_1k/stage_b_final.pt`, which reaches Kodak-center bpp `0.01349`, LPIPS `0.43748`, DISTS `0.42283`; on DIV2K-center it reaches bpp `0.01428`, LPIPS `0.41999`, DISTS `0.41425`.

Mechanism status: forcing stage 1 open confirms it can carry useful residual information (`0.02003` bpp / DISTS `0.42222` on Kodak-center), but ordinary gate/rate fine-tuning prunes stage 1 almost entirely. A top-k 5% budget pilot (`oa3hchyt`) now prevents collapse and transmits a sparse fine stage (`0.01391` bpp / DISTS `0.42378`, stage1 bpp `0.00034`, gate mean `0.04883`). This is the cleanest scratch realization of the original sparse-residual idea, but it is still not the performance lead. A deterministic top-k sweep from 2% to 20% confirms this is not a simple budget issue: Kodak-center DISTS stays near `0.42378-0.42380` while bpp rises from `0.01371` to `0.01519`. Next scratch work should make the selected top-k residual positions carry DISTS-useful corrections through a hard-gate-aware improvement objective or error/texture-conditioned gate supervision. The last short top-k10% strong-hinge pilot (`7dyy6dpq`) improves Kodak-center DISTS to `0.42219` at `0.01504` bpp, but LPIPS worsens and DIV2K-center is only `0.41361` at `0.01641` bpp, so it is a scratch quality-side signal rather than a paper lead.

Scratch selected-region update: the new extra-stage-only top-k10% selected-region objective with stage1 scale guard (`e6a0sh06`) gives the best scratch Kodak DISTS so far: `0.01377` bpp / LPIPS `0.44009` / DISTS `0.42195`, with only `0.00056` bpp in stage1 and gate mean `0.09961`. DIV2K-center is `0.01424` bpp / LPIPS `0.42176` / DISTS `0.41391`. This improves the method-faithful scratch story, but it is still a proxy-bpp center-crop sanity result, not a replacement for the paper-facing real-codec pretrained branch.


## 2026-06-21 05:35 JST - Stage-quant q1 low-rate target sweep

Goal: improve the paper-facing complete-design stage-quant curve near the low-rate end. Since q0-specific gates previously reduced bpp but hurt quality, I tested whether q1 can be shifted left to act as a safer q0.5-like point.

Runs:

| run | W&B | init | setting |
|---|---|---|---|
| `v4_stage_quant_v1q1_rhotarget110_quality_hinge_lR28_lp12_dists12_rt20_700` | `u9i3v479` | q1 quality checkpoint | rho target 1.10, stage_rho_max 2.0 |
| `v4_stage_quant_v1q1_rhotarget108_rhomax12_quality_hinge_lR24_lp12_dists12_rt15_700` | `09p6txjw` | q1 quality checkpoint | rho target 1.08, stage_rho_max 1.2 |

Training signal:

- target1.10 final: W&B A/B `delta_bpp_y=-0.00164`, PSNR `18.8447 -> 18.6981`, rho mean/max `1.0999/1.2448`.
- target1.08/rhomax1.2 final: W&B A/B `delta_bpp_y=-0.00139`, PSNR `18.9299 -> 18.8299`, rho mean/max `1.0789/1.1707`.

Kodak8 real-codec diagnostic, q1 only:

| run | bpp | PSNR | LPIPS | DISTS | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GLC q1 | 0.03104 | 21.3857 | 0.1852 | 0.1086 | 56.0845 | 0.0043 | anchor |
| stage-quant q1 quality | 0.03044 | 21.3211 | 0.1855 | 0.1081 | 56.1878 | 0.0044 | keep current q1 |
| target1.10 final | 0.02942 | 21.1612 | 0.1905 | 0.1109 | 56.7827 | 0.0045 | reject as replacement |
| target1.08/rhomax1.2 final | 0.02968 | 21.2243 | 0.1891 | 0.1111 | 56.9105 | 0.0046 | reject as replacement |

Read:

- Both target runs reduce real serialized bpp substantially, so the decoder-recomputable residual precision control works.
- However, Kodak8 LPIPS/DISTS regress enough that neither is suitable as a q1 curve replacement.
- Restricting local `stage_rho_max` reduces rho spikes but does not fix perceptual degradation. The failure is not only extreme local rho; low-rate q1 residual precision itself is close to the quality floor.

Decision:

- Keep the existing stage-quant q1 quality checkpoint in the paper-facing complete-design branch.
- Treat q1 target1.08/1.10 as upper-rate-knob ablations only.
- If q1 is revisited, use explicit GLC reconstruction distillation or a sendability/texture teacher that predicts where coarsening is safe, not a global rho target.


## 2026-06-21 05:45 JST - Stage-quant q1 sendability teacher trial

Implementation update:

- Ported a training-only sendability teacher from `train_v2.py` into `scripts/train_v1.py` for `predictor_param_mode=stage_quant_gate`.
- Added CLI: `--lambda_gate_send`, `--gate_send_tau`, `--gate_send_texture_weight`, and `--gate_send_edge_weight`.
- Added `stage_gate_p_from_rho_target()` to map a desired `rho_target` to the corresponding StageQuantGate `p_tex` mean under the softplus rho parameterization.
- Smoke test passed: `gate_send` is nonzero, target mean/std are logged, and inference remains side-info-free.

Run:

| run | W&B | init | setting |
|---|---|---|---|
| `v4_stage_quant_v1q1_send_rhotarget108_rhomax12_lR24_lp12_dists12_rt15_send5_700` | `0fwt4hk1` | q1 quality checkpoint | rho target 1.08, stage_rho_max 1.2, sendability BCE weight 5, texture 0.2, edge 0.1 |

Training was stopped after the 500 checkpoint because A/B PSNR stayed weak: q1 validation `delta_bpp_y=-0.0014`, PSNR `19.40 -> 19.15`.

Kodak8 real-codec diagnostic, q1 only:

| run | bpp | PSNR | LPIPS | DISTS | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GLC q1 | 0.03104 | 21.3857 | 0.1852 | 0.1086 | 56.0845 | 0.0043 | anchor |
| stage-quant q1 quality | 0.03044 | 21.3211 | 0.1855 | 0.1081 | 56.1878 | 0.0044 | keep current q1 |
| target1.08/rhomax1.2 final | 0.02968 | 21.2243 | 0.1891 | 0.1111 | 56.9105 | 0.0046 | reject |
| sendability target1.08 250 | 0.02958 | 21.1704 | 0.1898 | 0.1124 | 56.3093 | 0.0044 | reject |
| sendability target1.08 500 | 0.02953 | 21.1695 | 0.1900 | 0.1126 | 56.0714 | 0.0043 | reject |

Read:

- The sendability teacher improves distribution metrics relative to the non-send target run, with FID/KID near GLC at lower bpp.
- It does not protect DISTS/LPIPS or PSNR. The teacher's low-error/texture/edge proxy is too distribution-oriented and not sufficiently structure/perceptual-local for q1.
- This mirrors the scratch branch: sparse/coarsened residual decisions need an explicit DISTS/feature-local payoff model, not only a heuristic sendability map.

Decision:

- Do not replace the current stage-quant q1 quality checkpoint.
- Keep sendability-stage-quant as an implementation tool, but the next improvement should use a learned or measured local sensitivity teacher: e.g., compare baseline reconstruction against a synthetically coarsened reconstruction and train the gate toward positions where DISTS/LPIPS change is small.

## 2026-06-21 rho1.16 DISTS Fine-Tune

Purpose: improve the current paper-facing `rho1.16` real-codec lead without changing the zero-side-bit residual-suppression mechanism. Resumed from `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/train_state.pt` at iteration 6000 and fine-tuned to 9000 with `lambda_dists=2`, lower LR `3e-5`, same `rho_target=1.16`, same always-on sendability teacher. W&B run: `kytm8hb5`.

Checkpoint: `experiments/v2_gate_send_lR10_lp4_dists2_rho14_target116_send5_all_ft3k_from_lead/v2_final.pt`.

Kodak forward metrics suggested a small DISTS gain versus the existing lead. Real-codec Kodak evaluation confirmed the gain with exact arithmetic coding and forward/decode consistency (`max_abs=0.000e+00` for all checked images):

| run | DISTS BD | LPIPS BD | PSNR BD | MS-SSIM BD | FID BD | KID BD |
|---|---:|---:|---:|---:|---:|---:|
| `gp_rho116_real` existing lead | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.14% |
| `gp_rho116_dists2_ft` | -5.62% | -0.32% | -0.72% | +0.39% | -3.27% | -7.54% |

Per-q real metrics for `gp_rho116_dists2_ft` on Kodak:

| q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.023763 | 21.0708 | 0.2126 | 0.1145 | 29.2696 | 0.00358 |
| 1 | 0.027473 | 21.5020 | 0.1886 | 0.1064 | 26.3078 | 0.00294 |
| 2 | 0.032044 | 21.8904 | 0.1752 | 0.0990 | 24.9496 | 0.00262 |
| 3 | 0.036273 | 22.1608 | 0.1653 | 0.0952 | 23.8925 | 0.00229 |

Interpretation: DISTS fine-tuning gives a real-codec Kodak improvement over the current lead in DISTS/FID/KID, while LPIPS BD weakens. This is aligned with the short-track R-P emphasis but must be validated on DIV2K and CLIC before replacing `rho1.16` as paper lead. DIV2K real evaluation is currently running at `experiments/real_codec/div2k_gp_reslc_rho116_dists2_ft/`.



## DISTS Fine-Tune Decision Update

Final decision after Kodak, DIV2K, and CLIC validation: do not replace the paper lead. A gentler `lambda_dists=1` run (W&B `8lct9ym0`) improves Kodak real-codec DISTS to `-5.96%` BD versus GLC, compared with `-4.47%` for the lead and `-5.62%` for the `lambda_dists=2` run. However, on DIV2K real codec it gives DISTS `-10.47%`, worse than the existing lead `-10.79%`; `lambda_dists=2` is also worse on DISTS (`-10.53%`) despite slightly stronger FID. On CLIC professional validation, both DISTS fine-tunes are worse than `send5all` on DISTS BD (`+0.63%` for lambda 1, `+0.11%` for lambda 2; lower is better). The `lambda_dists=1` 6500 checkpoint is also worse (`+1.36%`).

Updated status:

- Main paper lead remains `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`.
- `lambda_dists=1` is a Kodak-DISTS auxiliary checkpoint, not a CLIC/DIV2K lead.
- `lambda_dists=2` is a Kodak/FID auxiliary checkpoint, not a CLIC/DIV2K lead.
- The next credible route to larger official-curve gains is not more global perceptual fine-tuning; it is a local sensitivity/sendability teacher that preserves spatial selectivity while reducing residual precision.


## 2026-06-21 10:00 JST - Latest Screening Decision

Two additional upper-bound attempts were rejected. The stage-quant measured-sensitivity teacher reduced Kodak8 q1 bpp but failed to preserve DISTS versus the existing stage-quant quality checkpoint (`0.1116` and `0.1097` DISTS versus `0.1081`). A pretrained `rho_target=1.18` fine-tune from the lead also failed to replace `rho1.16`: on Kodak real codec it gives DISTS/FID BD-rate `-4.03% / -0.64%` versus GLC, weaker than `rho1.16` (`-4.47% / -1.70%`) and the Kodak-only DISTS fine-tune. Keep `rho1.16` as the paper lead.


## 2026-06-21 10:55 JST - Pretrained gate-only measured-sensitivity screening

Implementation update:

- Ported the measured LPIPS-spatial gate teacher from `scripts/train_v1.py` into `scripts/train_v2.py`.
- Added `--lambda_gate_measured_sens`, `--gate_measured_sens_until`, `--gate_measured_sens_tau`, `--gate_measured_sens_margin`, and `--gate_measured_sens_edge_weight`.
- Added `--freeze_q_embed` so gate-only allocation experiments can freeze both `prior_predictor` and q conditioning.

Runs:

- `v2_gate_meassens_rho116_lR10_lp4_ms1_edge01_ft1500_from_lead`, W&B `r5lcsjhu`: predictor/q_embed trainable. Rejected early. At 500 A/B bpp_y became worse than GLC by about `+0.025` and PSNR collapsed; stopped around 600. Interpretation: measured teacher destabilizes the prior predictor when it is allowed to move.
- `v2_gateonly_meassens_rho116_lR10_lp4_ms05_edge01_ft1000_from_lead`, W&B `43kunfkw`: predictor and q_embed frozen, only gate trained. Real codec consistency passed (`max_abs=0` for Kodak q0-q3).

Kodak real-codec metrics for gate-only measured (`patch=64`, `split=2` for FID/KID, matching existing Kodak CSV protocol):

| q | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.023752 | 21.0371 | 0.7320 | 0.2122 | 0.1152 | 28.9731 | 0.0035 |
| 1 | 0.027518 | 21.4991 | 0.7566 | 0.1880 | 0.1065 | 26.6874 | 0.0031 |
| 2 | 0.032067 | 21.8782 | 0.7737 | 0.1744 | 0.1001 | 24.9466 | 0.0027 |
| 3 | 0.036274 | 22.1651 | 0.7860 | 0.1653 | 0.0965 | 24.0547 | 0.0023 |

BD-rate vs local GLC on Kodak:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| `gp_rho116_real` | -4.47 | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |
| `gp_gateonly_meassens_ms05_1000` | -4.16 | -0.89 | -0.35 | +0.34 | -0.59 | -3.16 |

Decision: reject as paper lead. It slightly improves LPIPS BD but worsens DISTS/FID/KID and does not help official-curve strength. The useful finding is that gate-only updates are stable; future pretrained allocation fine-tunes should freeze predictor/q_embed and use direct perceptual objectives rather than measured teacher alone.


## 2026-06-21 11:05 JST - Pretrained gate-only DISTS direct fine-tune screening

Run: `v2_gateonly_dists1_rho116_lR10_lp4_ft1000_from_lead`, W&B `0pgb0zn7`.

Setup: resumed from the `rho1.16` paper lead, froze `prior_predictor` and `q_embed`, trained only the decoder-computable perceptual gate with direct DISTS loss (`lambda_dists=1.0`) plus the original LPIPS/rate terms. This isolates zero-side-bit spatial allocation and avoids the measured-sensitivity instability seen when the predictor moved.

Real codec: Kodak q0-q3, exact arithmetic coding, all `max_abs=0`.

Kodak metrics using the existing local protocol (`patch=64`, `split=2` for FID/KID):

| q | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.023754 | 21.0409 | 0.7324 | 0.2121 | 0.1149 | 28.8918 | 0.0035 |
| 1 | 0.027509 | 21.4910 | 0.7566 | 0.1884 | 0.1070 | 26.5137 | 0.0030 |
| 2 | 0.032053 | 21.8862 | 0.7739 | 0.1746 | 0.0999 | 24.9132 | 0.0027 |
| 3 | 0.036257 | 22.1692 | 0.7858 | 0.1652 | 0.0962 | 23.8719 | 0.0022 |

BD-rate vs local GLC on Kodak:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| `gp_rho116_real` | -4.47 | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |
| `gp_rho116_dists1_ft` | -5.96 | -0.68 | -1.35 | +0.19 | -2.28 | -6.08 |
| `gp_rho116_dists2_ft` | -5.62 | -0.32 | -0.72 | +0.39 | -3.27 | -7.54 |
| `gp_gateonly_dists1_1000` | -3.95 | -0.65 | -0.44 | +0.24 | -2.01 | -5.04 |

Decision: reject as lead. It is stable and slightly improves FID over the paper lead, but DISTS/LPIPS BD are weaker. Next attempt: stronger DISTS-only gate allocation with `lambda_align=0` so the gate actually follows DISTS/rate rather than being numerically dominated by the frozen CE term.


## Pretrained Gate-only DISTS-heavy Screening

Run: `v2_gateonly_dists4_rho116_lR10_lp2_align0_ft1000_from_lead`, W&B `6dg3powh`.

Setup: resumed from the `rho1.16` paper lead, froze `prior_predictor` and `q_embed`, trained only the decoder-computable gate with stronger direct DISTS pressure (`lambda_dists=4`, `lambda_lpips=2`, `lambda_align=0`). This isolates whether the current lead can be improved by re-allocating zero-side-bit residual transmission toward DISTS-sensitive positions without moving the entropy model.

Exact Kodak real-codec points:

| q | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.023770 | 21.0491 | 0.7324 | 0.2120 | 0.1154 | 29.0295 | 0.0035 |
| 1 | 0.027508 | 21.4996 | 0.7564 | 0.1882 | 0.1069 | 26.4583 | 0.0030 |
| 2 | 0.032036 | 21.8827 | 0.7737 | 0.1748 | 0.1003 | 24.9985 | 0.0026 |
| 3 | 0.036225 | 22.1682 | 0.7859 | 0.1654 | 0.0955 | 24.0121 | 0.0023 |

BD-rate vs local real-codec GLC on Kodak: DISTS `-4.02%`, LPIPS `-0.66%`, PSNR `-0.54%`, MS-SSIM `+0.35%`, FID `-1.80%`, KID `-5.99%`.

Decision: reject as lead. Compared with the current paper lead (`rho1.16`: DISTS `-4.47%`, LPIPS `-0.79%`, PSNR `-0.87%`, MS-SSIM `+0.45%`, FID `-1.70%`, KID `-6.14%`), DISTS-heavy gate-only training improves only FID slightly and weakens the main perceptual curve. This suggests the gate alone is close to saturated under the frozen predictor; further gains likely require either a constrained predictor update or a q-specific objective rather than only increasing DISTS weight.


## Pretrained q0/q1 Baseline-Hinge Screening

Run: `v2_gateonly_q01_basehinge_rhomin1_rt112_lR10_lp4_d1_hinge10_1000_from_lead`, W&B `mdoc94xd`.

Implementation changes: `scripts/train_v2.py` now supports `--q_choices`, `--resume_weights_only`, `--lambda_dists_distill`, `--lambda_lpips_hinge`, and `--lambda_dists_hinge`. This enables q-specific gate-only fine-tuning from a pretrained V2 checkpoint with a fresh optimizer and explicit GLC-baseline perceptual safety constraints.

Setup: resumed weights from the `rho1.16` paper lead, froze `prior_predictor` and `q_embed`, trained only the decoder-computable gate on q0/q1 with `rho_min=1.0`, `rho_target=1.12`, LPIPS/DISTS loss, and LPIPS/DISTS hinge against frozen GLC. A failed first attempt (`ne8nvlck`) omitted `rho_min=1.0`, let rho fall below 1, and was stopped because it increased bits; do not use it.

Kodak real-codec BD vs local GLC: DISTS `-4.90%`, LPIPS `-1.00%`, PSNR `-1.43%`, MS-SSIM `-0.21%`, FID `-2.01%`, KID `-4.67%`. This is better than the paper lead on most Kodak metrics except KID.

DIV2K real-codec BD vs local GLC: DISTS `-8.69%`, LPIPS `-0.87%`, PSNR `-1.10%`, MS-SSIM `-0.46%`, FID `-4.97%`, KID `-8.02%`. This is worse than the paper lead on the primary DISTS/FID metrics (`rho1.16`: DISTS `-10.79%`, FID `-5.61%`), though it improves LPIPS/MS-SSIM/KID.

Decision: keep as an auxiliary Kodak/LPIPS checkpoint, not as the VCIP lead. The experiment is useful because it shows baseline-hinge safety can improve Kodak and LPIPS without breaking the exact codec, but it gives back too much rate saving on DIV2K to strengthen the official-curve claim.


### q0/q1 Weak Hinge Follow-up

A weaker q0/q1 hinge run (`v2_gateonly_q01_weakhinge_rhomin1_rt114_lR10_lp4_d05_hinge5_800_from_lead`, W&B `8xffwful`) was tested to preserve more rate saving. Kodak real codec was exact (`max_abs=0`) and gave DISTS `-4.78%`, LPIPS `-0.99%`, PSNR `-1.12%`, MS-SSIM `+0.03%`, FID `+0.26%`, KID `-3.14%` vs local GLC. Decision: reject. It is not worth CLIC/DIV2K evaluation because FID worsens and it is weaker than the conservative q01 hinge auxiliary run.


## Predictor-Only Mean Correction Probe

A constrained predictor-only fine-tune from the rho1.16 lead was tested to move closer to the original GP-ResLC design without destabilizing the successful gate. The run `v2_predonly_mean_b003_lR6_lp4_d1_hinge_from_lead_1200` (W&B `06468x7k`) freezes the lead gate and q embedding, then trains only a bounded `predictor_param_mode=mean` correction (`delta_bound=0.003`). Kodak real-codec decoding remains exact (`max_abs=0`) and bpp moves slightly lower than rho1.16, but DISTS/LPIPS are marginally weaker while FID/KID improve.

Decision: keep rho1.16 as the paper lead. The predictor-only run is useful evidence that the original residual-prediction axis has signal, but the current global mean head is too blunt. The next complete-design attempt should make the mean predictor stage-aware or sensitivity-aware, not simply unfreeze the old prior predictor globally.


## DIV2K Predictor-Only Mean Probe

`predonly_b003` was evaluated on full DIV2K real codec. It lowers bpp strongly and improves FID/KID/LPIPS versus rho1.16, but DISTS BD-rate is `+1.12%` versus rho1.16, so it does not replace the DISTS paper lead. Keep rho1.16 as the primary checkpoint and treat predonly_b003 as an auxiliary distribution-quality ablation.


## CLIC2020 Predictor-Only Mean Probe

Full CLIC2020 test evaluation is complete for `v2_predonly_mean_b003_lR6_lp4_d1_hinge_from_lead_1200` using the exact real codec. q1-q3 decode exactly (`max_abs=0`) with bpp `0.02241 / 0.02689 / 0.03089` and official-style patch count `28,650` for FID/KID.

A hybrid curve using rho1.16 q0 plus predonly_b003 q1-q3 gives local real-codec BD-rate versus GLC: DISTS `-9.22%`, FID `-7.19%`, KID `-5.36%`, LPIPS `-0.13%`, PSNR `-1.17%`, MS-SSIM `-0.07%`. Against rho1.16, however, it is weaker on DISTS (`+1.15%`) and slightly weaker on FID (`+0.22%`) and KID (`+2.50%`). Official graph-extracted CLIC GLC comparison is also weaker than rho1.16: DISTS `-8.00%`, FID `-5.99%`.

Decision: keep rho1.16 as the VCIP paper lead. The predictor-only mean branch is not a lead replacement, but it is strong evidence for the original GP-ResLC axis. Decoder-computable mean prediction can lower serialized y bits under the exact codec; the missing piece is selectivity. The next serious complete-design pretrained attempt should make the predictor stage-aware or sensitivity-aware and pair it with the existing residual precision gate, rather than using a global mean correction or unfreezing GLC priors wholesale.


## Stage Residual Audit - 2026-06-21 15:25 JST

A more literal stage-aware residual predictor was implemented and audited with the exact real codec. The codec path initially had a serious protocol mismatch for stage-aware modes: it skipped the perceptual gate that `train_forward` used. This produced nonzero `consistency_max_abs` around `1.4..2.3`, so those first reconstructions are invalid. The real codec now applies the gate before the four-part prior for stage modes, and Kodak debug checks give `max_abs=0`.

The `b006` stage residual checkpoint is rejected as a paper lead. On Kodak real codec it gives DISTS BD-rate `-2.32%` versus local GLC, much weaker than the existing rho1.16 lead at `-4.47%`, and it worsens LPIPS/FID/PSNR/MS-SSIM. Interpretation: this mechanism is conceptually on-axis but currently too unconstrained; it removes bits in a way that harms perceptual reconstruction. A conservative follow-up (`b002`, W&B `o7mnj6mq`) is running to test whether tiny decoder-predictable residual means can preserve quality.


## Stage Residual Follow-Up Decision - 2026-06-21 15:55 JST

Two follow-ups confirm that post-hoc stage residual mean correction is not the current paper lead. The conservative `b002` run is worse than GLC on Kodak DISTS (`+0.65%` BD), and the delta-L1 run improves only to `-2.52%` DISTS BD while worsening LPIPS/FID/PSNR/MS-SSIM. The rho1.16 lead remains stronger at `-4.47%` Kodak DISTS BD and much stronger on CLIC/DIV2K.

Failure analysis shows the cause: bounded stage residual deltas saturate or broadly shift many locations while having weak correlation with the actual residual target. This means the current stage residual branch is conceptually on-axis but not behaviorally selective. The next serious route is not more post-hoc mean-shift tuning; it is a full/staged design that jointly adapts the entropy model and decoder, or adds an uncertainty/perceptual-sendability gate so only genuinely generator-predictable residual is suppressed.


## Scratch Stage-A Basis Decision - 2026-06-21 16:00 JST

Checked the newer 30k Stage-A and a new DISTS-heavy decoder-only continuation. Neither improves the scratch foundation. Kodak center-crop DISTS is `0.45180` for the 30k soft-entropy Stage-A, `0.44128` for the new DISTS-heavy decoder-only best, and `0.43546` for the existing attention-refined Stage-A. Keep `experiments/scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` as the scratch basis.

Implication: scratch improvement should not spend more cycles on simple Stage-A decoder-only loss reweighting. The next scratch move should either improve generator capacity/training more substantially, or continue from the attention-refined Stage-A and focus on residual-stage objectives that make the sparse transmitted residual perceptually useful.

## 2026-06-21 16:00 JST - Stage-quant gate placement sensitivity audit

Purpose: diagnose why the complete-design `stage_quant_gate` branch is on-axis but weaker than the paper-facing `rho1.16` lead on CLIC/DIV2K. I analyzed the quality-preserving q1/q2/q3 stage-quant gates on Kodak against local reconstruction error, LPIPS-spatial delta, texture variance, and image gradient. Artifacts:

- `experiments/analysis/stage_quant_quality_q1_gate_sensitivity_kodak.csv/json`
- `experiments/analysis/stage_quant_quality_q2_gate_sensitivity_kodak.csv/json`
- `experiments/analysis/stage_quant_quality_q3_gate_sensitivity_kodak.csv/json`

Summary over 24 Kodak images:

| q | delta bpp_y | rho mean | corr rho/base error | corr rho/texture | corr rho/gradient | corr rho/LPIPS delta | high-rho LPIPS delta | low-rho LPIPS delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.00032 | 1.015 | +0.231 | +0.236 | +0.209 | +0.013 | 0.00142 | -0.00011 |
| 2 | -0.00113 | 1.061 | +0.237 | +0.244 | +0.216 | +0.024 | 0.00458 | 0.00151 |
| 3 | -0.00205 | 1.118 | +0.200 | +0.178 | +0.172 | +0.027 | 0.00437 | 0.00152 |

Interpretation: the gate reduces real y-stream bits, but its spatial allocation is not the intended one. High-rho locations are positively correlated with local baseline error, texture, and gradient, and they also show larger local LPIPS-spatial degradation than low-rho locations. This means the learned gate is using entropy pressure to coarsen difficult/high-energy regions, not safely generator-predictable regions. That explains why stage-quant is method-faithful and real-codec exact, yet still underperforms the global rho1.16 shortcut on the main CLIC/DIV2K curves.

Decision: do not spend more time on unconstrained stage-quant rate pressure. The next complete-design experiment should invert this placement bias: either train a measured local sensitivity teacher from synthetic coarsening trials, add an explicit high-rho penalty on edge/texture/high-LPIPS-delta regions, or jointly train the synthesis path so high-rho regions become genuinely generator-recoverable.

## 2026-06-21 16:15 JST - q2 stage-quant spatial-guard diagnostic

Purpose: test whether the bad `stage_quant_gate` placement found above can be corrected by a direct LPIPS-spatial high-rho penalty. This was a short diagnostic, not a paper-lead candidate.

Run:

- `v4_stage_quant_v1q2_spgate_fixalloc_rhotarget106_lR20_lp10_dists10_sp80_500`
- W&B: `d096tn7c`
- Init: weights-only resume from `v3_stage_quant_v1q2_quality_hinge_fast_lR35_rhomax20_3k/train_state.pt`
- Key setting: `lambda_lpips_spatial_gate_hinge=80`, `rho_target=1.06`, fixed GLC, q2 only.

Training A/B at iteration 250: baseline `bpp_y=0.03239`, ours `bpp_y=0.03134`, delta `-0.00104`, baseline PSNR `19.2848`, ours PSNR `19.2611`. The penalty roughly halves the rate saving but almost closes the PSNR gap.

Gate-placement audit on Kodak after the fine-tune:

| run | delta bpp_y | rho mean | corr rho/base error | corr rho/texture | corr rho/gradient | corr rho/LPIPS delta | high-rho LPIPS delta | low-rho LPIPS delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q2 quality | -0.00113 | 1.061 | +0.237 | +0.244 | +0.216 | +0.024 | 0.00458 | 0.00151 |
| q2 spatial-guard | -0.00055 | 1.028 | -0.057 | +0.027 | -0.048 | +0.004 | 0.00129 | 0.00109 |

Kodak8 real-codec q2 diagnostic for `q2 spatial-guard`:

- real bpp `0.03470`, bpp_y `0.02959`, encode/decode `0.1269s/0.1092s`, consistency `max_abs=0`.
- PSNR `21.5381`, MS-SSIM `0.7671`, LPIPS `0.1731`, DISTS `0.1028`, FID `54.2086`, KID `0.0038`.
- Artifacts: `experiments/analysis/stage_quant_q2_spgate_fixalloc_sensitivity_kodak.json`, `experiments/real_codec/kodak8_stage_quant_q2_spgate_fixalloc_metrics.csv`, `experiments/real_codec/kodak8_stage_quant_q2_spgate_fixalloc_recon/`.

Interpretation: the spatial guard successfully flips the gate placement away from high-error/high-gradient regions and improves local safety. However, it also gives back roughly half of the y-stream saving and lands between GLC q2 and the existing stage-quant quality q2. This is not a lead, but it proves the placement problem is controllable. The next credible complete-design experiment should combine this guard with a stronger generator/reconstruction path or a two-objective budget: keep a fixed bpp_y saving while moving high rho only to low-sensitivity regions.

### Kodak full q2 result for spatial-guard diagnostic

The q2 spatial-guard diagnostic was expanded from Kodak8 to all 24 Kodak images with the exact real codec:

| run | q | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage-quant quality | 2 | 0.03263 | 0.02753 | 21.9264 | 0.7757 | 0.1725 | 0.0993 | n/a | n/a |
| q2 spatial-guard | 2 | 0.03366 | 0.02855 | 21.9877 | 0.7792 | 0.1705 | 0.0993 | 24.6837 | 0.0025 |
| GLC local | 2 | 0.03472 | n/a | 22.0767 | 0.7819 | 0.1671 | 0.0979 | n/a | n/a |

Artifact: `experiments/real_codec/kodak_stage_quant_q2_spgate_fixalloc_metrics.csv`.

Read: the spatial guard trades back about `0.0010` bpp compared with the existing stage-quant q2 point, but improves LPIPS, PSNR, and MS-SSIM while keeping DISTS essentially tied. It is still not enough to beat the full GLC q2 quality point at equal quality, and it is not a curve lead. The important result is mechanistic: once high-rho placement is moved away from high-error/high-gradient regions, perceptual quality recovers, so the next version should preserve the corrected placement while enforcing a fixed rate-saving budget.

