# 12h GP-ResLC Research Sprint Summary

Date: 2026-06-21 JST

## Paper-facing lead

The VCIP package lead remains the pretrained real-codec GLC gate/rho branch, not scratch. Use the real codec results as source-of-truth:

- CLIC2020 full test: DISTS BD-rate `-10.28%`, FID BD-rate `-7.30%` versus local real-codec GLC.
- DIV2K validation: DISTS BD-rate `-10.79%`, FID BD-rate `-5.61%`.
- Kodak: DISTS BD-rate `-4.47%`.
- Official graph positioning also supports CLIC/DIV2K improvement, but local paired real-codec comparison is cleaner.

## Scratch complete-design progress

The scratch branch now better matches the original GP-ResLC idea: an 8x8 semantic/generative stream (`0.00977` bpp on 256 crops) plus a residual stream that should carry only unpredictable information.

Current scratch points:

| role | checkpoint | Kodak bpp | LPIPS | DISTS | note |
|---|---|---:|---:|---:|---|
| low-rate lead | `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` | 0.01321 | 0.43869 | 0.42313 | best low-rate scratch point |
| top-k 5% mechanism | `scratch_stage_b_progressive2_finedec_stage1warm_topk005_from_attnA_r8_q1q05_lR0p5_si5_1k/stage_b_final.pt` | 0.01391 | 0.43889 | 0.42378 | sparse stage1 works, not a lead |
| quality-side DISTS | `scratch_stage_b_progressive2_finedec_stage1warm_topk010_si20_b64_from_attnA_r8_q1q05_lR0p5_1k/stage_b_final.pt` | 0.01504 | 0.44118 | 0.42219 | best scratch Kodak DISTS, but higher bpp/worse LPIPS |

## Main research conclusion

Top-k sparse residual gating is mechanically solved enough to keep stage 1 open without side information. However, simply increasing the fine-stage budget from 2% to 20% raises bpp without improving DISTS. Stronger stage-improvement pressure can improve Kodak DISTS, but it shifts cost into stage0 and worsens LPIPS/generalization.

Next best experiment: selected-region hard-gated improvement loss plus anti-stage0-leak regularization, then only promote if both Kodak and DIV2K improve.

## Key files

- `docs/current_vcip_status.md`
- `docs/scratch_results_summary.md`
- `docs/scratch_progressive_residual_notes.md`
- `docs/scratch_gp_reslc_design.md`
- `docs/experiment_log.md`
- `docs/lic_sota_survey_2026.md`

## Follow-up After Sprint Start

A selected-region top-k fine-residual objective was implemented after the first 12h summary. The best new scratch checkpoint is `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt`: Kodak-center `0.01377` bpp / LPIPS `0.44009` / DISTS `0.42195`; DIV2K-center `0.01424` bpp / LPIPS `0.42176` / DISTS `0.41391`. This is a scratch DISTS update but still proxy-bpp, not paper-facing real codec.

## Late-Sprint Real-Codec Audit

The paper-facing lead remains `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt` (W&B `a2w5fjt4`). All current claims should continue to use real arithmetic-coded codec bpp, not estimated bpp. A codec mismatch for `stage_latent_residual` and `stage_quant_gate` was fixed: stage modes now still apply q-conditioned `z_cond` and the decoder-side `perceptual_gate` before the four-part prior. After the fix, stage-residual real-codec checks are exact with `max_abs=0.000e+00`.

Three post-hoc `stage_latent_residual` attempts were evaluated on Kodak with the corrected real codec:

| run | W&B | Kodak DISTS BD vs GLC | Kodak FID BD vs GLC | decision |
|---|---|---:|---:|---|
| `stage_resid_b006` | `07r6rjhl` | -2.32% | +4.34% | reject |
| `stage_resid_b002` | `o7mnj6mq` | +0.65% | +4.84% | reject |
| `stage_resid_b006_stageL20` | `9ovci19c` | -2.52% | +5.22% | reject |

The failure mode is informative. The bounded stage residual can lower y-rate, but saturation analysis shows broad bound-seeking shifts and weak correlation with the intended stage residual target. This is not selective enough to support the original claim that only generator-unpredictable information is transmitted. Do not promote post-hoc stage residual as the VCIP lead.

## Scratch Basis Audit

The scratch complete-design branch was also audited. The best Stage-A basis remains `experiments/scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` at fixed semantic bpp `0.00977`, because it gives the best Kodak-center DISTS among available Stage-A candidates. The longer 30k VQ run and the new DISTS-heavy decoder-only continuation did not improve the basis:

| checkpoint | Kodak-center LPIPS | Kodak-center DISTS | decision |
|---|---:|---:|---|
| `scratch_stage_a_vq1024_b80_z160_down5_softent_restart_from6000_30k/stage_a_best.pt` | 0.45730 | 0.45180 | reject |
| `scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` | 0.45767 | 0.43546 | keep |
| `scratch_stage_a_decoder_only_from_attn_best_d3_lp08_l103_4k/stage_a_best.pt` | 0.45655 | 0.43722 | reject |
| `scratch_stage_a_decoder_only_dists5_lp03_l102_from_attn_best_plus1200/stage_a_final.pt` | 0.46312 | 0.44256 | reject |

The research direction is now clearer: further gains are unlikely from post-hoc global mean/precision edits alone. The next high-upside branch should train the predictable/residual decomposition jointly, with a decoder-side uncertainty or local perceptual sensitivity signal that decides where residual bits are worth sending.

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

