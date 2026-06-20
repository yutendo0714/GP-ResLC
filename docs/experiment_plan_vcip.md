# VCIP Experiment Plan

## Goal

Demonstrate that GP-ResLC reduces bitrate at the same perceptual quality by sending only the residual information that the GLC generative prior cannot predict.

Primary claim:

> Given the transmitted GLC hyper/VQ code `z_hat`, a learned generative-prior predictor estimates the recoverable component of `y`; entropy coding only the residual lowers `bit_y` and total bpp at matched DISTS/LPIPS/FID.

## Phase V0: Reproduce GLC

Status: Kodak completed. CLIC remains.

Commands:

```bash
.venv/bin/python test_image.py --q_indexes 0 1 2 3 \
  --model_path pretrained/GLC_image.pth.tar \
  --input_path /dpl/kodak \
  --output_path experiments/v0_glc_kodak \
  --fid_patch_size 64

.venv/bin/python test_image.py --q_indexes 0 1 2 3 \
  --model_path pretrained/GLC_image.pth.tar \
  --input_path /dpl/clic/professional \
  --output_path experiments/v0_glc_clic_professional \
  --fid_patch_size 256
```

## Phase V1: Pθ only

Run q=2 first, then all q if signal appears.

Grid:

| run | q | lambda_d | lambda_lpips | lambda_align | unfreeze_fusion | purpose |
|---|---:|---:|---:|---:|---|---|
| v1_q2_base | 2 | 1.0 | 1.0 | 1.0 | no | initial signal; use `scale_mean` |
| v1_q2_perc | 2 | 0.1 | 1.0 | 0.1 | no | less MSE/CE pressure under `scale_mean` |
| v1_q2_noalign | 2 | 0.1 | 1.0 | 0.0 | no | isolate rate/distortion under `scale_mean` |
| v1_q2_fusion | 2 | 0.1 | 1.0 | 0.1 | yes | if Pθ alone too weak; keep quant_step frozen |

Success criteria:
- `ab/delta_bpp_y < 0` without large PSNR/LPIPS degradation.
- Full Kodak inference shows negative BD-rate for DISTS/LPIPS vs baseline.
- CLIC confirms trend.

## Phase V1 inference

```bash
.venv/bin/python scripts/test_v1.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --input /dpl/kodak \
  --out experiments/baseline_kodak \
  --method baseline \
  --q_indexes 0 1 2 3

.venv/bin/python scripts/test_v1.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --ckpt experiments/v1_q2_smoke/prior_predictor_final.pt \
  --input /dpl/kodak \
  --out experiments/ours_v1_q2_kodak \
  --method ours \
  --q_indexes 2
```

For a full curve, train separate V1 checkpoints for q=0..3, or use V2 q-conditioned model.

## Phase V2: q-conditioned + gate

Only after V1 has rate gains.

Questions:
- Does one q-conditioned Pθ match separate q models?
- Does perceptual gate improve DISTS/LPIPS at equal bpp, or does it just reduce bpp by damaging structure?

## Analysis checklist

- checkpoint evaluation on Kodak and CLIC.
- `bpp_y` vs total bpp plots.
- DISTS/LPIPS/FID/PSNR curves.
- `delta_params` distribution over training.
- `mu_pred` distribution vs `y` distribution on validation crops.
- Visual panels for failure cases: texture hallucination, object/semantic drift, color shift.
- MSE ablation: report as auxiliary, not headline.


## Diagnostic warning

The `all` predictor mode can lower bpp quickly but also changes quant_step, which can degrade quality and weaken the residual-coding claim. Use it as a diagnostic ablation only. The default `scale_mean` mode better matches the intended `p(y | z_hat)` refinement.


## 2026-06-19 Plan Revision

V1 P_theta-only is not the short-track lead anymore. The cleanest current VCIP path is V2 gate-only:

- freeze GLC and P_theta;
- learn only `q_embed` + `PerceptualGate`;
- exact zero-init identity;
- decoder recomputes `rho(z_hat, q)` without side bits;
- `rho>1` means coarser residual quantization where the generator can absorb missing detail.

Best current checkpoint:

```bash
experiments/v2_gateonly_rp_lR15_lp2_rho16_1500/v2_final.pt
```

Key Kodak numbers:

| method | q | bpp | LPIPS | DISTS | interpretation |
|---|---:|---:|---:|---:|---|
| GLC baseline | 2 | 0.0328 | 0.1680 | 0.0983 | reference |
| V2 balanced gate | 3 | 0.0320 | 0.1758 | 0.0987 | near-matched DISTS, 2.4% lower bpp |
| GLC baseline | 1 | 0.0282 | 0.1802 | 0.1040 | lower-rate reference |
| V2 balanced gate | 2 | 0.0291 | 0.1799 | 0.1004 | similar LPIPS, better DISTS than baseline q1 at slightly higher bpp |

Immediate next experiment:

```bash
.venv/bin/python scripts/train_v2.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --data /dpl/openimages/train \
  --val /dpl/kodak \
  --iters 12000 \
  --bs 2 \
  --num_workers 4 \
  --lr 4e-5 \
  --lambda_R 15 \
  --lambda_d 0.08 \
  --lambda_lpips 2 \
  --lambda_align 0 \
  --rho_max 1.6 \
  --freeze_predictor \
  --predictor_param_mode mean \
  --log_every 100 \
  --eval_every 1000 \
  --out experiments/v2_gateonly_rp_lR15_lp2_rho16_12k \
  --wandb_project gp-reslc-vcip \
  --wandb_name v2_gateonly_rp_lR15_lp2_rho16_12k \
  --wandb_mode online
```

Evaluation after longer run:

1. Kodak q0-q3 with `scripts/test_v2.py`.
2. Kodak metrics for all q with `scripts/eval_metrics.py`.
3. CLIC professional subset/full depending runtime.
4. Generate R-P plots for bpp vs DISTS/LPIPS/FID/KID.
5. Save gate/rho maps for qualitative appendix.

Method improvement tasks:

- Add optional differentiable DISTS training loss.
- Add `scripts/analyze_gate_maps.py`: dump rho heatmaps, rho histograms, and correlation with absolute reconstruction error.
- Add monotone coarsening parameterization for `rho>=1` that still has useful gradient at identity; current centered sigmoid can learn `rho<1` if LPIPS pressure dominates.
- Revisit stage-aware P_theta only after V2 curve is solid.


## 2026-06-19 Result Update After 12k V2 Run

Current best short-track checkpoint:

- checkpoint: experiments/v2_gateonly_rp_lR15_lp2_rho16_12k/v2_6000.pt
- W&B source run: zbuykb7n
- Kodak metrics source: experiments/eval_v2_gateonly_lR15_lp2_12k/metrics.json

Best defensible claim at this point:

| method | q | bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---:|---|
| GLC baseline | 2 | 0.032781 | 0.168006 | 0.098278 | official GLC Kodak point |
| GP-ResLC V2 gate | 3 | 0.0319 | 0.1747 | 0.0981 | near-matched DISTS, about 2.7% lower bpp |

4-point Kodak BD-rate versus official GLC:

| checkpoint | BD-rate DISTS | BD-rate LPIPS | BD-rate PSNR |
|---|---:|---:|---:|
| v2_3000 | -2.60% | +0.72% | +0.36% |
| v2_6000 | -3.94% | +0.57% | +1.43% |
| v2_final | -2.09% | +0.18% | +0.87% |

Decision:

- Use v2_6000 as the current paper-candidate checkpoint.
- Do not claim broad LPIPS or PSNR superiority yet.
- Frame the contribution as rate-perception control under DISTS, with RD metrics reported honestly as auxiliary.
- Longer training alone is not enough; after 6k the curve is mostly saturated.

Next experiments in priority order:

1. Add differentiable DISTS or DISTS-proxy training term and rerun the same V2 gate setup.
2. Evaluate v2_6000 on CLIC professional to test whether the Kodak DISTS gain generalizes.
3. Generate rho heatmaps for v2_6000 and include examples where coarsening avoids semantic edges.
4. Add monotone rho>=1 parameterization so the method always means residual suppression, not quality-seeking finer quantization at q0.
5. Only after this, revisit P_theta as stage-aware four-mask residual prediction for the full R-D-P GP-ResLC version.


## 2026-06-19 Current Best After Monotone Gate

Current best short-track checkpoint:

- checkpoint: experiments/v2_gateonly_min1_lR15_lp2_rho14_6k/v2_final.pt
- W&B run: 2lewp5k6
- metrics: experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/metrics.json

Best claim:

| method | q | bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---:|---|
| GLC baseline | 2 | 0.032781 | 0.168006 | 0.098278 | official GLC Kodak point |
| GP-ResLC monotone gate | 3 | 0.0320 | 0.1747 | 0.0981 | near-matched DISTS, about 2.4% lower bpp |

4-point Kodak BD-rate versus official GLC:

| method | BD-rate DISTS | BD-rate LPIPS | BD-rate PSNR |
|---|---:|---:|---:|
| previous best rho16 ckpt_6000 | -3.94% | +0.57% | +1.43% |
| monotone rho14 final | -4.38% | -0.07% | +1.80% |

Why this is stronger for VCIP:

- rho_min=1.0 makes the mechanism strictly residual-suppressing: the gate never sends extra residual detail.
- rho_max=1.4 prevents the over-coarsening seen with rho_max=1.6 and DISTS-loss runs.
- q0 stays near identity, while q2/q3 keep meaningful bitrate reduction.
- LPIPS BD-rate is no longer positive, although the gain is tiny.

Updated cross-dataset status:

| dataset | BD-rate DISTS | BD-rate LPIPS | BD-rate PSNR | q3 bpp delta |
|---|---:|---:|---:|---:|
| Kodak | -4.38% | -0.07% | +1.80% | -13.49% |
| CLIC professional valid | -8.45% | +1.15% | +2.05% | -13.33% |

Interpretation for VCIP short track:

- The main claim should be DISTS/FID/KID-oriented R-P improvement, not universal perceptual superiority.
- The strongest sentence is now: at q3, GP-ResLC achieves equal-or-better DISTS than GLC with about 13% lower bpp on both Kodak and CLIC professional valid.
- LPIPS is near-neutral on Kodak but worse on CLIC; do not make LPIPS the primary metric unless a follow-up run fixes it.
- PSNR/MS-SSIM are worse by design; use them to clearly delimit the method as short-track R-P, while the full GP-ResLC target remains R-D-P.

Next priority:

1. Select qualitative examples and gate heatmaps from q3 for the DISTS-matched / lower-bpp story.
2. Rho ablations are complete: `(rho_min=0.5,rho_max=1.4)` gives DISTS BD-rate -3.65%, `(rho_min=1.0,rho_max=1.6)` gives -3.19%, while the selected `(rho_min=1.0,rho_max=1.4)` gives -4.38%. Keep monotone suppression with a conservative cap.
3. Add OpenImages validation-crop sanity evaluation if runtime permits, mainly to check that the gate is not overfitting Kodak/CLIC textures.
4. Improve LPIPS without losing DISTS. Current evidence: LPIPS worsens per-image on Kodak/CLIC/OpenImages32, even though DISTS BD-rate is negative. Try a lower rate-pressure / higher LPIPS-weight run before paper freeze.
5. Keep MSE/PSNR as auxiliary diagnostics; do not overclaim RD in the short paper.


OpenImages32 sanity:

- Fixed subset: `data_subsets/openimages_v6_test_32`.
- Current best versus GLC: DISTS BD-rate -5.18%, FID BD-rate -6.12%, but q3 DISTS is slightly worse and KID is unstable.
- Use only as robustness evidence; primary tables should remain Kodak and CLIC professional valid.


LPIPS recovery attempt:

- `v2_gateonly_min1_lR10_lp4_rho14_6k` was stopped at about 5k because rho stayed exactly 1 and A/B bpp delta stayed zero.
- Conclusion: lower rate pressure plus hard monotone clamp causes an identity-gate failure.
- Next LPIPS work should use a soft monotone gate or a staged schedule, not simple loss reweighting.


## 2026-06-19 Sendability Teacher Update

The current strongest VCIP short-track checkpoint is now:

- `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`
- W&B: `a2w5fjt4`
- Main idea: monotone `rho>=1` gate with `rho_init=1.16`, explicit mean-rate target, and a training-only local sendability teacher. The decoder still recomputes the gate from `z_hat`; no side map is transmitted.

Main CLIC professional valid result versus GLC:

| metric | BD-rate |
|---|---:|
| DISTS | -13.06% |
| FID | -13.89% |
| KID | -10.53% |
| LPIPS | -0.48% |
| PSNR | -0.51% |

CLIC point result:

| q | bpp reduction | DISTS change | FID change | LPIPS change |
|---:|---:|---:|---:|---:|
| 0 | -11.39% | -0.0010 | -0.0960 | +0.0130 |
| 1 | -10.31% | -0.0009 | -0.2478 | +0.0087 |
| 2 | -8.60% | -0.0017 | -0.3317 | +0.0053 |
| 3 | -7.86% | -0.0015 | -0.3689 | +0.0032 |

Kodak is mixed, but q3 is a useful point claim: bpp 0.0370 -> 0.0342 (-7.48%) and DISTS 0.0954 -> 0.0949.

Next experiments:

1. Evaluate the sendability checkpoint on OpenImages32 to test cross-dataset sanity.
2. Try a lower LPIPS penalty or add a weak LPIPS-protection teacher because DISTS/FID improved but pointwise LPIPS is still worse.
3. Compare `rho_target=1.12` and `rho_target=1.20` to map the DISTS/LPIPS tradeoff.
4. Export gate maps for the top CLIC/Kodak qualitative examples and compute correlation between rho and local reconstruction error.
5. Prepare paper framing around rate-perception with DISTS/FID/KID as primary and LPIPS/PSNR/MS-SSIM as honest auxiliary metrics.


### OpenImages32 sanity result

The sendability checkpoint also passed a small OpenImages32 sanity check: BD-rate DISTS -7.11%, FID -7.98%, KID -4.24%, with q2 improving DISTS/FID/KID at -7.72% bpp. Treat this as sanity only because the subset is small.


### rho_target tradeoff

`rho_target=1.16` remains the strongest CLIC R-P headline. `rho_target=1.12` gives a cleaner balanced setting: CLIC BD-rate DISTS -9.86%, FID -10.38%, KID -11.10%, with pointwise LPIPS degradation reduced to +0.0023 at q3. Use both as evidence that GP-ResLC exposes a controllable rate-perception knob.


### Alex-LPIPS loss ablation

The `--train_lpips_net alex` ablation did not solve pointwise LPIPS degradation. It is useful as evidence that the issue is not only a train/eval backbone mismatch:

| run | dataset | BD DISTS | BD LPIPS | BD FID | BD KID | role |
|---|---|---:|---:|---:|---:|---|
| `rho_target=1.12`, VGG-loss | Kodak | -4.92% | -0.40% | +2.22% | -5.75% | balanced knob |
| `rho_target=1.12`, Alex-loss | Kodak | -5.66% | -0.67% | +0.20% | -6.64% | ablation |
| `rho_target=1.12`, VGG-loss | CLIC valid | -9.86% | -1.00% | -10.38% | -11.10% | balanced knob |
| `rho_target=1.12`, Alex-loss | CLIC valid | -9.78% | -1.01% | -11.23% | -2.59% | ablation |

Implications:

1. Use DISTS/FID/KID as the primary short-track R-P metrics.
2. Report LPIPS honestly as an auxiliary metric; do not claim pointwise LPIPS superiority.
3. The next technical improvement should preserve perceptual-sensitive residuals more explicitly, for example by adding a boundary/texture preservation regularizer or a nonuniform `rho_target` schedule, rather than only increasing LPIPS loss weight.


### Gate-analysis update

The latest LPIPS-protection ablations did not replace the lead checkpoint:

| run | Kodak BD DISTS | Kodak BD LPIPS | Kodak BD FID | decision |
|---|---:|---:|---:|---|
| `target116`, texture-free teacher | -5.23% | -0.21% | -0.02% | ablation only |
| `target116`, baseline LPIPS distill | -5.46% | -0.51% | +1.05% | ablation only |
| `target116`, original sendability teacher | -4.72% Kodak / -13.06% CLIC | -0.85% Kodak / -0.48% CLIC | +0.80% Kodak / -13.89% CLIC | lead |

The strongest new paper evidence is the q3 gate correlation analysis. On both Kodak and CLIC, rho is negatively correlated with local reconstruction error, texture variance, and image gradient. High-rho regions have roughly half the local error/gradient of low-rho regions. This directly supports the mechanism that GP-ResLC coarsens generator-predictable regions and protects hard residual regions.

Next priority:

1. Use `target116` original sendability as lead and `target112` as controllable balanced knob.
2. Include gate-correlation statistics and qualitative rho overlays in the paper draft.
3. Keep baseline distillation code for future variants, but do not spend more short-track time on LPIPS-only fixes unless a stronger hypothesis appears.
4. If training time remains, try a nonuniform target map that explicitly sets low rho on high-gradient regions while preserving the CLIC DISTS/FID gains.


### Edge-guard ablation decision

`--gate_send_edge_weight 0.25` gives a clean q3 point but does not beat the original lead curve. On CLIC q3 it reaches bpp 0.0329 vs GLC 0.0354 with DISTS 0.0790 vs 0.0804 and FID 9.5403 vs 9.8799, but Kodak q0-q2 still degrade. Treat it as an appendix/control showing that the method can explicitly protect gradients, not as the main checkpoint.


### rho_target upper-bound decision

`rho_target=1.20` cuts more bits but over-coarsens Kodak q0-q2. It is useful to show the controllable knob has an aggressive endpoint, but the paper should not lead with it. Keep `rho_target=1.16` as the strong R-P setting and `rho_target=1.12` as the balanced setting.
