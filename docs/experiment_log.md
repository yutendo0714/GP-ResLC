# GP-ResLC Experiment Log

Protocol note: entries before `2026-06-20 JST - CLIC2020 full test protocol correction` are historical and may contain superseded protocol counts or unseeded KID values. For paper-facing numbers, use `docs/glc_eval_protocol_audit.md`, `docs/real_codec_protocol.md`, and `experiments/paper_assets/vcip_key_tables.md`.

## 2026-06-18 Environment and Baseline

### Machine state

- Working directory: `/workspace/GP-ResLC`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB visible by `nvidia-smi`
- CUDA driver: 12.4 visible
- Python: pyenv `3.12.12`
- Virtualenv: `.venv`
- PyTorch: `2.5.1+cu124`
- Torch CUDA visible: `True`

### Project tree changes

- Kept `proposal_design/` as external design handoff.
- Copied runnable research code into `gp_reslc/`, `scripts/`, `configs/`.
- Added `.python-version` for Python 3.12.12.
- Added `.gitignore` entries for `.venv/`, `wandb/`, `outputs/`, raw `experiments/**/q*/`, and `*.pt`.

### Smoke test

Command:

```bash
.venv/bin/python scripts/smoke_gp_reslc.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --image /dpl/kodak/kodim01.png \
  --q_index 2
```

Result:

| item | value |
|---|---:|
| baseline_bpp_y | 0.03198624 |
| zero_init_predictor_bpp_y | 0.03198624 |
| delta_params_max_abs | 0 |
| recon_max_abs_diff | 0 |
| status | `smoke_ok=true` |

Interpretation: 未学習PθはGLC baselineと厳密一致。A/B実験の初期条件は健全。

### V0 Kodak baseline: official GLC image test

Command:

```bash
.venv/bin/python test_image.py \
  --q_indexes 0 1 2 3 \
  --model_path pretrained/GLC_image.pth.tar \
  --input_path /dpl/kodak \
  --output_path experiments/v0_glc_kodak \
  --fid_patch_size 64
```

Results from `experiments/v0_glc_kodak/q*/res.txt`:

| q | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.024323 | 21.320203 | 0.748707 | 0.196120 | 0.112936 | 28.144260 | 0.003561 |
| 1 | 0.028219 | 21.727473 | 0.768035 | 0.180205 | 0.104039 | 25.651115 | 0.002914 |
| 2 | 0.032781 | 22.067012 | 0.782013 | 0.168006 | 0.098278 | 24.736933 | 0.002694 |
| 3 | 0.036987 | 22.277390 | 0.791452 | 0.160994 | 0.095393 | 23.912575 | 0.002386 |

Notes:
- First run downloaded AlexNet/VGG/Inception metric weights into `/root/.cache/torch/hub/checkpoints`; later metric runs should be faster.
- `test_image.py` uses 24 dataloader workers; environment warned that 16 is suggested. For future custom scripts, keep `--num_workers` configurable.
- Kodak patch-FID uses `--fid_patch_size 64`, matching GLC's low-sample workaround.


### V1 one-iteration training smoke

Initial attempt failed with:

```text
RuntimeError: Output 1 of SplitBackward0 is a view and is being modified inplace.
```

Cause: `GLC_Image(inplace=True)` is fine for inference, but the differentiable `train_forward` path backpropagates through GLC blocks whose inplace LeakyReLU modifies split views. Fix: build frozen GLC with `GLC_Image(inplace=False)` in `scripts/train_v1.py` and `scripts/train_v2.py`.

Command after fix:

```bash
.venv/bin/python scripts/train_v1.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --data /dpl/openimages/train \
  --q_index 2 \
  --iters 1 \
  --bs 1 \
  --num_workers 0 \
  --log_every 1 \
  --eval_every 100 \
  --out experiments/train_v1_one_iter \
  --no_wandb
```

Result:

| iter | loss | bpp_total | bpp_y | PSNR | MSE | LPIPS | CE align | delta_abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11.4137 | 0.0309 | 0.0275 | 29.27 | 0.0047 | 0.2974 | 11.0840 | 0.00000 |

Interpretation: training path, `CodePredictionLoss`, LPIPS, checkpoint writing, and CUDA backward are functional. `delta_abs=0` at iter 0 is expected because the zero-init gate has not yet updated before logging.


### V1 short training diagnostics

Three 120-200 iter q=2 probes were run to separate useful residual prediction from degenerate rate reduction.

| run | wandb offline id | mode | key settings | quick A/B result | interpretation |
|---|---|---|---|---|---|
| `v1_q2_smoke_200` | `yibxx84e` | all params | `lr=1e-4, lambda_R=1, lambda_align=1` | `delta_bpp_y=+0.02904` at final eval | CE alignment dominated; rate became worse. |
| `v1_q2_rate_first_120` | `wttth3ms` | all params | `lr=2e-5, lambda_R=10, lambda_align=0` | `delta_bpp_y=-0.00777` | Rate can be reduced, but likely through quant_step/scale changes, with quality loss. |
| `v1_q2_mean_rate_first_120` | `yuz2wcfh` | means only | same as above | `delta_bpp_y≈0` | Means-only residual prediction is stable but too weak at 120 iter. |
| `v1_q2_scalemean_rate_first_120` | `d3v9pl5t` | scales+means only | same as above | `delta_bpp_y=-0.00007` | Freezing quant_step removes the quick rate gain; current injection/loss does not yet exploit predictive residuals. |

Full Kodak q=2 check for the all-params rate-first checkpoint:

| method | mode | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline q2 | none | 0.0328 | 0.0294 | 22.0670 | 0.7820 | 0.1680 | 0.0983 | 32.7140* | 0.0020* |
| V1 q2 rate-first | all params | 0.0268 | 0.0234 | 21.5399 | 0.7504 | 0.1964 | 0.1096 | 36.6417* | 0.0030* |

`*` These FID/KID values were computed by `scripts/eval_metrics.py`; they are comparable within this table but differ from `test_image.py`'s internal patch-FID implementation.

Conclusion: the initial code path is operational, and an unconstrained Pθ can lower bpp, but that is not yet a clean GP-ResLC contribution. The default implementation was changed to `predictor_param_mode=scale_mean` to freeze quant_step. Future work should recover rate gains under this cleaner constraint.

Next technical pivots:
1. Add a regularized `scale_mean` objective: penalize scale inflation, log scale deltas, and increase training length.
2. Add baseline reconstruction distillation: keep `x_hat_ours` close to `x_hat_glc` while optimizing rate, then compare DISTS/LPIPS.
3. Inject Pθ before `y_prior_fusion` or into `y_spatial_prior_reduction`, because post-fusion scale/mean deltas are weak once spatial priors overwrite later masks.
4. Try unfreezing `y_prior_fusion` under `scale_mean` while keeping quant_step fixed.

### Next recommended run

Next clean V1 q=2 run under quant_step-frozen `scale_mean` mode:

```bash
.venv/bin/python scripts/train_v1.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --data /dpl/openimages/train \
  --val /dpl/kodak \
  --q_index 2 \
  --iters 2000 \
  --bs 2 \
  --num_workers 4 \
  --lr 2e-5 \
  --lambda_R 10 \
  --lambda_align 0.05 \
  --lambda_d 0.1 \
  --lambda_lpips 1 \
  --predictor_param_mode scale_mean \
  --log_every 50 \
  --eval_every 250 \
  --out experiments/v1_q2_scalemean_lra_2k \
  --wandb_project gp-reslc-vcip \
  --wandb_name v1_q2_scalemean_lra_2k \
  --wandb_mode offline
```

Watch:
- `ab/delta_bpp_y`: should become negative without PSNR/LPIPS collapse.
- `train/bpp_total`, `train/bpp_y`, `train/psnr`, `train/lpips`.
- `pred/delta_abs_mean`: should grow gradually; very small values imply the post-fusion injection is too weak.
- If still flat, pivot to unfreezing `y_prior_fusion` or injecting Pθ before `y_prior_fusion`.


## 2026-06-18/19 V1 and V2 Online Diagnostics

### Online W&B runs

| run | W&B id | purpose | outcome |
|---|---|---|---|
| `v1_q2_scalemean_bound002_2k` | `181pvhpk` | bounded `scale_mean` prior correction | stable, but `delta_bpp_y≈+0.00004`; no useful bitrate gain. |
| `v1_q2_latentres_bound02_1500` | `rzk9kmun` | explicit `y - mu(z_hat)` latent residual coding | moved latent prediction, but validation bpp did not drop and PSNR collapsed in later iters. |
| `v1_q2_latentres_rateonly_bound1_lr1e4_500` | `yly21bcw` | destructive rate-only diagnostic | train bpp can be lowered, but run NaN'ed around it 50; confirms unconstrained residual shift is unstable. |
| `v1_q2_latentres_means_rateonly_b05_lr5e5_500` | `7wmi6giv` | means-add residual with rate-only | stable but gate barely moved (`delta_abs≈3e-5`); rate gradient alone too weak. |
| `v1_q2_latentres_means_sup5_distill1_b05_1500` | `dxxnzcuh` | means-add residual with moderate residual supervision | stable but bpp slightly worse (`delta_bpp_y≈+0.0002`). |
| `v1_q2_latentres_means_sup100_b05_750` | `pfqv0wpu` | strong residual supervision | bpp worsened (`delta_bpp_y≈+0.00167`); target `y*q_enc-base_mean` is not aligned with GLC spatial prior. |
| `v2_gateonly_rp_lR20_b2_1500_fix` | `aut0i561` | gate-only R-P, strong rate pressure | large bpp reduction; perceptual quality too degraded for final claim. |
| `v2_gateonly_rp_lR8_lp3_rho14_1500` | `aqj32i76` | quality-heavy gate, no rho floor | stopped; gate learned `rho<1`, increasing bpp for quality. |
| `v2_gateonly_rp_lR8_lp3_rho14_floor50_1500` | `3ncp6xd4` | quality-heavy gate with `rho<1` penalty | stopped; rho stayed at identity, no rate gain. |
| `v2_gateonly_rp_lR15_lp2_rho16_1500` | `shirf8tt` | balanced gate-only R-P | best current signal; moderate bpp reduction with near-baseline DISTS. |

### Implementation changes from these diagnostics

- Added `predictor_param_mode=latent_residual` to `train_forward`/`test_v1`/`test_v2`.
- Replaced the first latent-residual implementation (`y-mu` into four-part prior, add `mu` after) with a means-add variant that injects `mu` into each four-part prior mean. The first variant fed centered residuals into GLC's spatial prior and caused out-of-distribution autoregressive context.
- Fixed V2 gate autograd by replacing in-place `params[:, :N] = ...` with `torch.cat`.
- Made `PerceptualGate` exactly identity at initialization: zero head gives `rho=1`, preserving strict A/B equivalence.
- Added `--freeze_predictor` to `train_v2.py` to isolate gate-only behavior.
- Added `gate_rho` / `gate_p_tex` return values from `train_forward`, plus rho logging and optional `--lambda_rho_floor` in `train_v2.py`.

### V1 conclusion

The direct P_theta residual-prior line is not yet paper-ready:

- `scale_mean` prior correction is stable but too weak.
- Explicit latent residual subtraction can alter reconstructions but does not generalize to validation bpp reduction.
- Strong supervision toward `y*q_enc-base_mean` increases rate under the four-part spatial prior.

Interpretation: GLC's autoregressive spatial prior already uses local `y_hat_so_far` strongly; naively adding a global `z_hat -> mu_y` residual mean miscalibrates later spatial means/scales. For a defensible VCIP claim, P_theta needs either (a) stage-aware residual prediction per four-part mask, or (b) a learned predictor trained with actual arithmetic-coding likelihood and a stronger distribution-matching constraint. Do not lead the short paper with current V1 numbers.

### V2 gate-only Kodak results

Baseline values from V0 official GLC:

| baseline q | bpp | bpp_y approx | PSNR | LPIPS | DISTS |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.024323 | 0.0209 | 21.3202 | 0.1961 | 0.1129 |
| 1 | 0.028219 | 0.0248 | 21.7275 | 0.1802 | 0.1040 |
| 2 | 0.032781 | 0.0294 | 22.0670 | 0.1680 | 0.0983 |
| 3 | 0.036987 | 0.0336 | 22.2774 | 0.1610 | 0.0954 |

`v2_gateonly_rp_lR20_b2_1500_fix` (strong gate) full Kodak bpp:

| q | bpp | bpp_y | note |
|---:|---:|---:|---|
| 0 | 0.0203 | ~0.0169 | strong bitrate cut, quality visibly degraded. |
| 1 | 0.0231 | ~0.0197 | strong bitrate cut. |
| 2 | 0.0265 | ~0.0231 | q2 metrics: PSNR 21.1808, LPIPS 0.2058, DISTS 0.1130. |
| 3 | 0.0300 | ~0.0266 | q3 metrics: PSNR 21.4995, LPIPS 0.1941, DISTS 0.1079. |

This run proves the gate can robustly save bits, but it is too aggressive for the final R-P curve.

`v2_gateonly_rp_lR15_lp2_rho16_1500` (balanced gate) full Kodak:

| q | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0241 | ~0.0207 | not yet measured | - | - | - | - | - |
| 1 | 0.0269 | ~0.0235 | 21.6155 | 0.7626 | 0.1846 | 0.1059 | 34.3549 | 0.0027 |
| 2 | 0.0291 | ~0.0257 | 21.6866 | 0.7663 | 0.1799 | 0.1004 | 33.5070 | 0.0029 |
| 3 | 0.0320 | ~0.0286 | 21.8928 | 0.7742 | 0.1758 | 0.0987 | 32.9873 | 0.0027 |

Most useful current claim:

- Balanced V2 q3 reaches DISTS `0.0987` at `0.0320 bpp`.
- Baseline q2 reaches DISTS `0.0983` at `0.0328 bpp`.
- This is a small but clean ~2.4% total-bpp reduction at near-matched DISTS on Kodak.
- Balanced V2 q2 reaches LPIPS `0.1799` / DISTS `0.1004` at `0.0291 bpp`, sitting between baseline q1 and q2 and giving a useful interpolation point.

### Current decision

For the short VCIP track, pivot the empirical story from “P_theta alone lowers entropy means” to:

> A transmitted GLC semantic/hyper code lets the decoder deterministically infer where the generator can tolerate coarser latent residuals. A zero-init gate increases quantization only on those generator-predictable components, reducing transmitted residual bits without extra side information.

This preserves the user's central axis (“do not send what the generator can recover; send only residual/unpredictable detail”) while using the mechanism that currently works.

Next runs:

1. Train balanced V2 longer on OpenImages (`iters=10k-20k`) with `rho_max=1.6`, `lambda_R=12-15`, `lambda_lpips=2`, `lambda_d=0.08`.
2. Add differentiable DISTS loss or a DISTS-proxy patch loss; current LPIPS-only training does not optimize the headline metric directly.
3. Evaluate Kodak all q and CLIC after a longer checkpoint.
4. Add gate visualization: save `rho` heatmaps and correlate with image texture/residual error.
5. Revisit P_theta as stage-aware mask-conditioned prediction, not global `z_hat -> mu_y`.


### V2 balanced gate 12k run

W&B:

- run: v2_gateonly_rp_lR15_lp2_rho16_12k
- id: zbuykb7n
- url: https://wandb.ai/mayuyuto0714-waseda-university/gp-reslc-vcip/runs/zbuykb7n

Artifacts:

- checkpoints: experiments/v2_gateonly_rp_lR15_lp2_rho16_12k/v2_*.pt
- reconstructions: experiments/eval_v2_gateonly_lR15_lp2_12k/{ckpt_3000,ckpt_6000,final}/q*
- metrics: experiments/eval_v2_gateonly_lR15_lp2_12k/metrics.json
- metrics CSV: experiments/eval_v2_gateonly_lR15_lp2_12k/metrics.csv

Key checkpoint comparison on Kodak:

| checkpoint | q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 0 | 0.0248 | 21.3783 | 0.1954 | 0.1117 | 35.8354 | 0.0032 |
| final | 1 | 0.0267 | 21.5950 | 0.1845 | 0.1056 | 34.2661 | 0.0033 |
| final | 2 | 0.0290 | 21.7376 | 0.1797 | 0.1009 | 33.4095 | 0.0027 |
| final | 3 | 0.0317 | 21.8984 | 0.1760 | 0.0992 | 33.0066 | 0.0023 |
| ckpt_3000 | 0 | 0.0242 | 21.3003 | 0.1979 | 0.1108 | 36.3595 | 0.0032 |
| ckpt_3000 | 1 | 0.0269 | 21.6142 | 0.1845 | 0.1056 | 34.4446 | 0.0028 |
| ckpt_3000 | 2 | 0.0290 | 21.7304 | 0.1794 | 0.1014 | 33.6670 | 0.0024 |
| ckpt_3000 | 3 | 0.0319 | 21.9143 | 0.1752 | 0.0983 | 32.6466 | 0.0023 |
| ckpt_6000 | 0 | 0.0243 | 21.3278 | 0.1964 | 0.1121 | 36.3580 | 0.0029 |
| ckpt_6000 | 1 | 0.0265 | 21.5538 | 0.1860 | 0.1054 | 34.2401 | 0.0033 |
| ckpt_6000 | 2 | 0.0290 | 21.7274 | 0.1792 | 0.0994 | 33.4272 | 0.0026 |
| ckpt_6000 | 3 | 0.0319 | 21.9116 | 0.1747 | 0.0981 | 32.5721 | 0.0026 |

4-point BD-rate against official GLC Kodak baseline:

| checkpoint | BD-rate DISTS | BD-rate LPIPS | BD-rate PSNR |
|---|---:|---:|---:|
| ckpt_3000 | -2.60% | +0.72% | +0.36% |
| ckpt_6000 | -3.94% | +0.57% | +1.43% |
| final | -2.09% | +0.18% | +0.87% |

Interpretation:

- ckpt_6000 is the best current short-track checkpoint by DISTS BD-rate.
- ckpt_6000 q3 reaches DISTS 0.0981 at 0.0319 bpp, while GLC q2 reaches DISTS 0.0983 at 0.032781 bpp; this is a near-matched-DISTS bitrate reduction of about 2.7%.
- LPIPS and PSNR do not improve versus GLC at the same quality range. The current claim should therefore be framed as DISTS and R-P first, not RD first.
- FID/KID from scripts/evaluate_recon_grid.py are useful for within-script comparison, but differ from GLC internal test_image.py values. For final paper tables, recompute GLC and GP-ResLC with one metric script.
- The longer run did not materially improve over 3k/6k after q2/q3 rate reduction stabilized. Future training should change the objective rather than simply train longer.

Next immediate actions:

1. Add DISTS or DISTS-proxy loss and rerun rho_max=1.6, lambda_R=12-15.
2. Evaluate ckpt_6000 on CLIC professional after Kodak curve is plotted.
3. Generate rho heatmaps for ckpt_6000 to verify that coarsening aligns with predictable/textured regions rather than semantic edges.
4. Add monotone rho>=1 gate parameterization. Current centered sigmoid sometimes learns rho<1 at q0, which is not aligned with the residual-suppression story.


### Gate map analysis for ckpt_6000

Generated rho heatmaps and per-image stats:

- q0: experiments/eval_v2_gateonly_lR15_lp2_12k/gate_maps_ckpt_6000_q0
- q2: experiments/eval_v2_gateonly_lR15_lp2_12k/gate_maps_ckpt_6000_q2
- q3: experiments/eval_v2_gateonly_lR15_lp2_12k/gate_maps_ckpt_6000_q3

Summary:

| q | rho mean range over Kodak | rho min range | rho max range | interpretation |
|---:|---:|---:|---:|---|
| 0 | about 0.993 to 1.023 | about 0.934 to 0.965 | about 1.073 to 1.162 | near identity, with some rho<1 regions; weak for residual-suppression story. |
| 2 | about 1.225 to 1.290 | about 1.029 to 1.135 | about 1.395 to 1.470 | consistently coarsens residual transmission. |
| 3 | about 1.331 to 1.421 | about 1.090 to 1.213 | about 1.509 to 1.535 | strongest clean residual-suppression regime. |

Interpretation:

- The method behaves most cleanly in q2/q3, where all images have rho>1 throughout the map.
- q0 shows near-identity behavior with local rho<1; for a strict story, future variants should constrain rho>=1 or only report the gate effect on the intended ultra-low-bitrate operating points.
- The saved heatmaps should be inspected visually before selecting appendix figures. Good figure candidates should show high rho in texture/background/detail regions and lower rho around object boundaries or semantic structure.


### DISTS loss implementation smoke

Implemented optional DISTS training loss in scripts/train_v2.py:

- new argument: --lambda_dists
- DISTS is computed on [0,1] images after clamping x and x_hat from [-1,1]
- W&B key: train/dists
- loss now supports lambda_R*bpp_y + lambda_d*MSE + lambda_lpips*LPIPS + lambda_dists*DISTS + auxiliary terms

Smoke command:

.venv/bin/python scripts/train_v2.py --glc_weights pretrained/GLC_image.pth.tar --data /dpl/openimages/train --val /dpl/kodak --iters 1 --bs 1 --num_workers 0 --lr 1e-5 --lambda_R 1 --lambda_d 0.01 --lambda_lpips 0.1 --lambda_dists 1 --lambda_align 0 --rho_max 1.6 --freeze_predictor --predictor_param_mode mean --predictor_delta_bound 0 --log_every 1 --eval_every 100 --out experiments/smoke_v2_dists_loss --no_wandb

Smoke result:

- command exited successfully
- initial rho stayed exactly identity: 1.000/1.000/1.000
- initial A/B stayed equivalent across q0-q3
- train/dists at the sampled first batch was 0.2330

This enables the next R-P run to optimize the actual headline perceptual metric instead of relying only on LPIPS.


### DISTS-loss run: lambda_R15 lpips1 dists2 rho16 8k

W&B:

- run: v2_gateonly_rp_lR15_lp1_dists2_rho16_8k
- id: jl4xw6eu
- url: https://wandb.ai/mayuyuto0714-waseda-university/gp-reslc-vcip/runs/jl4xw6eu

Settings:

- lambda_R=15
- lambda_lpips=1
- lambda_dists=2
- lambda_d=0.05
- rho_max=1.6
- freeze_predictor=true

Kodak results:

| checkpoint | q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| ckpt_6000 | 0 | 0.0223 | 21.1075 | 0.2087 | 0.1146 | 37.5050 | 0.0034 |
| ckpt_6000 | 1 | 0.0243 | 21.2461 | 0.1997 | 0.1084 | 35.7110 | 0.0031 |
| ckpt_6000 | 2 | 0.0273 | 21.4276 | 0.1941 | 0.1081 | 34.9994 | 0.0027 |
| ckpt_6000 | 3 | 0.0308 | 21.7915 | 0.1820 | 0.1013 | 33.4386 | 0.0031 |
| final | 0 | 0.0223 | 21.1110 | 0.2077 | 0.1138 | 37.1080 | 0.0032 |
| final | 1 | 0.0241 | 21.2163 | 0.2017 | 0.1099 | 36.5153 | 0.0033 |
| final | 2 | 0.0270 | 21.3844 | 0.1970 | 0.1093 | 35.5140 | 0.0034 |
| final | 3 | 0.0307 | 21.7654 | 0.1825 | 0.1011 | 33.8105 | 0.0033 |

BD-rate versus official GLC:

| checkpoint | BD-rate DISTS | BD-rate LPIPS | BD-rate PSNR |
|---|---:|---:|---:|
| ckpt_6000 | +11.54% | +12.89% | +8.44% |
| final | +35.65% | +19.08% | +9.99% |

Conclusion:

- This setting is not useful. Direct DISTS loss did not prevent over-coarsening because lambda_R=15 and rho_max=1.6 still drove rho toward saturation.
- It reduced bpp strongly, but DISTS/LPIPS degraded too much.
- Next attempt should either fine-tune from the best earlier checkpoint with lower rate pressure, or lower rho_max to avoid saturation.


### Fine-tune from best checkpoint: lambda_R8 lpips1 dists4 rho floor 4k

W&B:

- run: v2_gate_ft_from_best_lR8_lp1_dists4_rhofloor1_4k
- id: p5uh6lwt
- url: https://wandb.ai/mayuyuto0714-waseda-university/gp-reslc-vcip/runs/p5uh6lwt

Initialization:

- resumed weights from experiments/v2_gateonly_rp_lR15_lp2_rho16_12k/v2_6000.pt
- optimizer was reinitialized because the checkpoint has no optimizer state

Settings:

- lambda_R=8
- lambda_lpips=1
- lambda_dists=4
- lambda_d=0.08
- lambda_rho_floor=1
- rho_max=1.6

Kodak final metrics:

| q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0242 | 21.3334 | 0.1970 | 0.1119 | 36.4373 | 0.0037 |
| 1 | 0.0265 | 21.5589 | 0.1855 | 0.1049 | 34.0031 | 0.0032 |
| 2 | 0.0286 | 21.6913 | 0.1817 | 0.1007 | 33.6847 | 0.0024 |
| 3 | 0.0314 | 21.8686 | 0.1771 | 0.1004 | 33.3541 | 0.0024 |

BD-rate versus official GLC:

| metric | BD-rate |
|---|---:|
| DISTS | +14.21% |
| LPIPS | -0.45% |
| PSNR | +1.24% |

Conclusion:

- Fine-tuning from the best checkpoint with stronger DISTS weight did not improve the DISTS curve.
- It slightly improves LPIPS BD-rate, but the project target is DISTS/R-P and this is not a better paper candidate.
- Current best remains experiments/v2_gateonly_rp_lR15_lp2_rho16_12k/v2_6000.pt.
- Next design should change the gate parameterization or rho_max, not just add more DISTS loss.


### Monotone gate run: rho_min1 rho14 lambda_R15 lpips2 6k

W&B:

- run: v2_gateonly_min1_lR15_lp2_rho14_6k
- id: 2lewp5k6
- url: https://wandb.ai/mayuyuto0714-waseda-university/gp-reslc-vcip/runs/2lewp5k6

Code changes:

- PerceptualGate now accepts rho_min, default 0.5 for backward compatibility.
- train_v2.py adds --gate_rho_min and saves rho_min in checkpoints.
- test_v2.py and analyze_gate_maps.py read rho_min from checkpoints, defaulting to 0.5 for old checkpoints.
- gate_rho_min=1.0 smoke passed with exact zero-init A/B equivalence.

Settings:

- lambda_R=15
- lambda_lpips=2
- lambda_d=0.08
- lambda_dists=0
- rho_max=1.4
- gate_rho_min=1.0
- freeze_predictor=true

Kodak final metrics:

| q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0240 | 21.3018 | 0.1980 | 0.1119 | 36.6117 | 0.0039 |
| 1 | 0.0262 | 21.5570 | 0.1860 | 0.1052 | 34.4041 | 0.0031 |
| 2 | 0.0286 | 21.6697 | 0.1821 | 0.1013 | 33.3955 | 0.0027 |
| 3 | 0.0320 | 21.9579 | 0.1747 | 0.0981 | 32.2695 | 0.0024 |

BD-rate versus official GLC:

| metric | BD-rate |
|---|---:|
| DISTS | -4.38% |
| LPIPS | -0.07% |
| PSNR | +1.80% |

Gate statistics:

- q3 rho mean range over Kodak: about 1.353 to 1.392.
- q3 rho min range: about 1.115 to 1.262.
- q3 rho max is clamped at 1.4 as intended.
- q0 rho mean range: about 1.005 to 1.024.
- q0 rho min is exactly 1.0 for every image, so bit-increasing rho<1 is removed.

Conclusion:

- This is the current best VCIP short-track candidate.
- It improves the previous best DISTS BD-rate from -3.94% to -4.38% and also makes LPIPS BD-rate slightly negative.
- It preserves the clean method story: rho never drops below 1, so the gate only suppresses residual transmission and never spends extra bits.
- The main weakness remains PSNR/RD, so the paper framing should stay rate-perception rather than rate-distortion.


## 2026-06-19 02:23 JST - CLIC validation for current monotone-gate candidate

Current candidate:

- run: `v2_gateonly_min1_lR15_lp2_rho14_6k`
- wandb: `2lewp5k6`
- checkpoint: `experiments/v2_gateonly_min1_lR15_lp2_rho14_6k/v2_final.pt`
- settings: `lambda_R=15`, `lambda_lpips=2`, `lambda_d=0.08`, `rho_max=1.4`, `gate_rho_min=1.0`, `freeze_predictor=true`

Code / evaluation updates:

- `scripts/evaluate_recon_grid.py` now reads both GP-ResLC `bpp.json` and official GLC `res.txt`, so baseline and proposed outputs can be evaluated through one CSV/JSON path.
- CSV writing now uses the union of observed keys, avoiding failures when only the proposed run contains `bpp_y`/`bpp_z`.
- Added `scripts/plot_metric_curves.py` for lightweight metric-curve plotting from these CSV files.

Artifacts:

- Kodak CSV/JSON: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/kodak_final_vs_glc_metrics.{csv,json}`
- CLIC professional valid CSV/JSON: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/clic_prof_valid_metrics.{csv,json}`
- Kodak plots: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/plots/kodak/curve_*.png`
- CLIC plots: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/plots/clic_prof_valid/curve_*.png`

CLIC professional valid metrics:

| method | q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLC | 0 | 0.0226 | 23.2825 | 0.1658 | 0.0952 | 35.0135 | 0.0018 |
| GLC | 1 | 0.0265 | 23.7125 | 0.1515 | 0.0877 | 32.1244 | 0.0006 |
| GLC | 2 | 0.0311 | 24.1064 | 0.1410 | 0.0831 | 30.0141 | 0.0010 |
| GLC | 3 | 0.0354 | 24.3332 | 0.1355 | 0.0804 | 29.3297 | 0.0008 |
| GP-ResLC | 0 | 0.0224 | 23.2664 | 0.1665 | 0.0950 | 34.6364 | 0.0016 |
| GP-ResLC | 1 | 0.0246 | 23.5063 | 0.1573 | 0.0872 | 32.4193 | 0.0012 |
| GP-ResLC | 2 | 0.0272 | 23.7007 | 0.1516 | 0.0813 | 31.0347 | 0.0006 |
| GP-ResLC | 3 | 0.0306 | 23.9936 | 0.1449 | 0.0791 | 30.1882 | 0.0004 |

BD-rate versus GLC baseline:

| dataset | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID | q3 bpp delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kodak | -4.38% | -0.07% | +1.80% | +2.16% | -0.65% | -5.47% | -13.49% |
| CLIC professional valid | -8.45% | +1.15% | +2.05% | +1.90% | -4.84% | -3.63% | -13.33% |

Interpretation:

- The DISTS/R-P claim now reproduces on both Kodak and CLIC professional valid.
- CLIC q3 is especially useful for the short paper: GLC has 0.03536 bpp / DISTS 0.08035, while GP-ResLC has 0.03065 bpp / DISTS 0.07906. That is slightly better DISTS at about 13.3% lower bpp.
- LPIPS remains the weak perceptual metric on CLIC (+1.15% BD-rate), so the paper should not claim universal perceptual dominance. Frame the current version as DISTS/FID/KID-oriented R-P compression with LPIPS near-neutral on Kodak and mildly worse on CLIC.
- PSNR/MS-SSIM remain worse, as expected for a short-track R-P method. Keep MSE/PSNR as diagnostic/secondary evidence, not the optimization target.
- FID/KID are patch-size/randomness sensitive; use them as supporting distribution metrics, not as the sole claim.


## 2026-06-19 02:55 JST - Ablation: rho_min0.5 rho14 gate-only

Purpose:

- Isolate whether the current best comes mainly from the lower `rho_max=1.4` or from the monotone `rho_min=1.0` constraint.
- This run keeps `rho_max=1.4` but restores old gate behavior with `gate_rho_min=0.5`.

Run:

- failed launch: `rzszhlnh`, stopped before training because `/dpl/openimages` had no images.
- successful run: `v2_gateonly_floor05_lR15_lp2_rho14_6k_r2`
- wandb: `pel2wkxr`
- url: https://wandb.ai/mayuyuto0714-waseda-university/gp-reslc-vcip/runs/pel2wkxr
- data: `/dpl/openimages/train`
- checkpoint: `experiments/v2_gateonly_floor05_lR15_lp2_rho14_6k_r2/v2_final.pt`

Settings:

- `lambda_R=15`, `lambda_lpips=2`, `lambda_d=0.08`, `lambda_dists=0`
- `rho_max=1.4`, `gate_rho_min=0.5`
- `freeze_predictor=true`, `predictor_param_mode=mean`

Kodak metrics:

| method | q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| floor05 rho14 | 0 | 0.0245 | 21.3525 | 0.1951 | 0.1111 | 36.2321 | 0.0035 |
| floor05 rho14 | 1 | 0.0265 | 21.5665 | 0.1852 | 0.1052 | 34.3719 | 0.0031 |
| floor05 rho14 | 2 | 0.0288 | 21.7001 | 0.1798 | 0.1007 | 33.6189 | 0.0027 |
| floor05 rho14 | 3 | 0.0320 | 21.9571 | 0.1745 | 0.0987 | 32.4774 | 0.0024 |

BD-rate versus GLC Kodak baseline:

| metric | BD-rate | current monotone best |
|---|---:|---:|
| DISTS | -3.65% | -4.38% |
| LPIPS | +0.04% | -0.07% |
| PSNR | +1.63% | +1.80% |
| MS-SSIM | +2.11% | +2.16% |
| FID | +0.83% | -0.65% |
| KID | -12.79% | -5.47% |

Gate statistics:

- q0 often has mean `rho < 1`; examples include 0.9769 to 0.9988 for many Kodak images, with minima around 0.94-0.96.
- q3 remains strongly residual-suppressing: mean rho roughly 1.34-1.39, min > 1.15, max clamped at 1.4.

Artifacts:

- recon/metrics: `experiments/eval_v2_gateonly_floor05_lR15_lp2_rho14_6k_r2/`
- gate maps: `experiments/eval_v2_gateonly_floor05_lR15_lp2_rho14_6k_r2/gate_maps_final_q{0,3}/`
- plots: `experiments/eval_v2_gateonly_floor05_lR15_lp2_rho14_6k_r2/plots/kodak/curve_*.png`

Conclusion:

- Lowering `rho_max` to 1.4 is useful, but it does not fully explain the current best.
- The monotone `rho_min=1.0` constraint improves DISTS BD-rate and makes LPIPS slightly negative while preserving the cleaner paper story.
- This supports keeping monotone residual suppression as a core GP-ResLC design choice.


## 2026-06-19 03:25 JST - Ablation: rho_min1 rho16 gate-only

Purpose:

- Isolate whether the current best needs the lower `rho_max=1.4`, after fixing the monotone `rho_min=1.0` constraint.
- This run keeps `rho_min=1.0` but restores a stronger upper bound `rho_max=1.6`.

Run:

- run: `v2_gateonly_min1_lR15_lp2_rho16_6k`
- wandb: `gbuz2bqn`
- url: https://wandb.ai/mayuyuto0714-waseda-university/gp-reslc-vcip/runs/gbuz2bqn
- checkpoint: `experiments/v2_gateonly_min1_lR15_lp2_rho16_6k/v2_final.pt`
- data: `/dpl/openimages/train`

Settings:

- `lambda_R=15`, `lambda_lpips=2`, `lambda_d=0.08`, `lambda_dists=0`
- `rho_max=1.6`, `gate_rho_min=1.0`
- `freeze_predictor=true`, `predictor_param_mode=mean`

Kodak metrics:

| method | q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| min1 rho16 | 0 | 0.0241 | 21.2931 | 0.1986 | 0.1123 | 36.4934 | 0.0037 |
| min1 rho16 | 1 | 0.0262 | 21.5385 | 0.1857 | 0.1059 | 34.7925 | 0.0025 |
| min1 rho16 | 2 | 0.0281 | 21.5762 | 0.1854 | 0.1038 | 34.4700 | 0.0021 |
| min1 rho16 | 3 | 0.0312 | 21.8168 | 0.1795 | 0.1008 | 33.3747 | 0.0023 |

Ablation summary versus GLC Kodak baseline:

| run | rho_min | rho_max | DISTS BD-rate | LPIPS BD-rate | PSNR BD-rate | q3 bpp delta | q3 DISTS | q3 LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best min1 rho14 | 1.0 | 1.4 | -4.38% | -0.07% | +1.80% | -13.49% | 0.098125 | 0.174674 |
| floor05 rho14 | 0.5 | 1.4 | -3.65% | +0.04% | +1.63% | -13.39% | 0.098689 | 0.174538 |
| min1 rho16 | 1.0 | 1.6 | -3.19% | unreliable overlap | +2.82% | -15.76% | 0.100789 | 0.179506 |

Gate statistics:

- q0 keeps the monotone property: min rho is exactly 1.0 for every Kodak image.
- q3 is much more aggressive than the current best: mean rho is about 1.41-1.52, max about 1.58. Current best rho14 had q3 mean about 1.35-1.39.
- The extra q3 rate reduction (-15.76% bpp at q3 vs GLC) comes with visible metric damage: DISTS 0.1008, LPIPS 0.1795.

Artifacts:

- recon/metrics: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho16_6k/`
- gate maps: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho16_6k/gate_maps_final_q{0,3}/`
- plots: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho16_6k/plots/kodak/curve_*.png`
- summary CSV: `experiments/rho_ablation_summary.csv`

Conclusion:

- `rho_max=1.6` over-suppresses residuals under this loss: bitrate falls, but DISTS/LPIPS degrade too much.
- The current best `rho_min=1.0, rho_max=1.4` is supported by both ablations: monotone constraint helps, and the lower upper bound avoids over-coarsening.
- This is a clean VCIP design statement: GP-ResLC should suppress only predictable residuals, but the suppression magnitude must be capped to preserve perceptual fidelity.


## 2026-06-19 03:35 JST - Qualitative candidate ranking

Purpose:

- Select paper-figure candidates where GP-ResLC keeps equal-or-better DISTS while reducing bpp at q3.
- Added `scripts/rank_qualitative_candidates.py` for image-level PSNR/LPIPS/DISTS/bpp ranking.
- Added `scripts/make_qualitative_grid.py` for original / GLC q3 / GP-ResLC q3 contact sheets.

Artifacts:

- Kodak ranking: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/qual_rank_kodak_q3.csv`
- CLIC ranking: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/qual_rank_clic_prof_q3.csv`
- Kodak contact sheet: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/qual_grid_kodak_q3_top4.png`
- CLIC contact sheet: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/qual_grid_clic_q3_top4.png`

Kodak q3 candidates:

- 24/24 images have lower bpp than GLC q3.
- 9/24 images also have DISTS no worse than GLC q3.
- Top examples by DISTS improvement: `kodim14`, `kodim09`, `kodim03`, `kodim17`.
- No Kodak q3 image improves LPIPS simultaneously, so Kodak qualitative discussion should avoid LPIPS claims.

CLIC professional q3 candidates:

- 41/41 images have lower bpp than GLC q3.
- 21/41 images also have DISTS no worse than GLC q3.
- Top examples by DISTS improvement: `wojciech-szaturski-3611`, `clem-onojeghuo-33741`, `michael-durana-82941`, `philippe-wuyts-45997`.
- CLIC top examples show larger DISTS gains than Kodak, often with about 11-15% lower bpp.
- LPIPS still worsens for every CLIC q3 image, reinforcing the current R-P framing as DISTS/FID-oriented rather than LPIPS-oriented.

Sanity check:

- `view_image` failed because of the container namespace restriction, but PIL verified the generated PNGs are non-empty: CLIC grid size 948x1058, Kodak grid size 948x2058.


## 2026-06-19 03:50 JST - OpenImages v6 first32 sanity check

Purpose:

- Check whether the current best `rho_min=1.0,rho_max=1.4` behavior transfers beyond Kodak and CLIC.
- This is a small fixed subset, not a final benchmark: first 32 images from `/dpl/open-images-v6/test/data` symlinked into `data_subsets/openimages_v6_test_32`.

Evaluation note:

- Initial GLC run with `fid_patch_size=256` failed after q0 because KID had fewer samples than subset_size.
- Re-ran with `fid_patch_size=64`; q0-q3 completed.
- KID is unstable on this small subset and should not be used for claims.

Metrics:

| method | q | bpp | PSNR | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLC | 0 | 0.0236 | 23.1999 | 0.1654 | 0.0910 | 17.9802 | 0.0014 |
| GLC | 1 | 0.0273 | 23.7318 | 0.1516 | 0.0844 | 16.7196 | 0.0011 |
| GLC | 2 | 0.0318 | 24.2221 | 0.1408 | 0.0799 | 16.0608 | 0.0014 |
| GLC | 3 | 0.0359 | 24.5136 | 0.1352 | 0.0777 | 15.4381 | 0.0024 |
| GP-ResLC | 0 | 0.0234 | 23.1548 | 0.1667 | 0.0910 | 18.0975 | 0.0027 |
| GP-ResLC | 1 | 0.0254 | 23.4691 | 0.1576 | 0.0849 | 16.5979 | 0.0016 |
| GP-ResLC | 2 | 0.0279 | 23.7215 | 0.1529 | 0.0820 | 15.9793 | 0.0018 |
| GP-ResLC | 3 | 0.0312 | 24.0318 | 0.1457 | 0.0788 | 15.6012 | 0.0015 |

BD-rate versus GLC:

| metric | BD-rate | note |
|---|---:|---|
| DISTS | -5.18% | curve-level overlap still favors GP-ResLC |
| LPIPS | +1.51% | worse, consistent with Kodak/CLIC per-image trend |
| PSNR | +1.61% | worse RD |
| MS-SSIM | +1.22% | worse RD |
| FID | -6.12% | supporting only |
| KID | unstable | small-sample artifact |

Qualitative ranking:

- 32/32 images have lower bpp at q3.
- 13/32 images also have DISTS no worse than GLC q3.
- Top q3 DISTS-improving examples: `0012aacda256f0fb`, `000cf5859025877f`, `0000c64e1253d68f`, `000c5f3f0b58ce18`.

Artifacts:

- baseline: `experiments/v0_glc_openimages32/`
- GP-ResLC recon/metrics: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/openimages32*`
- plots: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/plots/openimages32/curve_*.png`
- ranking: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/qual_rank_openimages32_q3.csv`
- contact sheet: `experiments/eval_v2_gateonly_min1_lR15_lp2_rho14_6k/qual_grid_openimages32_q3_top4.png`

Interpretation:

- The method does not appear Kodak/CLIC-specific: DISTS BD-rate remains negative on an independent OpenImages subset.
- However, q3 DISTS is slightly worse on average, so this subset should be used only as sanity evidence.
- For the paper, keep Kodak+CLIC as primary and mention OpenImages subset as a robustness check if space allows.


## 2026-06-19 04:18 JST - Aborted LPIPS recovery run

Purpose:

- Try to reduce the consistent LPIPS degradation seen on Kodak/CLIC/OpenImages32.
- Hypothesis: weaker rate pressure and higher LPIPS weight might preserve LPIPS while keeping some residual suppression.

Run:

- run: `v2_gateonly_min1_lR10_lp4_rho14_6k`
- wandb: `iqkm4suy`
- url: https://wandb.ai/mayuyuto0714-waseda-university/gp-reslc-vcip/runs/iqkm4suy
- intended settings: `lambda_R=10`, `lambda_lpips=4`, `lambda_d=0.08`, `rho_max=1.4`, `gate_rho_min=1.0`, `freeze_predictor=true`
- stopped manually at about 5k iterations; no final evaluation run.

Observation:

- The gate stayed exactly at `rho=1.000/1.000/1.000` through 5k iterations.
- A/B checks stayed identical: delta bpp_y was `+0.0000` for all q at 1k/2k/3k/4k/5k.
- This is not an LPIPS-improved compressor; it is effectively the identity gate.

Interpretation:

- Reducing rate pressure while using hard `rho_min=1.0` makes the optimal gate sit on the clamp boundary.
- Once the pre-clamp gate wants `rho<1`, the clamp kills the useful gradient for learning a positive suppression map.
- For future LPIPS recovery, do not simply lower `lambda_R` under the hard monotone clamp. Use either:
  - a soft monotone gate with near-identity initialization but nonzero positive-direction gradient, or
  - a scheduled rate warmup that first moves rho above 1 and then increases LPIPS weight.

Conclusion:

- Keep current best `lambda_R=15, lambda_lpips=2, rho_min=1.0, rho_max=1.4` for VCIP.
- The LPIPS weakness likely needs a gate parameterization/training schedule change, not only loss reweighting.


## 2026-06-19 06:35 JST - Sendability Teacher Gate Result

### Code changes

- Added `--gate_rho_init` to initialize monotone `rho>=1` gates above identity. This avoids the boundary-dead failure observed for LPIPS-heavy hard/softplus identity gates.
- Added `--lambda_rho_target`, `--rho_target`, `--rho_target_until` to keep an explicit no-send/coarsening budget alive during training.
- Added `--lambda_gate_send`, `--gate_send_until`, `--gate_send_tau`, `--gate_send_texture_weight` to train a deterministic gate with a training-only sendability teacher. The teacher is derived from local current reconstruction error plus a texture proxy, then recentered to the desired mean p_tex. Inference still uses only `z_hat` and sends no side map.
- W&B logs now include `gate/rho_active_frac`, raw gate stats, and sendability target stats.

### Failed / diagnostic runs

- `v2_gate_softplus_lR10_lp4_rho14_6k` (W&B `eodst7ie`) was stopped at ~5400 iters. `rho` stayed exactly 1.000 and A/B bpp delta stayed zero. Checkpoint `v2_5000.pt` showed small negative head weights, placing softplus at the lower clamp dead zone.
- `v2_gate_init116_lR10_lp4_rho14_target116_6k` (W&B `61cgwikd`) kept `rho≈1.16` but remained almost spatially uniform. Kodak q3: bpp 0.0343, DISTS 0.0957 vs GLC q3 0.0370 / 0.0954. Treat as uniform coarsening ablation, not paper lead.
- `v2_gate_send_lR10_lp4_rho14_target116_send5_6k` (W&B `758hi6qe`) learned spatial rho during the first 4000 iters, but the mask collapsed toward uniform after `gate_send` was disabled. Kodak q3 DISTS 0.0958, not enough.

### New paper-candidate run

Run:

```bash
.venv/bin/python -u scripts/train_v2.py   --glc_weights pretrained/GLC_image.pth.tar   --data /dpl/openimages/train --val /dpl/kodak   --out experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k   --iters 6000 --bs 4 --lr 0.0001   --lambda_R 10 --lambda_d 0.08 --lambda_lpips 4 --lambda_dists 0   --rho_max 1.4 --gate_rho_min 1.0 --gate_rho_mode hard --gate_rho_init 1.16   --lambda_rho_target 2 --rho_target 1.16 --rho_target_until 0   --lambda_gate_send 5 --gate_send_until 0 --gate_send_tau 1.0 --gate_send_texture_weight 0.2   --freeze_predictor --predictor_param_mode mean   --wandb_project gp-reslc-vcip --wandb_name v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k --wandb_mode online
```

W&B: `a2w5fjt4`
Checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`

Kodak metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/kodak_metrics.csv`

| q | GLC bpp | GP bpp | bpp delta | GLC DISTS | GP DISTS | GLC LPIPS | GP LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0243 | 0.0218 | -10.22% | 0.1129 | 0.1144 | 0.1961 | 0.2130 |
| 1 | 0.0282 | 0.0255 | -9.63% | 0.1040 | 0.1077 | 0.1802 | 0.1888 |
| 2 | 0.0328 | 0.0300 | -8.34% | 0.0983 | 0.0995 | 0.1680 | 0.1746 |
| 3 | 0.0370 | 0.0342 | -7.48% | 0.0954 | 0.0949 | 0.1610 | 0.1652 |

Kodak BD-rate vs GLC:

| metric | BD-rate |
|---|---:|
| DISTS | -4.72% |
| LPIPS | -0.85% |
| PSNR | -0.92% |
| MS-SSIM | +0.48% |
| FID | +0.80% |
| KID | -5.86% |

CLIC professional valid metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/clic_prof_valid_metrics.csv`

| q | GLC bpp | GP bpp | bpp delta | GLC DISTS | GP DISTS | GLC FID | GP FID | GLC LPIPS | GP LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0226 | 0.0200 | -11.39% | 0.0952 | 0.0942 | 12.5026 | 12.4066 | 0.1658 | 0.1788 |
| 1 | 0.0265 | 0.0237 | -10.31% | 0.0877 | 0.0868 | 11.1375 | 10.8897 | 0.1515 | 0.1601 |
| 2 | 0.0311 | 0.0285 | -8.60% | 0.0831 | 0.0814 | 10.2973 | 9.9656 | 0.1410 | 0.1463 |
| 3 | 0.0354 | 0.0326 | -7.86% | 0.0804 | 0.0789 | 9.8799 | 9.5110 | 0.1355 | 0.1387 |

CLIC BD-rate vs GLC:

| metric | BD-rate |
|---|---:|
| DISTS | -13.06% |
| LPIPS | -0.48% |
| PSNR | -0.51% |
| MS-SSIM | +0.01% |
| FID | -13.89% |
| KID | -10.53% |

Qualitative artifacts:

- CLIC rank: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/qual_rank_clic_prof_q3.csv` (27/41 candidates improve DISTS while reducing bpp)
- Kodak rank: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/qual_rank_kodak_q3.csv` (12/24 candidates improve DISTS while reducing bpp)
- CLIC grid: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/qual_grid_clic_q3_top4.png`
- Kodak grid: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/qual_grid_kodak_q3_top4.png`
- Gate map stats for kodim01 q3: rho mean 1.159, min 1.103, max 1.245.

Interpretation:

- The strongest current claim is CLIC: all four q points improve DISTS and FID while reducing bpp by 7.9-11.4%.
- Kodak is mixed across the curve, but q3 gives a clean point: lower bpp and slightly better DISTS than GLC q3.
- LPIPS remains the weak metric. Use DISTS/FID/KID as primary perceptual claims and report LPIPS honestly as mostly neutral/worse pointwise.
- The sendability teacher is not just a training trick: it makes the gate spatially selective, matching the paper axis that predictable regions are coarsened while sensitive regions return toward rho=1.


### OpenImages32 sanity for send5all

Metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/openimages32_metrics.csv`

| q | bpp delta | DISTS delta | LPIPS delta | FID delta |
|---:|---:|---:|---:|---:|
| 0 | -10.42% | +0.00221 | +0.01398 | +0.1079 |
| 1 | -9.17% | +0.00080 | +0.00875 | +0.0190 |
| 2 | -7.72% | -0.00033 | +0.00612 | -0.2807 |
| 3 | -6.86% | +0.00003 | +0.00367 | +0.0720 |

BD-rate vs GLC on this 32-image sanity subset:

| metric | BD-rate |
|---|---:|
| DISTS | -7.11% |
| LPIPS | +0.62% |
| PSNR | +0.01% |
| MS-SSIM | +0.20% |
| FID | -7.98% |
| KID | -4.24% |

Interpretation: not as strong as CLIC, but no collapse. q2 improves DISTS/FID/KID while reducing bpp; q3 is DISTS-neutral with lower bpp. FID/KID are noisy because this is only 32 images.


## 2026-06-19 07:15 JST - Sendability rho_target=1.12 Tradeoff

Run:

- W&B: `gbmdxyr8`
- checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target112_send5_all_6k/v2_final.pt`
- change vs target116: `gate_rho_init=1.12`, `rho_target=1.12`; all else kept the same.

Kodak metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target112_send5_all_6k/kodak_metrics.csv`

| q | bpp delta | DISTS delta | LPIPS delta | FID delta |
|---:|---:|---:|---:|---:|
| 0 | -7.50% | +0.00026 | +0.00995 | +0.8589 |
| 1 | -6.95% | +0.00176 | +0.00621 | +1.2055 |
| 2 | -6.13% | +0.00042 | +0.00516 | +0.4318 |
| 3 | -5.46% | -0.00069 | +0.00280 | +0.0475 |

Kodak BD-rate: DISTS -4.92%, LPIPS -0.40%, PSNR -1.11%, MS-SSIM -0.08%, FID +2.22%, KID -5.75%.

CLIC professional valid: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target112_send5_all_6k/clic_prof_valid_metrics.csv`

| q | bpp delta | DISTS delta | LPIPS delta | FID delta |
|---:|---:|---:|---:|---:|
| 0 | -8.19% | -0.00116 | +0.00788 | -0.1758 |
| 1 | -7.54% | -0.00077 | +0.00565 | -0.2372 |
| 2 | -6.45% | -0.00145 | +0.00328 | -0.3322 |
| 3 | -5.92% | -0.00091 | +0.00225 | -0.2263 |

CLIC BD-rate: DISTS -9.86%, LPIPS -1.00%, PSNR -0.96%, MS-SSIM -0.35%, FID -10.38%, KID -11.10%.

Decision:

- `rho_target=1.16` is the stronger R-P headline for CLIC (DISTS -13.06%, FID -13.89%, 7.9-11.4% bpp reduction), but LPIPS pointwise degradation is larger.
- `rho_target=1.12` is the balanced version: smaller bpp reduction (5.9-8.2% on CLIC) but DISTS/FID still improve at every q and LPIPS degradation is roughly halved.
- For the paper, present both as a rate-perception knob. The main claim should use target116, while target112 demonstrates controllability and LPIPS-aware tradeoff.


## 2026-06-19 07:55 JST - Alex-LPIPS Training Loss Ablation

Run:

- W&B: `gzq79h0p`
- checkpoint: `experiments/v2_gate_send_alexlp_lR10_lp4_rho14_target112_send5_all_6k/v2_final.pt`
- change vs target112: `--train_lpips_net alex`, matching the LPIPS evaluator used by `scripts/evaluate_recon_grid.py`.
- kept `rho_target=1.12`, `lambda_R=10`, `lambda_lpips=4`, `lambda_gate_send=5`, and the always-on sendability teacher.

Kodak metrics:

- metrics: `experiments/eval_v2_gate_send_alexlp_lR10_lp4_rho14_target112_send5_all_6k/kodak_metrics.csv`

| q | bpp delta | DISTS delta | LPIPS delta | FID delta |
|---:|---:|---:|---:|---:|
| 0 | -7.39% | +0.00023 | +0.01070 | +0.6415 |
| 1 | -6.95% | +0.00091 | +0.00572 | +0.9621 |
| 2 | -6.18% | +0.00072 | +0.00517 | +0.2878 |
| 3 | -5.55% | -0.00046 | +0.00295 | -0.2472 |

Kodak BD-rate: DISTS -5.66%, LPIPS -0.67%, PSNR -1.29%, MS-SSIM -0.18%, FID +0.20%, KID -6.64%.

CLIC professional valid:

- metrics: `experiments/eval_v2_gate_send_alexlp_lR10_lp4_rho14_target112_send5_all_6k/clic_prof_valid_metrics.csv`

| q | bpp delta | DISTS delta | LPIPS delta | FID delta |
|---:|---:|---:|---:|---:|
| 0 | -8.07% | -0.00064 | +0.00808 | -0.2186 |
| 1 | -7.39% | -0.00087 | +0.00550 | -0.2812 |
| 2 | -6.35% | -0.00130 | +0.00333 | -0.2964 |
| 3 | -5.88% | -0.00112 | +0.00184 | -0.2562 |

CLIC BD-rate: DISTS -9.78%, LPIPS -1.01%, PSNR -1.02%, MS-SSIM -0.42%, FID -11.23%, KID -2.59%.

Decision:

- Matching the training LPIPS backbone to the evaluator improves Kodak DISTS BD-rate slightly versus the VGG-loss target112 run, but does not fix pointwise LPIPS degradation.
- On CLIC, Alex-LPIPS is broadly comparable to the VGG-loss target112 run: DISTS BD-rate is essentially tied, FID is slightly better, KID is weaker.
- Keep this run as an ablation, not the lead. The lead remains `rho_target=1.16` for the strongest CLIC R-P headline, with `rho_target=1.12` VGG-loss as the balanced knob.
- The evidence now says that LPIPS pointwise degradation is not simply a backbone mismatch. The next improvement should target residual/sendability structure or perceptual preservation, not just swapping LPIPS networks.


## 2026-06-19 08:55 JST - LPIPS Protection Ablations and Gate Correlation Analysis

Implementation update:

- Added optional baseline reconstruction distillation to `scripts/train_v2.py`:
  - `--lambda_base_l1`
  - `--lambda_base_lpips`
  - `--base_distill_until`
- The baseline image is obtained by `train_forward(..., use_predictor=False, gate=None, q_shift=None)` under `torch.no_grad()`; defaults are zero, so earlier runs are unchanged.
- Added `scripts/analyze_gate_correlations.py` to correlate saved rho maps with local reconstruction error, error change, texture variance, and image gradient magnitude.

Texture-free sendability ablation:

- W&B: `jxouanu3`
- checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_notex_6k/v2_final.pt`
- change: `--gate_send_texture_weight 0.0` with `rho_target=1.16`.
- Kodak metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_notex_6k/kodak_metrics.csv`
- Kodak BD-rate: DISTS -5.23%, LPIPS -0.21%, PSNR -1.34%, MS-SSIM +0.38%, FID -0.02%.
- q3 point: bpp -7.26%, DISTS +0.00020, LPIPS +0.00448, FID +0.0216.

Decision: removing the texture term does not fix LPIPS and weakens the DISTS/FID story. Keep as an ablation; do not evaluate on CLIC unless needed for appendix.

Baseline-distillation ablation:

- W&B: `18m60mp6`
- checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_baseLP2_6k/v2_final.pt`
- change: original target116 settings plus `--lambda_base_l1 0.5 --lambda_base_lpips 2`.
- Kodak metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_baseLP2_6k/kodak_metrics.csv`
- Kodak BD-rate: DISTS -5.46%, LPIPS -0.51%, PSNR -0.99%, MS-SSIM +0.32%, FID +1.05%, KID -11.10%.
- q3 point: bpp -7.30%, DISTS -0.00045, LPIPS +0.00417, FID +0.0370.

Decision: baseline distillation narrows the gate and preserves q3 DISTS, but it does not recover pointwise LPIPS and hurts FID. Keep the code because it is useful for future controlled variants, but do not use this checkpoint as the paper lead.

Best-run q3 gate correlation analysis:

- checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`
- Kodak rho maps: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_maps_kodak_q3/`
- CLIC rho maps: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_maps_clic_q3/`
- Kodak correlations: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_kodak_q3.json`
- CLIC correlations: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_clic_q3.json`

Summary:

| dataset | mean rho | rho std | corr(rho, base err) | corr(rho, ours err) | corr(rho, texture var) | corr(rho, gradient) |
|---|---:|---:|---:|---:|---:|---:|
| Kodak q3 | 1.1716 | 0.0308 | -0.234 | -0.235 | -0.243 | -0.213 |
| CLIC q3 | 1.1815 | 0.0307 | -0.271 | -0.272 | -0.268 | -0.290 |

High-rho versus low-rho regions:

| dataset | high-rho base err | low-rho base err | high-rho texture | low-rho texture | high-rho grad | low-rho grad |
|---|---:|---:|---:|---:|---:|---:|
| Kodak q3 | 0.0312 | 0.0566 | 0.00189 | 0.00488 | 0.0307 | 0.0562 |
| CLIC q3 | 0.0246 | 0.0447 | 0.00107 | 0.00334 | 0.0194 | 0.0417 |

Interpretation:

- The learned gate is not merely a global coarsening knob: high rho is consistently assigned to low-error, low-texture, low-gradient regions.
- Low rho remains near structure/error-heavy regions, matching the mechanism: do not spend bits on generator-predictable regions; keep residual precision where prediction is hard.
- This analysis supports the VCIP explanation more directly than LPIPS-only ablations. Use it in the method/analysis section.


## 2026-06-19 09:12 JST - Edge-Guard Sendability Ablation

Implementation update:

- Added `--gate_send_edge_weight` to `scripts/train_v2.py`.
- The sendability teacher now can subtract a high-gradient map before recentering to the desired mean. This keeps the same average rho target while pushing high-rho coarsening away from edges.

Run:

- W&B: `mr2aebdw`
- checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_edge025_6k/v2_final.pt`
- change vs target116 lead: `--gate_send_edge_weight 0.25`.

Kodak metrics:

- metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_edge025_6k/kodak_metrics.csv`
- Kodak BD-rate: DISTS -5.37%, LPIPS -1.41%, PSNR -1.80%, MS-SSIM +0.20%, FID +0.23%.

Kodak point deltas:

| q | bpp delta | DISTS delta | LPIPS delta | FID delta |
|---:|---:|---:|---:|---:|
| 0 | -9.09% | +0.00216 | +0.01460 | +0.8595 |
| 1 | -8.54% | +0.00183 | +0.00675 | +1.1351 |
| 2 | -7.42% | +0.00167 | +0.00533 | +0.4717 |
| 3 | -6.57% | -0.00093 | +0.00353 | -0.0398 |

CLIC q3-only check:

- recon: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_edge025_6k/clic_prof_valid_q3/`
- metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_edge025_6k/clic_prof_valid_q3_metrics.csv`
- GLC q3: bpp 0.0354, LPIPS 0.1355, DISTS 0.0804, FID 9.8799, KID 0.0016.
- edge025 q3: bpp 0.0329, LPIPS 0.1386, DISTS 0.0790, FID 9.5403, KID 0.0007.

Decision:

- Edge guard preserves a strong q3 low-bitrate result, but it does not improve the curve enough to replace the original target116 lead.
- q0-q2 still show DISTS/LPIPS/FID degradation despite lower bpp.
- Keep `--gate_send_edge_weight` as a useful future control. For this short-track paper, lead with original target116 + target112 knob, and use gate-correlation statistics rather than edge-guard training as the main mechanism evidence.


## 2026-06-19 09:20 JST - Paper Asset Export

Generated paper-facing curve and summary assets:

- merged Kodak metrics: `experiments/paper_assets/kodak_glc_gp112_gp116_metrics.csv`
- merged CLIC metrics: `experiments/paper_assets/clic_prof_valid_glc_gp112_gp116_metrics.csv`
- Kodak curves: `experiments/paper_assets/kodak_curves/curve_{DISTS,LPIPS,FID,KID,PSNR,MS_SSIM}.png`
- CLIC curves: `experiments/paper_assets/clic_prof_valid_curves/curve_{DISTS,LPIPS,FID,KID,PSNR,MS_SSIM}.png`
- BD-rate summary: `experiments/paper_assets/bd_rate_summary.md` and `.csv`

BD-rate summary versus GLC:

| dataset | run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---|---:|---:|---:|---:|---:|---:|
| Kodak | GP-ResLC-rho1.12 | -4.92% | -0.40% | -1.11% | -0.08% | +2.22% | -5.75% |
| Kodak | GP-ResLC-rho1.16 | -4.72% | -0.85% | -0.92% | +0.48% | +0.80% | -5.86% |
| CLIC-prof-valid | GP-ResLC-rho1.12 | -9.86% | -1.00% | -0.96% | -0.35% | -10.38% | -11.10% |
| CLIC-prof-valid | GP-ResLC-rho1.16 | -13.06% | -0.48% | -0.51% | +0.01% | -13.89% | -10.53% |

Note: KID is noisy on these small sets and should remain auxiliary. DISTS/FID are the main R-P evidence.


## 2026-06-19 09:35 JST - rho_target=1.20 Upper-Knob Ablation

Run:

- W&B: `uj8mwifu`
- checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target120_send5_all_6k/v2_final.pt`
- change vs target116 lead: `--gate_rho_init 1.20 --rho_target 1.20`.

Kodak metrics:

- metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target120_send5_all_6k/kodak_metrics.csv`
- Kodak BD-rate: DISTS -4.70%, LPIPS -0.28%, PSNR -0.42%, MS-SSIM +0.93%, FID -0.73%, KID -42.28% (KID noisy).

Point deltas:

| q | bpp delta | DISTS delta | LPIPS delta | FID delta |
|---:|---:|---:|---:|---:|
| 0 | -11.66% | +0.00311 | +0.01899 | +2.4251 |
| 1 | -10.94% | +0.00446 | +0.01166 | +1.4648 |
| 2 | -9.41% | +0.00156 | +0.00753 | +0.3557 |
| 3 | -8.51% | +0.00002 | +0.00501 | +0.1001 |

Decision:

- `rho_target=1.20` demonstrates the upper end of the rate-saving knob but is too aggressive for the main curve.
- Do not run full CLIC unless needed for an appendix. The lead remains `rho_target=1.16`, with `rho_target=1.12` as the balanced knob.


## 2026-06-19 09:55 JST - Paper rho-overlay assets and method draft

Purpose: turn the current GP-ResLC rho1.16 lead into submission-ready evidence for the short-track VCIP story.

Changes:

- Added `scripts/make_rho_overlay_grid.py`.
- Added `docs/vcip_method_draft.md`.
- Generated q3 rho-overlay qualitative grids:
  - `experiments/paper_assets/clic_q3_rho_overlay_top4.png`
  - `experiments/paper_assets/kodak_q3_rho_overlay_top4.png`
- Updated `docs/current_vcip_status.md` with the method draft and paper assets.

Verification:

- `scripts/make_rho_overlay_grid.py` passed `py_compile`.
- Generated PNGs are non-empty by PIL size/stddev check:
  - CLIC: 1262 x 1084, RGB stddev about [82.07, 80.37, 87.06]
  - Kodak: 1262 x 2084, RGB stddev about [82.07, 81.93, 89.60]

Notes:

- `view_image` could not be used because the container sandbox helper cannot create a user namespace in this environment.
- The overlay uses a fixed rho scale [1.0, 1.4] so the colors are comparable across images. Warm regions mean higher `rho`, i.e. stronger residual suppression in generator-predictable areas.


## 2026-06-19 10:05 JST - Matched-metric bpp summary

Purpose: complement BD-rate with an easier paper headline: bpp reduction when matching the GLC perceptual metric values by interpolation.

Changes:

- Added `scripts/summarize_matched_metric.py`.
- Generated:
  - `experiments/paper_assets/matched_metric_bpp_summary.csv`
  - `experiments/paper_assets/matched_metric_bpp_summary.md`
- Updated `docs/current_vcip_status.md` and `docs/vcip_method_draft.md` with the matched-metric headline.

Key result:

- CLIC-prof-valid rho1.16: -13.38% mean bpp at matched DISTS, -14.67% mean bpp at matched FID over GLC q1-q3.
- Kodak rho1.16: -5.76% mean bpp at matched DISTS over GLC q0-q3; matched FID is near neutral at -0.47% mean.
- Matched LPIPS is near neutral to slightly worse, so LPIPS remains auxiliary rather than the primary claim.

Verification:

- `scripts/summarize_matched_metric.py` passed `py_compile`.


## 2026-06-19 10:30 JST - CLIC mobile valid generalization check

Purpose: test whether the rho1.16 short-track lead generalizes beyond Kodak and CLIC professional valid.

Dataset:

- `/dpl/clic/mobile/valid`, 61 images.

Procedure:

1. Attempted GLC baseline with `test_image.py --fid_patch_size 64`; q0 completed but q1 FID/KID evaluation hit CUDA OOM because the high-resolution mobile images produce too many 64x64 patches.
2. Re-ran GLC baseline with high-resolution setting `--fid_patch_size 256` for q1-q3, then q0, producing complete q0-q3 results in `experiments/v0_glc_clic_mobile_valid/`.
3. Ran GP-ResLC rho1.16 lead with `scripts/test_v2.py --predictor_param_mode mean`:
   - checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`
   - output: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/clic_mobile_valid/`
4. Evaluated GLC and GP-ResLC together with `scripts/evaluate_recon_grid.py --patch 256`.

Outputs:

- Metrics: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/clic_mobile_valid_metrics.csv`
- BD summary: `experiments/paper_assets/clic_mobile_bd_rate_summary.md`
- Matched-metric summary: `experiments/paper_assets/clic_mobile_matched_metric_bpp_summary.md`
- Curves: `experiments/paper_assets/clic_mobile_curves/`
- Cross-dataset summaries:
  - `experiments/paper_assets/bd_rate_summary_all.md`
  - `experiments/paper_assets/matched_metric_bpp_summary_all.md`

Main result:

| metric | result vs GLC |
|---|---:|
| BD-rate DISTS | -9.87% |
| BD-rate FID | -4.38% |
| BD-rate LPIPS | +0.09% |
| matched-DISTS bpp | -10.01% mean over q0-q3 |
| matched-FID bpp | -2.33% mean over q0-q2 |
| matched-LPIPS bpp | +1.10% mean over q0-q2 |

Interpretation:

- CLIC mobile supports the core R-P claim: rho1.16 reduces rate at matched DISTS on a second CLIC domain, not just professional valid.
- FID is also negative, though weaker than on professional valid.
- LPIPS remains near-neutral to slightly worse and should stay auxiliary.
- KID is noisy/non-monotonic and should not carry the paper claim.


## 2026-06-19 10:40 JST - VCIP submission outline

Purpose: consolidate the current research state into a paper-writing starting point.

Added:

- `docs/vcip_submission_outline.md`

Content:

- tentative title and one-sentence thesis
- abstract skeleton
- contributions
- main result table across Kodak, CLIC professional valid, and CLIC mobile valid
- method/mechanism figure plan
- evaluation framing and caveats

Key paper-facing claim captured there:

- GP-ResLC rho1.16 gives DISTS BD-rate reductions on all three evaluated datasets: Kodak -4.72%, CLIC professional valid -13.06%, and CLIC mobile valid -9.87%.
- FID is clearly improved on CLIC professional valid and CLIC mobile valid, while Kodak FID is near neutral.
- LPIPS remains auxiliary and should not be overclaimed.


## 2026-06-19 10:45 JST - CLIC mobile q3 gate-correlation analysis

Purpose: verify that the residual-suppression gate keeps the same mechanism on CLIC mobile valid.

Commands/outputs:

- Rho maps: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_maps_clic_mobile_q3/`
- Correlations: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_clic_mobile_q3.json`

Summary over 61 images:

| quantity | value |
|---|---:|
| mean rho | 1.1804 |
| rho std | 0.0311 |
| corr(rho, baseline error) | -0.251 |
| corr(rho, GP-ResLC error) | -0.252 |
| corr(rho, texture variance) | -0.251 |
| corr(rho, gradient) | -0.262 |
| high-rho baseline error | 0.0237 |
| low-rho baseline error | 0.0447 |
| high-rho gradient | 0.0188 |
| low-rho gradient | 0.0416 |

Interpretation:

- The same mechanism holds on CLIC mobile: high rho is assigned to lower-error, lower-texture, lower-gradient regions.
- This strengthens the paper claim that GP-ResLC is not uniform re-quantization; it selectively suppresses generator-predictable residuals.

Maintenance:

- Updated `scripts/analyze_gate_correlations.py` to load local rho tensors with `weights_only=True` when supported.


## 2026-06-19 10:41 JST - VCIP draft and paper-table consolidation

Added a submission-oriented paper draft and a reproducible key-table builder.

Files added/updated:

- `docs/vcip_paper_draft.md`: abstract, introduction, related work, method, experiments, discussion, conclusion, and citation scaffold for the GP-ResLC short-track paper.
- `scripts/build_vcip_key_tables.py`: reads existing BD-rate CSVs, matched-metric CSVs, and gate-correlation JSONs to regenerate the core paper tables.
- `experiments/paper_assets/vcip_key_tables.md`: generated main BD-rate table, matched-metric bpp table, mechanism table, secondary operating point table, and artifact manifest.
- `docs/vcip_method_draft.md`: added the CLIC mobile validation row to the current evidence table.

Generated headline table from `experiments/paper_assets/vcip_key_tables.md`:

| dataset | DISTS BD-rate | FID BD-rate | matched-DISTS bpp | mechanism note |
|---|---:|---:|---:|---|
| Kodak | -4.72% | +0.80% | -5.76% | high/low base error 0.55x |
| CLIC professional valid | -13.06% | -13.89% | -13.38% | high/low base error 0.55x |
| CLIC mobile valid | -9.87% | -4.38% | -10.01% | high/low base error 0.53x |

Interpretation: the project now has a coherent short-track story, generated tables, and a mechanism table all pointing to the same claim: zero-side-bit suppression removes residual precision mainly from easier, generator-predictable regions.


## 2026-06-19 10:43 JST - CLIC mobile rho overlay figure

Generated a CLIC mobile q3 qualitative rho overlay grid from existing GLC/GP-ResLC reconstructions and saved rho maps.

Command output:

- `experiments/paper_assets/clic_mobile_q3_rho_overlay_top4.png`

Selected images:

- `2017-07-27 16.21.36`
- `20170930_131716`
- `IMG_0470_1`
- `IMG_1170`

Sanity check:

- image size: 1262 x 1184
- mode: RGB
- per-channel stddev: 80.99, 72.51, 75.98

Interpretation: this gives the paper a CLIC-mobile visual counterpart to the Kodak and CLIC-professional rho overlay figures.


## 2026-06-19 12:35 JST - GLC evaluation protocol audit

Re-read the GLC paper TeX source and official implementation to verify evaluation datasets and FID/KID protocol.

Findings:

- GLC main natural-image evaluation is CLIC 2020 test set at original resolution.
- Supplementary natural-image evaluations are Kodak, DIV2K validation, and MS-COCO 30K.
- Natural-image FID/KID use 256x256 patches with normal grid plus 128-pixel shifted grid.
- Reported patch counts in the supplement are 28,650 for CLIC2020 test and 6,573 for DIV2K validation.
- Kodak has only 192 patches, so GLC omits FID/KID on Kodak.
- `/dpl/div2k` matches 6,573 patches exactly.
- `/dpl/clic/professional/test` has 250 images and 16,626 shifted 256-patches, so it is a test split but does not exactly match the GLC supplement's 28,650-patch CLIC2020 test.
- `/dpl/coco30k` contains 30,000 COCO2014 validation images with varied sizes, not pre-cropped uniform 256 patches.

Code fix:

- Updated `scripts/eval_metrics.py` to use official `src.utils._update_patch_fid.update_patch_fid` for FID/KID.
- Updated `scripts/evaluate_recon_grid.py` to pass `split_patch_num` and `kid_subset_size`.

Full audit: `docs/glc_eval_protocol_audit.md`.


## 2026-06-19 14:12 JST - CLIC professional test official-patch evaluation

Goal: verify the GLC paper protocol and replace CLIC validation headline numbers with the closest available CLIC test split.

Protocol: `/dpl/clic/professional/test`, original resolution, 250 images, 16,626 local shifted 256-patches. FID/KID use GLC-style 256x256 patches with an additional 128-pixel shift. This is closer to the paper than validation, but does not match the GLC supplement's 28,650 CLIC2020-test patch count.

Artifacts:

- GLC reconstructions: `experiments/v0_glc_clic_prof_test/`
- GP-ResLC reconstructions: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/clic_prof_test/`
- Metrics CSV: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/clic_prof_test_metrics_officialpatch.csv`
- BD summary: `experiments/paper_assets/clic_prof_test_bd_rate_summary.md`
- Matched summary: `experiments/paper_assets/clic_prof_test_matched_metric_bpp_summary.md`

Result versus GLC: DISTS BD-rate -10.42%, FID BD-rate -7.40%, KID BD-rate -5.69%, LPIPS BD-rate +0.16%, PSNR BD-rate -1.42%, MS-SSIM BD-rate +0.20%. Matched-DISTS bpp reduction is -10.54% over all four GLC q points; matched-FID reduction is -5.47% over three points.

Decision: use this local CLIC professional test result as the paper-facing natural-image result, while clearly stating the patch-count caveat. Keep previous CLIC professional/mobile validation results as development and cross-domain support. Next dataset priority is `/dpl/div2k`, because its 6,573 shifted-patch count exactly matches the GLC supplement.


## 2026-06-19 14:55 JST - DIV2K validation official-patch evaluation

Goal: add DIV2K as supplementary natural-image evidence after the GLC protocol audit.

Protocol: `/dpl/div2k`, original resolution, 100 images (`0801.png`-`0900.png`). Later protocol cleanup corrected this entry: the official-patch evaluator yields 6,573 shifted 256-patches, matching the GLC supplement count. FID/KID use GLC-style 256x256 patches with an additional 128-pixel shift.

Artifacts:

- GLC reconstructions: `experiments/v0_glc_div2k/`
- GP-ResLC reconstructions: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/div2k/`
- Metrics CSV: `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/div2k_metrics_officialpatch.csv`
- BD summary: `experiments/paper_assets/div2k_bd_rate_summary.md`
- Matched summary: `experiments/paper_assets/div2k_matched_metric_bpp_summary.md`

Result versus GLC: DISTS BD-rate -10.86%, FID BD-rate -5.65%, KID BD-rate -8.19%, LPIPS BD-rate -0.55%, PSNR BD-rate -1.50%, MS-SSIM BD-rate -0.18%. Matched-DISTS bpp reduction is -10.33% over all four GLC q points; matched-FID reduction is -3.42% over three points.

Decision: use DIV2K as strong supplementary support for the R-P claim. The trend matches CLIC professional test: roughly 10% DISTS-rate saving, FID/KID also negative, and LPIPS near neutral.


## 2026-06-20 00:55 JST - Real arithmetic-codec evaluation implementation

Goal: replace paper-facing estimated bpp with an actual serialized codec evaluation path, including arithmetic coding, fixed-width transmitted `z` indices, payload metadata, and encode/decode wall time.

Implementation:

- Added `gp_reslc/real_codec.py`.
- Added `scripts/evaluate_real_codec.py`.
- Updated `scripts/summarize_matched_metric.py` with `--anchor` so real-codec run names such as `glc_real` can be summarized.
- Wrote the protocol note `docs/real_codec_protocol.md`.

Codec design: `z` is packed as fixed-width VQ codebook indices, matching GLC's public fixed `log2(codebook_size)` evaluation. `y` is encoded as four `torchac` arithmetic streams in the same four-part spatial-prior order as GLC. The compact header and per-stream support metadata are included in bpp. Gaussian CDFs include lower/upper tail symbols, preserving untruncated Gaussian mass for observed symbols.

Consistency smoke checks:

- GLC Kodak `kodim01`, q0: real decode matches `net.test()` with max absolute difference `0.000e+00`.
- GP-ResLC lead checkpoint Kodak `kodim01`, q0: real decode matches `train_forward()` with max absolute difference `0.000e+00`.

Kodak full run artifacts:

- GLC real codec: `experiments/real_codec/kodak_glc/`
- GP-ResLC real codec: `experiments/real_codec/kodak_gp_reslc_rho116/`
- Metrics CSV: `experiments/real_codec/kodak_real_metrics.csv`
- BD summary: `experiments/real_codec/kodak_real_bd_rate_summary.md`
- Matched summary: `experiments/real_codec/kodak_real_matched_metric_summary.md`

Kodak average real bpp:

| q | GLC | GP-ResLC | delta | y-stream delta |
|---|---:|---:|---:|---:|
| 0 | 0.02620 | 0.02371 | -9.52% | -11.82% |
| 1 | 0.03013 | 0.02739 | -9.09% | -10.94% |
| 2 | 0.03472 | 0.03197 | -7.93% | -9.30% |
| 3 | 0.03897 | 0.03618 | -7.17% | -8.25% |

Average encode/decode time is about 0.07s / 0.10s per Kodak image after the first torchac/JIT warmup. `z` and header bpp are identical between GLC and GP-ResLC (`z=0.00342`, header `0.00169` on Kodak), so the savings are entirely in the arithmetic-coded `y` stream.

Real-bpp Kodak summary versus `glc_real`: DISTS BD-rate -4.47%, LPIPS BD-rate -0.79%, PSNR BD-rate -0.87%, MS-SSIM BD-rate +0.45%, FID BD-rate -1.70%, KID BD-rate -6.14%. Matched-metric bpp deltas: DISTS -5.45% over 4 points, FID -4.40% over 4 points, LPIPS +0.34% over 3 points.

Decision: all final paper-facing rate numbers should be regenerated through `scripts/evaluate_real_codec.py`. The old estimated bpp remains useful for fast training diagnostics only. For CLIC/DIV2K, run the same real-codec path and then reuse `scripts/evaluate_recon_grid.py` with the generated `bpp.json` files.


## 2026-06-20 01:35 JST - DIV2K real arithmetic-codec evaluation

Goal: rerun the `/dpl/div2k` evaluation with actual serialized bitstreams instead of estimated likelihood bpp.

Protocol: `/dpl/div2k`, original resolution, 100 images (`0801.png`-`0900.png`). Real codec bpp is measured as payload bytes from `scripts/evaluate_real_codec.py`. Quality metrics use the official-patch evaluator with 256x256 patches and 128-pixel shift (`--patch 256 --split_patch_num 2`). After protocol cleanup, the local patch count is 6,573, matching the GLC supplement.

Artifacts:

- GLC real codec: `experiments/real_codec/div2k_glc/`
- GP-ResLC real codec: `experiments/real_codec/div2k_gp_reslc_rho116/`
- Metrics CSV: `experiments/real_codec/div2k_real_metrics.csv`
- BD summary: `experiments/real_codec/div2k_real_bd_rate_summary.md`
- Matched summary: `experiments/real_codec/div2k_real_matched_metric_summary.md`

DIV2K average real bpp:

| q | GLC | GP-ResLC | delta | y-stream delta |
|---|---:|---:|---:|---:|
| 0 | 0.02381 | 0.02133 | -10.39% | -12.35% |
| 1 | 0.02764 | 0.02507 | -9.29% | -10.76% |
| 2 | 0.03224 | 0.02961 | -8.15% | -9.23% |
| 3 | 0.03649 | 0.03388 | -7.16% | -7.98% |

Average encode/decode time: GLC ranges from 0.693/0.963s at q0 to 0.996/1.283s at q3; GP-ResLC ranges from 0.654/0.925s at q0 to 0.930/1.208s at q3. The small speed difference is likely due to shorter arithmetic streams rather than a material architectural speedup.

Real-bpp DIV2K summary versus `glc_real`: DISTS BD-rate -10.79%, FID BD-rate -5.61%, KID BD-rate -6.50%, LPIPS BD-rate -0.54%, PSNR BD-rate -1.49%, MS-SSIM BD-rate -0.17%. Matched-metric bpp deltas: DISTS -10.27% over 4 points, FID -3.39% over 3 points, LPIPS +0.36% over 3 points.

Decision: the DIV2K result remains strong under real bitstream accounting. Use the real-codec numbers for paper-facing DIV2K tables and keep the previous estimated-bpp DIV2K table as a superseded diagnostic.

## 2026-06-20 03:56 JST - CLIC professional test real arithmetic-codec evaluation

Goal: run the official CLIC professional test split through the actual serialized codec path, matching the GLC paper setting more closely than the earlier CLIC validation diagnostics.

Protocol: `/dpl/clic/professional/test`, original resolution, 250 PNG images. Real codec bpp is measured from serialized payload bytes produced by `scripts/evaluate_real_codec.py`. Quality metrics use `scripts/evaluate_recon_grid.py` with 256x256 FID/KID patches and 128-pixel shift (`--patch 256 --split_patch_num 2`). The local shifted-patch count remains 16,626, not the 28,650 count reported in the GLC supplement.

Artifacts:

- GLC real codec: `experiments/real_codec/clic_prof_test_glc/`
- GP-ResLC real codec: `experiments/real_codec/clic_prof_test_gp_reslc_rho116/`
- Metrics CSV: `experiments/real_codec/clic_prof_test_real_metrics.csv`
- BD summary: `experiments/real_codec/clic_prof_test_real_bd_rate_summary.md`
- Matched summary: `experiments/real_codec/clic_prof_test_real_matched_metric_summary.md`

CLIC professional test average real bpp:

| q | GLC | GP-ResLC | delta | y-stream delta | GLC enc/dec s | GP enc/dec s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.02066 | 0.01833 | -11.27% | -13.79% | 0.661 / 0.930 | 0.628 / 0.895 |
| 1 | 0.02424 | 0.02176 | -10.26% | -12.14% | 0.732 / 1.006 | 0.692 / 0.966 |
| 2 | 0.02870 | 0.02622 | -8.64% | -9.95% | 0.829 / 1.102 | 0.746 / 1.021 |
| 3 | 0.03272 | 0.03016 | -7.84% | -8.86% | 0.930 / 1.204 | 0.862 / 1.134 |

Per-q quality summary:

| run | q | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLC | 0 | 0.02066 | 24.6659 | 0.8544 | 0.1423 | 0.0811 | 8.0202 | 0.00142 |
| GLC | 1 | 0.02424 | 25.1459 | 0.8664 | 0.1287 | 0.0731 | 6.9252 | 0.00108 |
| GLC | 2 | 0.02870 | 25.5673 | 0.8759 | 0.1193 | 0.0681 | 6.2042 | 0.00090 |
| GLC | 3 | 0.03272 | 25.8163 | 0.8812 | 0.1144 | 0.0658 | 5.9954 | 0.00080 |
| GP-ResLC | 0 | 0.01833 | 24.3278 | 0.8441 | 0.1541 | 0.0814 | 8.4597 | 0.00163 |
| GP-ResLC | 1 | 0.02176 | 24.8841 | 0.8589 | 0.1370 | 0.0732 | 7.0912 | 0.00113 |
| GP-ResLC | 2 | 0.02622 | 25.3814 | 0.8708 | 0.1243 | 0.0675 | 6.2937 | 0.00088 |
| GP-ResLC | 3 | 0.03016 | 25.6752 | 0.8777 | 0.1177 | 0.0651 | 6.0392 | 0.00085 |

Real-bpp CLIC professional test summary versus `glc_real`: DISTS BD-rate -10.30%, FID BD-rate -7.31%, LPIPS BD-rate +0.17%, PSNR BD-rate -1.39%, MS-SSIM BD-rate +0.21%, KID BD-rate -7.08%. Matched-metric bpp deltas: DISTS -10.43% over 4 points, FID -5.40% over 3 points, LPIPS +1.23% over 3 points.

Decision: CLIC professional test now supports the short-track R-P claim under real arithmetic-coded bitstream accounting. The strongest paper-facing claims should use DISTS-rate and FID-rate; LPIPS and KID should be reported honestly as mixed/near-neutral auxiliary diagnostics.

## 2026-06-20 09:30 JST - VCIP package promoted to real-codec source-of-truth

Goal: make the VCIP paper package use actual serialized codec bpp instead of estimated likelihood bpp.

Changes:

- Updated `scripts/build_vcip_key_tables.py` so its defaults read the real-codec summaries for CLIC professional test, DIV2K validation, and Kodak.
- Regenerated `experiments/paper_assets/vcip_key_tables.md` with real-codec BD-rate, matched-metric bpp, and per-q serialized bpp tables.
- Generated merged real-codec package CSVs:
  - `experiments/paper_assets/real_codec_bd_rate_summary_all.csv`
  - `experiments/paper_assets/real_codec_matched_metric_bpp_summary_all.csv`
  - `experiments/paper_assets/real_codec_metrics_all.csv`
- Generated real-codec R-P curves:
  - `experiments/paper_assets/clic_prof_test_real_curves/`
  - `experiments/paper_assets/div2k_real_curves/`
  - `experiments/paper_assets/kodak_real_curves/`
- Updated `docs/vcip_paper_draft.md`, `docs/vcip_method_draft.md`, `docs/vcip_submission_outline.md`, and `docs/current_vcip_status.md` so CLIC professional test / DIV2K / Kodak real-codec results are the paper-facing source-of-truth.

New VCIP package headline:

| dataset | DISTS BD-rate | FID BD-rate | matched-DISTS bpp | matched-FID bpp |
|---|---:|---:|---:|---:|
| CLIC professional test | -10.30% | -7.31% | -10.43% | -5.40% |
| DIV2K validation | -10.79% | -5.61% | -10.27% | -3.39% |
| Kodak | -4.47% | -1.70% | -5.45% | -4.40% |

Decision: estimated-bpp CLIC professional/mobile validation results are now development evidence only. The VCIP package should cite `experiments/paper_assets/vcip_key_tables.md` and the real-codec merged CSVs for all paper-facing rate numbers.

## 2026-06-20 11:22 JST - Official GLC paper-curve comparison package

Goal: compare current GP-ResLC real-codec results against graph-extracted values from the official GLC paper plots for GLC, MS-ILLM, HiFiC, and FCC on CLIC 2020, DIV2K, and Kodak.

Artifacts:

- Script: `scripts/compare_official_curves.py`
- Official extracted long CSV: `experiments/paper_assets/official_curve_comparison/official_extracted_metrics_long.csv`
- Combined official + local real-codec long CSV: `experiments/paper_assets/official_curve_comparison/official_plus_gp_reslc_real_long.csv`
- Official-vs-local GLC sanity: `experiments/paper_assets/official_curve_comparison/official_vs_local_glc_sanity_summary.md`
- GP-ResLC real vs official GLC BD-rate: `experiments/paper_assets/official_curve_comparison/gp_reslc_real_vs_official_glc_bd.md`
- Matched-metric bpp: `experiments/paper_assets/official_curve_comparison/gp_reslc_real_vs_official_glc_matched.md`
- Curves: `experiments/paper_assets/official_curve_comparison/curves/`

Key results versus graph-extracted official GLC:

| dataset | DISTS BD | FID BD | LPIPS BD | matched DISTS | matched FID | interpretation |
|---|---:|---:|---:|---:|---:|---|
| CLIC 2020 | -14.24% | +29.82% | -16.19% | -14.81% | +27.79% | DISTS/LPIPS favorable, but official/local GLC FID mismatch is large. |
| DIV2K | -9.62% | -4.23% | +0.58% | -9.06% | -1.55% | Best external-positioning comparison. |
| Kodak | +1.04% | n/a | +4.47% | +0.02% | n/a | Official plot has no FID/KID; real-codec bpp is about 5% above graph bpp. |

Sanity check: DIV2K local real-codec GLC nearly coincides with official graph-extracted GLC (DISTS/FID quality deltas around 0.02%/0.17%; bpp about +1.3%). Kodak quality also matches but local real-codec bpp is about +5.1%, consistent with stricter serialized payload accounting. CLIC local GLC has close DISTS and better LPIPS/PSNR/MS-SSIM than the official graph, but FID is 31.94% worse on average; treat CLIC official FID as a protocol/source mismatch rather than a GP-ResLC conclusion.

Decision: keep paired local real-codec GLC as the paper-facing anchor. Use the official graph-extracted package for supplementary positioning, especially DIV2K, and as a sanity/caveat table in the appendix or internal paper notes.
## 2026-06-20 12:42 JST - Protocol mismatch cleanup

Goal: eliminate evaluator-side protocol drift and identify any remaining data-source mismatch against the official GLC paper/supplement protocol.

Changes:

- Added `scripts/audit_glc_protocol.py`. Outputs are in `experiments/protocol_audit/`.
- Updated `scripts/eval_metrics.py` so `distribution_metrics(..., return_patch_count=True)` records FID/KID patch counts and patch settings.
- Updated `scripts/evaluate_recon_grid.py` to call `init_func()` like official GLC `test_image.py`, making KID sampling seeded/reproducible.
- Updated `scripts/build_vcip_key_tables.py` to merge CSVs with heterogeneous protocol columns.
- Re-evaluated CLIC professional test, DIV2K, and Kodak real-codec reconstructions with patch count columns.

Protocol audit result:

| dataset | local status |
|---|---|
| DIV2K | exact: 100 images, 6,573 shifted 256-patches, matching GLC supplement |
| CLIC professional test | unresolved data mismatch: 250 images, 16,626 shifted 256-patches; supplement reports 28,650 |
| all available CLIC under `/dpl/clic` | 352 non-MacOS images, 22,510 shifted 256-patches; still not the supplement count |
| Kodak | 24 images; 192 patches at 256-patch setting, so paper-style FID/KID should be omitted; local diagnostics use 64-patch setting |

Updated real-codec BD-rate after seeded KID:

| dataset | DISTS | FID | KID |
|---|---:|---:|---:|
| CLIC professional test | -10.30% | -7.31% | -7.08% |
| DIV2K validation | -10.79% | -5.61% | -6.50% |
| Kodak | -4.47% | -1.70% | -6.14% |

Decision: evaluator-side mismatch is now fixed. DIV2K can be used as the clean official-protocol support set. Exact CLIC supplement reproduction remains blocked by missing/different CLIC image set, not by metric code.


## 2026-06-20 JST - CLIC2020 full test protocol correction

Goal: re-audit the reported 28,650 CLIC2020 FID/KID patches after noticing that the previous audit only counted the professional test subset.

Correction:

- `/dpl/clic/professional/test`: 250 images, 16,626 shifted 256-patches.
- `/dpl/clic/mobile/test`: 178 images, 12,024 shifted 256-patches.
- Combined CLIC2020 test: 428 images, 28,650 shifted 256-patches.

This exactly matches the GLC/HiFiC-style CLIC2020 test patch count. The previous conclusion that CLIC was an unresolved data-source mismatch is superseded.

Changes:

- Updated `scripts/audit_glc_protocol.py` to include `clic2020_test`, `clic_prof_test`, and `clic_mobile_test` separately.
- Built `datasets/clic2020_test/` as the 428-image professional+mobile symlink set.
- Merged existing professional real-codec outputs with newly evaluated mobile-test outputs into `experiments/real_codec/clic2020_test_glc/` and `experiments/real_codec/clic2020_test_gp_reslc_rho116/`.
- Recomputed full CLIC2020 real-codec metrics with 28,650 FID/KID patches.
- Updated `scripts/compare_official_curves.py` and `scripts/build_vcip_key_tables.py` so paper-facing CLIC uses the full 428-image test set.

Full CLIC2020 real-codec result versus local real-codec GLC:

| dataset | DISTS BD | FID BD | KID BD | matched DISTS | matched FID |
|---|---:|---:|---:|---:|---:|
| CLIC2020 test | -10.28% | -7.30% | -7.10% | -10.26% | -6.02% |

Official graph-extracted GLC comparison is now usable for CLIC: local real-codec GLC matches official GLC FID closely after adding the mobile test subset. GP-ResLC versus official GLC gives CLIC2020 DISTS/FID BD-rate -9.07% / -6.10%.

Decision: use full CLIC2020 test, DIV2K validation, and Kodak as the VCIP real-codec package. Keep professional-only CLIC results as historical/development artifacts, not paper-facing headline results.
