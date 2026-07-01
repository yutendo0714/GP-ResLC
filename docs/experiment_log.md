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
- This is a small fixed subset, not a final benchmark: first 32 images from `/dpl/open-images-v6/test/data` symlinked into `data/subsets/openimages_v6_test_32`.

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
- Built canonical `data/clic2020_test_combined/` as the 428-image professional+mobile symlink set.
- Merged existing professional real-codec outputs with newly evaluated mobile-test outputs into `experiments/real_codec/clic2020_test_glc/` and `experiments/real_codec/clic2020_test_gp_reslc_rho116/`.
- Recomputed full CLIC2020 real-codec metrics with 28,650 FID/KID patches.
- Updated `scripts/compare_official_curves.py` and `scripts/build_vcip_key_tables.py` so paper-facing CLIC uses the full 428-image test set.

Full CLIC2020 real-codec result versus local real-codec GLC:

| dataset | DISTS BD | FID BD | KID BD | matched DISTS | matched FID |
|---|---:|---:|---:|---:|---:|
| CLIC2020 test | -10.28% | -7.30% | -7.10% | -10.26% | -6.02% |

Official graph-extracted GLC comparison is now usable for CLIC: local real-codec GLC matches official GLC FID closely after adding the mobile test subset. GP-ResLC versus official GLC gives CLIC2020 DISTS/FID BD-rate -9.07% / -6.10%.

Decision: use full CLIC2020 test, DIV2K validation, and Kodak as the VCIP real-codec package. Keep professional-only CLIC results as historical/development artifacts, not paper-facing headline results.

## 2026-06-20 JST - Latent-residual complete-design implementation start

Goal: move GP-ResLC closer to the original design: predict the generator-recoverable latent component from transmitted `z_hat` and q, then entropy-code only the unpredictable residual.

Changes:

- `gp_reslc/real_codec.py` now supports `predictor_param_mode=latent_residual` for actual arithmetic-coded real bitstreams.
- The real encoder codes symbols from `y_scaled - base_mean - mu_theta(z_hat, q)`, while the decoder recomputes `mu_theta(z_hat, q)` from transmitted `z_hat` and adds it back. No side map is transmitted.
- `scripts/evaluate_real_codec.py` now accepts `--predictor_param_mode latent_residual`.
- `scripts/train_v2.py` now has `--lambda_mean_pred` for q-conditioned latent residual training; in `latent_residual` mode it applies Smooth-L1 to `mu_theta(z_hat, q)` versus `y_scaled - base_mean`.

Verification:

- One-image Kodak real-codec smoke with existing checkpoint and `latent_residual` passed: real decode matches estimated train_forward with `max_abs=0.000e+00`.
- Two-iteration V2 training smoke passed with `--no_gate --predictor_param_mode latent_residual --lambda_mean_pred 0.05`.

Next run:

`v3_latent_residual_lR10_lp4_mp005_nogate_12k`: no gate, all-q q-conditioned latent residual predictor, frozen GLC, OpenImages training, Kodak quick A/B validation. This is a complete-design warm-start route rather than the current rho-gate shortcut.

## 2026-06-20 JST - V3 latent-residual direct R-P run stopped early

Run: `v3_latent_residual_lR10_lp4_mp005_nogate_12k`, W&B `sgs83602`.

Configuration: frozen GLC, q-conditioned V2, no gate, `predictor_param_mode=latent_residual`, `lambda_R=10`, `lambda_lpips=4`, `lambda_d=0.08`, `lambda_mean_pred=0.05`.

Observation through ~2.7k iterations:

- A/B at it=1000 and it=2000 showed positive `delta_bpp_y` on all q values, so the learned latent residual path was using slightly more y bits than baseline.
- `latent_pred_abs` grew only to roughly 0.009-0.011 while target residual magnitude was roughly 0.03-0.04.
- Direct R-P optimization is therefore too weak to learn the generator-predictable latent mean from scratch in the q-conditioned setting.

Decision: stop early and pivot to staged residual-target pretraining. Next run should strongly train `mu_theta(z_hat,q)` toward `y_scaled - base_mean` with weak rate pressure, then fine-tune perceptually after A/B bpp_y turns negative.

## 2026-06-20 JST - Latent-residual pretraining run stopped early

Run: `v3_latent_residual_pretrain_mp5_lR0p1_nogate_8k`, W&B `nju3here`.

Configuration: frozen GLC, q-conditioned V2, no gate, `predictor_param_mode=latent_residual`, `lambda_mean_pred=5.0`, `lambda_R=0.1`, no distortion/perceptual loss.

Observation through it=2000:

- `latent_pred_abs` increased to about 0.014-0.022 while target residual magnitude was about 0.03-0.045.
- A/B bpp_y became strongly worse: at it=2000, q0/q1/q2/q3 deltas were roughly +0.0045/+0.0040/+0.0047/+0.0054.
- Therefore forcing the predictor toward the full residual target breaks the frozen GLC four-part entropy model rather than reducing entropy.

Interpretation:

- In frozen GLC, `base_mean` and the spatial prior are co-adapted. A global `z_hat,q -> residual mean` added at every spatial-prior stage can shift contexts out of the distribution learned by GLC.
- The full design likely needs either a conservative bounded residual mean, q-specific training, stage-aware residual prediction, or partial unfreezing of the entropy/fusion modules.

Next run: q-specific V1 latent residual with `predictor_delta_bound=0.02`, stronger rate pressure, and no gate. This tests whether a small conservative residual subtraction can help without corrupting the spatial context.

## 2026-06-20 JST - V1 bounded latent-residual q2 stopped early

Run: `v3_latent_residual_v1q2_bound002_lR10_mp1_4k`, W&B `iv4elzaj`.

Configuration: q-specific V1, frozen GLC, no gate, `predictor_param_mode=latent_residual`, `predictor_delta_bound=0.02`, `lambda_R=10`, `lambda_mean_pred=1.0`.

Observation through it=1000:

- A/B `delta_bpp_y` stayed slightly positive: it=500 `+0.0001`, it=1000 `+0.0001`.
- The conservative global residual mean did not reduce the arithmetic model entropy, even when q-specific and bounded.

Decision:

- Stop this branch early. Frozen GLC does not appear to benefit from adding a decoder-recomputable residual mean that depends only on `z_hat`/q.
- Pivot to a stage-aware residual predictor that adds a small mean correction inside each four-part spatial prior stage, conditioned only on information already available at the decoder (`common_params` and, for stages 1-3, `y_hat_so_far`). This keeps the original GP-ResLC principle but moves the prediction to the correct autoregressive context.

Implementation note:

- Added `StageResidualPredictor` and `predictor_param_mode=stage_latent_residual` to `gp_reslc/prior_predictor.py` and `scripts/train_v1.py`.
- Smoke training passed; start state is exactly GLC-equivalent and checkpointing now saves `stage_residual_predictor`.

## 2026-06-20 JST - Stage-aware quantization gate real-codec q2 result

Motivation: latent-residual mean subtraction did not move rounded residual symbols because the learned mean corrections were much smaller than one quantized symbol. To keep the original principle while affecting actual transmitted bits, I added a decoder-recomputable stage-aware quantization gate: each four-part prior stage predicts `rho >= 1` from information already available to the decoder (`common_params`, and `y_hat_so_far` for stages 1-3). Larger `rho` means coarser residual quantization, i.e. predictable residuals are sent with fewer bits.

Implementation:

- Added `StageQuantGate` and `predictor_param_mode=stage_quant_gate`.
- Added a real arithmetic-codec path for stage quantization; one-image consistency passed with `max_abs=0.000e+00`.
- Added optional DISTS loss to `scripts/train_v1.py`.

Runs:

- Probe: `v3_stage_quant_v1q2_rhomax20_lR50_lr3e4_probe2k`, W&B `f40mzry2`. Strong rate pressure moved rho aggressively (`rho_mean` about 1.46) and reduced q2 Kodak crop `bpp_y` by about 0.006, but PSNR dropped about 0.7 dB. Full Kodak q2 real-codec metrics: `bpp=0.02922`, `DISTS=0.1070`, `LPIPS=0.1912`, `PSNR=21.39`. Too aggressive for q2-quality claims, but confirms the no-send mechanism works in real codec.
- Balanced DISTS run: `v3_stage_quant_v1q2_rhomax20_lR30_lr2e4_lp4_dists8_d02_3k`, W&B `y6z248p3`. W&B crop A/B summary: `bpp_y 0.03226 -> 0.02818`, `delta=-0.00408`, PSNR `19.93 -> 19.70`, `rho_mean=1.2446`. Full Kodak q2 real-codec metrics: `bpp=0.03088`, `DISTS=0.1002`, `LPIPS=0.1787`, `PSNR=21.71`, `MS-SSIM=0.7675`.

Comparison against existing Kodak real-codec table:

- GLC q1: `bpp=0.03013`, `DISTS=0.1040`, `LPIPS=0.1802`, `PSNR=21.73`. Stage-quant q2 at similar bpp improves DISTS/LPIPS.
- GLC q2: `bpp=0.03472`, `DISTS=0.0983`, `LPIPS=0.1680`, `PSNR=22.07`. Stage-quant q2 is about 11% lower bpp with modest quality loss.
- GP-ResLC rho1.16 q2: `bpp=0.03197`, `DISTS=0.0995`, `LPIPS=0.1746`, `PSNR=21.88`. Stage-quant q2 is lower bpp but slightly worse DISTS/LPIPS; this is promising but not yet dominant over the current best curve.

Decision:

- Stage-aware quantization is the first complete-design variant that clearly moves actual arithmetic-coded bits while staying on the original axis.
- Next: train q0/q1/q3 with the balanced DISTS setting to obtain a provisional real-codec curve, then compare BD-rate against GLC and rho1.16.

## 2026-06-20 JST - Stage-quant Kodak curve audit

Goal: check whether the more design-faithful stage-aware quantization gate actually beats the current rho1.16 real-codec package on a curve, not only at one visually plausible operating point.

Artifacts:

- Curve CSV: `experiments/real_codec/kodak_stage_quant_curve_metrics.csv`
- BD summary: `experiments/real_codec/kodak_stage_quant_bd_rate_summary.md`
- Matched-metric summary: `experiments/real_codec/kodak_stage_quant_matched_metric_summary.md`

Protocol note: Kodak FID/KID are intentionally excluded from this comparison because the stage-quant diagnostic runs and the historical Kodak real-codec table used different patch settings, and GLC itself does not use Kodak FID/KID as a main paper metric. The comparison below uses real serialized bpp and full-reference PSNR/MS-SSIM/LPIPS/DISTS.

Stage-quant real-codec points:

| q | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS | note |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.02620 | 0.02109 | 21.3202 | 0.7487 | 0.1961 | 0.1129 | no-op GLC q0 anchor; q0 learned gate did not move |
| 1 | 0.02908 | 0.02398 | 21.6100 | 0.7631 | 0.1843 | 0.1045 | learned stage gate |
| 2 | 0.03088 | 0.02578 | 21.7133 | 0.7675 | 0.1787 | 0.1002 | learned stage gate |
| 3 | 0.03366 | 0.02855 | 21.8304 | 0.7732 | 0.1752 | 0.0982 | learned stage gate |

Curve comparison versus GLC real codec on Kodak:

| run | DISTS BD-rate | LPIPS BD-rate | PSNR BD-rate | MS-SSIM BD-rate |
|---|---:|---:|---:|---:|
| rho1.16 real | -4.47% | -0.79% | -0.87% | +0.45% |
| stage-quant DISTS | -1.23% | +1.52% | +1.28% | +0.90% |

Matched-DISTS bpp summary:

- rho1.16 real: mean `-5.45%` over four GLC targets.
- stage-quant DISTS: mean `-2.03%` over three GLC targets.

Interpretation:

- Stage-quant is closer to the original GP-ResLC mechanism: the decoder predicts which residual precision is unnecessary, and no extra rho/mask side stream is sent.
- However, the current q-specific stage-quant training is not yet the paper lead. It underperforms rho1.16 on Kodak DISTS BD-rate and worsens LPIPS/PSNR/MS-SSIM curve summaries.
- The main failure mode is not codec mismatch. It is optimization/control: the gate can reduce real y-stream bits, but the reconstruction penalty is not keeping the generated image on the same perceptual manifold as well as rho1.16.

Next decision:

- Keep rho1.16 as the current paper-facing real-codec baseline.
- Continue stage-quant as the complete-design branch, but change the training objective from "reduce rate while paying DISTS/LPIPS" to "match or improve GLC/rho1.16 perceptual quality while reducing residual precision." Concretely, try a quality-preserving curriculum: start from rho=1, use strong GLC reconstruction distillation plus DISTS/LPIPS constraints, then slowly increase rate pressure.

## 2026-06-20 JST - Quality-preserving stage-quant improves Kodak curve

Goal: fix the first stage-quant curve, which was closer to the GP-ResLC design but underperformed rho1.16 because it reduced precision too aggressively. I added and tested a quality-preserving training objective with rate pressure plus weak GLC reconstruction distillation and LPIPS/DISTS hinge penalties against the frozen GLC baseline.

Code changes:

- `scripts/train_v1.py` now supports `lambda_lpips_distill`, `lambda_dists_distill`, `lambda_lpips_hinge`, `lambda_dists_hinge`, `lambda_R_start`, and `rate_warmup_iters`.
- The failed warmup run `v3_stage_quant_v1q2_quality_hinge_rhomax17_lR18_5k` (W&B `ps0xtsz9`) showed no practical rho movement: `rho=1.000/1.000`, A/B `delta_bpp_y=0` through it=1500. Decision: too little rate pressure.
- The successful setting uses immediate `lambda_R=35`, `rho_max=2.0`, and mild quality constraints.

Successful W&B runs:

| q | run | W&B | final A/B delta bpp_y | final A/B PSNR delta | rho mean/max |
|---|---|---|---:|---:|---:|
| 1 | `v3_stage_quant_v1q1_quality_hinge_fast_lR35_rhomax20_3k` | `hhvfn387` | -0.00106 | -0.2207 dB | 1.0548 / 1.1574 |
| 2 | `v3_stage_quant_v1q2_quality_hinge_fast_lR35_rhomax20_3k` | `naog9hjt` | -0.00271 | -0.1629 dB | 1.1413 / 1.2526 |
| 3 | `v3_stage_quant_v1q3_quality_hinge_fast_lR35_rhomax20_3k` | `mirujwwo` | -0.00366 | -0.2561 dB | 1.2377 / 1.4168 |

Real-codec Kodak artifacts:

- q1: `experiments/real_codec/kodak_stage_quant_q1_quality_hinge_fast/`
- q2: `experiments/real_codec/kodak_stage_quant_q2_quality_hinge_fast/`
- q3: `experiments/real_codec/kodak_stage_quant_q3_quality_hinge_fast/`
- Merged curve: `experiments/real_codec/kodak_stage_quant_quality_curve_metrics.csv`
- BD summary: `experiments/real_codec/kodak_stage_quant_quality_bd_rate_summary.md`
- Matched summary: `experiments/real_codec/kodak_stage_quant_quality_matched_metric_summary.md`

All real-codec runs passed estimated/decode consistency with `max_abs=0.000e+00` on all Kodak images.

Quality-preserving stage-quant real-codec points:

| q | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.02620 | 0.02109 | 21.3202 | 0.7487 | 0.1961 | 0.1129 |
| 1 | 0.02956 | 0.02446 | 21.6777 | 0.7653 | 0.1811 | 0.1029 |
| 2 | 0.03263 | 0.02753 | 21.9264 | 0.7757 | 0.1725 | 0.0993 |
| 3 | 0.03535 | 0.03025 | 22.0524 | 0.7817 | 0.1676 | 0.0952 |

Kodak BD-rate versus GLC real codec:

| run | DISTS | LPIPS | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| rho1.16 real | -4.47% | -0.79% | -0.87% | +0.45% |
| stage-quant quality | -4.96% | -0.47% | +0.32% | +0.70% |

Matched-metric bpp deltas versus GLC real codec:

| run | DISTS mean | LPIPS mean | PSNR mean | MS-SSIM mean |
|---|---:|---:|---:|---:|
| rho1.16 real | -5.45% | +0.34% | -0.41% | +0.80% |
| stage-quant quality | -4.22% | +0.10% | +0.08% | +0.42% |

Interpretation:

- This is the first complete-design branch that beats the rho1.16 shortcut on Kodak DISTS BD-rate while also making LPIPS BD-rate negative.
- The method is now much closer to the original claim: decoder-recomputable stage gates decide where residual precision can be reduced, and the real arithmetic-coded y stream shrinks without any transmitted gate map.
- q3 is especially strong: `0.03535 bpp / DISTS 0.09521`, compared with GLC q3 `0.03897 / 0.09539`. This is equal-or-better DISTS at about 9.3% lower serialized bpp.
- Remaining risk: q0 is still a no-op GLC point, and matched-DISTS mean is slightly weaker than rho1.16. Next priority is transfer evaluation on DIV2K/CLIC and possibly q0/low-rate-specific training.

## 2026-06-20 JST - Stage-quant quality transfer check on DIV2K

Goal: test whether the Kodak-improved quality-preserving stage-quant branch transfers to DIV2K validation under the same real arithmetic codec protocol.

Artifacts:

- q1: `experiments/real_codec/div2k_stage_quant_q1_quality_hinge_fast/`
- q2: `experiments/real_codec/div2k_stage_quant_q2_quality_hinge_fast/`
- q3: `experiments/real_codec/div2k_stage_quant_q3_quality_hinge_fast/`
- Curve CSV: `experiments/real_codec/div2k_stage_quant_quality_curve_metrics.csv`
- BD summary: `experiments/real_codec/div2k_stage_quant_quality_bd_rate_summary.md`
- Matched summary: `experiments/real_codec/div2k_stage_quant_quality_matched_metric_summary.md`

All q1-q3 real-codec runs passed estimated/decode consistency with `max_abs=0.000e+00` on all DIV2K images.

DIV2K stage-quant quality points:

| q | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.02381 | 0.02004 | 21.5114 | 0.7836 | 0.1842 | 0.0905 |
| 1 | 0.02710 | 0.02334 | 21.8484 | 0.7981 | 0.1713 | 0.0831 |
| 2 | 0.03023 | 0.02646 | 22.1441 | 0.8081 | 0.1627 | 0.0773 |
| 3 | 0.03302 | 0.02925 | 22.3730 | 0.8139 | 0.1583 | 0.0745 |

DIV2K BD-rate versus GLC real codec:

| run | DISTS | LPIPS | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| rho1.16 real | -10.79% | -0.54% | -1.49% | -0.17% |
| stage-quant quality | -4.05% | -0.10% | -0.19% | +0.11% |

Matched-metric bpp deltas versus GLC real codec:

| run | DISTS mean | LPIPS mean | PSNR mean | MS-SSIM mean |
|---|---:|---:|---:|---:|
| rho1.16 real | -10.27% | +0.36% | -1.16% | +0.68% |
| stage-quant quality | -5.64% | +0.26% | -0.20% | +0.54% |

Interpretation:

- The complete-design stage-quant branch transfers beyond Kodak: q1-q3 all give lower bpp than GLC at equal-or-better DISTS.
- DIV2K is not yet a stage-quant win over rho1.16. The rho1.16 shortcut remains much stronger on DIV2K DISTS-rate.
- The best stage-quant DIV2K point is q3: `0.03302 bpp / DISTS 0.07454`, beating both GLC q3 (`0.03649 / 0.07563`) and rho1.16 q3 (`0.03388 / 0.07508`) pointwise on DISTS and bpp, but the full curve is held back by q0 no-op and smaller q1/q2 savings.


## 2026-06-20 JST - Stage-quant quality branch on CLIC2020 full test

Goal: evaluate the decoder-recomputable stage-wise quantization gate on the official-protocol CLIC2020 full test set. This branch is closer to the original GP-ResLC claim than the global rho1.16 shortcut because each four-part prior stage predicts a no-side-bit rho >= 1 from already decoded context and reduces only residual precision that the decoder-side generator is expected to absorb.

Training runs:

- q1: v3_stage_quant_v1q1_quality_hinge_fast_lR35_rhomax20_3k, W&B hhvfn387
- q2: v3_stage_quant_v1q2_quality_hinge_fast_lR35_rhomax20_3k, W&B naog9hjt
- q3: v3_stage_quant_v1q3_quality_hinge_fast_lR35_rhomax20_3k, W&B mirujwwo
- q3 tuned: v3_stage_quant_v1q3_quality_tune_lR32_lp8_dists4_2k, W&B j9cottz9

Artifacts:

- Main CLIC curve: experiments/real_codec/clic2020_test_stage_quant_quality_curve_metrics.csv
- Main BD summary: experiments/real_codec/clic2020_test_stage_quant_quality_bd_rate_summary.md
- Main matched summary: experiments/real_codec/clic2020_test_stage_quant_quality_matched_metric_summary.md
- q3-tuned curve: experiments/real_codec/clic2020_test_stage_quant_quality_q3_tuned_curve_metrics.csv
- q3-tuned BD summary: experiments/real_codec/clic2020_test_stage_quant_quality_q3_tuned_bd_rate_summary.md
- q3-tuned official comparison: experiments/paper_assets/official_curve_comparison_stage_quant_q3_tuned/

All real-codec CLIC runs used the combined professional+mobile 428-image test set and shifted 256-patch FID/KID protocol, producing 28,650 patches. The real codec consistency check passed with max_abs=0.000e+00 for q1-q3 and for the q3-tuned run.

Main stage-quant CLIC points:

| q | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02134 | 0.01757 | 24.0733 | 0.8362 | 0.1542 | 0.08219 | 6.2655 | 0.001333 |
| 1 | 0.02456 | 0.02079 | 24.4729 | 0.8475 | 0.1413 | 0.07388 | 5.2823 | 0.001054 |
| 2 | 0.02771 | 0.02394 | 24.7660 | 0.8552 | 0.1338 | 0.06831 | 4.6622 | 0.000812 |
| 3 | 0.03043 | 0.02666 | 24.9472 | 0.8594 | 0.1301 | 0.06579 | 4.4788 | 0.000769 |

BD-rate versus local GLC real codec:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| rho1.16 real | -10.28% | +0.19% | -0.98% | +0.38% | -7.30% | -7.10% |
| stage-quant quality | -3.56% | +0.06% | +0.25% | +0.45% | -1.81% | +0.37% |
| stage-quant q3-tuned | -3.41% | +0.01% | +0.28% | +0.42% | -2.03% | -0.57% |

Matched-metric bpp deltas versus local GLC real codec:

| run | DISTS mean | FID mean | LPIPS mean | note |
|---|---:|---:|---:|---|
| rho1.16 real | -10.26% | -6.02% | +1.24% | strongest current CLIC headline |
| stage-quant quality | -5.29% | -2.48% | +0.63% | more faithful complete-design branch |
| stage-quant q3-tuned | -5.13% | -2.48% | +0.57% | improves q3 FID/KID/LPIPS but weakens q3 DISTS |

Official graph-extracted GLC comparison for q3-tuned CLIC curve:

| metric | BD-rate | matched bpp delta |
|---|---:|---:|
| FID | -0.74% | -0.92% over 3 points |
| KID | -0.55% | -3.77% over 3 points |
| DISTS | -2.02% | -3.97% over 4 points |
| LPIPS | +1.26% | +1.96% over 2 points |

Interpretation:

- The q3-tuned run is not a clean replacement for the main q3. It recovers FID from 4.4788 to 4.4611, KID from 0.000769 to 0.000736, and LPIPS from 0.13006 to 0.12976, but DISTS worsens from 0.06579 to 0.06590 and bpp rises from 0.03043 to 0.03059.
- For the VCIP short-track R-P story, the main stage-quant curve is still the cleaner DISTS result, while q3-tuned is evidence that quality recovery is possible but needs a better objective than simply increasing LPIPS/DISTS pressure.
- The complete-design branch is technically aligned with the original claim, but it is not yet competitive with the rho1.16 shortcut or large enough against official GLC. Next research should relax frozen-GLC limits: train a residual-predictive latent decomposition from pretrained weights first, then attempt scratch once the pretrained path shows a real-codec CLIC gain beyond the shortcut.


## 2026-06-20 JST - Complete-design escalation checks after CLIC stage-quant

Goal: test whether the more faithful GP-ResLC mechanisms can be pushed beyond the current rho1.16 shortcut without breaking the real arithmetic-codec accounting. These checks focused on two routes: residual mean prediction with partially trainable entropy modules, and stage-wise decoder-recomputable quantization with stronger rho targets.

Implementation updates:

- `gp_reslc/real_codec.py` now supports `predictor_param_mode=stage_latent_residual` for real arithmetic-coded payloads.
- `scripts/evaluate_real_codec.py` can attach `StageResidualPredictor`, load `stage_latent_residual` checkpoints, and restore optional `model_state_dict` snapshots.
- `scripts/train_v1.py` now supports `--unfreeze_entropy`, `--unfreeze_hyper_dec`, `--save_model_state`, `--freeze_aux_module`, and `--lambda_rho_target/--rho_target` for stage-quant control.
- Real-codec smoke for `stage_latent_residual` passed on a one-image Kodak check with `max_abs=0.000e+00`.

Runs and outcomes:

| branch | run | W&B | outcome |
|---|---|---|---|
| latent residual + entropy unfreeze | `v4_latres_v1q2_unfreeze_entropy_b005_lR8_lp4_dists1_mp005_6k` | `raeuurk3` | A/B estimated bpp moved slightly negative, but Kodak8 real codec was worse than GLC: bpp `0.03639` vs `0.03572`, DISTS `0.10246` vs `0.10171`. Reject. |
| stage residual + entropy unfreeze | `v4_stage_residual_v1q2_unfreeze_entropy_b005_lR8_lp4_dists1_mp05_4k` | `rz0olu2p` | Similar failure: Kodak8 bpp `0.03606`, DISTS `0.10324`, both worse than GLC q2. Reject. |
| stage quant + entropy unfreeze | `v4_stage_quant_v1q2_unfreeze_entropy_from_quality_lR24_lp8_dists4_2k` | `0s1p2c25` | rho collapsed toward identity and the rate saving disappeared. Reject. |
| stage quant, gate frozen, entropy only | `v4_stage_quant_v1q2_entropy_only_from_quality_lR12_lp8_dists8_1500` | `i97xp9i7` | Even with the gate frozen, changing the GLC entropy features changed the gate inputs and collapsed effective rho. Reject. |
| stage quant + entropy unfreeze + rho target | `v4_stage_quant_v1q2_unfreeze_entropy_rhotarget112_lR12_lp8_dists8_rt30_1500` | `k7cci11q` | Estimated A/B looked good (`delta_bpp_y≈-0.0029`, `rho≈1.12`), but Kodak8 real bpp rose to about `0.0378`. Real arithmetic length exposed a mismatch. Reject. |
| stage quant, fixed GLC, rho target 1.22 | `v4_stage_quant_v1q2_rhotarget122_quality_hinge_lR30_lp8_dists8_rt20_1500` | `ukqx3wbn` | Real Kodak8 bpp dropped to `0.03228` versus stage-quant-quality q2 `0.03360` and GLC q2 `0.03572`, but DISTS worsened to `0.10352` and LPIPS to `0.17888`. Keep as an upper-rate knob, not a lead. |

Interpretation:

- Unfreezing GLC entropy/prior modules is dangerous under the current pretrained decomposition. It can improve estimated or crop-level A/B likelihood, but the serialized bitstream gets longer once arithmetic support, CDF calibration, and actual symbol lengths are counted.
- Direct residual mean prediction is now real-codec correct, but not yet competitive. The likely cause is that pretrained GLC did not learn a clean `generator-predictable component + residual` factorization, so adding a mean correction perturbs the four-part spatial context rather than simplifying it.
- The fixed-GLC stage-quant route remains the only complete-design branch that reliably reduces real y-stream bits. However, stronger rho targets trade too much perceptual quality for rate.

Decision:

- For the VCIP short-track package, keep `rho1.16` as the headline because it is robust on CLIC2020 full test, DIV2K, Kodak, and against the official GLC curves.
- Keep stage-quant as the method-faithful secondary branch and continue with fixed-GLC, no-unfreeze training. The next useful sweep is an intermediate q2 target (`rho_target≈1.17-1.18`) with stronger DISTS/LPIPS hinge protection, not entropy unfreezing.
- Scratch GP-ResLC remains high-upside, but the pretrained branch shows that the decomposition must be trained jointly from the start; it should be developed as a separate staged branch rather than by forcing frozen GLC priors to absorb large residual-prediction changes.


## 2026-06-20 JST - Stage-quant q2 intermediate rho-target check

Goal: test whether the method-faithful stage-quant branch can gain additional q2 rate saving without the quality loss seen at `rho_target=1.22`.

Run:

- `v4_stage_quant_v1q2_rhotarget117_quality_hinge_lR24_lp10_dists10_rt20_1500_r2`
- W&B: `7wab1b19`
- initialization: weights-only resume from `v3_stage_quant_v1q2_quality_hinge_fast_lR35_rhomax20_3k/train_state.pt`
- settings: fixed GLC, `stage_quant_gate`, `rho_target=1.17`, `lambda_rho_target=20`, `lambda_R=24`, `lambda_lpips=10`, `lambda_dists=10`, strong LPIPS/DISTS hinges.

Training summary:

- The gate stayed near the intended target: final W&B `rho_mean=1.1708`, `rho_max=1.3410`.
- Kodak A/B estimated check stayed positive from a rate perspective: `delta_bpp_y=-0.00287`, PSNR `19.24 -> 19.27`.
- Real-codec consistency passed on Kodak8 with `max_abs=0.000e+00`.

Kodak8 real-codec comparison at q2:

| run | bpp | PSNR | LPIPS | DISTS | FID | decision |
|---|---:|---:|---:|---:|---:|---|
| GLC q2 | 0.03572 | 21.6311 | 0.1706 | 0.1017 | 53.77 | anchor |
| stage-quant quality q2 | 0.03360 | 21.4731 | 0.1748 | 0.1030 | 55.06 | current stage q2 |
| stage-quant rho_target=1.17 | 0.03299 | 21.4174 | 0.1775 | 0.1041 | 55.70 | reject as q2 replacement |
| stage-quant rho_target=1.22 | 0.03228 | 21.3334 | 0.1789 | 0.1035 | 55.67 | upper-rate knob |

Interpretation:

- The intermediate target successfully reduces serialized bpp, but the quality protection is not sufficient. It is worse than the existing q2 quality checkpoint on DISTS/LPIPS/FID and not clearly better than the more aggressive target1.22 in perceptual quality.
- This argues against pushing q2 harder. The stage-quant curve's larger weakness is q0 being a no-op anchor, so the next experiment should target a conservative q0 stage gate (`rho_target≈1.08-1.10`) with strict quality hinges.


## 2026-06-20 JST - Stage-quant q0 rho_target=1.08 check

Goal: improve the stage-quant curve's weakest point. Previous stage-quant quality curves used q0 as a no-op GLC anchor, which hurts BD-rate. This run tested whether a conservative q0 gate can reduce real bpp while preserving perceptual quality.

Run:

- `v4_stage_quant_v1q0_rhotarget108_quality_hinge_lR18_lp12_dists12_rt30_2k`
- W&B: `v7s5xquu`
- stopped early after the 1000-iteration checkpoint because A/B PSNR degradation persisted.
- settings: fixed GLC, `stage_quant_gate`, `rho_target=1.08`, `lambda_rho_target=30`, `lambda_R=18`, strong LPIPS/DISTS hinges.

Training / real-codec summary:

- rho reached target quickly and stayed near `1.08`.
- Kodak A/B at it=500/1000: `delta_bpp_y≈-0.0015..-0.0016`, but PSNR dropped by about `0.19-0.22 dB`.
- Real-codec consistency passed on Kodak8 with `max_abs=0.000e+00`.

Kodak8 real-codec q0 comparison:

| run | bpp | PSNR | LPIPS | DISTS | FID | decision |
|---|---:|---:|---:|---:|---:|---|
| GLC q0 | 0.02699 | 20.8504 | 0.2028 | 0.1190 | 61.1080 | anchor |
| stage-quant q0 target1.08 | 0.02564 | 20.7668 | 0.2088 | 0.1198 | 60.7882 | not a clean DISTS/LPIPS point |

Interpretation:

- The q0 gate does reduce real serialized bpp by about 5%, so the mechanism works even at the lowest rate point.
- However, q0 has little perceptual slack. DISTS and LPIPS worsen enough that this checkpoint is risky as a curve replacement.
- Next: lower the q0 target to `rho_target≈1.05` and reduce rate pressure. The goal is a smaller 2-3% rate cut with DISTS closer to neutral.


## 2026-06-20 JST - Stage-quant q0 rho_target=1.05 check

Goal: recover a safer q0 stage-quant point after `rho_target=1.08` reduced real bpp but worsened DISTS/LPIPS.

Run:

- `v4_stage_quant_v1q0_rhotarget105_quality_hinge_lR12_lp12_dists14_rt30_1200`
- W&B: `7r0j5hy3`
- settings: fixed GLC, `stage_quant_gate`, `rho_target=1.05`, `lambda_rho_target=30`, lower `lambda_R=12`, tighter LPIPS/DISTS hinges.

Training / real-codec summary:

- Final W&B `rho_mean=1.0498`, `rho_max=1.0615`.
- A/B at it=400: `delta_bpp_y=-0.0010`, PSNR equal. A/B at it=800 and final summary: same bpp saving with about `0.04 dB` PSNR loss.
- Real-codec consistency passed on Kodak8 with `max_abs=0.000e+00`.

Kodak8 real-codec q0 comparison:

| run | bpp | PSNR | LPIPS | DISTS | FID | decision |
|---|---:|---:|---:|---:|---:|---|
| GLC q0 | 0.02699 | 20.8504 | 0.2028 | 0.1190 | 61.1080 | anchor |
| stage q0 target1.05 | 0.02610 | 20.7860 | 0.2051 | 0.1203 | 60.4096 | reject for DISTS curve |
| stage q0 target1.08 | 0.02564 | 20.7668 | 0.2088 | 0.1198 | 60.7882 | also not clean |

Interpretation:

- Lowering the q0 target reduced the LPIPS damage but did not protect DISTS. On this Kodak8 subset, q0 is too rate-starved for simple stage-wise coarsening to be a clean DISTS improvement.
- Do not replace the stage-quant q0 anchor with either target1.05 or target1.08 yet.
- If q0 is revisited, add explicit baseline reconstruction distillation or a sendability teacher; otherwise keep q0 as GLC/no-op and focus complete-design improvements on q1-q3.


## 2026-06-20 JST - Stage-quant q0 rho_target=1.04 with baseline distillation stopped

Goal: check whether explicit GLC-reconstruction distillation can make a very conservative q0 stage gate usable after target1.08 and target1.05 both hurt DISTS.

Run:

- `v4_stage_quant_v1q0_rhotarget104_baseDist_lR10_lp10_dists14_rt30_1000`
- W&B: `zgu6f2zj`
- settings: fixed GLC, `rho_target=1.04`, `lambda_R=10`, `lambda_lpips_distill=4`, `lambda_dists_distill=8`, strong LPIPS/DISTS hinges.
- stopped after the 500-iteration A/B check.

Observation:

- rho reached about `1.04` as intended.
- A/B at it=500: baseline `bpp_y=0.0226`, PSNR `18.50`; ours `bpp_y=0.0221`, PSNR `18.31`, so the saving was only `-0.0006` bpp_y with about `-0.19 dB` PSNR.

Decision:

- Do not continue this q0 direction for now. Even with conservative rho and baseline distillation, q0 does not offer a clean rate-perception tradeoff on the quick Kodak A/B signal.
- Keep q0 as a no-op GLC anchor in the stage-quant curve until a better q0-specific mechanism exists. Continue complete-design work on q1-q3 or move to a jointly trained scratch decomposition.


## 2026-06-20 JST - Scratch GP-ResLC Stage-A scaffold

Goal: start the high-upside scratch branch that can learn the original GP-ResLC decomposition without being constrained by pretrained GLC latents. Stage A learns a compact semantic/generative VQ code `s`; later stages will add `mu_theta(s)` and entropy-code only the unpredictable residual.

Implementation:

- Added `gp_reslc/scratch/vq_autoencoder.py` with:
  - residual Conv encoder/decoder,
  - straight-through `VectorQuantizer`,
  - fixed semantic index bpp reporting,
  - default 16x16 latent grid for 256x256 crops.
- Added `scripts/train_scratch_stage_a.py` with L1 + LPIPS + DISTS + VQ loss, W&B logging, validation panels, checkpointing, and a GPU guard.
- Added `gp_reslc/scratch/__init__.py` exports.

Smoke test:

```bash
.venv/bin/python scripts/train_scratch_stage_a.py \
  --data /dpl/openimages/train --val /dpl/kodak \
  --out experiments/scratch_stage_a_smoke \
  --iters 2 --bs 1 --base_ch 32 --latent_dim 64 --codebook_size 128 \
  --num_workers 0 --log_every 1 --eval_every 1 --no_wandb
```

Result:

- `py_compile` passed for `gp_reslc/scratch/vq_autoencoder.py` and `scripts/train_scratch_stage_a.py`.
- Two-iteration smoke completed on CUDA.
- Semantic bpp for the small smoke model was `0.02734` because `codebook_size=128` gives 7 bits/index on a 16x16 grid.
- Validation panel images and `stage_a_final.pt` were written to `experiments/scratch_stage_a_smoke/`.

Interpretation:

- This is not yet a codec result. It is the first runnable scaffold for the scratch semantic branch.
- The next pilot should use `codebook_size=1024`, giving semantic fixed-index bpp about `0.03906` for 256 crops, which matches the GLC ultra-low-bitrate operating range.
- Exit criterion for Stage A is not PSNR; it is whether `s`-only reconstructions become perceptually plausible without codebook collapse. Then Stage B can add `y = mu_theta(s) + r`.


## 2026-06-20/21 JST - Scratch Stage-A VQ collapse and soft-entropy fix

Goal: start the scratch semantic-code branch and check whether a 0.039 bpp VQ semantic code can train without codebook collapse.

Runs:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_a_vq1024_b64_z128_lp1_dists1_3k` | `429wuhqo` | VQ-1024, base64, z128, no entropy regularization | stopped around it=700; hard perplexity collapsed to about 2-3. |
| `scratch_stage_a_vq1024_b64_z128_entropy03_beta01_3k` | `tfc1gins` | hard one-hot entropy regularization | stopped around it=500; hard entropy has no useful gradient through argmin and collapse persisted. |
| `scratch_stage_a_vq1024_b64_z128_softent_tau001_lam05_1500` | `sxfjozwa` | differentiable soft assignment entropy, `tau=0.01`, `lambda_codebook_entropy=0.5`, `vq_beta=0.1` | completed 1500 iters; hard perplexity stayed around 30-40 instead of collapsing. |

Implementation updates:

- `VectorQuantizer` now exposes soft assignment entropy from `softmax(-dist/tau)` in addition to hard code usage entropy.
- `ScratchVQAutoencoder` accepts `vq_beta` and `vq_entropy_tau`.
- `scripts/train_scratch_stage_a.py` logs hard/soft perplexity, hard/soft entropy, usage fraction, and supports `--lambda_codebook_entropy`.

Best scratch Stage-A pilot so far:

- run: `scratch_stage_a_vq1024_b64_z128_softent_tau001_lam05_1500`
- W&B: `sxfjozwa`
- fixed semantic index bpp: `0.03906` for 256 crops.
- validation at it=500: L1 `0.1301`, LPIPS `0.5493`, DISTS `0.4470`, hard perplexity `34.0`, soft perplexity `63.6`.
- validation at it=1000: L1 `0.0946`, LPIPS `0.5097`, DISTS `0.4589`, hard perplexity `31.0`, soft perplexity `34.6`.
- final train summary: hard perplexity `37.3`, hard entropy norm `0.522`, usage fraction about `0.055`.

Interpretation:

- Stage A is now runnable and does not immediately collapse with soft entropy regularization.
- The reconstruction quality is far from GLC and not paper-usable yet. This is expected: the model is small, trained only 1500 iterations, and has no GAN/perceptual decoder pretraining.
- The key technical lesson is that hard assignment entropy is not a valid anti-collapse loss; soft assignment entropy or EMA/codebook reset is required.
- Next scratch steps: add EMA or dead-code restart, lower the VQ loss instability, train Stage A longer, then introduce Stage B residual decomposition only after semantic reconstructions are plausible.


## 2026-06-21 JST - Scratch Stage-A dead-code restart and low-rate grid support

Goal: make the scratch semantic-code branch usable enough for the full GP-ResLC design. The key question is whether the VQ code can avoid collapse and whether the semantic stream can be made cheap enough that a residual stream can still fit below the official GLC curve.

Run:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_a_vq1024_b64_z128_softent_restart_2k` | `2d7yi3uk` | VQ-1024, base64, z128, `vq_beta=0.1`, soft entropy `tau=0.01`, `lambda_codebook_entropy=0.5`, dead-code restart every 200 iters | completed 2000 iters; dead-code restart raised hard perplexity and usage far above the no-restart pilot. |

Key scalar observations:

- fixed semantic index bpp for the 16x16 grid remains `0.03906`.
- validation at it=500: L1 `0.1654`, LPIPS `0.5334`, DISTS `0.5119`, hard perplexity `120.4`, hard entropy norm `0.691`, usage fraction `0.209`.
- validation at it=1000: L1 `0.1770`, LPIPS `0.6269`, DISTS `0.4900`, hard perplexity `18.6`, hard entropy norm `0.422`, usage fraction `0.133`; this was a transient post-restart instability.
- validation at it=1500: L1 `0.0980`, LPIPS `0.4610`, DISTS `0.4752`, hard perplexity `143.7`, hard entropy norm `0.717`, usage fraction `0.303`.
- final train summary at it=1950: L1 `0.1096`, LPIPS `0.4211`, DISTS `0.4547`, hard perplexity `150.5`, hard entropy norm `0.723`, usage fraction `0.314`.

Decision:

- Dead-code restart is useful and should remain in Stage A. It improves codebook utilization much more than soft entropy alone, whose previous hard perplexity was around 30-40.
- This branch is still not paper-leading. Reconstruction quality is far from GLC and the 16x16 fixed semantic cost is too high once a residual stream is added.
- Added configurable `num_down` to `ScratchVQAutoencoder`: `num_down=4` gives a 16x16 semantic grid at `0.03906` bpp, while `num_down=5` gives an 8x8 grid at `0.00977` bpp for 256 crops. The latter is much closer to the intended full design: cheap semantic code plus unpredictable residual only.
- Added `--resume` and `--num_down` to `scripts/train_scratch_stage_a.py`. Next high-upside experiment should train the 8x8 semantic code, then Stage B should learn `y = mu_theta(s) + r` on top of it.


## 2026-06-21 JST - Scratch Stage-B semantic-conditioned residual proof signal

Goal: implement and test the full GP-ResLC decomposition more directly than the pretrained GLC gate branch: transmit a cheap semantic code `s`, predict `mu_theta(s)` at the decoder, and entropy-code only the unpredictable residual `r = y - mu_theta(s)`.

Implementation:

- Added `gp_reslc/scratch/residual_autoencoder.py`.
- Added `scripts/train_scratch_stage_b.py`.
- Added `scripts/evaluate_scratch_stage_b.py` for deterministic Kodak center-crop evaluation.
- Stage B freezes the Stage-A VQ semantic autoencoder, predicts `mu_theta(z_s)`, quantizes residual symbols with hard rounding at eval time, estimates residual rate with a Gaussian entropy proxy, and reconstructs through a residual decoder.
- Important fix: residual decoder final convolution is zero-initialized, so the initial Stage-B reconstruction is exactly the Stage-A base reconstruction. This makes base/ours comparisons clean.
- Important design fix: residual bottleneck now uses independent `residual_dim`; using the semantic latent width (`160ch`) made the residual stream start around `0.28 bpp`, which is incompatible with ultra-low-rate operation. `residual_dim=16`, `quant_step=1.0` starts around `0.02 residual bpp`.

Stage-A source checkpoint:

- `experiments/scratch_stage_a_vq1024_b80_z160_down5_softent_restart_8k/stage_a_0006000.pt`
- W&B Stage A: `75mqqysy`
- deterministic Kodak center Stage-A base: fixed semantic bpp `0.00977`, LPIPS `0.4578`, DISTS `0.4526`.

Stage-B runs:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_b_down5_r16_q1_lR2_5k` | `7nk1zgmf` | `residual_dim=16`, `quant_step=1.0`, `lambda_R=2.0`, `lambda_pred=0.1` | stopped around it=1050; residual collapsed to near-zero hard bpp and validation did not improve over base. Negative result: rate pressure too strong and/or predictor collapse. |
| `scratch_stage_b_down5_r16_q1_lR0p1_pred001_3k` | `8fgx365x` | `residual_dim=16`, `quant_step=1.0`, `lambda_R=0.1`, `lambda_pred=0.01` | completed 3000 iters; hard-quantized residual improves LPIPS and DISTS on deterministic Kodak center evaluation. |

Deterministic Kodak center evaluation for `scratch_stage_b_down5_r16_q1_lR0p1_pred001_3k`:

| ckpt | semantic bpp | residual bpp | total bpp | base LPIPS | ours LPIPS | base DISTS | ours DISTS | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `stage_b_0001000.pt` | 0.00977 | 0.01388 | 0.02364 | 0.45782 | 0.44830 | 0.45264 | 0.45627 | LPIPS improves, DISTS worsens. |
| `stage_b_0002000.pt` | 0.00977 | 0.02365 | 0.03342 | 0.45782 | 0.43959 | 0.45264 | 0.44981 | first clean LPIPS+DISTS improvement. |
| `stage_b_final.pt` | 0.00977 | 0.02872 | 0.03848 | 0.45782 | 0.43485 | 0.45264 | 0.43711 | strongest current Stage-B quality; rate is high. |

Decision:

- This is a real proof signal for the original design axis: a cheap semantic stream plus hard-quantized residual can improve perceptual quality, and the residual is explicitly represented as `y - mu_theta(s)`.
- It is not yet competitive with GLC; absolute DISTS is still around `0.44`, far from the pretrained GLC real-codec curve. Keep pretrained rho/stage-quant as the VCIP safety lead.
- Next scratch research should optimize the Stage-B tradeoff, not just train longer: try `lambda_R=0.3-0.5`, higher DISTS weight, and possibly `residual_dim=8/16` sweeps. Then add a real residual entropy coder only after the proxy curve is meaningful.


## 2026-06-21 JST - Scratch Stage-B DISTS-weighted tradeoff improvement

Goal: reduce the Stage-B residual bpp while keeping the hard-quantized residual useful for both LPIPS and DISTS.

Run:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_b_down5_r16_q1_lR0p3_d2_3k` | `2ii44jvx` | `residual_dim=16`, `quant_step=1.0`, `lambda_R=0.3`, `lambda_lpips=0.7`, `lambda_dists=2.0`, `lambda_pred=0.01` | completed 3000 iters; much better rate-quality tradeoff than the previous `lambda_R=0.1` run. |

Deterministic Kodak center evaluation:

| ckpt | semantic bpp | residual bpp | total bpp | base LPIPS | ours LPIPS | base DISTS | ours DISTS | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `stage_b_0001000.pt` | 0.00977 | 0.00628 | 0.01604 | 0.45782 | 0.45291 | 0.45264 | 0.44885 | efficient early point. |
| `stage_b_0002000.pt` | 0.00977 | 0.00861 | 0.01838 | 0.45782 | 0.45164 | 0.45264 | 0.44077 | best current Stage-B tradeoff. |
| `stage_b_final.pt` | 0.00977 | 0.01015 | 0.01991 | 0.45782 | 0.44792 | 0.45264 | 0.44211 | better LPIPS, slightly worse DISTS than 2000. |

Comparison to previous Stage-B `lambda_R=0.1` final:

- Previous final: total bpp `0.03848`, LPIPS `0.43485`, DISTS `0.43711`.
- New 2000 ckpt: total bpp `0.01838`, LPIPS `0.45164`, DISTS `0.44077`.

Decision:

- `lambda_R=0.3` + DISTS-heavy loss is the better Stage-B operating region for the current weak Stage-A base. It gives most of the DISTS gain at less than half the residual rate.
- The next sweep should test `residual_dim=8` and maybe `quant_step=0.75/1.0` to see whether the residual stream can be kept near `0.005-0.008 bpp` without losing the DISTS gain.
- This remains a scratch proof-of-concept, not a GLC-competitive curve.


## 2026-06-21 JST - Scratch Stage-B residual_dim=8 sweep

Goal: test whether the residual stream can be narrowed below 16 channels while preserving the DISTS-heavy Stage-B gain.

Run:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_b_down5_r8_q1_lR0p3_d2_3k` | `r925d692` | `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.3`, `lambda_lpips=0.7`, `lambda_dists=2.0`, `lambda_pred=0.01` | completed 3000 iters; strongest DISTS improvement so far, with slightly higher bpp than the best r16 efficiency point. |

Deterministic Kodak center evaluation:

| ckpt | semantic bpp | residual bpp | total bpp | base LPIPS | ours LPIPS | base DISTS | ours DISTS | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `stage_b_0001000.pt` | 0.00977 | 0.01010 | 0.01987 | 0.45782 | 0.45169 | 0.45264 | 0.44540 | improves both, but weaker than r16 2000. |
| `stage_b_0002000.pt` | 0.00977 | 0.01163 | 0.02140 | 0.45782 | 0.45110 | 0.45264 | 0.43932 | slightly better DISTS than r16 2000, higher bpp. |
| `stage_b_final.pt` | 0.00977 | 0.01369 | 0.02345 | 0.45782 | 0.44438 | 0.45264 | 0.43024 | strongest current Stage-B perceptual result. |

Current scratch Stage-B Pareto points:

| model | total bpp | LPIPS | DISTS | interpretation |
|---|---:|---:|---:|---|
| Stage-A base | 0.00977 | 0.45782 | 0.45264 | semantic-only generator. |
| r16 DISTS-heavy 1000 | 0.01604 | 0.45291 | 0.44885 | efficient first residual point. |
| r16 DISTS-heavy 2000 | 0.01838 | 0.45164 | 0.44077 | best efficiency point. |
| r8 DISTS-heavy final | 0.02345 | 0.44438 | 0.43024 | best current quality point. |

Decision:

- `residual_dim=8` is not too narrow; it can produce the strongest DISTS improvement, likely because the narrow bottleneck regularizes the residual decoder and avoids sending broad noisy corrections.
- For the next scratch run, test `residual_dim=8`, `lambda_R=0.5`, or `quant_step=1.25` to seek a point near total bpp `0.018-0.020` with DISTS closer to `0.43`.
- Longer-term blocker remains Stage-A generator quality. Even the best Stage-B scratch result is far from GLC, so this branch is method-faithful but not yet competitive.


## 2026-06-21 JST - Scratch Stage-B residual_dim=8 lambda_R=0.5 sweep

Goal: tighten the `residual_dim=8` DISTS-heavy run and search for a lower-bpp point than `lambda_R=0.3` while preserving perceptual gains.

Run:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_b_down5_r8_q1_lR0p5_d2_3k` | `wwi995cn` | `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.5`, `lambda_lpips=0.7`, `lambda_dists=2.0`, `lambda_pred=0.01` | completed 3000 iters; best current low-rate Stage-B sweep. |

Deterministic Kodak center evaluation:

| ckpt | semantic bpp | residual bpp | total bpp | base LPIPS | ours LPIPS | base DISTS | ours DISTS | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `stage_b_0001000.pt` | 0.00977 | 0.00655 | 0.01631 | 0.45782 | 0.45442 | 0.45264 | 0.44100 | low-rate point, DISTS clearly improves. |
| `stage_b_0002000.pt` | 0.00977 | 0.00798 | 0.01775 | 0.45782 | 0.45258 | 0.45264 | 0.43456 | best current DISTS-efficiency point. |
| `stage_b_best.pt` | 0.00977 | 0.00826 | 0.01803 | 0.45782 | 0.45001 | 0.45264 | 0.43529 | random-val best, fixed eval slightly behind 2000. |
| `stage_b_final.pt` | 0.00977 | 0.00838 | 0.01815 | 0.45782 | 0.44694 | 0.45264 | 0.43709 | LPIPS-best among low-rate r8 points. |

Updated scratch Stage-B Pareto:

| model | total bpp | LPIPS | DISTS | interpretation |
|---|---:|---:|---:|---|
| Stage-A base | 0.00977 | 0.45782 | 0.45264 | semantic-only generator. |
| r8 lR0.5 1000 | 0.01631 | 0.45442 | 0.44100 | lowest useful residual point. |
| r8 lR0.5 2000 | 0.01775 | 0.45258 | 0.43456 | best current DISTS-efficient point. |
| r8 lR0.5 final | 0.01815 | 0.44694 | 0.43709 | best current LPIPS-efficient low-rate point. |
| r8 lR0.3 final | 0.02345 | 0.44438 | 0.43024 | best current DISTS quality point. |

Decision:

- `residual_dim=8`, `lambda_R=0.5`, DISTS-heavy loss is the best current scratch Stage-B operating region.
- This produces a coherent low-rate residual curve from total bpp `0.0163` to `0.0235`, all improving DISTS over the semantic-only base.
- This is still far from GLC absolute quality, so the next high-impact work is Stage-A/generator improvement or adding a stronger perceptual generator/discriminator, not squeezing Stage-B proxy further.


## 2026-06-21 JST - Scratch Stage-A continuation check

Goal: test whether simply continuing the down5 Stage-A semantic generator improves the absolute base quality that currently bottlenecks Stage B.

Run:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_a_down5_from6000_continue30k_lr1e4` | `2j544u3v` | resumed `stage_a_0006000.pt`, lower lr `1e-4`, planned 30k but stopped after 10k checkpoint due weak validation trend | no meaningful fixed-eval improvement; simple continuation is not the right next lever. |

Deterministic Kodak center evaluation:

| checkpoint | bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---|
| original `stage_a_0006000.pt` | 0.00977 | 0.45782 | 0.45264 | Stage-B source checkpoint. |
| continued `stage_a_best.pt` | 0.00977 | 0.45730 | 0.45180 | tiny DISTS/LPIPS improvement only. |
| continued `stage_a_0010000.pt` | 0.00977 | 0.44929 | 0.45899 | LPIPS improves, DISTS worsens. |

Decision:

- Do not spend more time on plain Stage-A continuation with the same loss. It does not materially improve the generator bottleneck.
- Next Stage-A work should change the objective/model, e.g. adversarial fine-tuning, stronger decoder, or multi-scale perceptual losses.


## 2026-06-21 JST - Scratch Stage-A adversarial fine-tuning negative result

Goal: test whether a lightweight PatchGAN fine-tune can improve the weak Stage-A generator that currently bottlenecks scratch Stage-B absolute quality.

Implementation:

- Added `gp_reslc/scratch/discriminator.py` with a spectral-normalized PatchDiscriminator.
- Added `scripts/train_scratch_stage_a_adv.py`.
- Added checkpoint compatibility fallback so adversarial Stage-A checkpoints can be loaded by `evaluate_scratch_stage_a.py` and Stage-B scripts.

Run:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_a_adv_down5_ladv001_3k` | `7uwfab18` | resumed Stage-A 6000, `lambda_adv=0.01`, PatchGAN, reconstruction+LPIPS+DISTS retained | stopped around it=1150; validation DISTS/LPIPS worsened. |

Observed validation:

- val0: LPIPS `0.4609`, DISTS `0.4173`.
- val500: LPIPS `0.4950`, DISTS `0.4480`.
- val1000: LPIPS `0.4731`, DISTS `0.4912`.

Decision:

- This adversarial setting is not useful for the current Stage-A objective. The discriminator becomes strong quickly and DISTS degrades.
- Do not continue this GAN direction without a more careful setup: lower `lambda_adv`, delayed adversarial start, feature matching, or a discriminator trained on larger crops/multi-scale patches.
- For now, the best scratch route remains Stage-B residual factorization with the current Stage-A base; Stage-A generator improvement needs a more deliberate redesign.


## 2026-06-21 JST - Scratch Stage-A DISTS-heavy fine-tune

Goal: improve the Stage-A generator bottleneck without GAN by applying the DISTS-heavy objective that worked for Stage B.

Run:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_a_down5_from6000_dists2_lp05_12k` | `zd8omzv0` | resumed Stage-A 6000, `lambda_dists=2.0`, `lambda_lpips=0.5`, lr `1e-4`; stopped after 8000 checkpoint | small but real fixed-eval improvement; better than plain continuation. |

Deterministic Kodak center evaluation:

| checkpoint | bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---|
| original `stage_a_0006000.pt` | 0.00977 | 0.45782 | 0.45264 | previous Stage-B source. |
| DISTS-heavy `stage_a_best.pt` | 0.00977 | 0.45757 | 0.45266 | random-val best did not transfer. |
| DISTS-heavy `stage_a_0008000.pt` | 0.00977 | 0.45221 | 0.44797 | best fixed Stage-A so far. |

Decision:

- DISTS-heavy Stage-A fine-tuning is mildly useful and should replace the original 6000 checkpoint for the next Stage-B sweep.
- The gain is small, so it does not solve the scratch absolute-quality gap by itself.



## 2026-06-21 JST - Scratch Stage-B from DISTS-heavy Stage-A, lower rate pressure

Goal: test whether the improved Stage-A base (`stage_a_0008000.pt` from the DISTS-heavy fine-tune) gives a better residual decomposition when Stage B is allowed to spend more residual bits.

Run:

| run | W&B | Stage-A source | setting |
|---|---|---|---|
| `scratch_stage_b_from_stageA_d2_8000_r8_q1_lR0p1_d2_3k` | `9gbu1r38` | `experiments/scratch_stage_a_down5_from6000_dists2_lp05_12k/stage_a_0008000.pt` | `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.1`, `lambda_lpips=0.7`, `lambda_dists=2.0`, `lambda_pred=0.01` |

Deterministic Kodak center evaluation:

| checkpoint | total bpp | residual bpp | base LPIPS | LPIPS | base DISTS | DISTS |
|---|---:|---:|---:|---:|---:|---:|
| `stage_b_0001000.pt` | 0.02098 | 0.01121 | 0.45221 | 0.44131 | 0.44797 | 0.44371 |
| `stage_b_0002000.pt` | 0.02052 | 0.01075 | 0.45221 | 0.43713 | 0.44797 | 0.44182 |
| `stage_b_best.pt` | 0.02257 | 0.01280 | 0.45221 | 0.43354 | 0.44797 | 0.43681 |
| `stage_b_final.pt` | 0.02212 | 0.01236 | 0.45221 | 0.43832 | 0.44797 | 0.43195 |

Interpretation:

- The hard-quantized residual stream again improves both LPIPS and DISTS over the Stage-A base, so the residual decomposition mechanism remains valid.
- This run does not update the scratch Pareto frontier: the previous `lambda_R=0.5` r8 run gives DISTS `0.43456` at bpp `0.01775`, and the r8 `lambda_R=0.3` final gives DISTS `0.43024` at bpp `0.02345`.
- Lowering `lambda_R` to `0.1` spends bits less efficiently on this Stage-A source. The better Stage-A base helps absolute DISTS slightly, but the residual model does not convert the extra bpp into a clear quality-rate win.

Decision:

- Do not promote this run as the scratch lead.
- Keep the current scratch lead as `scratch_stage_b_down5_r8_q1_lR0p5_d2_3k` for low-rate efficiency and `scratch_stage_b_down5_r8_q1_lR0p3_d2_3k` for the higher-quality point.
- The next high-value experiment should change the generator/Stage-A architecture or objective rather than simply relaxing the residual rate term.



## 2026-06-21 JST - Scratch Stage-A latent refinement and Stage-B Pareto update

Goal: improve the weak scratch Stage-A generator without changing semantic rate, then test whether the residual stream benefits from the stronger semantic generator.

Implementation:

- Added optional `decoder_attention` and `extra_decoder_blocks` to `ScratchVQAutoencoder`.
- The new modules live in `latent_refine` before the original decoder, so existing decoder weights keep identical names and can be fully reused.
- Added `--resume_partial` to `scripts/train_scratch_stage_a.py`; the attention/refine experiment loaded all 138 existing tensors from the DISTS-heavy Stage-A checkpoint and skipped 0 old tensors.
- New latent-refine blocks are identity-initialized, and attention output projection is zero-initialized. A direct output-difference check against the source checkpoint gave max/mean diff `0.0/0.0` before fine-tuning.

Stage-A run:

| run | W&B | source | setting |
|---|---|---|---|
| `scratch_stage_a_down5_attn_refine_from_d2_8000_6k` | `lbzhch1m` | `scratch_stage_a_down5_from6000_dists2_lp05_12k/stage_a_0008000.pt` | `decoder_attention`, `extra_decoder_blocks=2`, lr `5e-5`, DISTS-heavy objective |

Deterministic Kodak center Stage-A evaluation:

| checkpoint | bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---|
| source `stage_a_0008000.pt` | 0.00977 | 0.45221 | 0.44797 | previous best Stage-A base. |
| attn `stage_a_best.pt` / `stage_a_0002000.pt` | 0.00977 | 0.45767 | 0.43546 | strong DISTS gain, LPIPS worsens. |
| attn `stage_a_final.pt` | 0.00977 | 0.44733 | 0.45193 | LPIPS improves, DISTS worsens. |

Stage-B run from DISTS-best attention Stage-A:

| run | W&B | Stage-A source | setting |
|---|---|---|---|
| `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d2_3k` | `4a1jwvsw` | `scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` | `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.5`, `lambda_lpips=0.7`, `lambda_dists=2.0` |

Deterministic Kodak center Stage-B evaluation:

| checkpoint | total bpp | residual bpp | base LPIPS | LPIPS | base DISTS | DISTS |
|---|---:|---:|---:|---:|---:|---:|
| `stage_b_0001000.pt` | 0.01489 | 0.00512 | 0.45767 | 0.44398 | 0.43546 | 0.43239 |
| `stage_b_0002000.pt` | 0.01315 | 0.00339 | 0.45767 | 0.43685 | 0.43546 | 0.42912 |
| `stage_b_best.pt` | 0.01390 | 0.00414 | 0.45767 | 0.43918 | 0.43546 | 0.42890 |
| `stage_b_final.pt` | 0.01328 | 0.00352 | 0.45767 | 0.43770 | 0.43546 | 0.42446 |

Interpretation:

- This is the first clear scratch Pareto update. The previous scratch low-rate lead was `0.01775` bpp / DISTS `0.43456`; the new final point reaches `0.01328` bpp / DISTS `0.42446`.
- The result directly supports the original decomposition: a cheap 8x8 semantic/generator code (`0.00977` bpp) plus only `0.0035` residual proxy bpp improves both LPIPS and DISTS over the stronger Stage-A base.
- Absolute quality remains far below the pretrained GLC real-codec lead, so this is not the submission lead yet. It is now a credible complete-design branch rather than just a proof-of-concept.

Decision:

- Promote `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d2_3k/stage_b_final.pt` as the current scratch low-rate lead.
- Next: run a lower-rate-pressure Stage-B from the same attention Stage-A (`lambda_R=0.3` or `0.2`) to see whether a quality-side scratch point can move below DISTS `0.42` while staying under roughly `0.02` bpp.



## 2026-06-21 JST - Scratch Stage-B quality-side sweep from attention Stage-A

Goal: after the strong `lambda_R=0.5` low-rate update, test whether lower rate pressure gives a useful quality-side scratch point from the same attention-refined Stage-A.

Run:

| run | W&B | Stage-A source | setting |
|---|---|---|---|
| `scratch_stage_b_from_attnA_best_r8_q1_lR0p3_d2_3k` | `vo5d3dkz` | `scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` | `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.3`, `lambda_lpips=0.7`, `lambda_dists=2.0` |

Deterministic Kodak center evaluation:

| checkpoint | total bpp | residual bpp | base LPIPS | LPIPS | base DISTS | DISTS |
|---|---:|---:|---:|---:|---:|---:|
| `stage_b_0001000.pt` | 0.01761 | 0.00784 | 0.45767 | 0.44674 | 0.43546 | 0.42673 |
| `stage_b_0002000.pt` | 0.01417 | 0.00440 | 0.45767 | 0.43985 | 0.43546 | 0.43012 |
| `stage_b_best.pt` | 0.01847 | 0.00871 | 0.45767 | 0.44850 | 0.43546 | 0.43254 |
| `stage_b_final.pt` | 0.01588 | 0.00611 | 0.45767 | 0.43752 | 0.43546 | 0.42396 |

Interpretation:

- `lambda_R=0.3` final gives a slightly better DISTS point than `lambda_R=0.5` final (`0.42396` vs `0.42446`) at higher bpp (`0.01588` vs `0.01328`).
- The gain is small, but it forms a reasonable second point for a scratch rate-perception curve.
- The random-val `stage_b_best.pt` did not transfer to fixed Kodak evaluation, so fixed deterministic evaluation remains necessary for checkpoint selection.

Decision:

- Keep `lambda_R=0.5` final as the best low-rate scratch point.
- Keep `lambda_R=0.3` final as the current scratch quality-side point.
- Further gains are more likely from residual modeling/progressive residual coding than simply lowering `lambda_R` again.



## 2026-06-21 JST - Scratch Stage-B continuation from lambda_R 0.5 lead

Goal: continue the best `lambda_R=0.5` Stage-B model at lower lr to see whether the scratch lead can improve without changing architecture.

Implementation:

- Added `--resume` support to `scripts/train_scratch_stage_b.py`.
- Continued `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d2_3k/stage_b_final.pt` from it=3000 to it=6000 with lr `1e-4`.

Run:

| run | W&B | resume | setting |
|---|---|---|---|
| `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k` | `vektoxqk` | `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d2_3k/stage_b_final.pt` | same objective, lr `1e-4`, total iters `6000` |

Deterministic Kodak center evaluation:

| checkpoint | total bpp | residual bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---:|---|
| source final | 0.01328 | 0.00352 | 0.43770 | 0.42446 | previous scratch low-rate lead. |
| continued `stage_b_0004000.pt` | 0.01321 | 0.00345 | 0.43869 | 0.42313 | best DISTS update. |
| continued `stage_b_0005000.pt` | 0.01310 | 0.00333 | 0.43657 | 0.42641 | better LPIPS, worse DISTS. |
| continued `stage_b_best.pt` | 0.01328 | 0.00352 | 0.43733 | 0.42427 | random-val best; small. |
| continued `stage_b_final.pt` | 0.01338 | 0.00362 | 0.43546 | 0.42642 | best LPIPS among this group. |

Interpretation:

- Continued training gives a small DISTS update at 4000: `0.01321` bpp / DISTS `0.42313`.
- Later checkpoints move toward LPIPS/MSE improvement but sacrifice DISTS.
- This reinforces that checkpoint selection should be metric-specific. For R-P/DISTS, use the 4000 checkpoint; for LPIPS auxiliary reporting, final/5000 can be referenced but not promoted.

Decision:

- Promote `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` as the current scratch DISTS lead.
- Keep the original/continued final checkpoints only as auxiliary LPIPS-oriented variants.



## 2026-06-21 JST - Scratch lead DIV2K center-crop generalization check

Goal: verify that the current scratch DISTS lead is not only improving Kodak center crops.

Evaluation note: this is not the official GLC/HiFiC DIV2K full-resolution shifted-patch FID protocol. It is the scratch evaluator's deterministic 256x256 center-crop sanity check over `/dpl/div2k` validation images.

Run evaluated:

| checkpoint | dataset | images | total bpp | residual bpp | base LPIPS | LPIPS | base DISTS | DISTS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` | DIV2K val center crop | 100 | 0.01364 | 0.00388 | 0.44078 | 0.42058 | 0.42494 | 0.41563 |

Interpretation:

- The same checkpoint improves LPIPS and DISTS on DIV2K center crops, so the scratch improvement is not Kodak-only.
- The absolute quality is still far below pretrained GLC, but the decomposition signal generalizes across at least Kodak and DIV2K center crops.



## 2026-06-21 JST - Scratch Stage-B residual_dim=4 negative result

Goal: test whether a narrower residual latent can create an even lower-rate scratch point.

Implementation:

- Updated scratch GroupNorm handling to choose a valid group count for small channel widths. Existing 8-divisible channels still use GroupNorm(8), so previous checkpoints remain compatible.
- Trained `residual_dim=4` from the attention-refined Stage-A.

Run:

| run | W&B | setting |
|---|---|---|
| `scratch_stage_b_from_attnA_best_r4_q1_lR0p5_d2_3k` | `2sb82ffg` | `residual_dim=4`, `quant_step=1.0`, `lambda_R=0.5`, DISTS-heavy objective |

Deterministic Kodak center evaluation:

| checkpoint | total bpp | residual bpp | LPIPS | DISTS | scale mean |
|---|---:|---:|---:|---:|---:|
| `stage_b_0001000.pt` | 0.01540 | 0.00564 | 0.44545 | 0.42986 | 0.62947 |
| `stage_b_best.pt` | 0.01376 | 0.00399 | 0.43780 | 0.42853 | 0.96634 |
| `stage_b_final.pt` | 0.01404 | 0.00427 | 0.43943 | 0.42657 | 1.13993 |

Interpretation:

- Narrowing to `residual_dim=4` does not beat the r8 lead (`0.01321` bpp / DISTS `0.42313`).
- The model compensates for the narrow residual by increasing scale and residual magnitude; lower dimension does not translate into better bitrate-quality efficiency.
- r8 currently looks like the smallest useful residual width for this architecture.

Decision:

- Do not pursue narrower residual dimensions before changing residual coding structure.
- Next residual-side idea should be progressive/RVQ residual stages or better entropy conditioning, not smaller single bottleneck width.



## 2026-06-21 JST - Scratch Stage-B residual_dim=16 ablation

Goal: test whether a wider residual bottleneck gives a better quality-side point from the attention-refined Stage-A.

Run:

| run | W&B | setting |
|---|---|---|
| `scratch_stage_b_from_attnA_best_r16_q1_lR0p5_d2_3k` | `kivq0tki` | `residual_dim=16`, `quant_step=1.0`, `lambda_R=0.5`, DISTS-heavy objective |

Deterministic Kodak center evaluation:

| checkpoint | total bpp | residual bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---:|---|
| `stage_b_0001000.pt` | 0.01147 | 0.00171 | 0.44868 | 0.43433 | very low rate, small gain. |
| `stage_b_best.pt` | 0.03115 | 0.02138 | 0.45881 | 0.43478 | random-val best does not transfer; high bpp wasted. |
| `stage_b_final.pt` | 0.01258 | 0.00281 | 0.43866 | 0.43298 | low-rate auxiliary point, not a quality update. |

Interpretation:

- With `lambda_R=0.5`, r16 is regularized so strongly that it mostly collapses residual transmission; it does not produce the desired quality-side point.
- It can form an ultra-low-rate auxiliary point, but r8 remains much better at comparable bpp-quality tradeoff.
- A wider residual bottleneck only makes sense with a different rate schedule, progressive stages, or lower `lambda_R`; simple r16 is not enough.

Decision:

- Do not promote r16 as quality lead.
- Current best remains r8 continued 4000: `0.01321` bpp / DISTS `0.42313`.



## 2026-06-21 JST - Scratch Stage-B quant_step=0.5 negative result

Goal: test whether finer residual quantization improves the quality side of the r8 scratch curve.

Run:

| run | W&B | setting |
|---|---|---|
| `scratch_stage_b_from_attnA_best_r8_q0p5_lR0p5_d2_3k` | `5e0rulf9` | `residual_dim=8`, `quant_step=0.5`, `lambda_R=0.5`, DISTS-heavy objective |

Deterministic Kodak center evaluation:

| checkpoint | total bpp | residual bpp | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| `stage_b_0002000.pt` | 0.01329 | 0.00353 | 0.43616 | 0.43199 |
| `stage_b_best.pt` | 0.01592 | 0.00615 | 0.44979 | 0.43322 |
| `stage_b_final.pt` | 0.01339 | 0.00362 | 0.43954 | 0.43302 |

Interpretation:

- Finer residual quantization does not improve fixed Kodak DISTS. It is consistently worse than `quant_step=1.0` r8.
- The model appears to adjust residual magnitude/scale so the proxy bpp remains similar, but the residual correction is less perceptually efficient.

Decision:

- Keep `quant_step=1.0` for the current scratch Stage-B.
- Do not spend more time on scalar quant_step sweeps until residual representation is changed.



## 2026-06-21 JST - Scratch Stage-B stronger-DISTS objective negative result

Goal: push the current r8/q1 Stage-B toward a better DISTS point by increasing DISTS weight and reducing LPIPS weight.

Run:

| run | W&B | setting |
|---|---|---|
| `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d3_lp05_3k` | `hanx5zoe` | `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.5`, `lambda_dists=3.0`, `lambda_lpips=0.5` |

Deterministic Kodak center evaluation:

| checkpoint | total bpp | residual bpp | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| `stage_b_best.pt` | 0.01341 | 0.00364 | 0.44364 | 0.42692 |
| `stage_b_final.pt` | 0.01331 | 0.00354 | 0.44963 | 0.42984 |

Interpretation:

- Increasing DISTS loss weight does not improve fixed Kodak DISTS. The model still gravitates to a similar residual bpp but less effective correction.
- The previous objective (`lambda_dists=2.0`, `lambda_lpips=0.7`) remains better.

Decision:

- Keep the current Stage-B objective. Further DISTS gains need architecture/residual-coding changes, not a simple DISTS weight increase.


## 2026-06-21 Scratch Progressive Residual Experiments

Implemented `ScratchProgressiveResidualBottleneck` with stage-wise bpp logging, optional decoder-side hard gate, and soft-train/hard-eval gating. The implementation can initialize from the best single-stage Stage-B checkpoint.

W&B:
- `337eca40`: non-gated two-stage progressive residual, `lambda_R=0.6`.
- `ig60pxg2`: hard-gated constant-init pilot, stopped because stage 1 stayed closed.
- `faev11ea`: hard-gated random-init pilot, stopped because stage 1 collapsed closed.
- `4ht20cqw`: soft-train/hard-eval gated progressive residual.

Kodak center-crop fixed evaluation:
- Non-gated progressive 2000: bpp 0.01954, LPIPS 0.43337, DISTS 0.41948, stage1 bpp 0.00694. Quality improves, but bpp is too high.
- Non-gated progressive final: bpp 0.02000, LPIPS 0.42979, DISTS 0.42299. LPIPS improves but DISTS not better enough.
- Gated soft-train 1000: bpp 0.01299, LPIPS 0.43538, DISTS 0.42373, stage1 bpp near zero. Good lower-rate curve point, but stage 1 is not yet doing useful work.
- Gated soft-train final: bpp 0.01312, LPIPS 0.43677, DISTS 0.42500.

Conclusion: progressive residual is not yet the scratch lead. It exposes the next research need: stage 1 needs a stage-specific correction decoder or an explicit improvement hinge so it learns a non-redundant residual role instead of being pruned away by the rate term.


## 2026-06-21 Progressive Gate Threshold Sweep

Added `--gate_threshold_override` to `scripts/evaluate_scratch_stage_b.py` to test whether the decoder-side gate threshold can act as a no-side-info rate knob. The gated soft-train 1000 checkpoint was evaluated on Kodak center crops:

- threshold 0.20: bpp 0.01299, LPIPS 0.43538, DISTS 0.42373, stage1 bpp ~0.
- threshold 0.15: bpp 0.01326, LPIPS 0.43525, DISTS 0.42371, stage1 bpp 0.00028.
- threshold 0.10: bpp 0.01478, LPIPS 0.43448, DISTS 0.42360, stage1 bpp 0.00179.
- threshold 0.05: bpp 0.01658, LPIPS 0.43284, DISTS 0.42352, stage1 bpp 0.00359.

The threshold knob is valid mechanically, but the current fine residual is not DISTS-efficient. Also tested a stage-specific fine correction decoder; 1000-step Kodak result was bpp 0.01323, LPIPS 0.43615, DISTS 0.42441, stage1 bpp 0. The next fix should explicitly train stage 1 to improve a stage-0 reconstruction.


## 2026-06-21 Stage-Improvement Hinge Pilot

Implemented `stage0_x_hat` in `ScratchProgressiveResidualBottleneck` and added `--lambda_stage_improve` / `--stage_improve_margin` to Stage-B training. Pilot W&B run `9g72335u` used fine correction decoder + soft train/hard eval gate + `lambda_stage_improve=5.0`, margin `0.001`; stopped after 1000 steps because hard stage 1 remained closed.

Fixed evaluation:
- Kodak center: bpp 0.01422, LPIPS 0.43778, DISTS 0.42232, stage1 bpp 0. This updates the scratch quality-side point but not the low-rate lead.
- Kodak center threshold 0.10: bpp 0.01611, LPIPS 0.43780, DISTS 0.42234, stage1 bpp 0.00189. Opening stage 1 does not help.
- DIV2K center: bpp 0.01553, LPIPS 0.42011, DISTS 0.41333, stage1 bpp 0.

Conclusion: the hinge helps the residual model improve DISTS, but still through stage 0. Fine stage specialization needs a warmup or hard-gate-aware training objective.


## 2026-06-21 Stage-1 Warmup and Gate Fine-Tune

Added `--train_only_extra_stages` to freeze the base residual path and train only the extra stage modules. Ran W&B `rubquyfn` with stage 1 forced open. Kodak center final: bpp 0.02003, LPIPS 0.43720, DISTS 0.42222, stage1 bpp 0.00681. This confirms the fine stage can carry useful residual information, but it is too expensive when always transmitted.

Then ran W&B `v0jqpxyq`, gate/rate fine-tuning from the warmup checkpoint. Kodak center final: bpp 0.01349, LPIPS 0.43748, DISTS 0.42283, stage1 bpp ~0.00001. Threshold 0.10 opens stage1 to 0.00186 bpp but does not improve DISTS. DIV2K center final: bpp 0.01428, LPIPS 0.41999, DISTS 0.41425.

Conclusion: warmup -> gate fine-tune improves the scratch quality-side point, but the learned gate still prunes stage 1 almost completely. Future work should preserve sparse high-value stage1 positions, likely with a target gate budget or top-k gate constraint during fine-tuning.


## 2026-06-21 Additional Gate Fine-Tune Sweep

Ran W&B `12dmxux7`: warmup -> gate fine-tune with `lambda_R=0.3`. Kodak center final was bpp 0.01503, LPIPS 0.43645, DISTS 0.42452, stage1 bpp 0. The lower rate penalty did not preserve useful hard-gated stage1 residuals; it is worse than the `lambda_R=0.6` fine-tune.


## 2026-06-21 Kodak Per-Image Scratch Comparison

Created `experiments/scratch_per_image_comparison_kodak.md` comparing the current single-stage scratch lead against the newer quality-side checkpoints.

Against `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt`:
- `stage_impr` improves DISTS on 14/24 Kodak images, mean ΔDISTS `-0.000815`, median `-0.000386`; best gains are `kodim22`, `kodim23`, `kodim03`, `kodim01`, `kodim15`.
- `warm_gateft` improves DISTS on 13/24 images, mean ΔDISTS `-0.000300`; LPIPS improves on 13/24 images.

Interpretation: the new scratch quality-side checkpoints are real but fragile. They improve some images clearly, while hurting others. Next work should inspect the best/worst images to learn whether gains correlate with texture, structure, or Stage-A failure modes.

## 2026-06-21 Top-k Gate Budget Pilot

Added `--gate_topk_frac` to `ScratchProgressiveResidualBottleneck`, training, and evaluation. The gate keeps exactly the highest-scoring fine-stage positions per sample and requires no transmitted side map because it is computed from decoder-available context. Ran W&B `oa3hchyt` from the forced-open stage-1 warmup checkpoint with `gate_topk_frac=0.05`, fine correction decoder, `lambda_R=0.5`, and stage-improvement hinge.

Kodak center final:

| total bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 gate mean |
|---:|---:|---:|---:|---:|---:|
| 0.01391 | 0.43889 | 0.42378 | 0.00380 | 0.00034 | 0.04883 |

Per-image comparison versus the single-stage scratch lead:
- mean ΔDISTS `+0.000646`, median `+0.000252`, wins `12/24`.
- mean ΔLPIPS `+0.000201`, median `-0.000404`, wins `12/24`.
- best DISTS gains: kodim22.png:-0.0093, kodim16.png:-0.0052, kodim03.png:-0.0046, kodim21.png:-0.0034, kodim23.png:-0.0032.
- worst DISTS losses: kodim18.png:+0.0115, kodim20.png:+0.0085, kodim11.png:+0.0050, kodim06.png:+0.0046, kodim07.png:+0.0043.

Conclusion: top-k gate budget successfully prevents gate collapse and realizes a true sparse residual mechanism, but the selected fine residual does not yet improve the Pareto point. Next work should train the selected subset with a hard-gate-aware DISTS/LPIPS improvement loss or make the gate conditional on stage-0 reconstruction error/texture proxies, then repeat the 5-10% budget sweep.

## 2026-06-21 Top-k Gate Budget Sweep

Evaluated the same top-k checkpoint with different deterministic gate budgets on Kodak center crops:

| top-k frac | bpp | LPIPS | DISTS | stage1 bpp | stage1 gate mean |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.01371 | 0.43889 | 0.42378 | 0.00014 | 0.01953 |
| 0.05 | 0.01391 | 0.43889 | 0.42378 | 0.00034 | 0.04883 |
| 0.10 | 0.01433 | 0.43888 | 0.42378 | 0.00077 | 0.09961 |
| 0.20 | 0.01519 | 0.43888 | 0.42380 | 0.00163 | 0.19922 |

DIV2K center at the trained 5% budget gives bpp `0.01515`, LPIPS `0.42036`, DISTS `0.41508`, stage1 bpp `0.00036`, gate mean `0.04883`.

Interpretation: increasing the fine-stage budget from 2% to 20% mostly increases bpp while DISTS stays around `0.42378-0.42380`. LPIPS improves only in the fourth decimal place. This rules out a simple budget issue: the current fine stage needs a stronger hard-gated correction objective, not just a wider gate.

## 2026-06-21 Top-k 10% Strong Stage-Improvement Pilot

A final short pilot tested whether stronger hard-gated stage-improvement can make the selected fine residual positions useful.

- Failed start: W&B `s1tbidjt` used `base_ch=128` by mistake, loaded only 73 tensors, produced invalid high-bpp validation, and was interrupted.
- Correct run: W&B `7dyy6dpq`, output `experiments/scratch_stage_b_progressive2_finedec_stage1warm_topk010_si20_b64_from_attnA_r8_q1q05_lR0p5_1k/`.
- Init: forced-open stage-1 warmup, fully compatible (`230 tensors`, `missing=0`, `skipped=0`).
- Config: `gate_topk_frac=0.10`, `lambda_stage_improve=20.0`, `lambda_R=0.5`, fine correction decoder.

Fixed center-crop results:

| dataset | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | gate mean |
|---|---:|---:|---:|---:|---:|---:|
| Kodak | 0.01504 | 0.44118 | 0.42219 | 0.00449 | 0.00078 | 0.09961 |
| DIV2K | 0.01641 | 0.42225 | 0.41361 | 0.00587 | 0.00077 | 0.09961 |

Interpretation: this is the best scratch Kodak DISTS point so far, but it is not the low-rate lead and LPIPS worsens. DIV2K does not beat the earlier stage-improvement checkpoint. The useful conclusion is that hard-gated sparse residuals can improve DISTS if the improvement pressure is strong enough, but the objective needs better regularization so gains do not come mainly from higher stage0 bpp and worse LPIPS.

## 2026-06-21 Selected-Region Top-k Fine-Residual Update

Implemented selected-region improvement loss for progressive Stage-B. The loss upsamples the decoder-side `stage1_gate_map` to image space and penalizes locations where the final reconstruction does not improve local L1 error over detached stage-0 reconstruction. This targets the original GP-ResLC axis more directly: if a fine residual position is selected for transmission, it must carry useful unpredictable correction. Also added a stage-1 scale guard to prevent entropy-scale inflation.

Runs:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_from_warm_lR0p5_2k` | `r3i0z4f3` | extra-stage-only, top-k 10%, selected loss 20, no scale guard | Kodak DISTS improved, but stage1 scale inflated after 500-1000 steps. |
| `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500` | `e6a0sh06` | same, plus `lambda_stage1_scale_guard=0.2`, `stage1_scale_target=0.8` | best scratch Kodak DISTS so far with controlled stage1 scale. |

Fixed center-crop results:

| checkpoint | dataset | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| no-guard 500 | Kodak | 0.01385 | 0.43758 | 0.42292 | 0.00345 | 0.00064 | 0.775 | good early point |
| no-guard 1000 | Kodak | 0.01389 | 0.43964 | 0.42279 | 0.00345 | 0.00067 | 1.429 | DISTS improves, scale inflates |
| scale-guard 500 | Kodak | 0.01386 | 0.43846 | 0.42274 | 0.00345 | 0.00065 | 0.680 | balanced update |
| scale-guard 1000 | Kodak | 0.01377 | 0.44009 | 0.42195 | 0.00345 | 0.00056 | 0.687 | new scratch Kodak DISTS lead |
| scale-guard final | Kodak | 0.01371 | 0.43921 | 0.42253 | 0.00345 | 0.00050 | 0.662 | lower rate, slightly worse DISTS |
| scale-guard 1000 | DIV2K center | 0.01424 | 0.42176 | 0.41391 | 0.00388 | 0.00060 | 0.678 | lower-bpp DIV2K quality-side point |

Per-image Kodak comparison for scale-guard 1000 versus the previous single-stage scratch lead:

- DISTS: mean delta `-0.001177`, median `-0.001159`, wins `18/24`.
- LPIPS: mean delta `+0.001398`, median `+0.001165`, wins `7/24`.
- Best DISTS gains: `kodim22:-0.0071`, `kodim04:-0.0062`, `kodim17:-0.0051`, `kodim23:-0.0030`, `kodim02:-0.0030`.
- Worst DISTS losses: `kodim20:+0.0037`, `kodim13:+0.0036`, `kodim16:+0.0027`, `kodim18:+0.0015`, `kodim09:+0.0004`.

Decision: promote `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt` as the current scratch DISTS lead, but not as the LPIPS lead. The new objective confirms that sparse fine-stage residuals can help when the selected positions are explicitly trained and stage1 scale is guarded. Next work should add LPIPS/feature-region guidance or a DISTS-aligned local proxy so the gain is not purely DISTS-biased.


## 2026-06-21 05:12 JST - Scratch LPIPS-balanced follow-ups

### Runs
- W&B `4sr7hpua`: `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_lp12_from_warm_lR0p5_1500`
  - Aim: increase global LPIPS weight from 0.7 to 1.2 under selected-region + stage1 scale guard.
  - Kodak fixed final: bpp 0.013646, LPIPS 0.438245, DISTS 0.423436.
  - DIV2K fixed final: bpp 0.014156, LPIPS 0.420427, DISTS 0.415380.
  - Read: recovers LPIPS, but gives back the DISTS gain. Simple global LPIPS weighting is not enough.
- W&B `njfmi964`: `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_stageLP6_from_sel1000_lR0p5_1k`
  - Code change: added `--lambda_stage_lpips_improve` and `--stage_lpips_improve_margin`, a detached stage0-vs-final LPIPS hinge.
  - Kodak fixed 500: bpp 0.013739, LPIPS 0.438604, DISTS 0.423288.
  - Kodak fixed final: bpp 0.013715, LPIPS 0.438228, DISTS 0.423569.
  - Read: LPIPS no-regression is active and improves LPIPS, but it still does not preserve the selected-region DISTS lead. Current scratch DISTS lead remains `...s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt`.

### Decision
The selected fine-residual stage is useful for DISTS when guarded against scale inflation, but LPIPS and DISTS pull the correction decoder in different directions. Next scratch step should move from whole-image perceptual weighting to spatially targeted feature no-regression or gate selection based on residual unpredictability/texture value, not just selected-region L1.


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


### 2026-06-21 q1 base-sendability target sweep real-codec result

Run: `v4_stage_quant_v1q1_basesend_rhotarget108_rhomax12_lR24_lp12_dists12_rt15_send5_500` (W&B `2wkwx8vd`).

Purpose: test whether a baseline-reconstruction sendability teacher can lower q1 bpp while preserving perceptual quality. This directly probes the GP-ResLC axis: do not send information that the generator/baseline can already recover.

Kodak8 real codec / q1:

| model | bpp | PSNR | LPIPS | DISTS | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GLC | 0.0310 | 21.3857 | 0.1852 | 0.1086 | 56.0845 | 0.0043 | reference |
| stage-quant quality q1 | 0.0304 | 21.3211 | 0.1855 | 0.1081 | 56.1878 | 0.0044 | keep |
| base-send 250 | 0.0296 | 21.1773 | 0.1900 | 0.1119 | 56.0310 | 0.0043 | reject |
| base-send final | 0.0296 | 21.1761 | 0.1891 | 0.1120 | 56.6404 | 0.0044 | reject |

Conclusion: global rho target plus heuristic sendability can reduce bpp, but it does not preserve LPIPS/DISTS. It is a useful negative result: sendability cannot be approximated by simple reconstruction-error/texture/edge heuristics. The next viable direction is a measured local perceptual-sensitivity teacher or a stronger residual branch that explicitly checks whether suppressing a residual worsens local perceptual features.


### 2026-06-21 q1 local LPIPS-spatial gate hinge

Implemented `--lambda_lpips_spatial_gate_hinge` in `scripts/train_v1.py`. The loss uses a spatial LPIPS map and penalizes high stage-gate probability where the current reconstruction locally worsens versus frozen GLC baseline. This gives the decoder-side gate a local perceptual no-regression signal instead of only whole-image LPIPS/DISTS hinges.

Smoke: `experiments/stage_quant_spatial_gate_smoke` passed.

Run: `v4_stage_quant_v1q1_spgate_rhotarget108_rhomax12_lR22_lp12_dists14_rt15_sp20_700` (W&B `guyrrgo8`).

Kodak8 real codec / q1 comparison:

| run | bpp | PSNR | LPIPS | DISTS | FID | KID | note |
|---|---:|---:|---:|---:|---:|---:|---|
| stageq quality | 0.0304 | 21.3211 | 0.1855 | 0.1081 | 56.1878 | 0.0044 | current q1 keep |
| old target108 | 0.0296 | 21.2318 | 0.1905 | 0.1121 | 57.2862 | 0.0047 | too much quality loss |
| send108 | 0.0295 | 21.1695 | 0.1900 | 0.1126 | 56.0714 | 0.0043 | FID only |
| spgate108 final | 0.0297 | 21.2427 | 0.1890 | 0.1104 | 56.8244 | 0.0046 | improves old target108 but not enough |

Conclusion: the spatial hinge is directionally correct: at the same low-rate target it recovers a visible part of DISTS/LPIPS quality. It is not yet a q1 replacement because the existing quality checkpoint remains better. Next quick sweep: a moderate target (`rho_target=1.06`) to see whether the new local teacher gives a usable intermediate curve point.


### 2026-06-21 q1 spgate target106 negative result

Run: `v4_stage_quant_v1q1_spgate_rhotarget106_rhomax12_lR20_lp12_dists14_rt15_sp20_700` (W&B `5o1ta9hl`).

Kodak8 real codec / q1:

| run | bpp | PSNR | LPIPS | DISTS | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| stageq quality | 0.0304 | 21.3211 | 0.1855 | 0.1081 | 56.1878 | 0.0044 | keep |
| target104 final | 0.0303 | 21.2998 | 0.1866 | 0.1100 | 55.8816 | 0.0044 | FID-only auxiliary |
| spgate106 250 | 0.0300 | 21.2480 | 0.1879 | 0.1110 | 56.4058 | 0.0045 | reject |
| spgate106 final | 0.0300 | 21.2607 | 0.1883 | 0.1103 | 56.5339 | 0.0046 | reject |
| spgate108 final | 0.0297 | 21.2427 | 0.1890 | 0.1104 | 56.8244 | 0.0046 | reject |

Conclusion: the LPIPS-spatial gate hinge improves the overly aggressive target108 run, but moderate target106 still pays too much DISTS/LPIPS. Do not promote any q1 target-sweep checkpoint. Keep `v3_stage_quant_v1q1_quality_hinge_fast_lR35_rhomax20_3k` as the q1 stage-quant point. The next on-axis improvement must estimate measured local sensitivity instead of relying on global rho targets plus proxy local LPIPS.


### 2026-06-21 Stage-quant local sensitivity analysis

Added `scripts/analyze_stage_quant_gate_sensitivity.py`. The analyzer runs GLC baseline and stage-quant reconstruction, upsamples `gate_rho`/`gate_p_tex`, computes local absolute error, texture variance, gradient, and Alex-LPIPS spatial delta, then reports correlations and high-rho/low-rho statistics.

Kodak8 / q1 summaries:

| checkpoint | delta bpp_y | rho mean | rho std | corr(rho, base err) | corr(rho, grad) | corr(rho, LPIPS delta) | high-rho LPIPS delta | low-rho LPIPS delta | read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stageq quality | -0.00059 | 1.0314 | 0.0089 | +0.193 | +0.165 | -0.016 | -0.00029 | +0.00156 | best q1; high-rho does not worsen LPIPS spatial |
| target104 final | -0.00078 | 1.0395 | 0.0053 | +0.006 | -0.006 | -0.002 | +0.00147 | +0.00061 | gate becomes almost uniform; quality worsens |
| spgate108 final | -0.00131 | 1.0739 | 0.0058 | -0.011 | -0.021 | +0.013 | +0.00579 | +0.00422 | stronger rate target becomes uniform and locally harmful |

Interpretation: the failed q1 target sweeps are not failing merely because of rho magnitude. They lose spatial selectivity. The current q1 quality checkpoint coarsens more in difficult regions by absolute-error/gradient statistics, but those high-rho locations are not the places where LPIPS spatial worsens; that is why it survives. When a global target forces higher mean rho, the gate field becomes near-uniform and high-rho locations have worse LPIPS spatial delta.

Next method implication: a stronger full-design stage-quant teacher should not be a global mean rho target. It should optimize a per-location budget: keep average bpp reduction, but explicitly push rho away from local LPIPS/DISTS-sensitive regions. The newly added analyzer provides the diagnostic needed for such a teacher.


### 2026-06-21 q1 LPIPS-sensitivity teacher result

Implemented `--lambda_gate_lpips_sens`, which builds a spatial teacher from the frozen GLC baseline LPIPS map. The teacher is recentered to the desired p-map mean implied by `rho_target`, so it reallocates the gate spatially without changing the intended average rate budget. Smoke test: `experiments/stage_quant_lpips_sens_smoke`.

Run: `v4_stage_quant_v1q1_lpsens_rhotarget108_rhomax12_lR22_lp12_dists14_rt15_lps10_e01_700` (W&B `vec5867e`).

Kodak8 real codec / q1:

| run | bpp | PSNR | LPIPS | DISTS | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| stageq quality | 0.0304 | 21.3211 | 0.1855 | 0.1081 | 56.1878 | 0.0044 | keep |
| target104 final | 0.0303 | 21.2998 | 0.1866 | 0.1100 | 55.8816 | 0.0044 | FID-only auxiliary |
| spgate108 final | 0.0297 | 21.2427 | 0.1890 | 0.1104 | 56.8244 | 0.0045 | reject |
| lpsens108 250 | 0.0298 | 21.2174 | 0.1892 | 0.1114 | 55.8860 | 0.0044 | reject |
| lpsens108 final | 0.0297 | 21.2051 | 0.1895 | 0.1111 | 56.0831 | 0.0045 | reject |

Conclusion: baseline LPIPS spatial maps alone are not sufficient sendability targets. They can improve distribution metrics/FID slightly, but do not protect DISTS/LPIPS enough. The q1 low-rate sweep is now closed: keep the q1 quality checkpoint. A stronger teacher must directly measure the effect of local coarsening, not infer it from baseline perceptual error.

## 2026-06-21 Scratch selected-loss audit and LPIPS-spatial follow-up

Discovered and fixed a critical scratch Stage-B issue: `selected_region_improvement_loss` in `scripts/train_scratch_stage_b.py` was accidentally decorated with `@torch.no_grad()`. The selected-region L1 loss therefore did not backpropagate in earlier selected-region runs. The previous scratch DISTS lead remains a valid measured checkpoint, but its mechanism should not be described as caused by the selected L1 term.

Re-ran selected-region experiments after the fix:

| run | W&B | setting | Kodak center result | decision |
|---|---|---|---|---|
| `scratch_stage_b_progressive2_selected_gradfix_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500` | `e1jku5vo` | selected L1=20, DISTS=2 | best checked final/1000 around bpp `0.01364-0.01370`, LPIPS `0.4388-0.4392`, DISTS `0.4232` | true selected L1 improves L1/LPIPS slightly but worsens DISTS; reject as main scratch lead |
| `scratch_stage_b_progressive2_selected_gradfix_extraonly_topk010_sel5_d4_s1scale08_from_warm_lR0p5_1500` | `7xk6rmuo` | selected L1=5, DISTS=4 | 500: `0.013694/0.440696/0.423637`; 1000: `0.013670/0.439949/0.423410`; final: `0.013642/0.441078/0.422958` for bpp/LPIPS/DISTS | lower rate and acceptable LPIPS, but still worse DISTS than old lead `0.421954`; reject as lead |

Interpretation: local L1 selected improvement is the wrong proxy for the scratch perceptual objective. It makes the transmitted fine residual more locally faithful, but that does not align with the DISTS/FID-style R-P claim. Added a new `--lambda_selected_lpips_improve` path that uses LPIPS spatial maps only on decoder-selected fine-stage regions. This is closer to the paper axis: selected residual positions must improve perceptual feature distance over the generator-only stage0 reconstruction.

Current follow-up running:

| run | W&B | setting |
|---|---|---|
| `scratch_stage_b_progressive2_selected_lpipsmap_extraonly_topk010_sellp10_d4_s1scale08_from_warm_lR0p5_1500` | `tpauo0kk` | selected LPIPS-spatial=10, selected L1 disabled, DISTS=4, LPIPS=0.5, top-k 10%, extra-stage-only |

### LPIPS-Spatial Selected Loss Result

Evaluated `scratch_stage_b_progressive2_selected_lpipsmap_extraonly_topk010_sellp10_d4_s1scale08_from_warm_lR0p5_1500` (W&B `tpauo0kk`) on fixed Kodak center crops:

| checkpoint | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.013835 | 0.438671 | 0.423115 | 0.003448 | 0.000621 | 0.607 | not lead |
| 1000 | 0.013772 | 0.438635 | 0.423123 | 0.003448 | 0.000558 | 0.660 | not lead |
| final | 0.013716 | 0.438633 | 0.423080 | 0.003448 | 0.000503 | 0.701 | not lead |

Compared with the current scratch DISTS lead (`0.013768` bpp, LPIPS `0.440089`, DISTS `0.421954`), LPIPS-spatial selected loss improves LPIPS but loses DISTS. Decision: keep as an LPIPS-oriented auxiliary/reference, not the scratch lead. The selected fine-stage objective should next be DISTS/texture-statistic aligned, or the scratch branch should move to a stronger generator before spending more time on local selected losses.

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



### DISTS Fine-Tune Final Screening

Completed the follow-up screening for both `lambda_dists=2` and a gentler `lambda_dists=1` fine-tune from the paper lead. The gentler run is `v2_gate_send_lR10_lp4_dists1_rho14_target116_send5_all_ft3k_from_lead` (W&B `8lct9ym0`). It keeps the same zero-side-bit rho gate mechanism and exact decoder recomputation as the paper lead.

Real-codec Kodak comparison versus local GLC:

| run | DISTS BD | LPIPS BD | PSNR BD | MS-SSIM BD | FID BD | KID BD | read |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_rho116_real` | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.14% | current paper lead |
| `gp_rho116_dists2_ft` | -5.62% | -0.32% | -0.72% | +0.39% | -3.27% | -7.54% | best Kodak FID/KID among fine-tunes |
| `gp_rho116_dists1_ft` | -5.96% | -0.68% | -1.35% | +0.19% | -2.28% | -6.08% | best Kodak DISTS among fine-tunes |

Real-codec DIV2K comparison versus local GLC:

| run | DISTS BD | LPIPS BD | PSNR BD | MS-SSIM BD | FID BD | KID BD | read |
|---|---:|---:|---:|---:|---:|---:|---|
| `gp_rho116_real` | -10.79% | -0.54% | -1.49% | -0.17% | -5.61% | -6.50% | keep as DIV2K lead |
| `gp_rho116_dists2_ft` | -10.53% | -0.52% | -1.42% | -0.17% | -5.99% | -5.46% | slightly better FID, worse DISTS/KID |
| `gp_rho116_dists1_ft` | -10.47% | -0.51% | -1.41% | -0.20% | -5.90% | -5.19% | worse than lead on DISTS/KID |

CLIC professional validation, compared to the existing `send5all` lead using DISTS/LPIPS/PSNR/MS-SSIM only because old and new FID/KID CSVs use different patch metadata:

| run | DISTS BD vs `send5all` | LPIPS BD | PSNR BD | MS-SSIM BD | decision |
|---|---:|---:|---:|---:|---|
| `ft_dists2` final | +0.11% | +0.20% | -0.07% | +0.02% | not a lead replacement |
| `ft_dists1` final | +0.63% | -0.04% | -0.17% | -0.05% | not a lead replacement |
| `ft_dists1_6500` | +1.36% | +0.09% | -0.32% | -0.11% | early checkpoint also worse |

Decision:

- Do not replace the paper-facing `rho1.16` lead with DISTS fine-tuned checkpoints.
- Keep `lambda_dists=1` as a Kodak-oriented auxiliary result: it is a real-codec improvement on Kodak DISTS, with exact decode consistency.
- Keep `lambda_dists=2` as a Kodak/FID-oriented auxiliary result and as evidence that DISTS-heavy fine-tuning can overfit small/easier benchmarks.
- For VCIP, the main CLIC2020/DIV2K story remains the original `rho1.16` real-codec result. Future improvement should target local sensitivity/sendability, not simply adding global DISTS loss during fine-tuning.


## 2026-06-21 10:00 JST - Stage-quant measured-sensitivity and rho1.18 upper-knob screening

Purpose: test two possible routes beyond the current paper lead: (1) a more method-faithful `stage_quant_gate` variant whose decoder-side gate is supervised by measured local LPIPS degradation versus frozen GLC, and (2) a pretrained global rho upper knob between the accepted `rho1.16` lead and rejected `rho1.20` ablation.

Implementation update:

- Added `--lambda_gate_measured_sens`, `--gate_measured_sens_tau`, `--gate_measured_sens_margin`, and `--gate_measured_sens_edge_weight` to `scripts/train_v1.py`.
- The measured teacher compares frozen GLC reconstruction and current gated reconstruction with spatial LPIPS, then allocates higher coarsening probability where the current local LPIPS delta is small or negative. The teacher is recentered to the desired mean implied by `rho_target`, so it changes spatial allocation rather than the average rate budget.
- Smoke test passed with `stage_quant_gate`; real-codec checks for evaluated checkpoints all had `max_abs=0.000e+00`.

Stage-quant q1 Kodak8 real-codec diagnostics:

| run | W&B | bpp | PSNR | LPIPS | DISTS | FID | KID | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| existing stage-quant q1 quality | - | 0.03044 | 21.3211 | 0.1855 | 0.1081 | 56.1878 | 0.0044 | anchor |
| measured-sens target1.08 | `syquq3py` | 0.02955 | 21.1616 | 0.1898 | 0.1116 | 55.9545 | 0.0044 | reject |
| measured-sens + distill target1.04 | `rzkqxp6a` | 0.03027 | 21.3107 | 0.1864 | 0.1097 | 55.8798 | 0.0042 | reject as DISTS replacement |

Read: the measured teacher can reduce serialized bpp and gives slightly better distribution metrics in conservative form, but it still fails to preserve DISTS/LPIPS versus the existing stage-quant q1 quality checkpoint. The bottleneck is not only spatial allocation; q1 residual precision is close to the perceptual floor under frozen GLC.

Pretrained rho branch screening:

- Run: `experiments/v2_gate_send_rho118_edge01_baseLP08_dists05_ft2k_from_lead`, W&B `x3sso91d`.
- Init: `rho1.16` lead checkpoint at it=6000.
- Setting: `rho_target=1.18`, `gate_send_edge_weight=0.1`, `lambda_base_l1=0.2`, `lambda_base_lpips=0.8`, `lambda_dists=0.5`. Stopped after the 7000 checkpoint because A/B trends stabilized.
- Kodak real-codec output: `experiments/real_codec/kodak_gp_reslc_rho118_edge01_baseLP08_dists05_7000/`.
- Metrics: `experiments/real_codec/kodak_gp_reslc_rho118_edge01_baseLP08_dists05_7000_metrics.csv`.

Kodak BD-rate versus local real-codec GLC:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| `gp_rho116_real` | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.14% |
| `gp_rho116_dists1_ft` | -5.96% | -0.68% | -1.35% | +0.19% | -2.28% | -6.08% |
| `gp_rho118_edge01_baseLP08_dists05_7000` | -4.03% | -0.13% | -0.46% | +0.68% | -0.64% | -5.30% |

Decision: reject rho1.18 as a paper lead. It reduces bpp but does not improve Kodak DISTS/FID enough, and q0 degrades. The main paper checkpoint remains `rho1.16`; DISTS fine-tunes remain Kodak-only auxiliary checkpoints because they fail DIV2K/CLIC validation.


## 2026-06-21 10:10 JST - Scratch selected VGG feature improvement loss

Implementation update:

- Added `--lambda_selected_vgg_improve`, `--selected_vgg_improve_margin`, and `--selected_vgg_layers` to `scripts/train_scratch_stage_b.py`.
- The new loss uses the VGG feature stages inside `DISTS_pytorch.DISTS.forward_once()` and penalizes decoder-selected fine-stage regions where the final reconstruction's local feature error is worse than detached stage-0. This is a DISTS-adjacent local selected-region proxy, intended to be closer to structure/texture fidelity than local L1 or LPIPS-spatial alone.
- Smoke test passed: checkpoint loading was fully compatible (`230 tensors`, `missing=0`, `skipped=0`) and `selvggimpr` was nonzero at startup.

Run:

- `experiments/scratch_stage_b_progressive2_selected_vgg_extraonly_topk010_selvgg10_d4_s1scale08_from_warm_lR0p5_1500`
- W&B: `0vkyk0iu`
- Setting: extra-stage-only, top-k 10%, selected VGG improvement weight 10, DISTS weight 4, LPIPS weight 0.5, stage1 scale guard 0.2/0.8.

Fixed Kodak center-crop results:

| checkpoint | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.014087 | 0.437148 | 0.424297 | 0.003718 | 0.000603 | 0.693 | not lead |
| 1000 | 0.014063 | 0.437081 | 0.424611 | 0.003718 | 0.000579 | 0.718 | not lead |
| final | 0.014053 | 0.437388 | 0.424407 | 0.003718 | 0.000569 | 0.669 | not lead |

Decision: reject as scratch lead. The VGG selected no-regression loss becomes zero quickly, so it mainly behaves like another DISTS-heavy extra-stage run. If this path is revisited, require a positive feature-improvement margin rather than only no-regression. Current scratch DISTS lead remains `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt` at Kodak bpp `0.013768`, LPIPS `0.440089`, DISTS `0.421954`.


## 2026-06-21 10:18 JST - Scratch selected VGG margin follow-up

Follow-up run:

- `experiments/scratch_stage_b_progressive2_selected_vggmargin_extraonly_topk010_selvgg30m003_d4_s1scale08_from_warm_lR0p5_1000`
- W&B: `bw5cnko1`
- Change from selected VGG no-regression: `lambda_selected_vgg_improve=30`, `selected_vgg_improve_margin=0.003`, 1000 iterations.

Fixed Kodak center-crop results:

| checkpoint | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.014102 | 0.437569 | 0.431100 | 0.003718 | 0.000618 | 0.666 | reject |
| final | 0.014070 | 0.437357 | 0.432706 | 0.003718 | 0.000587 | 0.737 | reject |

Decision: positive-margin VGG selected improvement is worse than no-regression and much worse than the current scratch lead. The feature-improvement pressure appears to fight the DISTS objective under the current weak generator/fine decoder. Stop this local VGG selected-loss path for now. Next scratch work should prioritize stronger Stage-A/generator quality or a true DISTS-statistic local proxy, not more VGG/L1 selected losses.


## 2026-06-21 10:35 JST - Scratch decoder-only and gate-error-target screening

### Stage-A decoder-only fine-tune from attention Stage-A best

- Run: `scratch_stage_a_decoder_only_from_attn_best_d3_lp08_l103_4k`
- W&B: `gsyo72t9`
- Change: froze Stage-A encoder and VQ codebook, trained decoder/latent-refine only with stronger perceptual loss (`lambda_dists=3.0`, `lambda_lpips=0.8`, `lambda_l1=0.3`).
- Outcome: stopped early at ~2500 iters because validation DISTS did not recover.
- Deterministic Kodak center:
  - `stage_a_best.pt`: bpp `0.0097656`, LPIPS `0.45655`, DISTS `0.43722`
  - `stage_a_0001000.pt`: bpp `0.0097656`, LPIPS `0.45191`, DISTS `0.45155`
  - `stage_a_0002000.pt`: bpp `0.0097656`, LPIPS `0.44913`, DISTS `0.44866`
- Decision: reject as Stage-A replacement. Decoder-only improves L1/LPIPS slightly but damages DISTS relative to the existing Stage-A best (`DISTS=0.43546`). This suggests the fixed 8x8 semantic latent is the bottleneck; decoder polishing alone cannot supply the missing perceptual structure.

### Stage-B decoder-side gate error-target auxiliary loss

- Code change: `ScratchProgressiveResidualBottleneck` now exposes `stage{i}_gate_prob` and `stage{i}_gate_logit`. `scripts/train_scratch_stage_b.py` adds `--lambda_gate_error_target` and `--gate_error_target_topk_frac`.
- Purpose: teach the decoder-computable fine-stage gate to select regions where Stage-A base reconstruction fails, without transmitting any extra side information.
- Run: `scratch_stage_b_gateerr_from_selected1000_ge02_1200`
- W&B: `k8a2znzk`
- Init: resumed from `experiments/scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt`.
- Key settings: `lambda_gate_error_target=0.2`, top-k `0.10`, extra-stage-only training, existing selected-region improvement losses retained.
- Deterministic Kodak center:
  - `stage_b_best.pt`/`stage_b_0001500.pt`: bpp `0.013935`, LPIPS `0.43914`, DISTS `0.42309`
  - `stage_b_0002000.pt`: bpp `0.013912`, LPIPS `0.43939`, DISTS `0.42327`
  - `stage_b_final.pt`: bpp `0.013940`, LPIPS `0.43895`, DISTS `0.42306`
- Comparison: previous scratch lead remains `stage_b_0001000.pt` from `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500` with bpp `0.013768`, LPIPS `0.44009`, DISTS `0.42195`.
- Decision: reject as DISTS lead, keep as useful ablation. Gate-target improves LPIPS slightly and is conceptually aligned, but DISTS worsens by ~0.0011 and bpp is slightly higher.


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


## 2026-06-21 - Pretrained gate-only DISTS-heavy screening

- Run: `v2_gateonly_dists4_rho116_lR10_lp2_align0_ft1000_from_lead`
- W&B: `6dg3powh`
- Change: resumed from the `rho1.16` real-codec lead, froze `prior_predictor` and `q_embed`, trained only the decoder-side gate with `lambda_dists=4`, `lambda_lpips=2`, `lambda_align=0`.
- Exact real codec: arithmetic compress/decompress, Kodak q0-q3, all decoded images matched the model output with `max_abs=0`.
- Kodak real-codec metrics:
  - q0: bpp `0.023770`, LPIPS `0.2120`, DISTS `0.1154`, FID `29.0295`, KID `0.0035`
  - q1: bpp `0.027508`, LPIPS `0.1882`, DISTS `0.1069`, FID `26.4583`, KID `0.0030`
  - q2: bpp `0.032036`, LPIPS `0.1748`, DISTS `0.1003`, FID `24.9985`, KID `0.0026`
  - q3: bpp `0.036225`, LPIPS `0.1654`, DISTS `0.0955`, FID `24.0121`, KID `0.0023`
- BD-rate vs local real-codec GLC on Kodak: DISTS `-4.02%`, LPIPS `-0.66%`, PSNR `-0.54%`, MS-SSIM `+0.35%`, FID `-1.80%`, KID `-5.99%`.
- Decision: reject as lead. FID improves slightly relative to the current `rho1.16` lead, but DISTS/LPIPS and most distortion-side curves weaken at essentially the same exact bpp.


## 2026-06-21 - V2 q0/q1 baseline-hinge gate-only screening

Implementation:

- Added `--q_choices` to `scripts/train_v2.py` so V2 can fine-tune selected rates instead of sampling all q uniformly.
- Added `--resume_weights_only` to load an existing V2 checkpoint with a fresh optimizer, needed when trainable modules change.
- Added GLC-baseline perceptual constraints: `--lambda_dists_distill`, `--lambda_lpips_hinge`, `--lambda_dists_hinge`, plus hinge margins.

Runs:

- Failed/aborted: `v2_gateonly_q01_basehinge_lR10_lp4_d1_hinge15_1500_from_lead`, W&B `ne8nvlck`. I forgot to restore `gate_rho_min=1.0`; rho collapsed below 1 and the model spent more bits, so the run was stopped.
- Valid: `v2_gateonly_q01_basehinge_rhomin1_rt112_lR10_lp4_d1_hinge10_1000_from_lead`, W&B `mdoc94xd`. Resumed from `rho1.16`, froze predictor/q_embed, trained gate only on q0/q1 with `rho_min=1.0`, `rho_target=1.12`, `lambda_R=10`, `lambda_lpips=4`, `lambda_dists=1`, DISTS hinge 10, LPIPS hinge 1.

Real codec consistency:

- Kodak q0-q3 and DIV2K q0-q3 completed with arithmetic compress/decompress and `max_abs=0.000e+00` against the forward path.
- DIV2K average bpp: q0 `0.02197`, q1 `0.02571`, q2 `0.03025`, q3 `0.03446`.

BD-rate vs local real-codec GLC:

| dataset | run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Kodak | q01 hinge | -4.90 | -1.00 | -1.43 | -0.21 | -2.01 | -4.67 | better than rho1.16 on most Kodak metrics, not enough alone |
| DIV2K | q01 hinge | -8.69 | -0.87 | -1.10 | -0.46 | -4.97 | -8.02 | reject as lead; DISTS/FID weaker than rho1.16 |

Interpretation: q-specific baseline hinge improves point quality and LPIPS, but it is too conservative for the main R-P claim. The current `rho1.16` paper lead remains stronger on DIV2K DISTS/FID and should remain the lead unless a future run preserves rate saving while adding only local safety.


## 2026-06-21 - V2 q0/q1 weak baseline-hinge follow-up

Run: `v2_gateonly_q01_weakhinge_rhomin1_rt114_lR10_lp4_d05_hinge5_800_from_lead`, W&B `8xffwful`.

Goal: recover more of the `rho1.16` rate saving than the conservative q01 hinge run by using `rho_target=1.14`, weaker DISTS/LPIPS hinges, and lower direct DISTS weight.

Real codec: Kodak q0-q3 completed with arithmetic compress/decompress and `max_abs=0.000e+00`.

Kodak BD-rate vs local real-codec GLC:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| `gp_rho116_real` | -4.47 | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |
| `q01 weak hinge` | -4.78 | -0.99 | -1.12 | +0.03 | +0.26 | -3.14 |

Decision: reject and do not spend CLIC/DIV2K time. It gives small Kodak DISTS/LPIPS improvements but FID becomes worse than GLC on the Kodak patch protocol, so it weakens the perceptual-compression claim.


## 2026-06-21 12:55 JST - V2 predictor-only mean correction from rho1.16 lead

Goal: test whether the original GP-ResLC axis can be strengthened beyond pure quantization gating by letting a decoder-computable `z_hat,q -> prior mean` correction remove a small predictable latent component, while keeping the successful rho1.16 gate fixed.

Implementation:

- Added `--freeze_gate` to `scripts/train_v2.py` so fine-tunes can freeze the loaded perceptual gate.
- Smoke passed with `max`-bounded mean correction and frozen q embedding.
- Run: `v2_predonly_mean_b003_lR6_lp4_d1_hinge_from_lead_1200`, W&B `06468x7k`.
- Init: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`.
- Settings: `predictor_param_mode=mean`, `predictor_delta_bound=0.003`, `freeze_gate=true`, `freeze_q_embed=true`, `lambda_R=6`, `lambda_lpips=4`, `lambda_dists=1`, DISTS hinge 3, LPIPS hinge 0.5 with margin 0.02.

Training read:

- The predictor remained small: `delta_abs` stayed around `9.2e-4`, well below the `0.003` bound.
- Kodak validation A/B stayed stable: q0-q3 `delta_bpp_y` roughly `-0.0026..-0.0028` vs frozen GLC, with no catastrophic PSNR collapse.

Kodak exact real-codec evaluation:

- Output: `experiments/real_codec/kodak_gp_reslc_predonly_mean_b003_1200/`.
- Codec consistency: `max_abs=0.000e+00` for all q/images.
- Average bpp q0-q3: `0.02364 / 0.02741 / 0.03194 / 0.03610`.
- Comparison output: `experiments/real_codec/kodak_predonly_mean_b003_compare/`.

BD-rate vs local GLC on Kodak:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| rho1.16 lead | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.33% |
| predonly_b003 | -4.28% | -0.70% | -0.96% | +0.36% | -2.30% | -8.97% |

BD-rate vs rho1.16 lead:

| metric | predonly_b003 |
|---|---:|
| PSNR | -0.19% |
| MS-SSIM | -0.06% |
| LPIPS | +0.55% |
| DISTS | +0.21% |
| FID | -0.45% |
| KID | -2.69% |

Decision:

- Do not replace the paper lead. DISTS and LPIPS are slightly weaker than rho1.16, and the gain is not large enough for the official-curve story.
- Keep as positive mechanism evidence: a tiny decoder-computable mean correction can reduce serialized bpp a bit and improve distribution metrics without breaking exact decoding. This suggests the full design should combine gate-based residual precision suppression with a better-trained, stage-aware residual mean predictor rather than a global mean head.


## 2026-06-21 13:16 JST - DIV2K real-codec evaluation for predictor-only mean b003

After the Kodak probe, I evaluated `v2_predonly_mean_b003_lR6_lp4_d1_hinge_from_lead_1200` on full-resolution DIV2K validation using the exact real codec.

Artifacts:

- Recon/payload manifests: `experiments/real_codec/div2k_gp_reslc_predonly_mean_b003_1200/`
- Metrics CSV: `experiments/real_codec/div2k_gp_reslc_predonly_mean_b003_1200_metrics.csv`
- Comparison CSV: `experiments/real_codec/div2k_predonly_mean_b003_compare_metrics.csv`
- BD summary vs GLC: `experiments/real_codec/div2k_predonly_mean_b003_compare_bd.md`
- BD summary vs rho1.16: `experiments/real_codec/div2k_predonly_mean_b003_compare_vs_rho116_bd.md`

Exact codec status:

- All q/images decode with `max_abs=0.000e+00` versus the differentiable forward path.
- Average bpp q0-q3: `0.02129 / 0.02505 / 0.02952 / 0.03372`.
- This is substantially lower than the rho1.16 DIV2K bpp curve (`0.02347 / 0.02728 / 0.03184 / 0.03601`).

DIV2K metrics for predonly_b003:

| q | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02129 | 21.2496 | 0.7697 | 0.1959 | 0.09106 | 15.0914 | 0.001258 |
| 1 | 0.02505 | 21.6995 | 0.7901 | 0.1779 | 0.08320 | 13.3201 | 0.000923 |
| 2 | 0.02952 | 22.1004 | 0.8064 | 0.1645 | 0.07779 | 12.4485 | 0.000749 |
| 3 | 0.03372 | 22.4531 | 0.8164 | 0.1563 | 0.07514 | 11.8356 | 0.000670 |

BD-rate vs local GLC on DIV2K:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| rho1.16 | -10.79% | -0.54% | -1.49% | -0.17% | -5.61% | -6.50% |
| predonly_b003 | -9.76% | -0.80% | -1.61% | -0.45% | -6.73% | -8.27% |

BD-rate predonly_b003 vs rho1.16:

| metric | BD-rate |
|---|---:|
| DISTS | +1.12% |
| LPIPS | -0.30% |
| PSNR | -0.16% |
| MS-SSIM | -0.25% |
| FID | -1.68% |
| KID | -2.69% |

Decision:

- Do not replace the DISTS lead: rho1.16 remains stronger on the primary DISTS curve.
- predonly_b003 is a useful FID/KID/LPIPS auxiliary variant. It shows that bounded decoder-computable mean prediction can lower serialized bits more aggressively and improve distribution metrics, but it slightly underperforms rho1.16 on DIV2K DISTS.
- CLIC full evaluation is optional rather than mandatory for the paper lead. If run, it should be framed as checking whether the FID gain transfers, not as a likely DISTS replacement.


## 2026-06-21 14:55 JST - CLIC2020 full-test real-codec evaluation for predictor-only mean b003

I completed full CLIC2020 test evaluation for `v2_predonly_mean_b003_lR6_lp4_d1_hinge_from_lead_1200` using the exact arithmetic real codec. This uses the 428-image CLIC2020 Professional+Mobile test union and official-style 256x256 patches with half-patch shift.

Artifacts:

- Recon/payload manifests: `experiments/real_codec/clic2020_test_gp_reslc_predonly_mean_b003_1200/`
- q1-q3 metrics CSV: `experiments/real_codec/clic2020_test_gp_reslc_predonly_mean_b003_1200_q123_metrics.csv`
- Hybrid metrics CSV: `experiments/real_codec/clic2020_test_predonly_mean_b003_hybrid_metrics.csv`
- Local BD summaries: `experiments/real_codec/clic2020_hybrid_rhoq0_predq123_vs_GLC_bd.md`, `experiments/real_codec/clic2020_hybrid_rhoq0_predq123_vs_rho1p16_bd.md`
- Official graph comparison: `experiments/paper_assets/official_curve_comparison_predonly_mean_b003/`

Exact codec status:

- q1, q2, q3 all decode with `max_abs=0.000e+00` versus the forward reconstruction.
- Average bpp q1-q3: `0.02241 / 0.02689 / 0.03089`.
- Average encode/decode time: q1 `0.728s / 1.013s`, q2 `0.821s / 1.106s`, q3 `0.912s / 1.196s` per full-resolution image.
- FID/KID patch count is exactly `28,650` for every q, matching the CLIC2020 test protocol.

CLIC2020 metrics for predonly_b003 q1-q3:

| q | bpp | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.02241 | 24.2608 | 0.8408 | 0.1489 | 0.07493 | 5.4279 | 0.001010 |
| 2 | 0.02689 | 24.7280 | 0.8536 | 0.1358 | 0.06880 | 4.7164 | 0.000764 |
| 3 | 0.03089 | 25.0132 | 0.8611 | 0.1288 | 0.06621 | 4.4755 | 0.000714 |

Hybrid curve uses rho1.16 q0 plus predonly_b003 q1-q3. BD-rate vs local real-codec GLC:

| metric | BD-rate |
|---|---:|
| DISTS | -9.22% |
| LPIPS | -0.13% |
| PSNR | -1.17% |
| MS-SSIM | -0.07% |
| FID | -7.19% |
| KID | -5.36% |

BD-rate of the same hybrid curve vs rho1.16 lead:

| metric | BD-rate |
|---|---:|
| DISTS | +1.15% |
| LPIPS | -0.20% |
| PSNR | -0.19% |
| MS-SSIM | -0.36% |
| FID | +0.22% |
| KID | +2.50% |

Official graph-extracted CLIC GLC comparison for the hybrid curve:

| metric | BD-rate |
|---|---:|
| DISTS | -8.00% |
| FID | -5.99% |
| KID | -4.59% |
| LPIPS | +1.12% |
| PSNR | +0.49% |
| MS-SSIM | +1.32% |

Decision:

- Do not replace the VCIP paper lead. The predictor-only hybrid is strong versus GLC, but it is weaker than rho1.16 on the primary CLIC DISTS curve and slightly weaker on FID/KID versus rho1.16.
- The experiment is valuable mechanistic evidence: bounded decoder-computable mean prediction reduces serialized y bits and works with exact arithmetic coding, but a global mean correction is too blunt for DISTS.
- Next high-upside pretrained direction: combine the rho1.16 residual-precision gate with a stage-aware or sensitivity-aware mean predictor, using explicit DISTS/LPIPS safety hinges. Avoid globally unfreezing entropy/prior modules until estimated bpp and serialized bpp are calibrated.


## 2026-06-21 15:25 JST - Stage residual real-codec audit

Goal: test a more literal GP-ResLC mechanism: a decoder-recomputable four-part residual-mean predictor, so bits are spent on the part of `y` not predictable from `z_hat`, GLC common prior parameters, and already decoded latent parts.

Implementation/protocol fix:
- Added `stage_latent_residual` support to `scripts/train_v2.py` and `gp_reslc/real_codec.py`.
- Fixed a real-codec graph mismatch: the real codec skipped the pretrained perceptual gate for stage-aware modes, while `train_forward` applied it before the four-part prior. Before the fix, Kodak real-codec checks showed `consistency_max_abs ~= 1.4..2.3`, so those outputs were invalid for paper claims. After applying the same gate in `_apply_gp_reslc_params`, the debug real-codec run gives `max_abs=0` for q0/q3 on Kodak.

Run audited:
- Checkpoint: `experiments/v2_stage_resid_b006_lR6_lp4_d1_mean05_hinge_from_lead_1k/v2_final.pt`
- W&B: `07r6rjhl`
- Real codec: `experiments/real_codec/kodak_gp_reslc_stage_resid_b006_1k_gatefix/`
- Metrics: `experiments/real_codec/kodak_gp_reslc_stage_resid_b006_1k_gatefix_metrics.csv`
- BD comparison: `experiments/real_codec/kodak_stage_resid_b006_compare_bd.md`

Kodak real-codec metrics after the fix:

| q | bpp | DISTS | LPIPS | PSNR | MS-SSIM | FID |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.021331 | 0.131313 | 0.248151 | 20.4510 | 0.6994 | 35.2075 |
| 1 | 0.024864 | 0.114594 | 0.212862 | 21.0427 | 0.7322 | 29.4786 |
| 2 | 0.029455 | 0.104397 | 0.189971 | 21.5613 | 0.7575 | 26.2882 |
| 3 | 0.033588 | 0.098321 | 0.175090 | 21.9389 | 0.7747 | 24.7416 |

BD-rate versus local real-codec GLC:
- rho1.16 lead: DISTS `-4.47%`, LPIPS `-0.79%`, PSNR `-0.87%`, MS-SSIM `+0.45%`, FID `-1.70%`, KID `-6.14%`.
- stage residual b006: DISTS `-2.32%`, LPIPS `+6.48%`, PSNR `+3.46%`, MS-SSIM `+5.93%`, FID `+4.34%`, KID `+1.23%`.

Decision: reject `b006` as paper lead. It is closer to the original residual-coding thesis than pure rho scaling, but the residual-mean predictor is too aggressive: it lowers bits by moving necessary latent content into a mean prediction that the generator/prior cannot reconstruct perceptually well. This is a useful negative result and suggests the complete design needs either a much more constrained residual predictor, a reconstruction-preserving distillation term, or a staged training schedule where the entropy model is adapted around the residual path instead of only attaching a predictor to a fixed GLC prior.

Follow-up launched:
- `experiments/v2_stage_resid_b002_lR6_lp4_d1_mean01_hinge_from_lead_1k/`, W&B `o7mnj6mq`.
- Changes: `predictor_delta_bound=0.002`, `lambda_mean_pred=0.1` to keep only a small decoder-predictable residual correction.


## 2026-06-21 15:55 JST - Conservative and delta-regularized stage residual follow-ups

Follow-up runs after rejecting the unconstrained `b006` stage residual:

1. Conservative bound run
- Checkpoint: `experiments/v2_stage_resid_b002_lR6_lp4_d1_mean01_hinge_from_lead_1k/v2_final.pt`
- W&B: `o7mnj6mq`
- Real codec: `experiments/real_codec/kodak_gp_reslc_stage_resid_b002_1k_gatefix/`
- Metrics: `experiments/real_codec/kodak_gp_reslc_stage_resid_b002_1k_gatefix_metrics.csv`
- BD table: `experiments/real_codec/kodak_stage_resid_b002_compare_bd.md`
- Result vs local GLC on Kodak: DISTS `+0.65%`, LPIPS `+7.30%`, PSNR `+4.84%`, MS-SSIM `+6.73%`, FID `+4.84%`, KID `-0.75%`.
- Decision: reject. Lowering the bound to `0.002` does not recover quality; DISTS is worse than GLC.

2. Delta-L1 regularized run
- Code change: added `--lambda_stage_delta_abs` to `scripts/train_v2.py` and W&B logging for `train/stage_delta_l1`.
- Checkpoint: `experiments/v2_stage_resid_b006_stageL20_lR6_lp4_d1_mean01_hinge_from_lead_800/v2_final.pt`
- W&B: `9ovci19c`
- Real codec: `experiments/real_codec/kodak_gp_reslc_stage_resid_b006_stageL20_800_gatefix/`
- Metrics: `experiments/real_codec/kodak_gp_reslc_stage_resid_b006_stageL20_800_gatefix_metrics.csv`
- BD table: `experiments/real_codec/kodak_stage_resid_b006_stageL20_compare_bd.md`
- Result vs local GLC on Kodak: DISTS `-2.52%`, LPIPS `+6.62%`, PSNR `+3.85%`, MS-SSIM `+5.96%`, FID `+5.22%`, KID `+1.54%`.
- Decision: reject as paper lead. It slightly improves DISTS over unconstrained `b006` (`-2.32%`) but remains far behind rho1.16 (`-4.47%`) and worsens the other metrics.

Stage residual failure analysis:
- CSV: `experiments/real_codec/stage_residual_saturation_kodak.csv`
- L1 run q0/q3 CSV: `experiments/real_codec/stage_residual_stageL20_saturation_kodak_q03.csv`
- `b002` saturates the bounded delta almost everywhere: stage-wise saturation fraction is roughly `0.84..0.96`, while delta-target correlation is only `0.03..0.10` on q0/q3.
- `b006` and `b006_stageL20` reduce saturation but still have weak image-wise delta/target correlation (`~0.05..0.14`) and broad bound-seeking shifts.

Interpretation: the pretrained fixed-GLC-prior attachment does not yet realize the intended principle. It can lower arithmetic-coded `y` bits, but the learned mean correction is not selective enough to represent only generator-predictable residual. For a complete GP-ResLC version, the residual predictor must be trained jointly with the entropy model / synthesis path, or gated by an uncertainty/sendability signal tied to perceptual sensitivity. This supports moving beyond post-hoc mean correction toward a scratch or staged full-model training path.


## 2026-06-21 16:00 JST - Scratch Stage-A basis audit

Purpose: check whether the scratch branch has a better semantic/generative Stage-A foundation than the current attention-refined Stage-A used by Stage-B residual experiments.

Deterministic Kodak center-crop Stage-A evaluations:

| checkpoint | semantic bpp | LPIPS | DISTS | note |
|---|---:|---:|---:|---|
| `scratch_stage_a_vq1024_b80_z160_down5_softent_restart_from6000_30k/stage_a_best.pt` | 0.00977 | 0.45730 | 0.45180 | long soft-entropy run; not better |
| `scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` | 0.00977 | 0.45767 | 0.43546 | best DISTS foundation among checked Stage-A models |
| `scratch_stage_a_decoder_only_from_attn_best_d3_lp08_l103_4k/stage_a_best.pt` | 0.00977 | 0.45655 | 0.43722 | slightly better LPIPS, worse DISTS |

Tried a new decoder-only DISTS-heavy continuation from the attention-refined Stage-A:
- Failed setup run: W&B `e7zk9101`, structure mismatch (`extra_decoder_blocks` omitted); no training, ignore.
- Correct run: `scratch_stage_a_decoder_only_dists5_lp03_l102_from_attn_best_plus1200`, W&B `n14pag52`.
- Settings: freeze encoder+quantizer, resume `scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt`, decoder attention + `extra_decoder_blocks=2`, `lambda_dists=5`, `lambda_lpips=0.3`, `lambda_l1=0.2`, +1200 steps.
- Deterministic Kodak center results: best checkpoint LPIPS/DISTS `0.46299/0.44128`, final `0.46312/0.44256`.

Decision: reject the new Stage-A continuation. DISTS-heavy decoder-only training does not improve the Stage-A basis and hurts LPIPS. Keep `scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` as the scratch Stage-A foundation. The scratch bottleneck's remaining limitation is not simply Stage-A decoder loss weighting; it likely needs either a stronger generator architecture/training schedule or better residual-stage objectives after Stage-A.

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

## 2026-06-21 Scratch-v2 top-conference pivot

- Direction: VCIP short-cycle optimization is no longer the main target. Move toward a top-conference GP-ResLC design: keep the semantic/generative stream extremely cheap, and spend residual bits only where the generator cannot plausibly recover content.
- Finding: pretrained GLC-based variants have a solid local lead but are structurally constrained by the frozen/partially frozen GLC generator and do not fully enforce the predictable-vs-unpredictable decomposition.
- Fix implemented: scripts/train_scratch_stage_a_adv.py now loads attention/refined Stage-A checkpoints correctly and supports delayed adversarial training, discriminator feature matching, and encoder/quantizer freezing.
- Smoke test: experiments/smoke_stage_a_adv_load ran 1 iteration from experiments/scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt; CUDA visible; semantic fixed bpp stayed 0.00977; Kodak quick-val LPIPS/DISTS were roughly 0.4067/0.3927 for the sampled batch.
- Next run: long Stage-A generator fine-tune with semantic encoder/codebook frozen. Hypothesis: a stronger generator at unchanged semantic rate lowers the residual burden in Stage-B and better realizes the GP-ResLC premise.



## 2026-06-21 17:10 JST - Top-conference pivot: GLC-latent residual scratch/full-design branch

Decision update:

- The project direction is now top-conference/full-paper oriented rather than VCIP-short-track first.
- The pretrained real-codec rho branch remains the controlled GLC anchor and source-of-truth for protocol-clean comparisons, but it is unlikely to produce a sufficiently large conceptual/performance jump by small q/rho sweeps alone.
- The scratch branch is method-faithful but limited by the weak Stage-A generator. The high-upside next branch should keep the original axis while using the strong pretrained GLC/VQGAN generator as a synthesis prior.

Implemented branch:

- Added `gp_reslc/scratch/glc_latent_residual.py`.
- Added `scripts/train_glc_latent_residual.py`.
- The model sends the existing 8x8 Stage-A semantic VQ code `s` and learns `mu_theta(s)` in the frozen GLC/VQGAN latent space. A low-dimensional entropy-modeled residual then corrects only the unpredictable component of the GLC latent.
- This is closer to the full GP-ResLC design than post-hoc global GLC prior shifts: the generator-predictable component is explicitly represented as `mu_theta(s)`, and the transmitted residual is modeled separately.

Smoke/pilot:

- Smoke command: `scripts/train_glc_latent_residual.py`, 2 iterations, `residual_dim=16`, `hidden_dim=128`, no W&B. CUDA forward/backward and checkpoint writing passed.
- Predictor-only pilot: `experiments/glc_latent_residual_predictor_pilot_80`, 80 iterations, no W&B. Validation logic was fixed so predictor-only warmup does not evaluate with random residuals.
- Pilot signal: Kodak validation DISTS dropped from about `0.77` at init to about `0.51` by 40 iterations with only the fixed semantic stream (`0.00977` bpp). This suggests the frozen GLC generator can be driven from the cheap semantic code, unlike the weaker scratch decoder.

Long run started:

```bash
.venv/bin/python -u scripts/train_glc_latent_residual.py \
  --glc_weights pretrained/GLC_image.pth.tar \
  --stage_a_ckpt experiments/scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt \
  --data /dpl/openimages/train \
  --val /dpl/kodak \
  --out experiments/glc_latent_residual_predictor_warmup_6k \
  --iters 6000 --bs 2 --num_workers 4 \
  --log_every 50 --eval_every 500 --save_every 2000 \
  --residual_dim 24 --hidden_dim 256 --predictor_only_iters 6000 \
  --lr 0.0002 --lambda_R 0 \
  --lambda_l1 0.2 --lambda_lpips 0.8 --lambda_dists 1.2 \
  --lambda_pred 2 --lambda_latent 2 \
  --wandb_project gp-reslc-research \
  --wandb_name glc_latent_residual_predictor_warmup_6k \
  --wandb_mode online
```

W&B:

- Project: `gp-reslc-research`
- Run: `glc_latent_residual_predictor_warmup_6k`
- Run id: `woye1ymw`

Early online status:

- Iteration 500 validation: total bpp `0.00977`, residual bpp `0`, LPIPS `0.7868`, DISTS `0.4716`, pred/latent loss `0.1826`.
- The predictor is still below the best scratch Stage-A DISTS, but it is learning quickly and uses a much stronger frozen generator. Continue warmup, then start a residual phase from the best checkpoint.

Next action after warmup:

1. Resume from `glc_latent_residual_best.pt` with `predictor_only_iters=0` and nonzero `lambda_R` to train the residual branch.
2. Use `lambda_R` sweep around `0.5, 1.0, 2.0`; promote only if residual bpp adds perceptual gain efficiently.
3. Add deterministic center-crop evaluator for this branch and compare against current scratch Stage-B and local GLC crops before any full-resolution real-codec work.
4. If the residual branch is promising, implement real arithmetic coding for the residual symbols and adaptive entropy coding for the semantic VQ indices.


## 2026-06-21 17:45 JST - GLC-latent residual: predictor/residual phase audit

Purpose: evaluate whether the new GLC-latent residual branch is actually sending unpredictable residual information, rather than only learning a better decoder-side deterministic correction.

Completed runs:

| run | init | quant step | rate weight | fixed Kodak center result | interpretation |
|---|---|---:|---:|---|---|
| `glc_latent_residual_predictor_warmup_6k` | Stage-A semantic + frozen GLC/VQGAN | n/a | 0 | no-res final: bpp `0.00977`, LPIPS `0.65444`, DISTS `0.42000`; best: LPIPS `0.67302`, DISTS `0.41993` | semantic-only predictor reaches the old scratch DISTS range at lower bpp, but LPIPS remains weak. |
| `glc_latent_residual_residual_lR1_lp2_d2_from_warm_3k` | predictor warmup final | 0.5 | 1.0 | no-res final: LPIPS/DISTS `0.58901/0.39862`; residual-on final: total bpp `0.009766`, LPIPS/DISTS `0.58593/0.39924` | large deterministic refinement gain, but hard-rounded residual symbols are effectively all zero. |
| `glc_latent_residual_residual_q025_lR07_lp2_d15_from_lR1_1500` | q0.5 residual final | 0.25 | 0.7 | no-res final: LPIPS/DISTS `0.57749/0.39775`; residual-on final: total bpp `0.009766`, LPIPS/DISTS `0.56636/0.39775` | q0.25 improves LPIPS while preserving DISTS, but transmitted residual bpp remains effectively zero. |

Key finding:

- The frozen GLC generator can be driven surprisingly far from the cheap 8x8 semantic code: Kodak-center DISTS improves from the Stage-A scratch basis `0.43546` to about `0.39775` at the same semantic bpp `0.00977`.
- However, the residual stream is not yet doing the intended job. With additive-noise quantization during training, residual symbols stay below the hard-rounding threshold, so evaluation rounds almost all residual coefficients to zero.
- The residual decoder still improves results because it receives `mu` and upsampled semantic features; with zero residual symbols it becomes a bit-free deterministic latent refinement network. This is useful as a generator-side predictor, but it does not yet prove the main GP-ResLC thesis of sending only unpredictable residuals.

Action taken:

- Added `quant_mode=ste` to `GLCLatentResidualBottleneck` and `scripts/train_glc_latent_residual.py` so training can use hard-rounded symbols with a straight-through gradient.
- Added `rounded_abs_mean` and `rounded_nonzero_frac` logs to training, validation, and deterministic evaluation. This makes residual payload collapse visible during training instead of discovering it only after fixed evaluation.

Next experiment:

- Resume from the q0.25 run with `--quant_mode ste`, modest residual rate pressure, and no W&B upload. Promote only if `rounded_nonzero_frac > 0` and perceptual gains survive fixed hard-round evaluation.


### STE residual diagnostic result

Run: `experiments/glc_latent_residual_residual_ste_q025_lR02_lp25_d15_from_q025_1k`

Settings: resumed from q0.25 final, `quant_mode=ste`, `quant_step=0.25`, `lambda_R=0.2`, LPIPS-heavy residual phase, no W&B upload.

Fixed Kodak center-crop results:

| checkpoint | mode | bpp | LPIPS | DISTS | rounded nonzero frac | read |
|---|---|---:|---:|---:|---:|---|
| `0011000` | no residual | 0.009766 | 0.57976 | 0.40251 | 0 | predictor degraded vs q0.25 parent |
| `0011000` | residual on | 0.009766 | 0.56710 | 0.40352 | 0 | LPIPS improves, DISTS worsens; no true payload |
| `final` | no residual | 0.009766 | 0.59107 | 0.40279 | 0 | predictor/refiner degrades further |
| `final` | residual on | 0.010189 | 0.57695 | 0.40246 | 0.000244 | first tiny nonzero payload; LPIPS improves, DISTS nearly tied |

Conclusion: STE is necessary but insufficient. The residual decoder can still produce a nonzero correction from `mu` and `z_up` even when `q_residual=0`, so the model escapes into bit-free deterministic refinement. The next implementation must force `delta_r=0` wherever the rounded residual payload is zero, while preserving a straight-through gradient so the residual encoder can learn to activate paid residual locations.


### Payload-gated and top-k residual diagnostics

Implementation update:

- Added `delta_gate_mode=payload_ste`: residual correction `delta_r` is zero wherever the rounded residual payload is zero. This blocks the previous bit-free deterministic-refinement escape path.
- Added `force_topk_frac`: encoder-side sparse residual warmup that forces only the highest-magnitude residual coefficients to nonzero rounded symbols. The full residual grid is still entropy-modeled; no separate free mask is assumed.

Runs:

| run | setting | fixed Kodak center result | decision |
|---|---|---|---|
| `glc_latent_residual_payloadste_q025_lR005_lp25_d15_from_q025_800` | payload gate, no forced top-k | residual-on and no-res are identical: bpp `0.009766`, LPIPS/DISTS `0.57767/0.40620`, `delta_active_frac=0` | gate is correct, but residual does not self-activate. |
| `glc_latent_residual_payloadste_topk001_q025_lR01_lp25_d15_from_q025_800`, checkpoint `0011200` | payload gate + top-k `0.001` | no-res: bpp `0.009766`, LPIPS/DISTS `0.56709/0.40019`; residual-on: bpp `0.011458`, LPIPS/DISTS `0.56678/0.39951`, `rounded_nonzero_frac=0.00098`, `delta_active_frac=0.02165` | first true paid-residual point with both LPIPS and DISTS improving, but gain is very small. |
| same, final | payload gate + top-k `0.001` | no-res: bpp `0.009766`, LPIPS/DISTS `0.57175/0.39984`; residual-on: bpp `0.011458`, LPIPS/DISTS `0.57130/0.39916`, `rounded_nonzero_frac=0.00098`, `delta_active_frac=0.02035` | same direction as 0011200; keep as a mechanism-positive branch, not a performance lead. |

Interpretation:

- The payload gate proves the previous residual gains were partly an architectural escape path: when zero-payload correction is forbidden, the model initially collapses to no-res output.
- Sparse top-k residual warmup gives the first clean signal that a tiny paid residual can improve both LPIPS and DISTS under the intended rule.
- The effect is too small for a paper claim. The next high-upside path is not more tiny top-k tuning alone; it should train a stronger predictor/residual pair with a scheduled sparse budget, e.g. start with top-k `0.002-0.005`, then anneal the budget/rate and add a perceptual improvement objective on active residual sites.


### DIV2K check and residual-budget upper-bound diagnostics

DIV2K center-crop check (`/dpl/div2k`, 100 images):

| checkpoint | mode | bpp | LPIPS | DISTS | note |
|---|---|---:|---:|---:|---|
| q0.25 parent final | no residual | 0.009766 | 0.56085 | 0.39429 | deterministic predictor only |
| q0.25 parent final | residual on | 0.009766 | 0.54868 | 0.39310 | improves, but this is payload-gate-free deterministic correction, not a valid paid-residual claim |
| top-k 0.001 final | no residual | 0.009766 | 0.54983 | 0.39302 | payload-gated predictor state |
| top-k 0.001 final | residual on | 0.011458 | 0.54976 | 0.39295 | paid residual effect generalizes, but is extremely small |

Residual-budget diagnostics on fixed Kodak center crops:

| run | top-k frac | bpp | LPIPS no-res -> res | DISTS no-res -> res | decision |
|---|---:|---:|---:|---:|---|
| `payloadste_topk001_q025_lR01_lp25_d15_from_q025_800` final | 0.001 | 0.00977 -> 0.01146 | 0.57175 -> 0.57130 | 0.39984 -> 0.39916 | best mechanism-positive paid-residual branch so far |
| `payloadste_topk005_q025_lR005_lp25_d15_from_q025_600` final | 0.005 | 0.00977 -> 0.01822 | 0.56502 -> 0.56415 | 0.39371 -> 0.39383 | more bits do not improve DISTS; not rate-efficient |
| `payloadste_topk005_activel1_q025_lR005_lp25_d15_from_q025_600` final | 0.005 | 0.00977 -> 0.01822 | 0.57829 -> 0.57831 | 0.40526 -> 0.40535 | active L1 lowers local pixel error but hurts perceptual metrics; reject |

Interpretation:

- `top-k=0.001` is the only branch that is both method-faithful and improves LPIPS/DISTS on Kodak and DIV2K, but the gain is much too small for a paper result.
- Increasing residual budget without better selection/objective wastes bits and can hurt DISTS.
- Active local L1 is the wrong proxy for perceptual residual usefulness, matching earlier scratch-stage findings that L1 selected losses do not align with DISTS/FID-style claims.

Next research decision:

1. Keep `top-k=0.001` as the clean mechanism proof.
2. Stop simple budget scaling and L1 active losses.
3. Move to either a perceptual active-site objective (spatial LPIPS/VGG/DISTS-proxy) or a residual-latent training phase that first teaches the residual decoder to reconstruct true latent residual under a sparse budget, then fine-tunes image perceptual quality.


### Active-latent sparse residual result and next pivot

Run: `glc_latent_residual_payloadste_topk001_activelat_q025_lR01_lp25_d15_from_q025_1000`

Settings: payload-gated top-k `0.001`, active latent no-regression loss (`lambda_active_latent_improve=10`, margin `0.001`).

Fixed Kodak center:

| checkpoint | mode | bpp | LPIPS | DISTS | decision |
|---|---|---:|---:|---:|---|
| `0010500` | no residual | 0.009766 | 0.57743 | 0.39724 | base |
| `0010500` | residual on | 0.011458 | 0.57685 | 0.39704 | improves both; best active-latent point |
| final | no residual | 0.009766 | 0.58267 | 0.39727 | base drifted |
| final | residual on | 0.011458 | 0.58269 | 0.39733 | reject; over-training hurts residual effect |

Fixed DIV2K center for `0010500`:

| mode | bpp | LPIPS | DISTS |
|---|---:|---:|---:|
| no residual | 0.009766 | 0.56066 | 0.39391 |
| residual on | 0.011458 | 0.56016 | 0.39379 |

Readout:

- Active-latent loss gives a cleaner early sparse residual point than plain top-k, and the sign generalizes to DIV2K.
- The gain is still too small, and longer training degrades the residual effect.
- The bigger limitation is likely upstream: the 8x8 Stage-A code was trained for the scratch decoder, not to predict GLC/VQGAN latent space. Keeping Stage-A frozen constrains the full-design branch.

Next pivot:

- Fine-tune the Stage-A encoder/codebook jointly with the GLC-latent predictor while keeping the semantic rate fixed. This directly learns a semantic code whose purpose is to let the frozen generator reconstruct predictable content, then residual coding can be revisited on top of a stronger semantic predictor.


### Stage-A joint fine-tune diagnostic

Run: `glc_latent_stagea_joint_pred_from_q025_2k`

Settings: resume q0.25 parent weights only, update Stage-A encoder/codebook plus GLC-latent predictor, predictor-only/no residual, fixed semantic bpp `0.00977`.

Fixed center-crop results for `glc_latent_residual_best.pt` (iteration 11500):

| dataset | bpp | LPIPS | DISTS | comparison |
|---|---:|---:|---:|---|
| Kodak | 0.009766 | 0.58188 | 0.39993 | worse than q0.25 parent no-res (`0.57749/0.39775`) |
| DIV2K | 0.009766 | 0.56311 | 0.39611 | worse than q0.25 parent no-res (`0.56085/0.39429`) |

Decision: reject this simple joint Stage-A fine-tune. Random validation looked promising, but deterministic Kodak/DIV2K show that moving the Stage-A encoder/codebook from a pretrained semantic basis hurts generalization. The next safer route is fixed Stage-A plus predictor-only fine-tuning, then revisit trainable semantic codes with a slower schedule, EMA/codebook regularization, or a separate Stage-II latent-code objective.


### Fixed Stage-A predictor-only continuation: new semantic lead

Run: `glc_latent_predictor_only_from_q025_d2_lp2_2k`

Settings: resume q0.25 parent weights only, keep Stage-A frozen, no residual, optimize the semantic-code-to-GLC-latent predictor with image perceptual and latent losses.

Best checkpoint: `glc_latent_residual_best.pt` at iteration `12000`.

Fixed center-crop results:

| dataset | checkpoint | bpp | LPIPS | DISTS | comparison to q0.25 parent no-res |
|---|---|---:|---:|---:|---|
| Kodak | predictor-only best | 0.009766 | 0.57334 | 0.39338 | improves from `0.57749/0.39775` |
| DIV2K | predictor-only best | 0.009766 | 0.56086 | 0.39113 | DISTS improves from `0.56085/0.39429`, LPIPS tied |

Decision:

- This is the strongest current full-design/scratch-direction result: no residual stream, no free correction, same fixed semantic bpp, better predictable-component reconstruction.
- The gain is larger and more stable than the paid sparse residual gains. This supports a research pivot: first maximize what the frozen generator can recover from the semantic code, then add paid residual only after the predictable component saturates.
- Next: continue this branch with a lower LR and DISTS-heavy but LPIPS-safe objective. Then re-attach payload-gated top-k residual from the improved predictor.


### Low-LR predictor continuation result

Run: `glc_latent_predictor_only_from_predbest_dists25_lp15_4k`

The 4k low-LR continuation from the semantic lead did not improve the fixed Kodak result:

| checkpoint | bpp | LPIPS | DISTS | decision |
|---|---:|---:|---:|---|
| previous predictor-only best | 0.009766 | 0.57334 | 0.39338 | keep as semantic lead |
| continuation `0015000` | 0.009766 | 0.57425 | 0.39715 | reject |
| continuation final | 0.009766 | 0.57373 | 0.39843 | reject |

Decision: do not continue this DISTS-heavy low-LR setting. The useful move is to keep `glc_latent_predictor_only_from_q025_d2_lp2_2k/glc_latent_residual_best.pt` as the semantic predictor lead and re-attach payload-gated sparse residual from that stronger predictable component.


## 2026-06-21 19:05 JST - GLC-latent hard-topk zero-center residual

Implemented two important corrections in the GLC-latent residual branch:

- `delta_gate_mode=zero_center`: residual correction is now `Decoder(q_residual, context) - Decoder(0, context)`. This preserves the core GP-ResLC constraint: if no residual payload is sent, the correction is exactly zero, while a sparse paid residual can still propagate through the convolutional decoder to neighboring latent positions.
- `--hard_topk`: after selecting top-k residual coefficients, all non-top-k rounded symbols are forced to zero. This fixes the previous `force_topk_frac` loophole where non-top-k symbols could become nonzero and silently increase bpp.

Also added:

- `--freeze_predictor`, so residual-only runs can test whether paid residual bits improve a fixed predictable component.
- `--reset_best_on_resume`, so continued runs can save a local best checkpoint.
- LPIPS/DISTS no-regression hinges against the no-residual base.
- residual/correction magnitude regularizers for later balancing.

### Key mechanism result

Run: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk001_perchinge_1500`

Settings:

- Resume: `experiments/glc_latent_predlead_topk001_activelat_lR01_1k/glc_latent_residual_final.pt`
- Predictor frozen, Stage-A frozen.
- `delta_gate_mode=zero_center`, `force_topk_frac=0.001`, `hard_topk=True`.
- Residual payload: roughly `0.002738` bpp, total `0.012504` bpp.

Fixed 256x256 center-crop proxy results. These are not official full-resolution real-codec evaluations.

| dataset/checkpoint | mode | bpp | LPIPS | DISTS | note |
|---|---|---:|---:|---:|---|
| Kodak | no residual | 0.009766 | 0.572118 | 0.391503 | fixed predictable component |
| Kodak best (`it=13500`) | residual on | 0.012504 | 0.570858 | 0.391440 | small but clean paid-residual improvement |
| Kodak final (`it=14500`) | residual on | 0.012504 | 0.589248 | 0.386138 | DISTS improves strongly, LPIPS worsens |
| DIV2K | no residual | 0.009766 | 0.557175 | 0.390806 | fixed predictable component |
| DIV2K best (`it=13500`) | residual on | 0.012504 | 0.556456 | 0.390581 | balanced improvement |
| DIV2K final (`it=14500`) | residual on | 0.012504 | 0.553252 | 0.384901 | strong improvement on both LPIPS and DISTS |
| CLIC2020 test center, 428 imgs | no residual | 0.009766 | 0.542022 | 0.376615 | professional 250 + mobile 178 |
| CLIC2020 test center, 428 imgs | residual on final | 0.012504 | 0.530758 | 0.370720 | strong improvement on both LPIPS and DISTS |

CLIC subset details:

| subset | mode | images | bpp | LPIPS | DISTS |
|---|---|---:|---:|---:|---:|
| professional test | no residual | 250 | 0.009766 | 0.533872 | 0.372375 |
| professional test | residual final | 250 | 0.012504 | 0.521344 | 0.367255 |
| mobile test | no residual | 178 | 0.009766 | 0.553470 | 0.382571 |
| mobile test | residual final | 178 | 0.012504 | 0.543980 | 0.375587 |

Interpretation:

- This is the cleanest scratch/full-design evidence so far for the original GP-ResLC axis: a fixed semantic/predictable stream is improved by sending only a strictly sparse residual payload.
- The final checkpoint generalizes well to CLIC and DIV2K, but Kodak LPIPS worsens despite better Kodak DISTS. Keep both `best` and `final` as curve/ablation points.
- The re-balance continuation from final (`experiments/glc_latent_zerocenter_hardtopk001_lpips_rebalance_from_final_1k`) did not fix Kodak LPIPS and is not promoted. It improves DIV2K but weakens Kodak enough that it should remain a rejected follow-up.

Next research steps:

1. Build a small rate/perception curve for `hard_topk` budgets, especially `0.0005`, `0.001`, `0.002`, and `0.005`, using zero-center residuals.
2. Add a non-saturating entropy proxy or scale lower-bound adjustment because `gaussian_bits` clamps probabilities and can under-penalize very large active symbols.
3. Add a lightweight LPIPS-safe objective or early stopping criterion based on a fixed validation set, since random Kodak-val batches are noisy and Kodak LPIPS can diverge from CLIC/DIV2K.
4. Once proxy behavior stabilizes, implement real entropy coding for the residual symbol stream and evaluate full-resolution CLIC/DIV2K/Kodak.


### Hard-topk budget 0.2% follow-up

Run: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk002_perchinge_1500`

Settings match the 0.1% run except `force_topk_frac=0.002`, `hard_topk=True`, and `lambda_R=0.1`.

Fixed center-crop proxy results:

| dataset/checkpoint | bpp | LPIPS | DISTS | decision |
|---|---:|---:|---:|---|
| DIV2K 0.2% `0014000` | 0.015241 | 0.553443 | 0.386538 | balanced but worse DISTS than 0.1% final |
| DIV2K 0.2% final | 0.015241 | 0.548306 | 0.389163 | better LPIPS, worse DISTS |
| CLIC2020 test center no-res | 0.009766 | 0.542022 | 0.376615 | reference |
| CLIC2020 test center 0.1% final | 0.012504 | 0.530758 | 0.370720 | current DISTS/L1 lead |
| CLIC2020 test center 0.2% final | 0.015241 | 0.527802 | 0.374115 | LPIPS-oriented point, not DISTS lead |

Interpretation: more sparse residual budget does not monotonically improve DISTS. The 0.2% payload learns stronger corrections and improves LPIPS on CLIC/DIV2K, but it harms DISTS relative to the 0.1% final point. This suggests the next improvement should not simply widen the top-k budget; it should improve residual objective/entropy calibration so stronger corrections stay perceptually aligned.


### Hard-topk budget curve update

Saved curve summary CSV: `experiments/glc_latent_hardtopk_curve_summary.csv`.

CLIC2020 test center-crop proxy, professional 250 + mobile 178:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.542022 | 0.376615 | 0.113911 | 0.026989 | predictable-only reference |
| hardtopk0005 | 0.011135 | 0.001369 | 0.530583 | 0.373010 | 0.108494 | 0.024987 | very efficient low payload point |
| hardtopk001 | 0.012504 | 0.002738 | 0.530758 | 0.370720 | 0.105636 | 0.023745 | current DISTS/L1/MSE lead |
| hardtopk002 | 0.015241 | 0.005475 | 0.527802 | 0.374115 | 0.107330 | 0.024625 | LPIPS-oriented, DISTS worsens |

DIV2K center-crop proxy:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE |
|---|---:|---:|---:|---:|---:|---:|
| no residual | 0.009766 | 0.000000 | 0.557175 | 0.390806 | 0.127053 | 0.032145 |
| hardtopk0005 | 0.011135 | 0.001369 | 0.550043 | 0.386773 | 0.122601 | 0.030408 |
| hardtopk001 | 0.012504 | 0.002738 | 0.553252 | 0.384901 | 0.119844 | 0.029408 |
| hardtopk002 | 0.015241 | 0.005475 | 0.548306 | 0.389163 | 0.121340 | 0.029877 |

Budget-curve interpretation:

- The curve is not monotonic across all perceptual metrics, which is useful evidence: more residual bits are not automatically better if the residual objective/entropy proxy allows overly strong local edits.
- `hardtopk0005` is the best efficiency point: very small residual payload produces large LPIPS/DISTS gains over no-residual on CLIC/DIV2K.
- `hardtopk001` is the best DISTS/L1/MSE point on CLIC and DIV2K.
- `hardtopk002` is LPIPS-oriented but not a good main point because DISTS degrades relative to `hardtopk001`.
- Next method work should improve entropy calibration and residual regularization, not simply increase top-k budget.


## 2026-06-21 20:05 JST - Stable bounded residual curve

Motivation: the previous hard-topk curve used the original clamped Gaussian bit proxy. That proxy is useful for fast screening, but it can under-penalize large active residual symbols. I added a stable entropy mode with a quadratic tail fallback and then constrained the transmitted residual symbols with `max_symbol_abs`. This makes the scratch/full-design branch closer to a real entropy-coded sparse residual stream.

Implementation changes:

- `gp_reslc/scratch/glc_latent_residual.py`: added `gaussian_bits_stable`, `entropy_mode`, and `max_symbol_abs`.
- `scripts/train_glc_latent_residual.py`: added CLI/config plumbing for stable entropy and bounded symbols.
- `scripts/evaluate_glc_latent_residual.py`: evaluation now restores `entropy_mode` and `max_symbol_abs` from checkpoint config.

Important negative result:

- Unbounded stable entropy with `topk=0.001` is unstable. Although nonzero positions remain fixed, active symbol magnitude grows and residual bpp can jump to multiple bpp. This confirms that position sparsity alone is insufficient; value bounding or a stronger entropy/real-codec constraint is required.

Stable bounded curve, fixed 256x256 center-crop proxy. These are not yet official full-resolution real-codec results.

CLIC2020 test center, 428 images:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.542011 | 0.376628 | 0.113910 | 0.026989 | reference |
| stable ternary topk0005 final | 0.010612 | 0.000847 | 0.537297 | 0.374999 | 0.111184 | 0.025975 | adopt low-rate point |
| stable ternary topk002 best | 0.013150 | 0.003384 | 0.529933 | 0.371677 | 0.107104 | 0.024467 | adopt main stable point |
| stable small-int2 topk001 best | 0.021877 | 0.012111 | 0.533375 | 0.373803 | 0.109661 | 0.025363 | reject: dominated |

DIV2K center:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.557210 | 0.390785 | 0.127048 | 0.032145 | reference |
| stable ternary topk0005 final | 0.010612 | 0.000847 | 0.553770 | 0.388299 | 0.124123 | 0.030960 | adopt low-rate point |
| stable ternary topk002 best | 0.013150 | 0.003384 | 0.550992 | 0.385331 | 0.121149 | 0.029947 | adopt main stable point |
| stable small-int2 topk001 best | 0.021877 | 0.012111 | 0.552500 | 0.387740 | 0.123059 | 0.030568 | reject: dominated by topk002 |

Kodak center:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.572011 | 0.391566 | 0.114565 | 0.024937 | reference |
| stable ternary topk0005 final | 0.010612 | 0.000847 | 0.569598 | 0.389898 | 0.113148 | 0.024466 | adopt low-rate point |
| stable ternary topk002 best | 0.013150 | 0.003384 | 0.575541 | 0.390750 | 0.110967 | 0.023804 | mixed: DISTS/L1 improve, LPIPS worsens |
| stable small-int2 topk001 best | 0.021877 | 0.012111 | 0.570009 | 0.390974 | 0.112631 | 0.024205 | reject: not rate efficient |

Interpretation:

- `stable_ternary_topk0005` is now the cleanest low-rate evidence for the GP-ResLC axis: only 0.00085 residual bpp improves LPIPS/DISTS/L1/MSE on CLIC, DIV2K, and Kodak.
- `stable_ternary_topk002` is the best stable perceptual point on CLIC/DIV2K, but Kodak LPIPS worsens. Keep it as the main stable curve point, not as the only visual candidate.
- `stable_smallint2_topk001` improves over no-residual but is dominated by `stable_ternary_topk002`; increasing symbol amplitude is worse than sending a few more ternary positions. This supports a simple design preference: sparse ternary residuals over high-amplitude sparse residuals.
- Next method direction: real-codec arithmetic coding for bounded residual symbols, fixed-validation checkpoint selection, and possibly topk0015/learned topk budget to bridge low-rate and main stable points.

Saved CSV: `experiments/glc_latent_stable_bounded_curve_summary.csv`.


## 2026-06-21 20:25 JST - TorchAC real residual codec bridge

Implemented `scripts/evaluate_glc_latent_residual_realcodec.py` for the GLC-latent scratch branch. This script keeps the current fixed semantic-code bpp accounting, but actually entropy-codes the bounded residual symbol tensor with `torchac` using the same Gaussian CDF helper as the GLC real codec. It then decodes symbols, reconstructs from decoded symbols, and reports exact byte-derived residual bpp plus decode consistency.

Important caveat: this is still a center-crop development codec, not the final full-resolution paper codec. The per-image stream header is intentionally counted separately because it is disproportionately large at 256x256. For full-resolution CLIC/DIV2K, that header overhead should be smaller.

CLIC2020 test center, 428 images:

| point | proxy total bpp | AC-only residual bpp | stream residual bpp | stream total bpp | LPIPS | DISTS | decode symbols |
|---|---:|---:|---:|---:|---:|---:|---|
| stable ternary topk0005 final | 0.010612 | 0.000854 | 0.001831 | 0.011597 | 0.537302 | 0.375008 | exact, max abs 0 |
| stable ternary topk002 best | 0.013150 | 0.003052 | 0.004028 | 0.013794 | 0.529926 | 0.371678 | exact, max abs 0 |

DIV2K center:

| point | proxy total bpp | AC-only residual bpp | stream residual bpp | stream total bpp | LPIPS | DISTS | decode symbols |
|---|---:|---:|---:|---:|---:|---:|---|
| stable ternary topk0005 final | 0.010612 | 0.000854 | 0.001831 | 0.011597 | 0.553742 | 0.388320 | exact, max abs 0 |
| stable ternary topk002 best | 0.013150 | 0.003052 | 0.004028 | 0.013794 | 0.551003 | 0.385368 | exact, max abs 0 |

Kodak center, topk0005 final:

| proxy total bpp | AC-only residual bpp | stream residual bpp | stream total bpp | LPIPS | DISTS | decode symbols |
|---:|---:|---:|---:|---:|---:|---|
| 0.010612 | 0.000854 | 0.001831 | 0.011597 | 0.569766 | 0.389893 | exact, max abs 0 |

Consistency note:

- `decode_symbol_max_abs` is 0 for all checked datasets and points.
- With `bs=1`, `forward_decode_max_abs` is also 0. With `bs=2`, small pixel differences around 0.002-0.003 appear due to batch-dependent floating point execution through the generator, but the decoded residual symbols are exact and metrics remain aligned with the proxy evaluator.

Interpretation:

- The stable proxy is well calibrated for the arithmetic-coded residual payload: topk0005 AC residual bpp is `0.000854` vs proxy `0.000847`; topk002 AC residual bpp is `0.003052` vs proxy `0.003384`.
- Header overhead is the main remaining mismatch in 256x256 center-crop proxy evaluation. This reinforces the need to move the scratch branch to full-resolution CLIC/DIV2K before paper claims.
- This is the first scratch/full-design result that is not merely estimated likelihood: the residual symbols themselves are serialized and decoded exactly. The semantic stream is still fixed-width counted, so the next real-codec step is to serialize semantic VQ indices and combine both streams into one payload.


### Semantic stream byte packing check

The real residual evaluator now also serializes Stage-A semantic VQ indices with fixed-width packing using `ceil(log2(codebook_size))` bits per index. For the current Stage-A checkpoint, `codebook_size=1024`, so the bit width is 10. The 8x8 semantic grid on 256x256 crops serializes to exactly 80 bytes, matching `semantic_real_bpp = 0.009765625`.

Sanity check on Kodak limit-4 with `bs=1`:

| field | value |
|---|---:|
| semantic_bpp formula | 0.009765625 |
| semantic_real_bpp bytes | 0.009765625 |
| semantic_bit_width | 10 |
| residual_ac_bpp | 0.000854492 |
| residual_stream_bpp | 0.001831055 |
| total_real_bpp, semantic + stream residual | 0.011596680 |
| decode_symbol_max_abs | 0 |
| forward_decode_max_abs | 0 |

This makes the scratch center-crop real-codec bridge byte-consistent for both transmitted streams. Remaining limitations: full-resolution tiling/padding, compact combined payload header, and official FID/KID patch extraction are still pending.


## 2026-06-21 21:15 JST - Full-resolution scratch real-codec and residual delta scaling

Implemented `scripts/evaluate_glc_latent_residual_fullres_realcodec.py`, a full-resolution development codec for the GLC-latent scratch branch. It follows the GLC padding/bpp protocol more closely than the previous center-crop evaluator: original-resolution images are replicate-padded to multiples of 64, transmitted bits are divided by original pixels, semantic VQ indices are fixed-width packed and decoded back through the Stage-A codebook, residual symbols are arithmetic-coded with `torchac`, and decoding uses only the decoded semantic/residual streams.

Key implementation checks:

- CLIC2020 test all uses canonical `data/clic2020_test_combined` with 428 symlinked images, i.e. professional 250 + mobile 178.
- Semantic decode max error is around `1e-8`; residual symbol decode max error is `0`.
- Current Stage-A semantic stream is 10 bits/index; bpp is about `0.01006` on high-resolution CLIC/DIV2K because padding/byte rounding is counted over original pixels.
- Residual AC stream is only about `0.00078` bpp for `topk0005`.

Full-resolution results for the original `stable ternary topk0005` checkpoint at `delta_scale=1.0` show the core mechanism but also a safety problem:

| dataset | bpp | residual AC bpp | LPIPS base -> residual | DISTS base -> residual | interpretation |
|---|---:|---:|---:|---:|---|
| CLIC2020 428 | 0.011000 | 0.000781 | 0.540915 -> 0.620295 | 0.341476 -> 0.323087 | strong DISTS gain, LPIPS/L1 unsafe |
| DIV2K 100 | 0.011025 | 0.000784 | 0.565526 -> 0.674794 | 0.353956 -> 0.345589 | DISTS gain, large LPIPS/L1 degradation |
| Kodak 24 | 0.011617 | 0.000753 | 0.576080 -> 0.577656 | 0.376250 -> 0.362915 | strong Kodak DISTS gain, slight LPIPS loss |

A short 512-crop safety fine-tune from this checkpoint (`experiments/glc_latent_fullres_safe512_topk0005_l1lp_from_final_1500`, W&B `2anv6fcl`) showed the tradeoff clearly: the final checkpoint improves Kodak LPIPS/L1/MSE but loses DISTS (`0.376250 -> 0.378034`), while the best checkpoint improves DISTS (`0.357377`) but damages LPIPS (`0.632410`). Simple no-regression fine-tuning is therefore too blunt.

The better simple fix is decoder-side residual delta scaling. This is a fixed model setting, not side information. Applying `latent_hat = mu + gamma * residual_delta` with the same transmitted semantic/residual bitstream gives:

| dataset / gamma | bpp | residual AC bpp | LPIPS base -> residual | DISTS base -> residual | LPIPS wins | DISTS wins |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 gamma=0.5 | 0.011000 | 0.000781 | 0.540915 -> 0.541775 | 0.341476 -> 0.326563 | 296/428 | 397/428 |
| DIV2K gamma=0.5 | 0.011025 | 0.000784 | 0.565526 -> 0.580409 | 0.353956 -> 0.338459 | 59/100 | 88/100 |
| Kodak gamma=0.5 | 0.011617 | 0.000753 | 0.576080 -> 0.573606 | 0.376250 -> 0.375262 | 16/24 | 17/24 |
| Kodak gamma=0.75 | 0.011617 | 0.000753 | 0.576080 -> 0.573079 | 0.376250 -> 0.370487 | 14/24 | 15/24 |

Decision:

- Promote `topk0005 + gamma=0.5` as the current full-resolution scratch/full-design candidate. It preserves the thesis: a ~0.00078 bpp residual stream improves DISTS at essentially the same LPIPS on CLIC, while using actual serialized semantic and residual streams.
- Keep `gamma=0.75` as a Kodak/DISTS-oriented auxiliary point, but use `gamma=0.5` as the safer global setting across CLIC/DIV2K/Kodak.
- Next method step should make `gamma` learnable or context-adaptive without side information, e.g. decoder-side confidence from semantic/predictor features, rather than relying on a manually fixed scalar.

Saved summary CSV: `experiments/fullres_realcodec_gamma_summary.csv`.


### Gamma=0.5 training adaptation follow-up

After the fixed decoder-side scale result, I added `delta_scale` to `GLCLatentResidualBottleneck.forward()` and `scripts/train_glc_latent_residual.py`, then trained a short 512-crop adaptation run with the same transmitted ternary top-k residual but `delta_scale=0.5` during training.

Run:

- `experiments/glc_latent_gamma050_adapt512_topk0005_balanced_1000`
- W&B: `g7ockfh4`
- Init: original `stable_ternary_topk0005` final checkpoint
- Key settings: 512 crops, frozen Stage-A and predictor, topk0005, stable entropy, ternary residual, `delta_scale=0.5`, moderate LPIPS/DISTS no-regression.

Kodak full-resolution real-codec results at `delta_scale=0.5`:

| checkpoint | bpp | LPIPS base -> residual | DISTS base -> residual | L1 base -> residual | decision |
|---|---:|---:|---:|---:|---|
| fixed original gamma=0.5 | 0.011617 | 0.576080 -> 0.573606 | 0.376250 -> 0.375262 | 0.093480 -> 0.092335 | current safe global candidate |
| adapt best | 0.011617 | 0.576080 -> 0.568699 | 0.376250 -> 0.377626 | 0.093480 -> 0.090852 | reject: DISTS lost |
| adapt final | 0.011617 | 0.576080 -> 0.567286 | 0.376250 -> 0.377817 | 0.093480 -> 0.091002 | reject: DISTS lost |

Decision: do not promote the trained gamma-adaptation checkpoints. The simple fixed-scale original checkpoint is better balanced. The adaptation run confirms that standard no-regression training tends to suppress the DISTS-useful residual too much. Next design should use a decoder-side confidence/gamma predictor or region-specific safety mechanism, not just global no-regression fine-tuning.


## 2026-06-21 21:55 JST - Full-resolution gamma sweep and adaptive decoder gate

- Completed real-codec full-resolution gamma=0.6 sweep for the scratch topk0005 ternary residual checkpoint.
- Same transmitted bitstream as gamma=0.5/1.0: semantic fixed-width stream plus torchac residual stream. `delta_scale` only changes decoder-side residual application strength and costs no side bits.
- CLIC2020 all 428: bpp=0.011000, residual_ac_bpp=0.000781, LPIPS 0.540915 -> 0.549742, DISTS 0.341476 -> 0.320686, L1 0.087467 -> 0.090442. Win counts: LPIPS 263/428, DISTS 394/428, L1 243/428.
- DIV2K 100: bpp=0.011025, LPIPS 0.565526 -> 0.594456, DISTS 0.353956 -> 0.335538. Win counts: LPIPS 48/100, DISTS 84/100.
- Kodak 24: bpp=0.011617, LPIPS 0.576080 -> 0.573031, DISTS 0.376250 -> 0.374235, L1 0.093480 -> 0.092191. Win counts: LPIPS 16/24, DISTS 18/24, L1 21/24.
- Interpretation: gamma=0.6 is stronger than gamma=0.5 for DISTS on CLIC/DIV2K/Kodak, but it degrades LPIPS/L1 on CLIC/DIV2K. Fixed gamma is a useful diagnostic, not sufficient as final top-conference method.
- Implemented decoder-side adaptive residual gate (`delta_scale_net`) in `gp_reslc/scratch/glc_latent_residual.py`, plus train/eval CLI support. The gate is recomputed from transmitted residual symbols, semantic features, and predicted latent mean, so it adds no side bits.
- Started gate-only fine-tune from the topk0005 checkpoint: `experiments/glc_latent_adaptive_gate_topk0005_from_final_512_balanced_14500to17500`, W&B run `pup0zpf3`. Failed setup runs: `ghs7ls69` (zero-iter due resume counter), `fh29dt65` (missing return key bug).


## 2026-06-21 22:08 JST - Adaptive gate-only balanced result

- Completed gate-only fine-tune from scratch topk0005 checkpoint: `experiments/glc_latent_adaptive_gate_topk0005_from_final_512_balanced_14500to17500`, W&B `pup0zpf3`.
- Gate is decoder-side only and recomputed from transmitted residual symbols + semantic features + predicted latent mean. It adds no side bits.
- Kodak full-resolution real-codec, final checkpoint: bpp=0.011617, residual_ac_bpp=0.000753, LPIPS 0.576080 -> 0.574754, DISTS 0.376250 -> 0.367728, L1 0.093480 -> 0.092705, adaptive scale mean=0.560.
- Kodak full-resolution real-codec, best checkpoint: bpp=0.011617, LPIPS 0.576080 -> 0.576465, DISTS 0.376250 -> 0.365664, L1 0.093480 -> 0.093375, adaptive scale mean=0.449.
- Interpretation: adaptive gate is safer than fixed gamma=1.0 for LPIPS/L1, and improves DISTS over gamma=0.5/0.6 on Kodak, but does not beat the original gamma=1.0 DISTS point. As a top-conference method, a scalar/gate-only residual usage controller is probably too incremental.
- Next decision: use this as evidence that decoder-side residual confidence is useful, then move to a stronger design: either DISTS-oriented gate with less LPIPS regularization, or progressive residual stages that separate DISTS-useful structural correction from LPIPS-damaging texture/detail correction.

## 2026-06-21 22:44 JST - GLC-latent progressive stage-specific top-k residual

Motivation: the previous decoder-side fixed/adaptive residual scale improved DISTS but remained a post-hoc strength control. It did not fully realize the GP-ResLC thesis that only generator-unpredictable residual information should be transmitted. The new experiment makes the residual stream progressive and allocates sparse symbols separately to stage1/stage2 channels, so the coarse and fine residual decoders cannot silently compete for the same global top-k budget.

Implementation:

- `gp_reslc/scratch/glc_latent_residual.py`: added `progressive_stage_topk`, `stage1_topk_frac`, and `stage2_topk_frac`. In progressive mode, the top-k mask is now selected independently for stage1 and stage2 residual channel groups.
- `scripts/train_glc_latent_residual.py`: added CLI/W&B logging for stage-specific top-k and stage-specific nonzero fractions.
- `scripts/evaluate_glc_latent_residual_fullres_realcodec.py`: added progressive checkpoint decode support. Real-codec evaluation now reconstructs progressive residuals through `residual_decoder_stage1` and `residual_decoder_stage2` instead of incorrectly falling back to the single residual decoder.
- Smoke checks: `py_compile` passed; tensor forward passed for both progressive residual and `use_residual=False` base path.

Active run:

- W&B: `1xwv0f6q`
- Name: `progressive_stagealloc_topk0008_512_14500to16500`
- Output: `experiments/glc_latent_progressive_stagealloc_topk0008_from_final_512_14500to16500/`
- Init: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk0005_stable_ternary_1500/glc_latent_residual_final.pt`
- Dataset: `/dpl/openimages/train`, validation `/dpl/kodak`, crop `512`, batch `1`.
- Key settings: frozen predictor, `quant_mode=ste`, `delta_gate_mode=zero_center`, `hard_topk`, stable entropy, `stage1_channels=12`, `stage1_topk_frac=0.0008`, `stage2_topk_frac=0.0008`, `lambda_lpips=1.5`, `lambda_dists=3.0`, base LPIPS/DISTS no-regression retained.

Early read:

- At iteration 14500, stage decoders are newly initialized, so final reconstruction equals the generator/predictor base, as expected.
- By iteration 14550-14650, final LPIPS begins to improve relative to stage1/base on some batches, indicating stage2 is receiving useful gradients under separated top-k selection.
- The key decision point is not crop validation alone. After training, evaluate with full-resolution real codec on Kodak, DIV2K, and CLIC test/all using the updated progressive decode path, then compare against the official GLC curve and the previous fixed-gamma/adaptive-gamma residual points.

## 2026-06-21 23:13 JST - Progressive decoder init and top-k score audit

This block tested whether the GLC-latent residual stream can be made more faithful to the GP-ResLC principle by either splitting the residual into two progressive stages or changing which sparse residual symbols are transmitted.

Implementation updates:

- Added stage-specific top-k allocation for progressive residual mode.
- Added `--init_progressive_decoders_from_single {stage1,both}` so old single residual decoder weights can initialize the progressive decoders instead of discarding the useful learned residual synthesis path.
- Fixed train/validation memory use: stage1 reconstruction is now generated only when stage1 losses/metrics are requested. This avoids 512-crop OOM during runs that do not use stage1 supervision.
- Added `--topk_score_mode {abs,latent_error,latent_error_sq}`. `latent_error` scores source-side symbols by `|symbol| * mean_c|target_latent - mu|`; `latent_error_sq` uses the squared spatial error. This tests whether choosing spatially less predictable positions improves the transmitted residual at the same top-k budget.

W&B / runs:

- `1xwv0f6q`: zero-initialized stage1/stage2 progressive decoder. Rejected; DISTS worsened and stage2 carried almost no delta.
- `o5bno8es`: stage1 initialized from old single decoder, stage2 zero. Better crop behavior but full-res Kodak DISTS worsened.
- `huipcyt9`: both stage decoders initialized from old single decoder, balanced fine-tune. LPIPS improved, DISTS worsened.
- `frh8u1tx`: both-init DISTS-heavy fine-tune. DISTS returned close to no-train but did not beat it.

Kodak full-resolution real-codec results:

| variant | payload bpp | residual AC bpp | LPIPS | DISTS | read |
|---|---:|---:|---:|---:|---|
| Stage-A base only | 0.009766 | 0 | 0.576080 | 0.376250 | generator/predictor base |
| previous single abs top-k, gamma=1 reference | ~0.011617 | ~0.00075 | ~0.5777 | ~0.3629 | still DISTS lead among these local tests |
| progressive both-init no-train | 0.011617 | 0.000753 | 0.578412 | 0.363322 | stage split itself is not destructive |
| progressive stage1-init best | 0.011658 | 0.000793 | 0.568868 | 0.377770 | LPIPS improves, DISTS fails |
| progressive both-init fine-tune best | 0.011617 | 0.000753 | 0.574547 | 0.370081 | LPIPS improves, DISTS regresses |
| progressive both-init fine-tune final | 0.011617 | 0.000753 | 0.568128 | 0.377328 | strongest LPIPS, poor DISTS |
| progressive both-init DISTS-heavy best | 0.011617 | 0.000753 | 0.578450 | 0.363348 | basically no better than no-train |
| single top-k score = latent_error | 0.011617 | 0.000753 | 0.574098 | 0.375504 | simple unpredictability score hurts DISTS |
| single top-k score = latent_error_sq | 0.011617 | 0.000753 | 0.574603 | 0.375561 | same failure mode |

Interpretation:

1. Progressive splitting is mechanically valid only when both stage decoders inherit the old residual decoder. Zero-init stage2 does not learn useful corrections in short fine-tunes.
2. Fine-tuning the progressive decoders tends to trade DISTS for LPIPS. This may be useful for a separate LPIPS point, but it is not a paper-facing DISTS/FID/KID improvement.
3. Hand-designed latent-error top-k is not a good proxy for perceptual payoff. It selects positions where the generator latent prediction is wrong, but those are not necessarily the positions that improve DISTS.
4. The next serious version should use a learned encoder-side selector trained against measured perceptual payoff, e.g. compare candidate residual masks by DISTS/LPIPS improvement, distill the payoff into a selector, then entropy-code the selected sparse residual symbols. That is closer to “send only unpredictable residuals” than raw latent-error weighting.

Decision:

Do not promote the progressive/top-k-score variants as the current lead. Keep the previous single residual real-codec point as the practical lead, and treat these experiments as evidence that the next improvement must be a learned residual-value selector rather than a manual top-k heuristic.

## 2026-06-21 23:35 JST - Encoder-side latent-gradient residual selection

A more direct test of the GP-ResLC principle was implemented after manual top-k heuristics failed. Instead of selecting sparse residual symbols only by magnitude, the encoder estimates each candidate residual symbol's first-order latent reconstruction payoff:

`score_i = | grad_{q_i} SmoothL1(mu + D_r(q), y) * candidate_q_i |`

where `mu` is the generator-predictable latent from the semantic stream, `D_r` is the learned residual decoder, and `y` is the frozen GLC/VQGAN target latent. The decoder receives the same sparse residual grid as before; no side map is transmitted. The cost is encoder-side backprop during analysis/encoding.

Implementation:

- Added `topk_score_mode=latent_grad` in `gp_reslc/scratch/glc_latent_residual.py`.
- Added CLI/evaluator support through `--topk_score_mode latent_grad`.
- The current implementation is for the single residual decoder, not progressive residual mode.
- `latent_error` and `latent_error_sq` hand-designed scores were also tested and rejected.

Kodak full-resolution real-codec:

| variant | payload bpp | LPIPS | DISTS | DISTS wins | LPIPS wins | read |
|---|---:|---:|---:|---:|---:|---|
| base semantic predictor | 0.009766 | 0.576080 | 0.376250 | - | - | no residual |
| prior abs top-k reference | ~0.011617 | ~0.5777 | ~0.3629 | - | - | previous DISTS lead |
| latent_grad, delta=1.0 | 0.011617 | 0.595405 | 0.354949 | 17/24 | 8/24 | strong DISTS, LPIPS worse |
| latent_grad, delta=0.8 | 0.011617 | 0.579667 | 0.362212 | 18/24 | 14/24 | balanced, slight DISTS lead |
| latent_grad, delta=0.6 | 0.011617 | 0.574957 | 0.371653 | 18/24 | 17/24 | LPIPS/L1 safer, weaker DISTS |
| latent_error | 0.011617 | 0.574098 | 0.375504 | - | - | rejects simple latent-error score |
| latent_error_sq | 0.011617 | 0.574603 | 0.375561 | - | - | also rejected |

DIV2K full-resolution real-codec, `latent_grad`, `delta=0.6`:

| payload bpp | base LPIPS | LPIPS | base DISTS | DISTS | DISTS wins | LPIPS wins |
|---:|---:|---:|---:|---:|---:|---:|
| 0.011024 | 0.565526 | 0.603688 | 0.353956 | 0.334945 | 83/100 | 45/100 |

CLIC2020 test all 428 full-resolution real-codec, `latent_grad`, `delta=0.6`:

| payload bpp | base LPIPS | LPIPS | base DISTS | DISTS | DISTS wins | LPIPS wins |
|---:|---:|---:|---:|---:|---:|---:|
| 0.011000 | 0.540915 | 0.559899 | 0.341476 | 0.319396 | 373/428 | 234/428 |

Interpretation:

- This is the strongest mechanism result so far for the original thesis. The residual stream is selected by decoder payoff rather than raw magnitude, and it improves DISTS on Kodak, DIV2K, and CLIC at the same byte-backed payload bpp.
- It is not yet an official-curve result. Absolute DISTS/LPIPS values are from the scratch/Stage-A semantic branch and are far worse than official GLC q0-q3. The contribution is mechanism-level: at fixed ultra-low payload, better residual selection improves DISTS substantially.
- LPIPS and L1 often worsen, especially on DIV2K. The method is currently DISTS-oriented. The next version needs a multi-objective payoff score or adaptive residual strength to trade DISTS/LPIPS per image.
- Encoder analysis time increases because each image requires one latent-gradient backprop. Decoder time and bitstream format are unchanged.

Next steps:

1. Add `latent_grad_mix` score: combine latent payoff with LPIPS/L1 safety proxy or a learned per-image `delta_scale` to reduce LPIPS damage.
2. Train a selector/distillation network to approximate latent-gradient top-k without test-time backprop, making the method practical.
3. Evaluate FID/KID patches for CLIC/DIV2K using reconstructions from `latent_grad delta=0.6`; DISTS improvements suggest FID/KID might improve, but this must be measured.
4. Keep the official-curve comparison separate. This branch currently validates the GP-ResLC design principle but does not yet beat GLC's official perceptual curve.



## 2026-06-21 23:59 JST - Latent-gradient residual FID/KID protocol check

Added `--disable_residual` to `scripts/evaluate_glc_latent_residual_fullres_realcodec.py` so the semantic/base generator reconstruction can be saved through the same full-resolution loader, padding, and VQGAN decode path as the residual reconstructions. This lets us measure whether encoder-side `latent_grad` residual selection improves distribution metrics over the Stage-A/base predictor, not only DISTS.

Saved-reconstruction runs:

| dataset | variant | payload bpp | LPIPS | DISTS | FID | KID | patches | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CLIC2020 test all 428 | base semantic only | 0.010058 | 0.540915 | 0.341476 | 118.364 | 0.06663 | 28,650 | no residual stream |
| CLIC2020 test all 428 | latent_grad, delta=0.6 | 0.011000 | 0.559899 | 0.319396 | 105.336 | 0.04996 | 28,650 | residual AC bpp 0.000781 |
| DIV2K validation | base semantic only | 0.010087 | 0.565526 | 0.353956 | 181.631 | 0.12152 | 6,573 | no residual stream |
| DIV2K validation | latent_grad, delta=0.6 | 0.011024 | 0.603688 | 0.334945 | 156.661 | 0.08339 | 6,573 | residual AC bpp 0.000784 |

Per-image deltas, residual minus base:

| dataset | DISTS mean/median | DISTS wins | LPIPS mean/median | LPIPS wins | L1 mean/median | L1 wins |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test all | -0.02208 / -0.01450 | 373/428 | +0.01898 / -0.00049 | 234/428 | +0.00382 / -0.00024 | 237/428 |
| DIV2K validation | -0.01901 / -0.01441 | 83/100 | +0.03816 / +0.00181 | 45/100 | +0.00910 / +0.00285 | 37/100 |

Interpretation:

- The official-style patch protocol is correct for both datasets: CLIC gives 28,650 shifted 256-patches and DIV2K gives 6,573 shifted 256-patches.
- `latent_grad` is not merely improving DISTS; it also improves FID/KID versus the semantic/base generator on both CLIC and DIV2K.
- Absolute FID/KID/DISTS remain far worse than official GLC because this is still the weak scratch/Stage-A semantic branch. Do not compare these absolute values to the pretrained GLC official curve as a claim of SOTA.
- The important research signal is mechanistic: at a fixed sparse residual budget and real serialized payload, selecting residual symbols by decoder-side latent payoff sends more distribution-useful information than sending no residual or selecting by raw latent error.
- The limitation is now sharp: the current payoff is DISTS/FID/KID oriented and can damage LPIPS/L1, especially on DIV2K. The next serious model should learn a multi-objective residual-value selector: latent/DISTS payoff with LPIPS/L1 no-regression or a decoder-safe residual strength predictor.

Artifacts:

- `experiments/realcodec_single_topk_latent_grad_notrain_clic2020_all_delta06_save/`
- `experiments/realcodec_single_topk_latent_grad_notrain_clic2020_all_base_save/`
- `experiments/realcodec_single_topk_latent_grad_notrain_div2k_delta06_save/`
- `experiments/realcodec_single_topk_latent_grad_notrain_div2k_base_save/`


## 2026-06-22 00:57 JST - Sign-aware and adaptive residual-strength pilots

Follow-ups after the CLIC/DIV2K FID/KID check tested whether the LPIPS/L1 damage of `latent_grad` could be reduced without losing the DISTS/FID mechanism.

### Sign-aware latent-gradient selector

Implemented `topk_score_mode=latent_grad_improve`, using only first-order latent-loss-improving candidates:

`score_i = relu(- grad_i * candidate_i)`

This produced the same Kodak full-resolution result as the previous absolute `latent_grad` selector: payload bpp `0.011617`, LPIPS `0.595405`, DISTS `0.354949`. Interpretation: the learned residual encoder already proposes mostly latent-loss-improving symbol signs, so the LPIPS damage is not caused primarily by selecting first-order harmful latent directions.

### Fixed residual-delta shaping

Tested no-side-bit decoder shaping of the latent residual delta.

| variant | dataset | LPIPS | DISTS | read |
|---|---|---:|---:|---|
| latent_grad delta=0.6 | Kodak | 0.574957 | 0.371653 | safest previous fixed scale |
| split k=3, low=0.8, high=0.3 | Kodak | 0.577646 | 0.364341 | good Kodak trade-off |
| split k=3, low=0.8, high=0.3 | DIV2K | 0.644967 | 0.337498 | worse than delta=0.6 on both LPIPS/DISTS; reject |
| lowpass k=3 | Kodak | 0.586429 | 0.357182 | strong Kodak DISTS, LPIPS worse |
| lowpass k=3 | DIV2K | interrupted after 25/100 | clearly bad early: LPIPS `0.5411 -> 0.8433`, DISTS `0.3503 -> 0.3813` on image 0825 | reject |

Conclusion: fixed low/high-frequency shaping is dataset-sensitive. It can improve Kodak but does not generalize to DIV2K. Do not promote as a method; use it only as evidence that residual frequency content matters and should be controlled adaptively.

### Adaptive delta-scale head

Ran a decoder-side adaptive residual-strength pilot that freezes all modules except `delta_scale_net`:

- W&B: `iw6gcvot`
- Run: `experiments/glc_latent_delta_gateonly_latentgrad_lpips_safe_14500to15300/`
- Training: OpenImages train, Kodak crop validation, 800 steps from `it=14500`.
- Loss: LPIPS/DISTS/L1 with base no-regression penalties and small mean-scale penalty.

Full-resolution Kodak real-codec results:

| checkpoint | adaptive scale mean | LPIPS | DISTS | decision |
|---|---:|---:|---:|---|
| final | 0.458 | 0.602286 | 0.360207 | reject; LPIPS too poor |
| best / 15200 | 0.620 | 0.625681 | 0.363172 | reject; worse LPIPS |

Interpretation: crop validation was misleading for the adaptive gamma head. The head learned unstable full-resolution behavior and did not solve the LPIPS/L1 regression. For future adaptive residual strength, evaluate Kodak full-res early and train with full-res or larger crops; otherwise the learned gamma overfits crop statistics.

Current practical conclusion remains:

- `latent_grad delta=0.6` is the best cross-dataset mechanism point because it improves CLIC/DIV2K FID/KID and DISTS with moderate, known LPIPS cost.
- The next serious direction is not fixed filtering or a tiny gamma head; it is a learned multi-objective residual-value selector trained from full-res/cross-dataset teacher signals, or a stronger Stage-A/generator so the residual does not need to perturb perceptual features so aggressively.


## 2026-06-22 02:10 JST - Mixed perceptual encoder selector, real-codec full-resolution check

Implemented an encoder-side mixed perceptual residual selector in `scripts/evaluate_glc_latent_residual_fullres_realcodec.py`. The goal was to keep the useful part of `latent_grad` while reducing LPIPS/L1 damage. The selector recomputes the sparse ternary residual mask by first-order improvement of:

`0.5 * L1 + 0.5 * LPIPS + 1.0 * DISTS + 0.25 * latent-L1`

Important implementation detail: direct full-resolution VQGAN backward OOMs on DIV2K. Added `--selector_latent_max_side 32`, so selector scoring is done through a downsampled latent/generator proxy while the actual bitstream, decode, saved reconstructions, and metric evaluation remain full-resolution.

Rejected direction:

- Adaptive delta-scale checkpoint `experiments/glc_latent_adaptive_gate_topk0005_from_final_512_balanced_14500to17500/glc_latent_residual_best.pt` looked promising on Kodak but failed on DIV2K early. It worsened both LPIPS and DISTS on several of the first 20 images, so it is not a generalizable method candidate.

Smoke tests:

- Kodak limit-2, `encoder_selector_loss=l1_latent`: DISTS improved but LPIPS worsened.
- Kodak limit-2, `encoder_selector_loss=mix`: both LPIPS and DISTS improved, so the full run used mixed payoff.

Full-resolution real-codec results:

| dataset | setting | payload bpp | residual AC bpp | LPIPS | DISTS | note |
|---|---|---:|---:|---:|---:|---|
| Kodak | mix, gamma=1.0 | 0.011617 | 0.000753 | 0.607516 | 0.355495 | strong DISTS, LPIPS too poor |
| Kodak | mix, gamma=0.6 | 0.011617 | 0.000753 | 0.575253 | 0.371978 | safest Kodak point |
| DIV2K | mix, gamma=0.6, latent32 | 0.011024 | 0.000784 | 0.588100 | 0.335700 | good DISTS, LPIPS cost moderate |
| DIV2K | mix, gamma=0.5, latent32 | 0.011024 | 0.000784 | 0.577254 | 0.340472 | safer default, DISTS wins 84/100 |
| CLIC2020 all | mix, gamma=0.5, latent32 | 0.011000 | 0.000781 | 0.543736 | 0.327340 | DISTS wins 391/428, LPIPS wins 277/428 |

Saved-reconstruction patch metrics:

| dataset | recon path | FID | KID | LPIPS | DISTS |
|---|---|---:|---:|---:|---:|
| CLIC2020 all | `experiments/eval_selector_mix_gamma05_lat32_clic2020_all_save/recon` | 108.4273 | 0.0562 | 0.5427 | 0.3268 |
| DIV2K | `experiments/eval_selector_mix_gamma05_lat32_div2k_save/recon` | 163.7456 | 0.0956 | 0.5766 | 0.3402 |

Read:

- The mixed selector is less aggressive than latent-gradient: it gives up some FID/DISTS improvement but keeps LPIPS much closer to the base generator.
- This is a cleaner top-conference story than manual `latent_error`, progressive splitting, or adaptive gamma. It directly operationalizes "send only residual symbols with perceptual innovation value" at fixed real payload.
- It is still not a final official-GLC improvement. The branch's Stage-A/generator quality is far below official GLC, so the next major step should be selector distillation or a stronger staged training path rather than more small gamma sweeps.


## 2026-06-22 03:55 JST - Learned selector distillation beats base under real codec

Implemented and trained the first practical learned residual-value selector.

Code changes:

- `gp_reslc/scratch/glc_latent_residual.py`: added `_ResidualSelectorNet`, `topk_score_mode=learned_selector`, and selector diagnostics in the model output.
- `scripts/train_glc_latent_residual.py`: added selector-only training, mixed perceptual teacher generation, BCE distillation loss, and W&B logs for selector precision/recall/score statistics.

Training:

- W&B run: `w1briam2` (`selector_mixdistill_topk0005_256_14500to15100`).
- Output: `experiments/glc_latent_selector_mixdistill_topk0005_256_14500to15100/`.
- Resume: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk0005_stable_ternary_1500/glc_latent_residual_final.pt`.
- Frozen: Stage-A, GLC/VQGAN, predictor, residual encoder/decoder, entropy scale model.
- Trainable: selector head only.
- Teacher: mixed first-order perceptual payoff, `0.5 L1 + 0.5 LPIPS + 1.0 DISTS + 0.25 latent-L1`.
- Length: 600 OpenImages crop steps, batch 1, crop 256, `topk=0.0005`, `delta_scale=0.5`.

Real-codec evaluation, no encode-time oracle:

| dataset | bpp | LPIPS base -> learned | DISTS base -> learned | FID/KID learned | note |
|---|---:|---:|---:|---:|---|
| Kodak | 0.011617 | 0.576080 -> 0.573336 | 0.376250 -> 0.375072 | not measured | improves LPIPS/L1 and slightly improves DISTS |
| DIV2K | 0.011023 | 0.565526 -> 0.569826 | 0.353956 -> 0.345658 | 172.6705 / 0.1087 | FID/KID improve from base 181.631 / 0.12152 |
| CLIC2020 all | 0.010999 | 0.540915 -> 0.538061 | 0.341476 -> 0.335407 | 114.3934 / 0.0631 | FID/KID improve from base 118.364 / 0.06663 |

Read:

- This is the first non-oracle selector that improves CLIC LPIPS, DISTS, FID, KID, and L1 together under real byte-backed payload accounting.
- DIV2K still has a small LPIPS/L1 cost, but DISTS and distribution metrics improve clearly.
- The learned selector does not match the oracle mixed/latent-gradient teacher yet; this is expected after only 600 steps and a small selector head.
- The scientific story is now stronger: GP-ResLC can be framed as perceptual innovation coding, with an oracle teacher upper bound and a learned codec-compatible selector.
- This still does not solve the official-GLC gap. The scratch semantic generator remains too weak, so the next high-value path is to transplant this learned selector idea into the stronger pretrained/GLC branch or train the full scratch stages much longer.

Next action:

Run a longer selector distillation with higher LPIPS no-regression pressure and possibly a larger selector head. Then evaluate CLIC/DIV2K/Kodak again and compare the learned selector against mixed-oracle and latent-gradient teacher curves.


## 2026-06-22 04:05 JST - LPIPS-heavy selector distillation rejected on Kodak

Follow-up run after the first successful learned selector:

- W&B: `b15117hn` (`selector_mixdistill_lpips1_topk0005_256_15100to17100`).
- Output: `experiments/glc_latent_selector_mixdistill_lpips1_topk0005_256_15100to17100/`.
- Init: first learned selector final checkpoint.
- Change: teacher weights moved toward LPIPS/L1 safety: `0.75 L1 + 1.0 LPIPS + 1.0 DISTS + 0.25 latent-L1`.

Kodak full-resolution real-codec:

| checkpoint | LPIPS | DISTS | L1 | decision |
|---|---:|---:|---:|---|
| first selector final | 0.573336 | 0.375072 | 0.092120 | current lead |
| LPIPS-heavy best | 0.572479 | 0.376071 | 0.092109 | LPIPS slightly better, DISTS gain nearly gone |
| LPIPS-heavy final | 0.573747 | 0.375379 | 0.092373 | worse than first selector on LPIPS/DISTS balance |

Decision: do not promote the LPIPS-heavy follow-up. It confirms teacher weighting can move the LPIPS/DISTS trade-off, but the first 600-step mixed selector remains the best practical learned-selector checkpoint so far.


## 2026-06-22 Learned selector residual-strength sweep

After the first learned selector succeeded, I swept the decoder-side global residual strength without changing the transmitted bitstream. This tests whether the learned selector can support separate operating modes: a safe LPIPS-preserving point and a DISTS/FID-oriented point.

Checkpoint: `experiments/glc_latent_selector_mixdistill_topk0005_256_14500to15100/glc_latent_residual_final.pt`.

Kodak real-codec sweep:

| delta_scale | LPIPS | DISTS | L1 | read |
|---:|---:|---:|---:|---|
| 0.5 | 0.573336 | 0.375072 | 0.092120 | safe learned-selector default |
| 0.6 | 0.573484 | 0.374590 | 0.091855 | slightly stronger, still safe |
| 0.8 | 0.573651 | 0.372755 | 0.091484 | best Kodak DISTS while keeping LPIPS below base |
| 1.0 | 0.577225 | 0.369601 | 0.092121 | DISTS strong, LPIPS worse than base |

CLIC2020 all 428 at `delta_scale=0.8`:

| payload bpp | base LPIPS | LPIPS | base DISTS | DISTS | base FID/KID | FID/KID | read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.010999 | 0.540915 | 0.548661 | 0.341476 | 0.320307 | 118.364 / 0.06663 | 106.092 / 0.0520 | DISTS/FID-oriented learned-selector point |

Comparison:

- Learned selector `delta=0.5`: CLIC LPIPS/DISTS/FID/KID = `0.538061 / 0.335407 / 114.393 / 0.0631`. This is the safe default because it improves all CLIC metrics versus base.
- Learned selector `delta=0.8`: CLIC LPIPS/DISTS/FID/KID = `0.548661 / 0.320307 / 106.092 / 0.0520`. This is close to latent-gradient oracle distribution quality while keeping LPIPS better than latent-gradient.
- Latent-gradient oracle `delta=0.6`: CLIC LPIPS/DISTS/FID/KID = `0.559899 / 0.319396 / 105.336 / 0.04996`.

Decision:

Use two paper-facing learned-selector operating points: `delta=0.5` as the conservative practical point and `delta=0.8` as the DISTS/FID-oriented point. Do not use `delta=1.0` as default because Kodak LPIPS crosses above the base. DIV2K at `delta=0.8` improves DISTS strongly (`0.334945`) but worsens LPIPS (`0.593723`), so it should be reported as a perceptual-distribution point, not as a safe all-metric point.


## 2026-06-22 05:00 JST - Stage-quant mixed local sensitivity teacher

Purpose: transplant the scratch learned-selector finding into the stronger pretrained GLC branch. The new `scripts/train_v1.py` option `--lambda_gate_mixed_sens` builds a training-only teacher from local L1 degradation, LPIPS-spatial degradation, texture, and edge strength. High gate probability means the current coarsening is locally safe. The teacher is recentered to the requested `rho_target`, so it changes spatial allocation without directly changing the average rate budget.

Implementation:

- Added `make_gate_mixed_sensitivity_target()` to `scripts/train_v1.py`.
- Added CLI options `--lambda_gate_mixed_sens`, `--gate_mixed_l1_weight`, `--gate_mixed_lpips_weight`, `--gate_mixed_texture_weight`, `--gate_mixed_edge_weight`, `--gate_mixed_sens_tau`, and `--gate_mixed_sens_margin`.
- Smoke test passed at `experiments/stage_quant_mixed_sens_smoke`.

q2 run:

- Run: `v5_stage_quant_q2_mixedsens_rt106_rhomax12_lR24_lp10_dists10_mix40_1200`
- W&B: `nzo1i9x1`
- Init: weights-only resume from `experiments/v3_stage_quant_v1q2_quality_hinge_fast_lR35_rhomax20_3k/train_state.pt`
- Setting: fixed GLC, `q_index=2`, `rho_target=1.06`, `stage_rho_max=1.2`, `lambda_gate_mixed_sens=40`, local teacher weights L1/LPIPS/texture/edge = `0.5/1.0/0.1/0.2`.

Training summary:

- Final W&B crop A/B: `delta_bpp_y=-0.00103`, baseline PSNR `20.038`, ours PSNR `19.877`.
- Final gate: `rho_mean=1.061`, `rho_max=1.103`.

Kodak q2 exact real codec:

| run | bpp | bpp_y | PSNR | MS-SSIM | LPIPS | DISTS | FID | KID | enc/dec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q2 mixed-sens | 0.03367 | 0.02832 | 22.0239 | 0.7798 | 0.1701 | 0.0995 | 32.8059 | 0.0026 | 0.089 / 0.107 |
| q2 spatial-guard | 0.03366 | 0.02855 | 21.9877 | 0.7792 | 0.1705 | 0.0993 | 24.6837 | 0.0025 |
| q2 stage-quant quality | 0.03263 | 0.02753 | 21.9264 | 0.7757 | 0.1725 | 0.0993 | n/a | n/a |
| GLC local q2 | 0.03472 | n/a | 22.0767 | 0.7819 | 0.1671 | 0.0979 | n/a | n/a |

Gate-placement audit on Kodak:

| run | delta bpp_y | rho mean | corr rho/base err | corr rho/texture | corr rho/gradient | corr rho/LPIPS delta | high-rho LPIPS delta | low-rho LPIPS delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q2 quality | -0.00113 | 1.061 | +0.237 | +0.244 | +0.216 | +0.024 | 0.00458 | 0.00151 |
| q2 spatial-guard | -0.00055 | 1.028 | -0.057 | +0.027 | -0.048 | +0.004 | 0.00129 | 0.00109 |
| q2 mixed-sens | -0.00104 | 1.056 | -0.305 | -0.372 | -0.284 | -0.016 | 0.00127 | 0.00379 |

Read:

The mixed teacher strongly fixes the placement problem: high-rho locations now have much lower baseline error, texture, and gradient, and high-rho LPIPS delta is lower than low-rho LPIPS delta. This validates the mechanism and is more on-axis than the previous spatial guard because it keeps roughly the same bpp_y saving as q2 quality. However, q2 Kodak DISTS/FID do not improve enough to become a lead. The next step is to move the same teacher to q3, where the rate budget is larger and distribution metrics may benefit more.

## 2026-06-22 q-conditioned mixed sensitivity teacher

Goal: move beyond a global rho gate by letting each GLC rate point use a different generator-predictability teacher. This supports the paper story that GP-ResLC should suppress residual precision where the generator can recover content, while preserving quality-sensitive residuals at higher-rate points.

Implementation:
- Added q_value() helper in scripts/train_v2.py.
- Added --rho_target_by_q and per-q mixed-teacher weights: --gate_mixed_l1_weight_by_q, --gate_mixed_lpips_weight_by_q, --gate_mixed_texture_weight_by_q, --gate_mixed_edge_weight_by_q.
- Existing single-value flags remain backward compatible.
- Training logs now include rho_target_q and active mixed-teacher weights for W&B traceability.

Validation:
- py_compile passed for scripts/train_v2.py.
- Write/saving smoke execution was rejected by the approval layer because escalated write commands are disabled in this environment.
- In-memory CUDA forward/backward validation from the rho1.16 lead checkpoint passed with no file writes. For q3 with rho_target_by_q=[1.20,1.18,1.14,1.10] and mixed weights L1/LPIPS=[0.50,1.00], target_mean=0.6173, gate_mean=0.7129, rho_mean=1.1703, BCE=0.7210, and perceptual-gate gradient L1 sum=76.88.

Interpretation:
- The q-conditioned machinery is functional and produces nonzero gradients.
- This is a research-enabling change, not a completed result. Full checkpoint training/evaluation must be run once the environment allows write-enabled training.

## 2026-06-22 safe-weighted mean prediction

Motivation: global mean prediction can move decoder prior means in quality-sensitive regions and collapse DISTS/LPIPS. To align with the GP-ResLC thesis, predictor supervision should concentrate on regions where the generator/decoder can recover the omitted residual.

Implementation:
- Added --lambda_mean_pred_safe to scripts/train_v2.py.
- Added weighted_spatial_smooth_l1(pred, target, weight).
- The safe weight is the mixed sensitivity target when available, otherwise the gate p-map.
- The loss trains corrected prior means or latent residual means mainly in safe/predictable regions.
- Logged train/mean_pred_safe to W&B.

Validation:
- py_compile passed.
- In-memory CUDA validation from the rho1.16 lead checkpoint passed with no file writes.
- q2 example with predictor+gate trainable, predictor_delta_bound=0.003: target_mean=0.6710, safe_mean=0.01364, gate_bce=0.63734, bpp_y=0.03263, prior_grad_l1=0.11335, gate_grad_l1=25.16, delta_abs=0 at initialization, rho_mean=1.1619.

Interpretation:
- This is a stronger rate-side GP-ResLC signal than the earlier global lambda_mean_pred.
- It operationalizes "send only unpredictable residual" by applying residual-mean prediction where the teacher marks coarsening/prediction as locally safe.
- Needs full write-enabled training and real-codec evaluation against official GLC curves.

## 2026-06-22 rate-potential mixed teacher

Motivation: the prior mixed teacher selects locally safe regions but does not know whether coarsening that region actually saves bits. Diagnostics showed the current lead gate can be negatively correlated with the estimated bit map, so it may suppress low-value regions.

Implementation:
- Added --gate_mixed_rate_weight and --gate_mixed_rate_weight_by_q to scripts/train_v2.py.
- make_gate_mixed_sensitivity_target now accepts rate_map/rate_weight.
- The training loop passes a detached spatial bit map computed from net.get_y_gaussian_bits(out["y_q"], out["scales_hat"]).mean(channel).
- High estimated-bit regions reduce the teacher score, increasing the target gate probability/rho only where rate saving is likely. No inference side map is sent.

In-memory diagnostics:
- q0, OpenImages crop, rho_target=1.24: current gate vs estimated-bit map corr=-0.286.
- Mixed teacher without rate term corr=-0.169.
- Mixed teacher with rate_weight=1.0 corr=+0.583.

In-memory 30-step pilots:
- q2/q3 quality-repair target: q3 DISTS improved slightly but bpp did not move materially; useful for quality repair, not rate.
- q0/q1 rate-heavy target without rate map: tiny bpp drop but DISTS/LPIPS worsened.
- q0/q1 rate-heavy target with rate map: target placement improved, but 30-step tiny pilot still worsened DISTS/LPIPS before meaningful bpp savings.

Interpretation:
- Rate-potential weighting is conceptually important and fixes a real diagnostic mismatch.
- It should be run with longer training, lower rate_weight, and explicit DISTS/LPIPS hinge or baseline distillation.
- Recommended first full run when write-enabled: q0/q1 only, rate_weight_by_q around [0.4,0.3,0,0], rho_target_by_q [1.20,1.18,1.12,1.08], lambda_dists_hinge > 0, lambda_mean_pred_safe small, and real-codec evaluation on Kodak/DIV2K before CLIC.

### Kodak q-wise gate-vs-rate diagnostic

No-write CUDA diagnostic on 8 Kodak center crops using the current rho1.16 lead checkpoint:

| q | corr(current gate p, estimated bit map) | corr(mixed teacher, bit map) | corr(rate-weighted teacher, bit map) |
|---|---:|---:|---:|
| q0 | -0.1964 | -0.0412 | +0.3985 |
| q1 | -0.2277 | -0.0378 | +0.4167 |
| q2 | -0.2721 | +0.0243 | +0.4475 |
| q3 | -0.2574 | -0.0608 | +0.3846 |

Interpretation: the current lead gate is good at avoiding local error/texture, but it is not rate-potential aware. This explains why rho sweeps saturate around modest official-curve gains. The next full training should include a mild rate-potential teacher plus DISTS/LPIPS hinges, rather than only increasing rho.

### Scratch selector readout for top-conference direction

Re-read scratch notes and current result summaries. The scratch branch is not competitive with official pretrained GLC in absolute quality, but it contains the strongest mechanism evidence for the GP-ResLC thesis.

Key facts from docs/scratch_results_summary.md:

- Latent-gradient residual selection improves fixed-payload DISTS/FID/KID on CLIC2020 and DIV2K, but hurts LPIPS.
- Mixed perceptual selector reduces that LPIPS damage while preserving DISTS/FID gains.
- Learned selector distillation is the first practical selector version: no test-time perceptual backprop, real payload accounting, and improved CLIC LPIPS/DISTS/FID/KID versus the scratch base.
- Learned selector at delta=0.5 on CLIC2020 all 428: LPIPS/DISTS/FID/KID = 0.538061 / 0.335407 / 114.393 / 0.0631 versus scratch base 0.540915 / 0.341476 / 118.364 / 0.06663.
- Learned selector at delta=0.8 gives stronger CLIC DISTS/FID = 0.320307 / 106.092, with LPIPS tradeoff.

Interpretation for pretrained GLC branch:

- Scratch should not be used as the main SOTA result yet.
- Its selector mechanism should be imported into the pretrained/official-GLC branch as a gate/mean-prediction teacher: select high perceptual-value, high-rate-potential residual information, not raw residual magnitude.
- The q-conditioned mixed teacher, safe-weighted mean prediction, and rate-potential teacher added to scripts/train_v2.py are the first step toward that transfer.

## 2026-06-22 q-wise objective weights and rate-correlation logging

Implemented q-wise loss weights in scripts/train_v2.py so one variable-rate model can train q0/q1 with stronger rate/rho pressure and q2/q3 with stronger perceptual repair. Added W&B logging for active q-specific weights, gate-rate correlation, mixed-teacher-rate correlation, and rate-map mean/std. No-write CUDA validation passed: q1 example produced finite loss, gate/rate corr -0.197, teacher/rate corr +0.170, and nonzero predictor/gate gradients. Short no-write sweeps showed hinge-only training raises bpp by lowering rho, while explicit rho hold gives small bpp reductions with quality tradeoff; this motivates q-wise objective scheduling for the next write-enabled run.

## 2026-06-22 staged gate-first controls

Implemented staged trainability controls in scripts/train_v2.py: --predictor_train_start, --gate_train_start, and --q_embed_train_start. No-write CUDA check confirmed predictor/q_embed gradients are zero before their start iteration and nonzero after activation. The current rho1.16 lead checkpoint has prior_predictor gate.weight/bias exactly zero, so a gate-first stage from that checkpoint is a true rate-allocation stage; q_embed is nonzero and should be controlled explicitly. All-q no-write qwise pilots showed joint training tends to lower rho and repair quality instead of preserving bpp savings, so the recommended next write-enabled run is gate-first for roughly 1000 iterations, then safe-mean predictor fine-tuning.

## 2026-06-30 stage-aware residual + quant-gate mainline

Motivation: the rho-only branch is a useful safety anchor, but it can be read as
adaptive quantization. The next mainline candidate should implement the original
GP-ResLC thesis more directly: at each GLC four-part prior stage, use only
decoder-available context to predict a generator-recoverable residual mean, then
entropy-code the remaining residual with decoder-computable precision control.

Implemented mode:

```text
predictor_param_mode = stage_residual_quant_gate

For each four-part stage i:
  rho_i = rho_theta_i(z_hat, q, decoded previous y parts)
  delta_i = gp_mu_theta_i(z_hat, q, decoded previous y parts)
  y_scaled_i = y / (quant_step * rho_i)
  encode residual around base_mean_i + delta_i
  decode and reconstruct with quant_step * rho_i
```

Important constraints:

- zero initialization reproduces the pretrained GLC graph up to numerical noise
- no side map is transmitted
- `delta_i` and `rho_i` are both recomputed by the decoder from already decoded
  information
- real codec encode/decode has explicit support for the combined mode
- the old global perceptual gate is disabled with `--no_gate` for this branch

Implementation files:

- `gp_reslc/prior_predictor.py`: added
  `forward_four_part_prior_with_stage_residual_quant_gate`.
- `gp_reslc/real_codec.py`: added matching encode/decode paths for the combined
  stage-aware residual/quant-gate graph.
- `scripts/train_v2.py`: added combined-mode training, checkpointing, logging.
- `scripts/evaluate_real_codec.py`: added combined-mode CLI support and
  recursive dataset image discovery.
- `scripts/train_v1.py`: kept compatible with combined-mode checkpointing.

Smoke validation:

- W&B run: `stage_residual_quant_gate_smoke`
  (`kzpby7b5`).
- Command used OpenImages train, Kodak validation, q2 only, 2 iterations.
- Forward/backward/checkpoint save passed.
- Real codec smoke on Kodak image 1, q2:
  `bpp=0.03746`, `y=0.03235`, `z=0.00342`, `header=0.00169`,
  `encode=0.613s`, `decode=0.112s`, `max_abs=0.000e+00`.

Next full run:

- all q training on OpenImages
- DISTS/LPIPS quality guard plus stage residual prediction loss
- real-codec Kodak quick curve at checkpoints
- promote only if it beats the current rho safety lead under counted bpp

### Full-run result: stage residual + quant gate, rate-heavy checkpoint

Run:

- training: `experiments/stage_residual_quant_gate_allq_rpteacher_lR14_20k`
- W&B: `stage_residual_quant_gate_allq_rpteacher_lR14_20k` (`u2lk41st`)
- promoted checkpoint for evaluation: `v2_2000.pt`
- mode: `stage_residual_quant_gate`
- dataset for transfer check: DIV2K validation images `0801`-`0900`
- codec: `scripts/evaluate_real_codec.py`, same four-part stage graph in encode/decode

Kodak full-resolution real-codec summary versus local GLC:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| rate-heavy `v2_2000` | -5.39% | -0.25% | -1.66% | -3.90% | -3.65% |
| balanced `v2_1000` | -4.00% | +0.20% | -0.57% | -0.88% | -2.29% |

DIV2K validation real-codec summary versus local GLC:

| metric | BD-rate | matched bpp delta |
|---|---:|---:|
| DISTS | -7.43% | -6.66% |
| LPIPS | +0.05% | +1.24% |
| FID | -3.78% | -1.31% |
| KID | -3.98% | -0.68% |
| PSNR | +0.65% | n/a |

DIV2K pointwise behavior:

- q0: bpp `0.02381 -> 0.02148`, but LPIPS/FID worsen.
- q1: bpp `0.02764 -> 0.02546`, DISTS slightly improves, LPIPS/FID worsen mildly.
- q2: bpp `0.03224 -> 0.03054`, DISTS improves slightly, LPIPS/FID worsen mildly.
- q3: bpp `0.03649 -> 0.03512`, DISTS is effectively tied and FID/KID are effectively tied.

Interpretation:

- This branch is more faithful to the GP-ResLC thesis than the earlier global rho shortcut because the residual mean and precision control are computed inside the GLC four-part prior order from decoder-available information.
- The result transfers to both Kodak and DIV2K under counted real bitstreams: DISTS/FID/KID BD-rate are negative on both sets, while LPIPS is near neutral on DIV2K and slightly improved on Kodak.
- The weakness is clear: q0 is over-compressed, and LPIPS is still the easiest perceptual metric to damage. The current decoder-computable `z_hat/q/context` gate cannot fully identify all safe-to-drop locations.
- The next full implementation should therefore move to the mainline "safe-to-drop / residual-control" design, not more rho-target or loss-weight cosmetics.

Next research step:

1. Add a tiny counted control stream for safe-to-drop or residual precision correction when decoder-only signals are insufficient.
2. Keep the stream extremely small and include it in the payload bpp.
3. Couple it with a learned residual/control entropy model, so the new bits are spent only where they buy DISTS/FID/KID or LPIPS safety.
4. Evaluate immediately with real-codec Kodak and DIV2K curves before expanding to CLIC.

### Tiny counted control stream implementation and first rejection

Implemented a counted protection-control stream on top of
`stage_residual_quant_gate`.

Design:

- The encoder predicts a 4-channel binary control map at z resolution.
- The maps are packed into one Bernoulli arithmetic stream and counted in the
  payload.
- A control symbol of 1 protects that coarse region by moving the effective
  stage rho back toward 1:

```text
rho_eff = 1 + (rho_stage - 1) * (1 - control)
```

- The decoder receives the control stream before decoding the y streams, so the
  real arithmetic codec is consistent.
- Real-codec smoke passed with `max_abs=0.000e+00`.

Important implementation correction:

- Four separate control streams wasted header bytes.
- The final implementation packs all four stage maps into one control stream.
- One-image Kodak q2 smoke after packing: `bpp=0.03603`, `control=0.00035`,
  header `0.00185`, `max_abs=0.000e+00`.

Training/evaluation:

- Run: `stage_residual_quant_gate_control_topk04_freezestage_from_rate2000_5k`
- W&B: `o089lp1x`
- Init: `stage_residual_quant_gate_allq_rpteacher_lR14_20k/v2_2000.pt`
- Stage residual predictor, stage quant gate, and q embedding were frozen.
- Only the tiny control encoder was trained.
- Top-k control budget: 4% of z-resolution control symbols.
- Evaluated checkpoint: `v2_2000.pt`

Kodak8 real-codec result versus local GLC:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|
| stage residual + quant gate | -6.01% | -1.20% | +1.02% | -2.74% |
| + top-k 4% control | -3.94% | +0.09% | -13.75% | -1.69% |

Interpretation:

- The counted control stream can improve distribution quality on Kodak8, as
  shown by the FID gain.
- It does not yet beat the stage-only branch on DISTS/LPIPS because the control
  overhead and protection placement cost more than the quality it recovers.
- This confirms that the control-stream idea is method-faithful but needs a
  smaller budget or a learned entropy/control prior before it can become the
  lead.

Next immediate test:

- Repeat the frozen-stage control training with a 2% top-k budget.
- Promote only if it keeps the stage-only DISTS/LPIPS curve while retaining some
  of the FID improvement.

### Tiny counted control stream 2% top-k result

Run:

- training: `experiments/stage_residual_quant_gate_control_topk02_freezestage_from_rate2000_2500`
- W&B: `stage_residual_quant_gate_control_topk02_freezestage_from_rate2000_2500` (`e1a3p24k`)
- init: `stage_residual_quant_gate_allq_rpteacher_lR14_20k/v2_2000.pt`
- frozen modules: stage residual predictor, stage quant gate, q embedding
- trained module: tiny paid control encoder only
- control budget: top-k 2% at z-resolution, packed into one counted Bernoulli stream
- evaluated checkpoint: `v2_final.pt`

Kodak8 real-codec result versus local GLC:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| stage residual + quant gate | -6.01% | -1.20% | +1.02% | +7.44% | -2.74% |
| + top-k 2% control | -3.33% | -0.08% | -4.56% | +10.21% | -1.27% |

Decision:

- Reject the current top-k control formulation as a lead branch.
- It is on-axis because all control bits are counted and decoder uses them before
  y arithmetic decoding, but the current protection labels are not worth their
  cost under DISTS/LPIPS.
- Do not continue with control-budget-only sweeps. Revisit control only after a
  stronger learned residual/control entropy model or teacher can decide where
  protection actually pays off.

### Stage-aware residual entropy predictor: mean + scale

Implemented a stronger mainline branch:

```text
GLC four-part stage prior:
  y_stage = base_mean_stage + gp_mu_stage(context) + residual_stage
  residual_stage ~ N(0, base_scale_stage * gp_scale_stage(context))
```

Properties:

- `gp_mu_stage` and `gp_scale_stage` use only decoder-available signals in the
  same four-part order as GLC.
- No side map and no control stream are transmitted.
- The real arithmetic codec uses the same residual scale multiplier as the
  training graph.
- Zero initialization gives `gp_mu=0`, `gp_scale=1`; old mean-only stage
  predictor weights are copied into the mean half when resuming from the
  current lead checkpoint.

Implementation:

- `StageResidualEntropyPredictor` added to `gp_reslc/prior_predictor.py`.
- New mode: `stage_residual_entropy_quant_gate`.
- Updated `scripts/train_v2.py`, `scripts/evaluate_real_codec.py`, and
  `gp_reslc/real_codec.py`.
- Smoke training and real-codec encode/decode passed.

First run:

- training: `experiments/stage_residual_entropy_quant_gate_from_rate2000_8k`
- W&B: `stage_residual_entropy_quant_gate_from_rate2000_8k` (`afiysst9`)
- init: `stage_residual_quant_gate_allq_rpteacher_lR14_20k/v2_2000.pt`
- frozen: stage quant gate, q embedding
- trained: stage residual entropy predictor
- stopped after checkpoint `v2_2000.pt` for quick real-codec evaluation

Kodak8 real-codec result versus local GLC:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| stage residual + quant gate | -6.01% | -1.20% | +1.02% | +7.44% | -2.74% |
| + residual entropy scale | -5.83% | -2.66% | -4.96% | +36.82% | -4.16% |

Interpretation:

- The scale-aware residual entropy branch reduces real y-stream bpp further
  without any side stream.
- It improves LPIPS and FID BD-rate on Kodak8, and matched-DISTS bpp is stronger
  than the stage-only branch.
- It narrowly loses DISTS BD-rate to the stage-only checkpoint because q0/q1 are
  over-compressed.
- This is a better mainline direction than tiny control: it directly strengthens
  residual entropy modeling and keeps the GP-ResLC story simple.

Next immediate test:

- Continue from this checkpoint with lower rate pressure and DISTS/LPIPS hinges,
  especially on q0/q1, to keep the scale predictor from collapsing quality while
  preserving the LPIPS/FID gain.
- Active run: `experiments/stage_residual_entropy_quant_gate_hinge_from_entropy2000_4k`
  / W&B `stage_residual_entropy_quant_gate_hinge_from_entropy2000_4k`.

Hinge follow-up:

- training: `experiments/stage_residual_entropy_quant_gate_hinge_from_entropy2000_4k`
- W&B: `stage_residual_entropy_quant_gate_hinge_from_entropy2000_4k` (`qtvizlgg`)
- evaluated checkpoint: `v2_2000.pt`

Kodak8 real-codec result:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| stage residual + quant gate | -6.01% | -1.20% | +1.02% | +7.44% | -2.74% |
| + residual entropy scale | -5.83% | -2.66% | -4.96% | +36.82% | -4.16% |
| + residual entropy scale + hinge | -5.52% | -2.58% | -4.09% | +13.53% | -4.38% |

Decision:

- Reject the hinge continuation as a lead. It improves KID stability but does
  not recover DISTS enough and slightly weakens LPIPS/FID relative to the
  non-hinge entropy-scale checkpoint.
- The core diagnosis is not simply loss weight; the residual scale multiplier is
  too free. A narrower scale range should keep the entropy-model benefit while
  avoiding q0/q1 overconfidence.

Next:

- Restart from the stage-only lead with `stage_scale_log_bound=0.3` instead of
  `0.7`.
- Keep stage quant gate and q embedding frozen.
- Promote only if it beats stage-only on DISTS or preserves stage-only DISTS
  while retaining the LPIPS/FID gains of the entropy-scale branch.

Scale-bound follow-up:

- training: `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k`
- W&B: `stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k` (`01bxxf6q`)
- init: `stage_residual_quant_gate_allq_rpteacher_lR14_20k/v2_2000.pt`
- frozen: stage quant gate, q embedding
- residual scale log bound: `0.3`
- evaluated checkpoint: `v2_2000.pt`

Kodak8 real-codec result:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| stage residual + quant gate | -6.01% | -1.20% | +1.02% | +7.44% | -2.74% |
| residual entropy scale, bound 0.7 | -5.83% | -2.66% | -4.96% | +36.82% | -4.16% |
| residual entropy scale, bound 0.3 | -5.95% | -2.67% | -7.23% | +20.66% | -4.52% |

Decision:

- Promote the scale-bound 0.3 checkpoint to a transfer check.
- It nearly matches the stage-only DISTS BD-rate while improving LPIPS and FID
  on Kodak8, and it has stronger matched-DISTS bpp than the stage-only branch.
- Because Kodak8 is small and FID/KID are unstable there, the next decision must
  use DIV2K validation real-codec evaluation.

DIV2K transfer check:

- dataset: `data/eval/div2k_valid_0801_0900` (DIV2K validation 100 images)
- compared runs:
  - local GLC real codec: `experiments/real_codec/div2k_glc`
  - stage residual + quant gate: `experiments/real_codec/div2k_stage_residual_quant_gate_rate_2000`
  - residual entropy scale, bound 0.3: `experiments/real_codec/div2k_stage_residual_entropy_quant_gate_scalebound03_2000`
- metrics:
  - `experiments/real_codec/div2k_stage_residual_entropy_quant_gate_scalebound03_2000_metrics.csv`
  - `experiments/real_codec/div2k_stage_residual_entropy_quant_gate_scalebound03_2000_bd.md`
  - `experiments/real_codec/div2k_stage_residual_entropy_quant_gate_scalebound03_2000_matched.md`

DIV2K real-codec result versus local GLC:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| stage residual + quant gate | -7.43% | +0.05% | -3.78% | -3.98% | -6.66% |
| residual entropy scale, bound 0.3 | -9.20% | -0.05% | -5.13% | -1.23% | -8.03% |

Decision:

- Promote `stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
  to the current mainline validation branch.
- The improvement is not just a q-index point comparison: the matched-DISTS
  interpolation also moves from `-6.66%` to `-8.03%` bpp versus GLC.
- LPIPS remains essentially neutral and KID is less strong than the stage-only
  branch, so the method should not be declared finished.
- Next full test is CLIC2020 combined test. If CLIC confirms the DIV2K trend,
  the next implementation should be a q/stage-conditioned scale schedule or
  safe-to-drop teacher for LPIPS/KID stability, not another loss-weight sweep.

CLIC evaluation safeguard:

- A first CLIC run was accidentally launched without an explicit
  `--predictor_param_mode`. Because `scripts/evaluate_real_codec.py` defaulted
  to `mean`, the run reproduced local GLC bpp exactly instead of using
  `stage_residual_entropy_quant_gate`.
- The duplicate output was moved to
  `experiments/real_codec/clic2020_test_stage_residual_entropy_quant_gate_scalebound03_2000_wrong_mode_mean_duplicate`.
- `scripts/evaluate_real_codec.py` was fixed so that, when a checkpoint stores
  `predictor_param_mode`, the evaluator adopts that mode automatically and
  prints it before evaluation.
- One-image CLIC smoke after the fix confirmed the correct path:
  baseline q0 first image `0.02637 bpp`, GP-ResLC q0 first image
  `0.02225 bpp`, no control stream.

CLIC2020 combined real-codec validation:

- dataset: `data/clic2020_test_combined` (428 images)
- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- evaluator mode: checkpoint-inferred
  `stage_residual_entropy_quant_gate`
- reconstructed output:
  `experiments/real_codec/clic2020_test_stage_residual_entropy_quant_gate_scalebound03_2000`
- metrics:
  - `experiments/real_codec/clic2020_test_stage_residual_entropy_quant_gate_scalebound03_2000_metrics.csv`
  - `experiments/real_codec/clic2020_test_stage_residual_entropy_quant_gate_scalebound03_2000_bd.md`
  - `experiments/real_codec/clic2020_test_stage_residual_entropy_quant_gate_scalebound03_2000_matched.md`

Real transmitted bpp:

| q | GLC bpp | GP-ResLC bpp | y bpp | z bpp | control bpp | encode s/img | decode s/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02134 | 0.01801 | 0.01424 | 0.00352 | 0.00000 | 0.645 | 0.924 |
| 1 | 0.02503 | 0.02187 | 0.01810 | 0.00352 | 0.00000 | 0.725 | 1.008 |
| 2 | 0.02958 | 0.02704 | 0.02327 | 0.00352 | 0.00000 | 0.827 | 1.111 |
| 3 | 0.03369 | 0.03159 | 0.02782 | 0.00352 | 0.00000 | 0.933 | 1.217 |

CLIC2020 quality/BD result versus local GLC:

| metric | BD-rate | matched-quality bpp delta |
|---|---:|---:|
| DISTS | -8.37% | -7.28% |
| LPIPS | +0.68% | +1.88% |
| FID | -5.50% | -2.42% |
| KID | -4.39% | -1.85% |
| PSNR | +0.28% | n/a |

Decision:

- Promote the scale-bound stage residual entropy branch as the current best
  full-benchmark mainline.
- The CLIC result confirms the DIV2K trend on the official-size natural-image
  test set: DISTS/FID/KID curves move left under real serialized codec bpp.
- This is not just a quality-index retuning result. Matched-quality DISTS uses
  all four GLC q targets inside the GP-ResLC range and gives `-7.28%` average
  bpp reduction.
- LPIPS is still the weak axis: BD-rate is slightly worse and matched-LPIPS bpp
  is positive. The next branch should optimize q/stage-conditioned residual
  entropy and precision control to preserve LPIPS while retaining the DISTS/FID
  gain.

Next:

- Start a q-conditioned full branch from this checkpoint.
- Unfreeze the stage quant gate and q embedding, keep the residual entropy
  scale bound at `0.3`, and use conservative learning rate.
- This is a mainline capacity increase, not a scalar sweep: the model should
  learn rate-specific residual predictability/precision behavior across q.

Q-conditioned full branch started:

- run:
  `experiments/stage_residual_entropy_quant_gate_qcond_full_from_sb03_20k`
- W&B:
  `stage_residual_entropy_quant_gate_qcond_full_from_sb03_20k`
  (`t2od6wbs`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate`
- training data:
  `/dpl/open-images-v6/train`
- validation:
  `data/eval/kodak8`
- iterations:
  `20000`
- batch size:
  `4`
- learning rate:
  `3e-6`
- q choices:
  `0 1 2 3`
- trainable:
  stage residual entropy predictor, stage quant gate, q embedding
- frozen:
  pretrained GLC backbone
- loss:
  `12 R + 0.03 MSE + 3.0 LPIPS + 1.8 DISTS + 1.0 LPIPS-hinge + 0.1 mean-pred + 0.01 stage-delta-abs`
- residual scale log bound:
  `0.3`

Initial log:

- trainable params: `10.31 M`
- initial q1 batch: bpp `0.0239`, bpp_y `0.0204`, LPIPS `0.2476`,
  DISTS `0.1304`, LPIPS hinge `0.0191`
- A/B at iteration 0 shows lower y-stream bpp for all q, but quality is still
  slightly behind the frozen GLC baseline, so the run must be judged by
  checkpoint real-codec validation, not training bpp alone.

Early stop decision:

- stopped manually after iteration `2200`
- checkpoints kept:
  - `v2_0.pt`
  - `v2_1000.pt`
  - `v2_2000.pt`
- reason:
  the global LPIPS hinge protected quality by pushing the learned stage rho
  almost back to `1.0`, erasing the y-stream saving that made the scale-bound
  branch strong.
- A/B at iteration `2000`:
  - q0 y delta: `-0.0007 bpp`
  - q1 y delta: `-0.0004 bpp`
  - q2 y delta: `-0.0005 bpp`
  - q3 y delta: `-0.0004 bpp`
- conclusion:
  reject this as a lead. It is too conservative and is likely to collapse the
  CLIC/DIV2K BD-rate gains.

Next branch:

- use the existing mixed safe-to-drop teacher instead of a global LPIPS hinge.
- The teacher estimates where current coarsening is locally safe from
  GLC-relative L1/LPIPS/texture/rate signals, then trains the decoder-computable
  stage gate map.
- This is closer to the GP-ResLC thesis: learn where the generator can recover
  details and keep residual precision for the unpredictable regions.

Mixed safe-to-drop teacher branch started:

- run:
  `experiments/stage_residual_entropy_quant_gate_mixedteacher_qcond_from_sb03_20k`
- W&B:
  `stage_residual_entropy_quant_gate_mixedteacher_qcond_from_sb03_20k`
  (`hazdh760`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate`
- trainable:
  stage residual entropy predictor, stage quant gate, q embedding
- central change:
  replace global LPIPS hinge with mixed safe-to-drop teacher:
  `0.5 BCE(p_stage, safe_target)`, where `safe_target` is built from
  GLC-relative local L1/LPIPS-spatial damage, texture/edge protection, and
  estimated rate map.
- rate/teacher setup:
  `lambda_R=14`, `rho_target=1.14`, `rho_target_until=12000`,
  `rho_max=1.5`, `stage_rho_max=1.5`, `stage_scale_log_bound=0.3`
- auxiliary residual constraints:
  `lambda_mean_pred_safe=0.10`, `lambda_predictor_unsafe_delta=0.02`,
  `lambda_stage_delta_abs=0.01`
- initial A/B:
  - q0 y delta: `-0.0037 bpp`
  - q1 y delta: `-0.0035 bpp`
  - q2 y delta: `-0.0032 bpp`
  - q3 y delta: `-0.0024 bpp`
- initial teacher stats:
  `mixed_target mean/std = 0.602/0.303`, `rho mean/min/max = 1.152/1.051/1.224`

Mid-run research summary, 2026-06-30 11:11 JST:

- Main verified result in this block:
  `stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
  is the current strongest full-benchmark checkpoint.
- CLIC2020 combined real-codec evaluation confirmed:
  DISTS BD-rate `-8.37%`, FID BD-rate `-5.50%`, KID BD-rate `-4.39%`,
  matched-DISTS bpp `-7.28%` versus local GLC.
- DIV2K validation had already shown the same trend:
  DISTS BD-rate `-9.20%`, FID BD-rate `-5.13%`, matched-DISTS bpp `-8.03%`.
- The fix to `scripts/evaluate_real_codec.py` is important:
  evaluator now adopts `predictor_param_mode` from the checkpoint, preventing
  silent fallback to `mean` mode and accidental GLC-duplicate evaluations.

Branches tried in this block:

1. `stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k`
   - purpose:
     strengthen stage-aware residual entropy modeling with a conservative scale
     multiplier bound.
   - result:
     promoted as current mainline because it improves real-codec curves on
     Kodak8, DIV2K, and CLIC2020.

2. `stage_residual_entropy_quant_gate_qcond_full_from_sb03_20k`
   - purpose:
     unfreeze q embedding and stage quant/residual modules, add LPIPS hinge,
     and learn q-conditioned full behavior.
   - early result:
     rejected after `2200` iterations because the LPIPS hinge drove `rho`
     back toward `1.0` and erased most y-stream saving.
   - lesson:
     global quality protection is too blunt; it protects quality by undoing the
     compression mechanism.

3. `stage_residual_entropy_quant_gate_mixedteacher_qcond_from_sb03_20k`
   - purpose:
     train a safe-to-drop stage gate using local GLC-relative L1/LPIPS-spatial
     damage, texture/edge protection, and estimated rate map.
   - current status:
     running. At `5000` iterations, A/B y-stream deltas remain strong:
     q0 `-0.0025`, q1 `-0.0028`, q2 `-0.0031`, q3 `-0.0030` bpp_y.
   - interpretation:
     unlike the LPIPS-hinge branch, this keeps the compression mechanism alive
     while attempting to learn where coarsening is safe.

External-method references used in this block:

- No new external GitHub repository was cloned or directly imported in this
  block.
- The design decisions are still grounded in existing methods:
  - GLC's four-part autoregressive prior and real codec evaluation path.
  - HiFiC/GLC natural-image FID/KID patch protocol.
  - Hyperprior/autoregressive entropy modeling practice from learned image
    compression.
  - Learned bit-allocation / safe-to-drop ideas from generative/perceptual
    compression, implemented here inside the pretrained GLC residual stream
    rather than by changing the backbone codec.

Next decision point:

- Evaluate `stage_residual_entropy_quant_gate_mixedteacher_qcond_from_sb03_20k`
  at checkpoint `v2_5000.pt` first on Kodak8 real codec.
- Promote to DIV2K/CLIC only if it keeps or improves the current mainline
  DISTS/FID gains without worsening LPIPS beyond the scale-bound checkpoint.
- If it fails quality, preserve the current mainline and design a less blunt
  teacher: either lower desired safe-map mean by q, or add a tiny counted
  protection stream for only the unpredictable residual/control.

Mixed teacher branch evaluation:

- stopped after `v2_6000.pt`
- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_mixedteacher_qcond_from_sb03_20k/v2_6000.pt`
- real-codec output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_mixedteacher_qcond_6000`
- metrics:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_mixedteacher_qcond_6000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_mixedteacher_qcond_6000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_mixedteacher_qcond_6000_matched.md`

Kodak8 real-codec result versus local GLC:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| residual entropy scale, bound 0.3 | -5.95% | -2.67% | -7.23% | +20.66% | -4.52% |
| mixed safe-to-drop teacher, qcond, 6000 | -3.97% | -2.34% | -6.20% | -3.07% | -3.21% |

Decision:

- Do not promote this mixed-teacher checkpoint.
- It still beats GLC, but it is weaker than the current scale-bound mainline on
  DISTS, LPIPS, FID, and matched-DISTS bpp.
- The result is informative: the local safe-to-drop teacher preserved the
  y-stream saving in training logs, but did not translate into a better
  perceptual real-codec curve.
- This suggests that a decoder-only safe map is not enough to protect all
  unpredictable residual regions. The next mainline attempt should use a tiny
  counted protection/control stream, where only genuinely decoder-unpredictable
  residual/control information is transmitted and charged to real bpp.

Implementation update: entropy-scale residual gate with tiny counted control:

- Added predictor mode:
  `stage_residual_entropy_quant_gate_control`
- Purpose:
  combine the current strongest entropy-scale residual branch with a tiny paid
  protection stream.
- Mechanism:
  - keep stage-aware residual mean prediction
  - keep residual entropy scale multiplier
  - keep decoder-computable stage precision gate
  - transmit sparse binary control symbols only where the decoder-only gate
    should be locally pulled back toward `rho=1`
  - count the control stream in real serialized bpp
- Smoke checks:
  - `py_compile` passed for `gp_reslc/prior_predictor.py`,
    `gp_reslc/real_codec.py`, `scripts/train_v2.py`,
    `scripts/evaluate_real_codec.py`
  - 1-iteration train smoke passed:
    `experiments/smoke_stage_residual_entropy_quant_gate_control/v2_0.pt`
  - real-codec smoke passed on 2 Kodak8 images:
    `experiments/real_codec/smoke_stage_residual_entropy_quant_gate_control`
  - observed q0 smoke:
    `bpp=0.02346`, `y=0.01794`, `control=0.00024`, `z=0.00342`

Next branch:

- Start from the current scale-bound mainline.
- First train only the tiny control encoder with the mainline frozen, so the
  experiment tests the new paid protection stream rather than drifting the
  whole codec.
- If this improves LPIPS/DISTS/FID per real bpp, follow with a joint fine-tune.

Tiny counted control branch started:

- run:
  `experiments/stage_residual_entropy_quant_gate_control_topk04_controlonly_from_sb03_8k`
- W&B:
  `stage_residual_entropy_quant_gate_control_topk04_controlonly_from_sb03_8k`
  (`ry6ibeqy`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate_control`
- trainable:
  tiny control encoder only, `1.45 M` params
- frozen:
  q embedding, stage residual entropy predictor, stage quant gate, pretrained
  GLC backbone
- control codec:
  fixed-prior Bernoulli arithmetic stream, counted in real bpp
- control setting:
  `topk_frac=0.04`, `control_prob_one=0.04`, `control_target_mean=0.04`
- initial log:
  - q3 batch `bpp_y=0.0320`, `bpp_control=0.00027`
  - control symbol/prob/max: `0.0469/0.0500/0.0500`
  - A/B y-stream deltas:
    q0 `-0.0030`, q1 `-0.0029`, q2 `-0.0022`, q3 `-0.0016`

Decision criterion:

- The branch is only useful if the paid control improves perceptual quality
  enough to offset its real bpp cost.
- First checkpoint target: `v2_2000.pt` or `v2_4000.pt` on Kodak8 real codec.

Tiny counted control branch evaluation:

- stopped after `v2_2000.pt`
- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_control_topk04_controlonly_from_sb03_8k/v2_2000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_topk04_controlonly_2000`
- metrics:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_topk04_controlonly_2000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_topk04_controlonly_2000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_topk04_controlonly_2000_matched.md`

Kodak8 real-codec result versus local GLC:

| run | DISTS BD-rate | LPIPS BD-rate | FID BD-rate | KID BD-rate | matched DISTS bpp |
|---|---:|---:|---:|---:|---:|
| residual entropy scale, bound 0.3 | -5.95% | -2.67% | -7.23% | +20.66% | -4.52% |
| tiny control topk 0.04, control-only | -2.98% | -0.70% | +1.94% | -1.26% | -1.96% |

Decision:

- Reject the control-only checkpoint.
- The paid stream is correctly counted (`~0.00024 bpp`) and technically works,
  but it does not improve the rate-perception curve enough to justify the
  extra bpp.
- The q0/q1 perceptual quality remains weak and FID worsens. The control
  encoder appears to learn a sparse location map, but the current teacher/topk
  target is not aligned well enough with actual perceptual benefit.
- Keep the code path because it is a valid full implementation of the original
  "send only unpredictable control" idea, but do not use it as the current lead.

Current lead after this branch:

- `stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
  remains the best checkpoint.

Next direction:

- Return to the no-side-information mainline and improve the residual entropy
  branch itself.
- A promising route is a q-conditioned fine-tune without global LPIPS hinge or
  mixed teacher: keep the scale-bound entropy predictor, allow q embedding and
  stage gate to adapt, but use a lower learning rate and real-codec checkpoint
  validation to prevent quality collapse.

### 2026-06-30 wrap-up: q-conditioned no-hinge branch

Started a final no-side-information branch to test whether q-conditioned stage
entropy/gate adaptation can improve the current lead without the two failure
modes observed above:

- no LPIPS hinge, because the hinge branch collapsed rho toward 1 and erased
  the bpp saving
- no mixed teacher, because the teacher branch improved over GLC but did not
  beat the current lead on Kodak8
- no counted control stream, because the paid control branch did not offset its
  added bpp

Run:

- experiment:
  `experiments/stage_residual_entropy_quant_gate_qcond_nohinge_from_sb03_6k`
- W&B:
  `stage_residual_entropy_quant_gate_qcond_nohinge_from_sb03_6k`
  (`dj9m57f2`)
- initialization:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate`
- trainable:
  q embedding, stage quant gate, stage residual entropy predictor
- frozen:
  pretrained GLC backbone
- checkpoint available:
  `v2_1000.pt`

At iteration 1000, the quick A/B y-stream deltas were still negative:

| q | y-stream delta |
|---|---:|
| q0 | -0.0029 |
| q1 | -0.0026 |
| q2 | -0.0020 |
| q3 | -0.0014 |

The branch was stopped at the user's requested wrap-up point after `v2_1000.pt`
was safely written. It has not yet been promoted because real-codec evaluation
has not been run.

Current lead remains:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- CLIC2020 combined real-codec evidence:
  - DISTS BD-rate: `-8.37%`
  - FID BD-rate: `-5.50%`
  - KID BD-rate: `-4.39%`
  - LPIPS BD-rate: `+0.68%`
- interpretation:
  the branch gives a real serialized bpp reduction on the main perceptual
  metrics, but LPIPS remains weak and should not be overclaimed.

Next action when resuming:

1. Real-codec evaluate
   `experiments/stage_residual_entropy_quant_gate_qcond_nohinge_from_sb03_6k/v2_1000.pt`
   on Kodak8 first.
2. Promote it only if it beats the current lead in DISTS/FID/KID without a
   meaningful LPIPS regression.
3. If it fails, keep the scale-bound checkpoint as the mainline and move to a
   stronger but still simple residual allocation design rather than tuning
   scalar loss weights.

System state at wrap-up:

- GPU visible: RTX 4070 Ti SUPER
- GPU utilization after stop: idle
- no long-running training/evaluation process left active

## 2026-06-30 mainline restart: residual entropy research

Policy context:

- The active research direction is no longer rho/loss-weight tuning.
- Keep the current scale-bound rho/stage-entropy checkpoint as a safety lead.
- Move toward stage-aware residual-variable coding and learned residual entropy
  modeling while preserving GLC's four-part decode order and real serialized bpp
  evaluation.

Current lead remains:

- `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`

### Pending q-conditioned no-hinge checkpoint evaluation

Evaluated the previously pending checkpoint:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_qcond_nohinge_from_sb03_6k/v2_1000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_qcond_nohinge_1000`
- mode:
  `stage_residual_entropy_quant_gate`
- real codec:
  byte-backed payload, no control stream, `max_abs=0` for decode consistency

Kodak8 average bpp:

| q | qcond no-hinge bpp | current lead bpp |
|---|---:|---:|
| q0 | 0.02416 | 0.02334 |
| q1 | 0.02843 | 0.02760 |
| q2 | 0.03377 | 0.03299 |
| q3 | 0.03856 | 0.03793 |

Summary files:

- `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_qcond_nohinge_1000_metrics.csv`
- `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_qcond_nohinge_1000_bd.md`
- `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_qcond_nohinge_1000_matched.md`

Kodak8 BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| qcond no-hinge 1000 | -5.01% | -1.70% | -6.56% | -12.12% |
| current lead | -5.95% | -2.67% | -7.23% | -12.27% |

Decision:

- Reject qcond no-hinge as a new lead.
- It is codec-correct and better than GLC, but weaker than the current lead on
  Kodak8 and uses more bpp at every q.
- Do not reopen q/rho/loss tuning from this branch.

### Real-codec entropy family implementation

Implemented a small residual entropy-model extension in `gp_reslc/real_codec.py`:

- existing default:
  `gaussian`
- added:
  `laplace`, `logistic`
- evaluator:
  `scripts/evaluate_real_codec.py --entropy_family {gaussian,laplace,logistic}`

Motivation:

- This tests whether the arithmetic-coded residual symbols are better matched by
  a heavier-tailed finite-support distribution than the existing Gaussian model.
- It is an entropy-model experiment, not a rho/quantization tuning experiment.
- The implementation follows the same GLC four-part stage order and preserves
  exact decode consistency.

Smoke result on the current lead, Kodak first two images, q0:

| family | avg bpp | decode consistency |
|---|---:|---:|
| gaussian | 0.02289 | `max_abs=0` |
| logistic | 0.02373 | `max_abs=0` |
| laplace | 0.02563 | `max_abs=0` |

Kodak8 full-q logistic check:

- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_scalebound03_2000_logistic`
- logistic bpp was higher than Gaussian at every q:
  q0 `0.02415`, q1 `0.02847`, q2 `0.03388`, q3 `0.03883`
- current Gaussian lead remains:
  q0 `0.02334`, q1 `0.02760`, q2 `0.03299`, q3 `0.03793`

Decision:

- Keep the entropy-family code path because it is default-compatible and useful
  for residual entropy experiments.
- Do not promote Laplace/logistic for the current lead. The existing Gaussian
  entropy family is shorter for the current residual symbols.

### New mainline run: residual entropy refinement only

Started a branch that freezes the current rho/stage-quant allocation and trains
only the stage-aware residual entropy predictor:

- run:
  `experiments/stage_residual_entropy_refine_only_from_sb03_10k`
- W&B:
  `stage_residual_entropy_refine_only_from_sb03_10k`
  (`h873kr0s`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate`
- trainable:
  stage residual mean/scale predictor only (`5.42 M` params)
- frozen:
  pretrained GLC backbone, current stage quant gate, q embedding, perceptual gate
- purpose:
  improve the predictable/unpredictable residual entropy model while preserving
  the current residual precision allocation.

Initial A/B at iteration 0:

| q | baseline y bpp | ours y bpp | delta |
|---|---:|---:|---:|
| q0 | 0.0229 | 0.0193 | -0.0036 |
| q1 | 0.0269 | 0.0234 | -0.0035 |
| q2 | 0.0319 | 0.0291 | -0.0029 |
| q3 | 0.0362 | 0.0338 | -0.0023 |

Decision rule:

- First checkpoint to evaluate: `v2_1000.pt`.
- Promote only if real-codec Kodak8 improves over the current lead, not merely
  over GLC.
- If this refinement only repairs quality by giving back bpp, reject it and move
  to a stronger stage-aware residual-variable codec path.

Residual entropy refine-only v2_1000 result:

- checkpoint:
  `experiments/stage_residual_entropy_refine_only_from_sb03_10k/v2_1000.pt`
- real-codec output:
  `experiments/real_codec/kodak8_stage_residual_entropy_refine_only_1000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_refine_only_1000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_refine_only_1000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_refine_only_1000_matched.md`

Kodak8 BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -7.23% | -12.27% |
| refine-only v2_1000 | -6.58% | -1.95% | -4.23% | -11.16% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | +1.04% | -0.97% | -5.85% |
| refine-only v2_1000 | -5.97% | -0.72% | -0.16% | -3.06% |

Interpretation:

- The refinement is not useless: it improves DISTS BD-rate and matched-DISTS bpp
  over the current lead on Kodak8.
- It is not a new lead: FID/KID BD-rate and LPIPS are weaker than the current
  lead.
- The training logs show `stage_delta_abs` remains far below the residual target
  (`~0.003-0.004` versus `~0.018-0.024`), so the model is still not strongly
  explaining predictable residual components with `gp_mu_stage`.

Decision:

- Do not promote `refine-only v2_1000`.
- Use it as evidence that residual entropy refinement can move DISTS, but a
  stronger stage-aware residual-variable objective is needed.

Next branch:

- Start a direct residual-mean branch from the current lead.
- Freeze the current stage quant gate and q embedding.
- Increase stage mean-prediction pressure so `gp_mu_stage` actually explains a
  larger fraction of `y_stage - base_mean_stage`.
- Evaluate quickly by real codec; promote only if DISTS/FID/KID improve without
  sacrificing LPIPS.

### 2026-06-30 JST - Normalized stage residual-mean loss check

Added a normalized residual explanation loss to the stage residual-entropy path:

- code:
  - `gp_reslc/prior_predictor.py`
  - `scripts/train_v2.py`
- new training option:
  `--lambda_stage_mean_norm`
- purpose:
  raw residual targets are small (`~0.018-0.024`), so the previous raw
  Smooth-L1 loss gave a weak gradient. The normalized loss asks `gp_mu_stage`
  to explain a larger fraction of the decoder-computable stage residual.

Run:

- `experiments/stage_residual_entropy_normmean_from_sb03_3k`
- W&B:
  `stage_residual_entropy_normmean_from_sb03_3k` (`ea9muskc`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- trained:
  stage residual entropy predictor only
- frozen:
  pretrained GLC, stage quant gate, q embedding, perceptual gate

Stopped after the `v2_1000.pt` checkpoint plus a short continuation to inspect
the trend:

- `stage_abs` stayed small at roughly `0.003-0.004`
- `stage_target_abs` stayed around `0.018-0.024`
- normalized loss remained around `0.62-0.71`

Real-codec Kodak8 evaluation:

- checkpoint:
  `experiments/stage_residual_entropy_normmean_from_sb03_3k/v2_1000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_normmean_1000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_normmean_1000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_normmean_1000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_normmean_1000_matched.md`

Kodak8 BD-rate versus local GLC in the same metric pass:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| normmean v2_1000 | -3.79% | -1.93% | -14.30% | -7.71% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| normmean v2_1000 | -3.06% | -2.00% | -0.49% | -1.37% |

Decision:

- Do not promote the normalized residual-mean branch.
- It slightly helps FID/KID on Kodak8, but it weakens the primary DISTS/LPIPS
  tradeoff compared with the current lead.
- More importantly, `stage_abs / stage_target_abs` still does not grow enough.
  The bottleneck is not just loss scale; the current after-the-fact stage mean
  predictor is not yet a strong residual-variable codec.

Next mainline direction:

- Move from residual-mean fine-tuning to a fuller safe-coarsening/control
  mechanism:
  make the decoder-computable stage gate more aggressive, then use a tiny
  counted control stream to protect only the unsafe/unpredictable locations.
- This tests the GP-ResLC thesis more directly than another residual mean-loss
  or rho-target sweep.

### 2026-06-30 JST - Joint aggressive gate + tiny counted control

Tested a fuller control-stream version of the mainline idea:

- make the stage quant gate more aggressive
- send a tiny counted control stream only for unsafe/unpredictable locations
- keep all transmitted control bits in the real payload

Run:

- `experiments/stage_residual_entropy_quant_gate_control_joint_rdo_topk02_from_sb03_4k`
- W&B:
  `stage_residual_entropy_quant_gate_control_joint_rdo_topk02_from_sb03_4k`
  (`diu5iu81`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate_control`
- trained:
  stage quant gate + tiny control encoder
- frozen:
  pretrained GLC, q embedding, stage residual entropy predictor, perceptual gate
- control stream:
  top-k binary symbols at z resolution, `control_topk_frac=0.02`,
  `control_prob_one=0.02`
- teacher:
  mixed local safety plus rate-potential weighting, with control target built
  from unsafe locations.

Stopped after `v2_1000.pt` because the real-codec A/B trend showed weaker
y-stream saving than the current lead and the control/header overhead moved the
curve right.

Real-codec Kodak8 evaluation:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_control_joint_rdo_topk02_from_sb03_4k/v2_1000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_joint_rdo_topk02_1000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_joint_rdo_topk02_1000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_joint_rdo_topk02_1000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_joint_rdo_topk02_1000_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| control joint v2_1000 | 0.02401 | 0.02825 | 0.03370 | 0.03865 |

Kodak8 BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| control joint v2_1000 | -2.52% | -0.70% | -10.98% | -3.28% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| control joint v2_1000 | -1.23% | -0.12% | +0.83% | +0.89% |

Decision:

- Reject this joint control checkpoint as a lead.
- The mechanism is codec-correct and the control stream is small
  (`~0.00016 bpp` on Kodak8), but it does not buy enough quality or additional
  y-stream reduction to beat the zero-side current lead.
- The result suggests that a paid stream should not merely protect a coarse rho
  map. The next paid-stream version should transmit a compact residual/control
  symbol that directly changes the reconstruction or entropy parameters where
  the decoder-only predictor is insufficient.

Next mainline implication:

- Keep tiny counted control infrastructure.
- Shift from "protect unsafe rho locations" to "encode a compact residual/control
  variable with a learned entropy model", closer to the original
  predictable/unpredictable residual allocation design.

### 2026-06-30 JST - Aggressive tiny-control top-k 0.01 check

After the joint top-k 0.02 control branch moved right, I tested whether a much
smaller paid stream plus a more aggressive stage gate can overcome the control
overhead.

Run:

- `experiments/stage_residual_entropy_quant_gate_control_aggr_topk01_from_sb03_2k`
- W&B:
  `stage_residual_entropy_quant_gate_control_aggr_topk01_from_sb03_2k`
  (`h474o0ek`)
- init:
  current scale-bound lead
- mode:
  `stage_residual_entropy_quant_gate_control`
- trained:
  stage quant gate + tiny control encoder
- frozen:
  pretrained GLC, q embedding, stage residual entropy predictor, perceptual gate
- control stream:
  top-k binary symbols, `control_topk_frac=0.01`,
  real bpp about `0.00010`
- target:
  more aggressive q-dependent coarsening, especially q0/q1/q2.

Stopped after `v2_1000.pt` for real-codec evaluation.

Real-codec Kodak8:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_control_aggr_topk01_from_sb03_2k/v2_1000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_aggr_topk01_1000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_aggr_topk01_1000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_aggr_topk01_1000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_aggr_topk01_1000_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| aggressive topk0.01 | 0.02273 | 0.02702 | 0.03286 | 0.03813 |

Kodak8 metrics:

| run | q0 DISTS | q1 DISTS | q2 DISTS | q3 DISTS |
|---|---:|---:|---:|---:|
| current lead | 0.1264 | 0.1118 | 0.1038 | 0.0983 |
| aggressive topk0.01 | 0.1360 | 0.1184 | 0.1040 | 0.0964 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| aggressive topk0.01 | -2.72% | -1.40% | -10.83% | -2.27% |

Decision:

- Do not promote this all-q aggressive branch.
- It proves that paid control plus stronger no-send pressure can move q0/q1/q2
  left, but q0/q1 perceptual quality collapses too much.
- Useful signal: q3 DISTS improves over the current lead despite similar bpp.

Next check:

- Train a q2/q3-focused control branch so the useful high-quality behavior is
  explored without letting q0/q1 dominate the shared gate update.

### 2026-06-30 JST - q2/q3-focused tiny-control branch

Because the aggressive all-q branch improved q3 DISTS but damaged q0/q1, I ran
a q2/q3-focused variant.

Run:

- `experiments/stage_residual_entropy_quant_gate_control_q23_topk01_from_sb03_2k`
- W&B:
  `stage_residual_entropy_quant_gate_control_q23_topk01_from_sb03_2k`
  (`28jaba1j`)
- q choices:
  `2 3`
- init:
  current scale-bound lead
- mode:
  `stage_residual_entropy_quant_gate_control`
- trained:
  stage quant gate + tiny control encoder
- frozen:
  pretrained GLC, q embedding, stage residual entropy predictor, perceptual gate
- control:
  top-k binary, `control_topk_frac=0.01`, counted real bpp.

Real-codec Kodak8:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_control_q23_topk01_from_sb03_2k/v2_1000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_q23_topk01_1000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_q23_topk01_1000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_q23_topk01_1000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_control_q23_topk01_1000_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| q23 topk0.01 | 0.02332 | 0.02749 | 0.03303 | 0.03815 |

DISTS by q:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.1264 | 0.1118 | 0.1038 | 0.0983 |
| q23 topk0.01 | 0.1293 | 0.1158 | 0.1028 | 0.0976 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| q23 topk0.01 | -4.25% | -1.67% | -11.42% | -2.19% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| q23 topk0.01 | -3.81% | +2.10% | +0.33% | +3.61% |

Decision:

- Do not promote q23 topk0.01.
- It improves high-quality DISTS points q2/q3, but q0/q1 still degrade through
  shared gate side effects, and FID/KID/LPIPS are weaker than the current lead.

Research conclusion for the control-protect family:

- A counted protection stream is codec-correct and on-axis, but as implemented
  it mostly trades quality between q regions instead of moving the full curve.
- The next implementation should make the paid stream carry a compact residual
  or entropy-parameter correction, not merely a binary "undo rho" decision.

### 2026-06-30 JST - Signed residual-control stream prototype

I implemented a codec-side prototype closer to the main GP-ResLC residual/control
story:

- mode:
  `stage_residual_entropy_quant_gate_residual_control`
- base checkpoint:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- payload:
  one extra arithmetic-coded signed ternary stream
  (`-1/0/+1`) before the four `y` streams
- control source:
  source-side top-k stage residual scores aggregated at `z_hat` resolution
- decoder action:
  signed control shifts the four-part stage prior mean before coding/decoding
  the `y` residual
- counted bits:
  control bytes are included in serialized bpp.

Smoke:

- `experiments/real_codec/smoke_residual_control_topk01_delta025_g1`
- q0, one Kodak image, real payload decode passed.

First attempt:

- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk01_delta025_g1`
- settings:
  `topk_frac=0.01`, `prob_nonzero=0.01`, `mean_delta=0.25`,
  `groups=1`
- result:
  codec-correct but visually/metric-wise broken. Kodak8 q0 DISTS jumped to
  `0.6410` and q3 to `0.6537`.

Interpretation:

- A single low-resolution sign shared across all latent channels is too coarse.
- Direct mean shifts at this scale are not safe without a learned or more
  channel-aware residual-control design.

Scale sanity check:

- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk01_delta0025_g1`
- settings:
  `topk_frac=0.01`, `prob_nonzero=0.01`, `mean_delta=0.025`,
  `groups=1`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk01_delta0025_g1_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk01_delta0025_g1_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk01_delta0025_g1_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| residual-control g1 | 0.02362 | 0.02786 | 0.03325 | 0.03819 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| residual-control g1 | -4.42% | -1.56% | -12.14% | -4.46% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| residual-control g1 | -3.29% | -0.82% | +0.12% | +0.18% |

Decision:

- Do not promote the single-group signed mean-control prototype.
- It validates the counted residual/control payload path, but the action is too
  coarse: strong control breaks reconstructions; weak control mostly adds
  overhead.

Next mainline action:

- Test channel-grouped signed residual-control to see whether the failure is
  caused by all-channel coupling.
- If grouped mean-control still cannot beat the current lead, switch the paid
  stream from "mean shift" to a direct residual-symbol/control-token coding
  design.

### 2026-06-30 JST - Channel-grouped signed mean-control check

To test whether the single-group residual-control failure came from coupling all
latent channels together, I evaluated a channel-grouped variant.

Run:

- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk0025_delta005_g16`
- settings:
  `topk_frac=0.0025`, `prob_nonzero=0.0025`, `mean_delta=0.05`,
  `groups=16`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk0025_delta005_g16_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk0025_delta005_g16_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_residual_control_topk0025_delta005_g16_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| residual-control g16 | 0.02395 | 0.02818 | 0.03361 | 0.03854 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| residual-control g16 | -4.53% | -1.35% | -11.92% | -4.46% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| residual-control g16 | -2.93% | -0.11% | +0.79% | +1.21% |

Decision:

- Do not promote grouped signed mean-control.
- q3 DISTS improved slightly (`0.0978` vs current `0.0983`), but the full curve
  is weaker once counted control bits are included.
- Mean-shift control appears structurally weak: weak shifts do not move quality
  enough, and strong shifts damage the generator latent.

Next:

- Replace mean-shift control with a direct latent residual-symbol stream:
  encode sparse signed corrections to the decoded latent representation itself,
  count the control bits, and test whether quality gain is large enough to move
  matched-metric curves.

### 2026-06-30 JST - Direct latent residual-control prototype

I replaced mean-shift control with a more direct residual stream:

- mode:
  `stage_residual_entropy_quant_gate_latent_control`
- base:
  current scale-bound lead
- payload:
  one counted signed ternary stream + four `y` streams
- encoder signal:
  sparse stage/channel-group signs from latent residual `y - y_hat_base`
- decoder action:
  add signed corrections directly to decoded `y_hat` before `net.dec` and VQGAN
  generation.

This is conceptually closer to "send only unpredictable residual/control" than
mean-control because the extra stream directly refines the transmitted latent.

Run:

- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_delta005_g16`
- settings:
  `topk_frac=0.0025`, `prob_nonzero=0.0025`, `delta=0.05`,
  `groups=16`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_delta005_g16_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_delta005_g16_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_delta005_g16_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| latent-control | 0.02397 | 0.02822 | 0.03363 | 0.03857 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| latent-control | -3.96% | -1.09% | -11.55% | -3.91% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| latent-control | -2.50% | +0.16% | +0.72% | +1.38% |

Decision:

- Do not promote the hand-built latent-control stream.
- The direct residual stream is codec-correct, but residual selection by raw
  latent error magnitude is not perceptually efficient enough to pay for the
  counted stream.

Research conclusion:

- "Send a small residual/control stream" is not sufficient by itself.
- The stream must be selected by a learned safe-coarsening / benefit-cost
  teacher, not by raw residual magnitude or hand-crafted top-k.
- Next step should train a teacher/RDO branch that learns where removing `y`
  precision is safe and where residual/control bits actually buy DISTS/LPIPS/FID
  benefit.

### 2026-06-30 JST - Trainable latent residual-control encoder

I implemented a trainable encoder-only signed latent-control stream.

Implementation:

- `StageLatentControlEncoder` in `gp_reslc/prior_predictor.py`
  - input:
    source-side `y` and decoder-available common prior features
  - output:
    sparse signed ternary symbols at `z_hat` resolution, grouped by GLC
    four-part stage and channel group
  - code length:
    fixed-prior ternary bits included in the training rate term
- `stage_residual_entropy_quant_gate_latent_control` mode in
  `train_forward`
  - first decodes the current stage residual entropy gate latent
  - then applies signed latent corrections before the synthesis transform
- real codec:
  - loads `latent_control_encoder` from checkpoint
  - serializes one signed ternary control stream plus the four `y` streams
  - counts control bytes in real bpp.

Smoke:

- run:
  `experiments/stage_residual_entropy_quant_gate_latent_control_from_sb03_smoke`
- W&B:
  `latent_control_smoke_from_sb03` (`iocjaffy`)
- result:
  10-iteration smoke passed. Trainable params were about `1.49M`.

Medium diagnostic:

- run:
  `experiments/stage_residual_entropy_quant_gate_latent_control_from_sb03_12k`
- W&B:
  `latent_control_from_sb03_12k` (`4f0dt1kl`)
- init:
  current scale-bound lead
- trained:
  latent-control encoder only
- frozen:
  pretrained GLC, stage residual entropy predictor, stage quant gate, q embedding,
  perceptual gate
- stopped:
  after `v2_2000.pt`

Reason for early stop:

- A/B PSNR on the fixed validation panel did not move from the initial state.
- The hard top-k stream kept the same nonzero rate; the network learned rankings
  but did not produce enough reconstruction gain to justify continuing the
  latent-control-only stage.

Real-codec Kodak8:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_latent_control_from_sb03_12k/v2_2000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_learned_2000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_learned_2000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_learned_2000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_learned_2000_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| latent-control learned | 0.02393 | 0.02818 | 0.03358 | 0.03853 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| latent-control learned | -4.11% | -1.21% | -11.68% | -4.13% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| latent-control learned | -2.72% | -0.02% | +0.60% | +1.10% |

Decision:

- Do not promote latent-control-only.
- It slightly improves some point metrics over hand-selected latent control, but
  not enough to pay for the counted stream.

Mainline implication:

- The paid residual/control stream should be trained jointly with a more
  aggressive no-send stage gate. Used alone, it is mostly an expensive quality
  repair; paired with additional y-stream reduction, it can test the actual
  GP-ResLC thesis: coarsen generator-recoverable detail and spend a tiny counted
  stream on the unpredictable residual.

### 2026-06-30 JST - Joint latent-control + stage gate check

I trained the signed latent-control stream jointly with the stage quant gate to
test whether the paid residual/control stream can justify itself by enabling a
shorter `y` payload.

Run:

- `experiments/stage_residual_entropy_quant_gate_latent_control_joint_from_lc2000_6k`
- W&B:
  `latent_control_joint_from_lc2000_6k` (`0ow87ouz`)
- init:
  `experiments/stage_residual_entropy_quant_gate_latent_control_from_sb03_12k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate_latent_control`
- trained:
  stage quant gate + latent-control encoder
- frozen:
  pretrained GLC, stage residual entropy predictor, q embedding, perceptual gate
- stopped:
  after `v2_2000.pt`

Real-codec Kodak8:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_latent_control_joint_from_lc2000_6k/v2_2000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_joint_2000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_joint_2000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_joint_2000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_joint_2000_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| latent-control joint | 0.02661 | 0.02926 | 0.03269 | 0.03669 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| latent-control joint | -3.89% | +0.55% | -13.73% | -8.01% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| latent-control joint | -2.46% | +1.04% | +1.47% | +1.54% |

Decision:

- Do not promote the joint latent-control checkpoint.
- It reduces counted bpp at q2/q3 even after adding the control stream, but
  q0/q1 move right and DISTS/LPIPS are weaker than the current lead.
- FID/KID improve in BD-rate, which suggests that sparse latent-control can
  help distributional realism, but the current fixed top-k stream is not a
  reliable rate-quality allocation mechanism.

Implementation correction:

- Added `latent_control_hard_mode={topk,threshold}` and
  `latent_control_threshold` to `StageLatentControlEncoder`.
- For threshold-mode latent control, the training rate term now uses expected
  fixed-prior ternary code length from the nonzero probability, while real
  evaluation still serializes the actual hard symbols and counts payload bytes.
- Rationale: fixed top-k learns only the ranking of control symbols; it cannot
  learn how many residual/control bits to spend. Threshold mode lets the model
  trade control payload against DISTS/LPIPS/FID benefit under the real codec.

Next:

- Run a threshold-mode latent-control branch focused on q2/q3, where the joint
  fixed-top-k experiment already showed counted bpp reduction. Promote only if
  it improves DISTS/LPIPS matched-bpp or BD-rate relative to the current lead.

### 2026-06-30 JST - Threshold/top-k latent-control teacher checks

I corrected and tested the paid latent-control training path.

Implementation fixes:

- `StageLatentControlEncoder` now supports `hard_mode={topk,threshold}`.
- Threshold mode uses a differentiable expected ternary code length during
  training and actual hard-symbol serialization during real-codec evaluation.
- The sparse control teacher was changed to build top-k targets over the full
  stage/channel/spatial control tensor. The previous target was effectively
  empty on 256 crops because the `z_hat` spatial grid is very small.

Threshold-mode probes:

- `experiments/stage_residual_entropy_quant_gate_latent_control_threshold_q23_from_sb03_4k`
- `experiments/stage_residual_entropy_quant_gate_latent_control_threshold02_teacher5_q23_probe`

Outcome:

- Without a control teacher, probabilities collapse to zero.
- With a strong teacher, probabilities cross the threshold but hard nonzero
  symbols jump to about `12.5%` on 256-crop batches.
- This is too much payload for the intended tiny residual/control stream.

Decision:

- Do not continue threshold-mode latent-control as currently implemented.
- It needs either channel/group-specific sparse targets, entropy-calibrated
  logits, or an explicit learned entropy model before it can be a reliable
  variable-rate control stream.

Top-k teacher branch:

- run:
  `experiments/stage_residual_entropy_quant_gate_latent_control_topk0025_teacher_q23_from_sb03_2500`
- W&B:
  `latent_control_topk0025_teacher_q23_from_sb03_2500` (`6haknuql`)
- checkpoint evaluated:
  `v2_1000.pt`
- output:
  `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_teacher_q23_1000`
- summaries:
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_teacher_q23_1000_metrics.csv`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_teacher_q23_1000_bd.md`
  - `experiments/real_codec/kodak8_stage_residual_entropy_quant_gate_latent_control_topk0025_teacher_q23_1000_matched.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| top-k teacher | 0.02422 | 0.02817 | 0.03277 | 0.03705 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| top-k teacher | -2.10% | -1.29% | -11.39% | -4.20% |

Matched-metric bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| top-k teacher | -2.60% | +1.90% | +0.69% | +1.97% |

Decision:

- Do not promote top-k teacher latent-control.
- The tiny counted budget is stable and codec-correct, and q2/q3 bpp move
  slightly left, but DISTS/LPIPS/FID degrade relative to the current lead.
- The issue is not only where to send control; fixed signed latent corrections
  do not buy enough reconstruction/perceptual benefit for their payload.

Mainline implication:

- A future paid stream should carry a residual variable with magnitude or a
  learned entropy-parameter/control symbol, not only a fixed-magnitude sign.
- Until that is implemented, the zero-side current lead remains stronger.

### 2026-06-30 JST - Stage residual entropy normalized-prediction check

I also tested whether the stage-aware residual entropy predictor can be pushed
closer to the target residual by adding the normalized residual-prediction loss.

Run:

- `experiments/stage_residual_entropy_normpred_q23_from_sb03_2000`
- W&B:
  `stage_residual_entropy_normpred_q23_from_sb03_2000` (`x5e540mz`)
- init:
  current scale-bound lead
- trained:
  stage residual entropy predictor
- frozen:
  pretrained GLC, q embedding, stage quant gate, perceptual gate
- q choices:
  `2 3`
- extra loss:
  `lambda_stage_mean_norm=0.35`

Outcome:

- Stopped manually around 900 iterations.
- `stage_abs` stayed around `0.003` while target residual magnitude stayed
  around `0.021-0.025`.
- The normalized loss did not make the stage predictor carry substantially more
  residual structure, and the bpp/quality logs did not show a promising trend.

Decision:

- Do not evaluate or promote this checkpoint.
- Simply adding residual-prediction supervision is not enough. The stage-aware
  residual-variable branch likely needs a stronger architectural change:
  residual symbolization, magnitude coding, or a dedicated residual entropy
  model rather than a small mean-delta correction alone.

### 2026-06-30 JST - Source-side multi-level latent-control upper-bound check

I completed the real-codec plumbing for counted multi-level signed control
symbols:

- `scripts/evaluate_real_codec.py` now accepts
  `--residual_control_levels`.
- `gp_reslc/real_codec.py` now serializes signed control symbols in
  `[-levels, +levels]` with torchac and decodes them before applying the latent
  correction.
- Source-side latent controls can quantize the pooled residual as
  `round(residual / delta)` instead of only sending a fixed `{-1,0,+1}` sign.
- `py_compile` passed for `gp_reslc/real_codec.py`,
  `scripts/evaluate_real_codec.py`, `scripts/train_v2.py`, and
  `gp_reslc/prior_predictor.py`.

Smoke:

- checkpoint:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate_latent_control`
- output:
  `experiments/real_codec/smoke_latent_control_multilevel_l2_d005`
- setting:
  `topk=0.0025`, `groups=16`, `levels=2`, `delta=0.05`, q3, one Kodak image
- result:
  real decode succeeded; control payload was `5` bytes
  (`0.00010 bpp`) on `kodim01`.

Kodak8 q2/q3 source-side probes:

| run | q2 bpp | q2 DISTS | q2 LPIPS | q2 FID | q3 bpp | q3 DISTS | q3 LPIPS | q3 FID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLC | 0.03572 | 0.10171 | 0.17060 | 53.77 | 0.04001 | 0.09862 | 0.16420 | 53.39 |
| current lead | 0.03299 | 0.10380 | 0.17558 | 54.77 | 0.03793 | 0.09827 | 0.16580 | 53.42 |
| levels2/delta0.05 | 0.03324 | 0.10406 | 0.17537 | 54.78 | 0.03818 | 0.09816 | 0.16596 | 53.35 |
| levels2/delta0.10 | 0.03322 | 0.10407 | 0.17538 | 54.77 | 0.03816 | 0.09816 | 0.16595 | 53.35 |
| levels3/delta0.02 | 0.03369 | 0.10404 | 0.17531 | 54.78 | 0.03863 | 0.09811 | 0.16592 | 53.31 |

Artifacts:

- `experiments/real_codec/kodak8_latent_control_source_multilevel_l2_d005`
- `experiments/real_codec/kodak8_latent_control_source_multilevel_l2_d010`
- `experiments/real_codec/kodak8_latent_control_source_multilevel_l3_d002`
- metric CSVs:
  - `experiments/real_codec/kodak8_latent_control_source_multilevel_l2_q23_metrics.csv`
  - `experiments/real_codec/kodak8_latent_control_source_multilevel_l3d002_q23_metrics.csv`

Interpretation:

- Multi-level source-side control is codec-correct, but it does not beat the
  current lead on the q2/q3 local curve.
- q3 DISTS/FID improve slightly when a paid control stream is added, but bpp
  also increases; q2 gets worse on DISTS/FID.
- `delta=0.10` produces almost no nonzero symbols, so it is effectively an
  empty control stream.
- `delta=0.02, levels=3` activates a real `~0.00053 bpp` control stream, but
  the quality gain is still too small for the payload.

Decision:

- Do not train the current low-resolution signed mean-control branch further.
- This negative result narrows the mainline: paying for coarse latent mean
  corrections is not enough.
- The next serious branch should implement residual-variable coding with a
  learned entropy model or a safe-coarsening/RDO teacher that controls the
  existing `y` stream directly, not a standalone low-resolution correction map.

### 2026-06-30 JST - Entropy family check for real y-stream coding

I tested whether changing only the arithmetic-coding distribution family can
reduce real serialized bpp without changing reconstruction. This is a codec
correctness and entropy-modeling check: the same quantized symbols are decoded,
but torchac uses Gaussian, Laplace, or Logistic CDFs.

Implementation status:

- `gp_reslc/real_codec.py` supports `entropy_family={gaussian,laplace,logistic}`.
- `scripts/evaluate_real_codec.py` passes the selected family into both
  compress and decompress.

Kodak8 average bpp:

| method | family | q0 | q1 | q2 | q3 |
|---|---|---:|---:|---:|---:|
| GLC | Gaussian | 0.02699 | 0.03104 | 0.03572 | 0.04001 |
| GLC | Laplace | 0.02997 | 0.03419 | 0.03891 | 0.04320 |
| GLC | Logistic | 0.02788 | 0.03196 | 0.03663 | 0.04090 |
| current lead | Gaussian | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| current lead | Laplace | 0.02604 | 0.03054 | 0.03613 | 0.04117 |
| current lead | Logistic | 0.02415 | 0.02847 | 0.03388 | 0.03883 |

Artifacts:

- `experiments/real_codec/kodak8_glc_laplace_allq`
- `experiments/real_codec/kodak8_current_laplace_allq`
- `experiments/real_codec/kodak8_glc_logistic_allq`
- `experiments/real_codec/kodak8_current_logistic_allq`

Decision:

- Keep Gaussian for the current real-codec path.
- The symbol residuals are not helped by a heavier-tailed fixed Laplace or
  Logistic CDF under the current scale parameters.
- The next entropy-modeling direction should be Gaussian scale/mean calibration
  or a learned residual entropy model, not a fixed family swap.

### 2026-06-30 JST - Global Gaussian scale calibration check

I added an optional `entropy_scale_factor` to the real codec:

- It multiplies the Gaussian/Laplace/Logistic CDF scale used by torchac.
- It is applied symmetrically during compress and decompress.
- It does not change quantized symbols or reconstruction, only serialized y
  stream length.
- Default is `1.0`, preserving all existing evaluations.

Implementation:

- `gp_reslc/real_codec.py`
- `scripts/evaluate_real_codec.py --entropy_scale_factor`
- `py_compile` passed.

Current-lead Kodak8 average bpp:

| scale | q0 | q1 | q2 | q3 |
|---:|---:|---:|---:|---:|
| 1.0 | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| 0.9 | 0.02340 | 0.02780 | 0.03329 | 0.03829 |
| 1.1 | 0.02361 | 0.02776 | 0.03312 | 0.03802 |

Artifacts:

- `experiments/real_codec/smoke_current_entropy_scale080`
- `experiments/real_codec/smoke_current_entropy_scale120`
- `experiments/real_codec/kodak8_current_entropy_scale090_allq`
- `experiments/real_codec/kodak8_current_entropy_scale110_allq`

Decision:

- Do not use a global entropy-scale multiplier as a method component.
- The current Gaussian scale is already close to the real-codec optimum under a
  single scalar multiplier.
- A useful residual entropy model must be local/stage/channel conditioned, not a
  global scale calibration.

### 2026-06-30 JST - Launch plan: safe-RDO stage gate refinement

The checks above ruled out three weak directions:

1. low-resolution signed mean-control stream,
2. fixed Laplace/Logistic entropy-family swap,
3. global Gaussian scale calibration.

Next mainline candidate:

```text
 current scale-bound stage residual/entropy lead
+ freeze stage residual predictor and q embedding
+ train only the decoder-computable stage quant gate
+ supervise gate placement with mixed local damage + rate-potential teacher
+ keep all bits in the existing y stream; no side map
+ evaluate by real codec
```

Rationale:

- The current lead already proves that decoder-computable residual precision
  suppression reduces real bpp.
- Prior diagnostics showed the remaining failure is placement: high-rho regions
  can overlap with high-error/high-gradient/high-texture locations.
- The mixed teacher directly asks where coarsening is locally safe, and the
  rate term asks whether coarsening there is worth bits.
- Freezing the stage residual predictor avoids destabilizing the learned
  entropy/mean path while the gate placement is repaired.

Planned run:

- output:
  `experiments/stage_residual_entropy_quant_gate_safe_rdo_gateonly_from_sb03_8k`
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- data:
  `/dpl/open-images-v6/train`
- validation:
  `/dpl/kodak`
- W&B:
  `safe_rdo_gateonly_from_sb03_8k`

Promotion condition:

- First check Kodak real-codec q0-q3.
- Promote to DIV2K/CLIC only if it improves the current lead, not merely local
  training loss.

### 2026-06-30 JST - Safe-RDO gate-only 2000-step result

Run:

- training:
  `experiments/stage_residual_entropy_quant_gate_safe_rdo_gateonly_from_sb03_8k`
- W&B:
  `safe_rdo_gateonly_from_sb03_8k` (`vz008xfj`)
- evaluated checkpoint:
  `v2_2000.pt`
- real codec:
  `experiments/real_codec/kodak8_safe_rdo_gateonly_from_sb03_2000`
- metrics:
  `experiments/real_codec/kodak8_safe_rdo_gateonly_from_sb03_2000_metrics.csv`

Kodak8 real-codec point metrics:

| run | q0 bpp/DISTS/FID | q1 bpp/DISTS/FID | q2 bpp/DISTS/FID | q3 bpp/DISTS/FID |
|---|---:|---:|---:|---:|
| current | 0.02334 / 0.12641 / 64.91 | 0.02760 / 0.11177 / 58.92 | 0.03299 / 0.10380 / 54.77 | 0.03793 / 0.09827 / 53.42 |
| safe-RDO | 0.02478 / 0.12252 / 61.71 | 0.02882 / 0.10961 / 57.20 | 0.03381 / 0.10160 / 54.39 | 0.03847 / 0.09802 / 52.97 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| safe-RDO | -5.52% | -1.51% | -14.34% | -6.45% |

BD-rate versus current lead:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| safe-RDO | +1.20% | +0.30% | -1.66% | -1.33% |

Matched-quality bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| safe-RDO | -5.12% | -3.47% | +0.10% | -1.99% |

Interpretation:

- The mixed damage + rate-potential teacher improves the pointwise quality at
  every q and gives stronger FID/KID than the current lead.
- It gives back too much rate, so DISTS/LPIPS BD-rate are weaker than the
  current lead.
- This is a useful mechanism result, not a new lead.

Decision:

- Do not promote `v2_2000.pt` as the main checkpoint.
- Continue with a rate-budgeted safe-RDO continuation: keep the improved
  placement, increase rate pressure and rho targets slightly, and evaluate again
  after a short continuation.

### 2026-06-30 JST - Rate-budgeted safe-RDO continuation result

Run:

- training:
  `experiments/stage_residual_entropy_quant_gate_safe_rdo_budget_from_safe2000_4k`
- W&B:
  `safe_rdo_budget_from_safe2000_4k` (`as78pb1c`)
- init:
  safe-RDO `v2_2000.pt`
- evaluated checkpoint:
  `v2_1000.pt`
- real codec:
  `experiments/real_codec/kodak8_safe_rdo_budget_from_safe2000_1000`
- metrics:
  `experiments/real_codec/kodak8_safe_rdo_budget_from_safe2000_1000_metrics.csv`

Kodak8 real-codec point metrics:

| run | q0 bpp/DISTS/FID | q1 bpp/DISTS/FID | q2 bpp/DISTS/FID | q3 bpp/DISTS/FID |
|---|---:|---:|---:|---:|
| current | 0.02334 / 0.12641 / 64.91 | 0.02760 / 0.11177 / 58.92 | 0.03299 / 0.10380 / 54.77 | 0.03793 / 0.09827 / 53.42 |
| safe-budget | 0.02408 / 0.12209 / 63.59 | 0.02829 / 0.11141 / 57.85 | 0.03359 / 0.10217 / 54.86 | 0.03832 / 0.09758 / 53.29 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| safe-budget | -5.50% | -1.60% | -13.25% | -6.16% |

BD-rate versus current lead:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| safe-budget | +0.69% | -0.57% | -1.33% | +0.15% |

Matched-quality bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| safe-budget | -5.35% | -1.16% | -0.07% | +0.50% |

Interpretation:

- Rate-budgeting recovered some bpp while preserving much of the safe-RDO
  quality repair.
- It still does not replace the current lead on the primary DISTS BD-rate.
- The branch is useful evidence that local safe/RDO supervision improves
  placement, but gate-only refinement is saturating.

Decision:

- Do not promote `safe-budget v2_1000.pt`.
- Stop gate-only safe-RDO continuation for now.
- Next mainline work should move the coded variable or entropy model itself:
  stage-aware residual-variable coding, local residual scale/mean modeling, or
  a real learned residual entropy model.

### 2026-06-30 JST - Safe-weighted stage residual mean predictor wiring

Motivation:

- The previous `lambda_mean_pred_safe` path was not fully aligned with the
  mainline GP-ResLC story for stage residual modes.  It used the averaged
  stage residual prediction loss, so the mixed safe/RDO teacher did not
  spatially select where the decoder-computable residual mean should be
  learned.
- This made the branch too close to generic residual/quantization tuning.  The
  intended mainline is more specific: learn `gp_mu_stage` mainly where the
  generator/decoder can safely absorb residual precision loss, and suppress
  predictor deltas in unsafe regions.

Code change:

- `gp_reslc/prior_predictor.py`
  - `forward_four_part_prior_with_stage_residual_quant_gate`
    and the counted-control variant now return full-size
    `stage_delta_map` and `stage_target_map` in addition to scalar diagnostics.
- `scripts/train_v2.py`
  - stage residual modes now use
    `weighted_spatial_smooth_l1(stage_delta_map, stage_target_map, safe_weight)`
    for `lambda_mean_pred_safe`.
  - `lambda_predictor_unsafe_delta` now also works for stage residual modes by
    penalizing `stage_delta_map` in mixed-teacher unsafe regions.
  - `stage_residual_entropy_quant_gate_latent_control` is treated consistently
    as a stage residual mode in these losses.

Smoke check:

- Command: one iteration from
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
  with `lambda_mean_pred_safe_by_q=1` and mixed teacher active.
- Result: passed forward/backward.  Logged `safe_mean=0.0061`,
  `unsafe_delta=0.00158`, and normal stage residual diagnostics.

Next experiment:

- Start from the current lead checkpoint and train a full stage residual +
  quant gate continuation with safe-weighted residual mean supervision active.
- Primary comparison remains Kodak8 real codec at q0-q3 against:
  local GLC baseline and the current lead.

### 2026-06-30 JST - Safe-weighted stage residual mean result

Run:

- training:
  `experiments/stage_residual_entropy_quant_gate_safe_weighted_mean_from_sb03_6k`
- W&B:
  `safe_weighted_stage_mean_from_sb03_6k` (`y0prf3hx`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- evaluated checkpoint:
  `v2_1000.pt`
- real codec:
  `experiments/real_codec/kodak8_safe_weighted_stage_mean_from_sb03_1000`
- metrics:
  `experiments/real_codec/kodak8_safe_weighted_stage_mean_from_sb03_1000_metrics.csv`

Kodak8 real-codec point metrics:

| run | q0 bpp/DISTS/FID | q1 bpp/DISTS/FID | q2 bpp/DISTS/FID | q3 bpp/DISTS/FID |
|---|---:|---:|---:|---:|
| current | 0.02334 / 0.12641 / 64.91 | 0.02760 / 0.11177 / 58.92 | 0.03299 / 0.10380 / 54.77 | 0.03793 / 0.09827 / 53.42 |
| safe-weighted | 0.02436 / 0.12246 / 62.61 | 0.02844 / 0.11191 / 56.62 | 0.03353 / 0.10188 / 54.61 | 0.03811 / 0.09945 / 52.92 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| safe-weighted | -4.74% | -2.16% | -18.84% | -16.61% |

BD-rate versus current lead:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| safe-weighted | +2.15% | -0.33% | -7.18% | -12.71% |

Matched-quality bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| safe-weighted | -4.30% | -4.43% | -0.76% | -4.31% |

Interpretation:

- The new safe-weighted wiring works technically and gives a clear naturalness
  repair: FID/KID improve substantially over the current lead.
- It does not replace the current lead because the primary DISTS curve is
  weaker and the rate increase is not compensated at q1/q3.
- This suggests that safe/RDO supervision is useful, but simply attaching it to
  the stage mean predictor still behaves like quality repair.  Larger gains
  likely require moving the coded variable/entropy model itself rather than
  more gate or residual-mean tuning.

Decision:

- Do not promote `safe-weighted v2_1000.pt`.
- Keep the code change because it is the correct mainline wiring for future
  residual predictor training.
- Next: implement or strengthen stage-aware residual-variable coding / learned
  residual entropy modeling so the bitstream itself carries a better centered
  variable, not only a safer precision gate.

### 2026-06-30 JST - OpenImages split alignment note

Dataset check:

- `/dpl/open-images-v6/train`: 300,000 images
- `/dpl/open-images-v6/test`: 125,436 images
- `/dpl/open-images-v6/validation`: 41,620 images

Interpretation:

- The 125K-image OpenImages split matches the reported GLC Stage II/III
  training split better than the 300K train split used in our exploratory runs.
- This is not final-evaluation leakage because the final natural-image
  evaluation sets remain CLIC2020 test, DIV2K validation, and Kodak.  It is a
  distribution/protocol alignment issue with the GLC pretrained baseline.
- The current OpenImages-train runs remain useful as exploration, but
  paper-facing or mainline confirmation runs should be repeated on
  `/dpl/open-images-v6/test/data`.

Decision:

- Switch upcoming mainline training to `/dpl/open-images-v6/test/data`.
- Keep documenting whether a run used OpenImages-train exploration or
  OpenImages-test GLC-aligned training.

### 2026-06-30 JST - DIV2K transfer check for refine-only checkpoint

Reason:

- `stage_residual_entropy_refine_only_from_sb03_10k/v2_1000.pt` was rejected on
  Kodak8 as a lead because it weakened FID/KID/LPIPS, but it slightly improved
  DISTS BD-rate.  I evaluated it on DIV2K before discarding the branch.

Run:

- checkpoint:
  `experiments/stage_residual_entropy_refine_only_from_sb03_10k/v2_1000.pt`
- real codec:
  `experiments/real_codec/div2k_stage_residual_entropy_refine_only_1000`
- metrics:
  `experiments/real_codec/div2k_stage_residual_entropy_refine_only_1000_metrics.csv`

DIV2K real-codec point bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| Current lead | 0.0204 | 0.0244 | 0.0297 | 0.0344 |
| Refine-only | 0.0204 | 0.0245 | 0.0297 | 0.0344 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Current lead | -9.20% | -0.05% | -5.13% | -8.85% |
| Refine-only | -9.24% | -0.03% | -5.24% | -2.22% |

BD-rate versus current lead:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Refine-only | -0.11% | -0.01% | -0.56% | -7.75% |

Matched-quality bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| Current lead | -8.03% | -1.58% | +1.10% | +1.38% |
| Refine-only | -8.06% | -1.78% | +1.11% | -0.81% |

Interpretation:

- Refine-only transfers without collapsing and is essentially tied with the
  current lead on DIV2K.
- The improvement is too small to replace the lead by itself.
- It is still useful evidence that residual entropy predictor refinement is a
  stable component and can be included in a stronger OpenImages-test-aligned
  continuation.

Decision:

- Do not promote refine-only as a new lead.
- Use the result to justify the next mainline run: repeat the current strongest
  stage residual entropy + quant gate training on the GLC-aligned
  OpenImages-test split, with residual entropy refinement kept in the model.

### 2026-06-30 JST - Start OpenImages-test-aligned mainline continuation

Hypothesis:

- Some of the remaining gap to the official GLC curve may come from training
  GP-ResLC adapters on `/dpl/open-images-v6/train` while GLC Stage II/III used
  an OpenImages test split.  The split is not an evaluation set in our protocol;
  it is the GLC training distribution to match.
- Re-aligning the mainline GP-ResLC modules to the 125K OpenImages-test images
  should reduce dataset-distribution mismatch without changing the GLC base.

Run plan:

- data:
  `/dpl/open-images-v6/test/data`
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate`
- trainable:
  stage residual entropy predictor + stage quant gate
- frozen:
  pretrained GLC backbone, q embedding, global perceptual gate
- goal:
  improve the real serialized bpp versus perceptual-quality curve without
  adding side information.
- first decision checkpoints:
  evaluate `v2_1000.pt` and `v2_2000.pt` on Kodak8 real codec, then DIV2K if
  it beats or ties the current lead.

This is not a rho/alpha sweep.  It is a data-protocol aligned continuation of
the current strongest mainline residual entropy + stage precision model.

### 2026-06-30 JST - OpenImages-test-aligned v2_2000 Kodak8 result

Run:

- training:
  `experiments/stage_residual_entropy_quant_gate_oitest_aligned_from_sb03_12k`
- W&B:
  `oitest_aligned_stage_res_entropy_gate_from_sb03_12k` (`u6pspfhg`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- data:
  `/dpl/open-images-v6/test/data`
- evaluated checkpoint:
  `v2_2000.pt`
- real codec:
  `experiments/real_codec/kodak8_oitest_aligned_stage_res_entropy_gate_2000`
- metrics:
  `experiments/real_codec/kodak8_oitest_aligned_stage_res_entropy_gate_2000_metrics.csv`

Kodak8 real-codec point metrics:

| run | q0 bpp/DISTS/FID | q1 bpp/DISTS/FID | q2 bpp/DISTS/FID | q3 bpp/DISTS/FID |
|---|---:|---:|---:|---:|
| Current | 0.02334 / 0.12641 / 64.91 | 0.02760 / 0.11177 / 58.92 | 0.03299 / 0.10380 / 54.77 | 0.03793 / 0.09827 / 53.42 |
| OITestAligned | 0.02449 / 0.12367 / 61.84 | 0.02857 / 0.11057 / 56.81 | 0.03363 / 0.10179 / 54.33 | 0.03829 / 0.09704 / 52.52 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Current lead | -5.95% | -2.67% | -13.22% | -5.52% |
| OITestAligned | -5.65% | -2.05% | -16.92% | -11.35% |

BD-rate versus current lead:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| OITestAligned | +1.07% | +0.21% | -4.62% | -6.57% |

Matched-quality bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| Current lead | -4.52% | -0.06% | -0.97% | -0.71% |
| OITestAligned | -5.45% | -5.49% | -0.74% | -3.74% |

Interpretation:

- Training on the GLC-aligned OpenImages-test split repairs perceptual quality
  at all four q points and strongly improves FID/KID.
- It gives back some rate, so DISTS BD-rate versus GLC is slightly weaker than
  the current lead, even though matched-DISTS versus GLC improves.
- The branch is promising but not promoted from Kodak8 alone.  It needs DIV2K
  validation to decide whether the quality repair generalizes or is just a
  small-set effect.

Decision:

- Evaluate `v2_2000.pt` on DIV2K validation real codec.
- If DIV2K DISTS also improves or ties while FID/KID improve, consider this the
  new mainline seed.  If not, keep it as a quality-repair branch and switch to
  a DISTS/rate-focused residual entropy design.

### 2026-06-30 JST - OpenImages-test-aligned v2_2000 DIV2K result

Run:

- evaluated checkpoint:
  `experiments/stage_residual_entropy_quant_gate_oitest_aligned_from_sb03_12k/v2_2000.pt`
- real codec:
  `experiments/real_codec/div2k_oitest_aligned_stage_res_entropy_gate_2000`
- metrics:
  `experiments/real_codec/div2k_oitest_aligned_stage_res_entropy_gate_2000_metrics.csv`
- BD summaries:
  `experiments/real_codec/div2k_oitest_aligned_stage_res_entropy_gate_2000_bd_vs_glc.md`,
  `experiments/real_codec/div2k_oitest_aligned_stage_res_entropy_gate_2000_bd_vs_current.md`

DIV2K real-codec point metrics:

| run | q0 bpp/DISTS/FID | q1 bpp/DISTS/FID | q2 bpp/DISTS/FID | q3 bpp/DISTS/FID |
|---|---:|---:|---:|---:|
| GLC | 0.0238 / 0.0905 / 14.3714 | 0.0276 / 0.0835 / 13.1837 | 0.0322 / 0.0780 / 12.1258 | 0.0365 / 0.0756 / 11.8415 |
| Current lead | 0.0204 / 0.0944 / 16.5453 | 0.0244 / 0.0839 / 13.8078 | 0.0297 / 0.0779 / 12.3473 | 0.0344 / 0.0756 / 11.8983 |
| OITestAligned | 0.0214 / 0.0915 / 15.3014 | 0.0253 / 0.0828 / 13.3414 | 0.0303 / 0.0774 / 12.2784 | 0.0348 / 0.0755 / 11.7975 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Current lead | -9.20% | -0.05% | -5.13% | -8.85% |
| OITestAligned | -8.36% | -0.73% | -5.39% | -3.94% |

BD-rate versus current lead:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| OITestAligned | +0.75% | -0.81% | -2.04% | -6.84% |

Matched-quality bpp versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| Current lead | -8.03% | -1.58% | +1.10% | +1.38% |
| OITestAligned | -7.73% | -3.86% | +0.37% | -2.52% |

Interpretation:

- OpenImages-test alignment consistently repairs quality: all four q points have
  better DISTS/LPIPS/FID than the current lead.
- The repair costs bitrate.  On DIV2K, the DISTS BD-rate versus GLC weakens
  from -9.20% to -8.36%, and the branch is +0.75% worse than the current lead
  on DISTS BD.
- FID and KID improve versus the current lead, but the main research objective
  is a clear left shift of the real-codec perceptual curve, not just quality
  repair at slightly higher bpp.

Decision:

- Do not promote OITestAligned as the new lead.
- Keep it as a quality-repair branch and possible seed for FID/KID-oriented
  tuning.
- Mainline should now move beyond rho/quality-weight tuning into a
  DISTS/rate-focused residual entropy design: decoder-computable
  q/stage-conditioned residual distribution calibration, residual-control
  entropy modeling, or stage-aware residual-variable coding with exact real
  codec accounting.

### 2026-06-30 JST - Implement stage scale calibrator branch

Motivation:

- OITestAligned repairs quality but gives back too much bitrate, so it is not a
  new lead.
- Paid control streams have been codec-correct but have not yet paid for their
  payload.
- The strongest next zero-side direction is to improve the residual entropy
  model itself: keep the current decoded symbols and reconstruction, but make
  the decoder-computable Gaussian CDF tighter for the actual residual symbols.

Implementation:

- Added `StageScaleCalibrator` in `gp_reslc/prior_predictor.py`.
- New mode:
  `stage_residual_entropy_quant_gate_scale_calib`.
- The calibrator is four-part-stage aware and uses the same decoder-available
  signals as GLC's stage prior:
  - stage 0: reduced hyperprior parameters,
  - stages 1-3: reduced hyperprior parameters plus decoded `y_hat_so_far`.
- It outputs only a bounded scale multiplier, initialized to exactly `1.0`.
- It does not change the mean, quantization step, decoded symbols, or
  reconstruction; it only changes `scales_hat` used by likelihood/real
  arithmetic coding.
- It is serialized in checkpoints as `stage_scale_calibrator` and loaded by
  `scripts/evaluate_real_codec.py`.
- The real codec applies the same calibrator in both compress and decompress.

Smoke:

- training smoke:
  `experiments/smoke_stage_scale_calibrator_from_sb03`
- W&B:
  `smoke_stage_scale_calibrator_from_sb03_rerun` (`hkliki23`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- result:
  10 iterations passed; checkpoint save/load path works.
- real-codec smoke:
  `experiments/real_codec/smoke_stage_scale_calibrator_from_sb03`
- q3, first 2 Kodak8 images:
  real decode matched train/eval forward with `max_abs=0.000e+00`.

Decision:

- Launch a full zero-side entropy-scale calibration run from the current lead.
- Train only `StageScaleCalibrator`; keep the pretrained GLC backbone, current
  stage residual entropy predictor, stage quant gate, and q embedding frozen.
- Promote only if actual serialized y-stream bpp decreases on Kodak/DIV2K
  without changing reconstruction or breaking real-codec consistency.

### 2026-06-30 JST - Stage scale calibrator 1k result

Run:

- training:
  `experiments/stage_scale_calibrator_from_sb03_1k`
- W&B:
  `stage_scale_calibrator_from_sb03_1k` (`4f87u6sb`)
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- mode:
  `stage_residual_entropy_quant_gate_scale_calib`
- trained:
  `StageScaleCalibrator` only
- frozen:
  pretrained GLC, stage residual entropy predictor, stage quant gate, q embedding,
  global perceptual gate disabled as in the current lead path.
- note:
  the first 12k launch failed at iteration 0 because W&B image logging timed
  out. The actual 1k chunk was run without validation images and completed.

Real-codec Kodak8:

- output:
  `experiments/real_codec/kodak8_stage_scale_calibrator_from_sb03_1k`
- consistency:
  `--check_estimated_consistency` passed for all q/images with
  `max_abs=0.000e+00`.

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| Current lead | 0.02334 | 0.02760 | 0.03299 | 0.03793 |
| ScaleCalib 1k | 0.02333 | 0.02761 | 0.03299 | 0.03792 |

Interpretation:

- The implementation is correct, but the actual payload gain after 1k is
  negligible on Kodak8.
- Because the calibrator only changes entropy scales, reconstruction quality is
  expected to be identical to the current lead; the decision is based on real
  bpp.
- This suggests the current stage residual entropy scale is already near the
  local optimum under a simple Gaussian CDF, consistent with the previous global
  scale-factor check.

Decision:

- Do not promote ScaleCalib 1k.
- Do not spend a long run on scale-only calibration unless later residual
  symbolization changes the residual distribution.
- Keep the implementation because it is codec-correct and can be reused as a
  small entropy-model component, but move the mainline to residual-variable
  coding / benefit-cost allocation rather than pure scale calibration.

### 2026-06-30 JST - Launch stronger stage residual-variable refinement

Reason:

- Scale-only calibration is codec-correct but too small to move the curve.
- The current best mainline still relies heavily on residual precision
  suppression.
- The original GP-ResLC thesis needs a stronger decoder-computable
  `gp_mu_stage(context)` so that the coded variable becomes a smaller
  unpredictable residual:

```text
y_stage = base_mean_stage + gp_mu_stage(context) + residual_stage
```

Run plan:

- output:
  `experiments/stage_residual_variable_refine_from_sb03_1500`
- init:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`
- data:
  `/dpl/open-images-v6/test/data`
- mode:
  `stage_residual_entropy_quant_gate`
- trained:
  stage residual entropy predictor only
- frozen:
  pretrained GLC, stage quant gate, q embedding, global perceptual gate.
- objective:
  rate + DISTS/LPIPS quality + stage residual mean prediction + normalized
  stage residual prediction.

This is not another rho sweep.  It directly strengthens the residual-variable
decomposition while preserving the existing decoder order and real codec path.

### 2026-06-30 JST - Stronger stage residual-variable refine 1500 result

Run:

- training:
  `experiments/stage_residual_variable_refine_from_sb03_1500`
- W&B:
  `stage_residual_variable_refine_from_sb03_1500` (`bjxbf9an`)
- evaluated checkpoint:
  `v2_final.pt`
- real codec:
  `experiments/real_codec/kodak8_stage_residual_variable_refine_from_sb03_1500`
- metrics:
  `experiments/real_codec/kodak8_stage_residual_variable_refine_from_sb03_1500_metrics.csv`

Kodak8 point result:

| run | q0 bpp/DISTS | q1 bpp/DISTS | q2 bpp/DISTS | q3 bpp/DISTS |
|---|---:|---:|---:|---:|
| Current lead | 0.0233 / 0.1264 | 0.0276 / 0.1118 | 0.0330 / 0.1038 | 0.0379 / 0.0983 |
| ResidualRefine | 0.0234 / 0.1273 | 0.0276 / 0.1133 | 0.0330 / 0.1023 | 0.0379 / 0.0992 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID |
|---|---:|---:|---:|
| Current lead | -5.95% | -2.67% | -1.01% |
| ResidualRefine | -5.72% | -1.60% | -1.62% |

Interpretation:

- q2 DISTS improved, but q0/q1/q3 weakened and the curve does not beat the
  current lead.
- Stage residual magnitude stayed far below the residual target during
  training, despite explicit mean-prediction and normalized residual losses.
- This suggests the current shallow stage residual predictor is capacity- or
  conditioning-limited; simply fine-tuning it is not enough.

Decision:

- Do not promote this checkpoint.
- Add a zero-init `StageResidualRefiner` branch stacked on top of the current
  predictor.  This preserves the current lead at initialization while giving a
  deeper additive decoder-computable `gp_mu_stage` path for residual-variable
  coding.

### 2026-06-30 JST - StageResidualRefiner implementation smoke

Implemented a zero-init `StageResidualRefiner` branch for the mainline
residual-variable path.

Purpose:

- Move beyond rho-only precision suppression.
- Add a deeper decoder-computable residual prediction branch on top of the
  current stage residual entropy predictor.
- Preserve baseline/current-lead behavior at initialization by zero-initializing
  the final refinement layers.

Files touched:

- `gp_reslc/prior_predictor.py`
- `gp_reslc/real_codec.py`
- `scripts/train_v2.py`
- `scripts/evaluate_real_codec.py`

Smoke:

- training:
  `experiments/smoke_stage_residual_refiner_from_sb03`
- checkpoint:
  `experiments/smoke_stage_residual_refiner_from_sb03/v2_final.pt`
- real codec:
  `experiments/real_codec/smoke_stage_residual_refiner_from_sb03`
- q:
  `3`
- images:
  first 2 Kodak8 images

Result:

- real codec decode consistency:
  `max_abs=0`
- average bpp:
  `0.03739`
- average encode/decode time:
  `0.284s / 0.114s`

Decision:

- Smoke passed.
- Launch a short but real mainline run from the current best checkpoint.
- Train only the new residual refiner while keeping the current stage residual
  predictor, stage quant gate, q embedding, global gate, and GLC backbone fixed.

### 2026-06-30 JST - StageResidualRefiner 1500 result

Run:

- training:
  `experiments/stage_residual_refiner_from_sb03_1500`
- W&B:
  offline run `bgaw0a8s`
- checkpoint:
  `experiments/stage_residual_refiner_from_sb03_1500/v2_final.pt`
- real codec:
  `experiments/real_codec/kodak8_stage_residual_refiner_from_sb03_1500`
- metrics:
  `experiments/real_codec/kodak8_stage_residual_refiner_from_sb03_1500_compare_metrics.csv`
- BD:
  `experiments/real_codec/kodak8_stage_residual_refiner_from_sb03_1500_compare_bd.md`
- matched-quality:
  `experiments/real_codec/kodak8_stage_residual_refiner_from_sb03_1500_compare_matched.md`

Real codec:

- q0-q3 all passed with `max_abs=0`.
- no control stream was sent (`bpp_control=0`).
- average bpp:
  q0 `0.02341`, q1 `0.02761`, q2 `0.03304`, q3 `0.03789`.

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Current lead | -5.95% | -2.67% | -7.23% | +7.04% |
| Refiner | -4.28% | -1.84% | -6.12% | +145.50% |

Matched-quality bpp delta versus local GLC:

| run | DISTS | LPIPS | FID |
|---|---:|---:|---:|
| Current lead | -4.52% | -0.97% | +1.04% |
| Refiner | -3.42% | -0.48% | +0.95% |

Interpretation:

- The deeper zero-init residual refiner is real-codec compatible, but it does
  not improve the curve.
- It slightly preserves q2/q3 local quality, but q0/q1 weaken and the matched
  DISTS/LPIPS savings are smaller than the current lead.
- This suggests that adding residual-predictor capacity on top of the current
  rho/quant branch is not enough.  The limiting factor is likely allocation:
  the model needs a stronger signal for which residual precision can be safely
  removed and where residual precision must be protected.

Decision:

- Do not promote this checkpoint.
- Keep the implementation as a reusable residual-variable component, but move
  the next mainline experiment to safe-coarsening / residual-RDO training rather
  than further capacity or loss-weight tuning.

### 2026-06-30 JST - Safe-RDO teacher implementation smoke

Implemented a training-only safe residual coarsening / RDO teacher in
`scripts/train_v2.py`.

Core idea:

- Compare frozen GLC reconstruction against the current GP-ResLC reconstruction.
- Compute local perceptual/fidelity damage from L1 and LPIPS-spatial deltas.
- Compute local estimated saved y bits from
  `base_rate_map - ours_rate_map`.
- Train the decoder-computable stage gate toward regions with high saved bits
  and low damage.

This is different from another rho sweep:

- the target is spatial and benefit-cost based;
- the target is recentered to the requested mean, so it reallocates residual
  precision instead of silently changing the global rate knob;
- no target map is sent at inference.

Implementation notes:

- Added `make_gate_rdo_sensitivity_target`.
- Added `--lambda_gate_rdo_sens` and per-q RDO teacher weight options.
- Corrected teacher target mean for stage gates to use `stage_rho_max` rather
  than the global-gate `rho_max`.
- Real codec path is unchanged.

Smoke:

- run:
  `experiments/smoke_stage_safe_rdo_teacher_from_sb03`
- checkpoint:
  `experiments/smoke_stage_safe_rdo_teacher_from_sb03/v2_final.pt`
- base checkpoint:
  `experiments/stage_residual_entropy_quant_gate_scalebound03_from_rate2000_5k/v2_2000.pt`

Result:

- smoke completed without NaN/shape errors.
- RDO teacher active:
  `rdo_sens ~= 0.66`, target mean `~0.62`, target std `~0.27-0.32`.
- local saved-rate map nonzero:
  `rdosave mean ~= 0.002-0.004 bits/symbol`.

Decision:

- Launch a real fine-tune from the current lead.
- Train only the stage quantization gate; keep the stage residual entropy
  predictor, q embedding, global gate, and frozen GLC backbone fixed.

### 2026-06-30 JST - Safe-RDO gate fine-tune 2000 Kodak8 result

Run:

- training:
  `experiments/stage_safe_rdo_gate_from_sb03_2000`
- W&B:
  `stage_safe_rdo_gate_from_sb03_2000` (`5r1hxauf`)
- checkpoint:
  `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`
- real codec:
  `experiments/real_codec/kodak8_stage_safe_rdo_gate_from_sb03_2000`
- metrics:
  `experiments/real_codec/kodak8_stage_safe_rdo_gate_from_sb03_2000_compare_metrics.csv`
- BD:
  `experiments/real_codec/kodak8_stage_safe_rdo_gate_from_sb03_2000_compare_bd.md`
- matched-quality:
  `experiments/real_codec/kodak8_stage_safe_rdo_gate_from_sb03_2000_compare_matched.md`

Real codec:

- q0-q3 all passed with `max_abs=0`.
- no control stream was sent (`bpp_control=0`).
- average bpp:
  q0 `0.02429`, q1 `0.02817`, q2 `0.03279`, q3 `0.03704`.

Point behavior:

- Compared with the current lead, Safe-RDO gives back some low-rate savings
  at q0/q1 but recovers quality.
- At q3 it reduces bpp and slightly improves DISTS.

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Current lead | -5.95% | -2.67% | -7.23% | +7.04% |
| Safe-RDO | -6.00% | -2.97% | -7.38% | -22.04% |

Matched-quality bpp delta versus local GLC:

| run | DISTS | LPIPS | FID |
|---|---:|---:|---:|
| Current lead | -4.52% | -0.97% | +1.04% |
| Safe-RDO | -5.19% | -1.45% | +1.60% |

Interpretation:

- This is a small but meaningful positive result for the mainline teacher:
  safe-RDO allocation improves Kodak8 DISTS and LPIPS curve summaries over the
  current lead while preserving exact serialized decoding.
- FID matched-quality remains unstable on Kodak8 and should not drive the
  decision.
- The next check is DIV2K.  Promote only if the improvement survives a larger
  natural-image set.

Decision:

- Evaluate Safe-RDO on DIV2K real codec.
- If DIV2K confirms DISTS/LPIPS or FID gains, make Safe-RDO the next lead.
- If DIV2K regresses, keep it as a useful teacher variant and tune the RDO
  target by q rather than doing blind rho sweeps.

### 2026-06-30 JST - Safe-RDO gate DIV2K result

Run:

- checkpoint:
  `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`
- real codec:
  `experiments/real_codec/div2k_stage_safe_rdo_gate_from_sb03_2000`
- metrics:
  `experiments/real_codec/div2k_stage_safe_rdo_gate_from_sb03_2000_metrics.csv`
- comparison metrics:
  `experiments/real_codec/div2k_stage_safe_rdo_gate_from_sb03_2000_compare_metrics.csv`
- BD:
  `experiments/real_codec/div2k_stage_safe_rdo_gate_from_sb03_2000_compare_bd.md`
- matched-quality:
  `experiments/real_codec/div2k_stage_safe_rdo_gate_from_sb03_2000_compare_matched.md`

Real codec:

- q0-q3 all passed with `max_abs=0`.
- no control stream was sent (`bpp_control=0`).
- average bpp:
  q0 `0.02107`, q1 `0.02485`, q2 `0.02928`, q3 `0.03348`.

DIV2K point metrics:

| q | bpp | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.02107 | 0.1988 | 0.0927 | 15.6416 | 0.0014 |
| 1 | 0.02485 | 0.1793 | 0.0832 | 13.5240 | 0.0011 |
| 2 | 0.02928 | 0.1652 | 0.0779 | 12.4174 | 0.0008 |
| 3 | 0.03348 | 0.1567 | 0.0749 | 11.8564 | 0.0007 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Previous current lead | -9.20% | -0.05% | -5.13% | -1.23% |
| Safe-RDO | -10.37% | -0.49% | -5.64% | -4.98% |

Matched-quality bpp delta versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Previous current lead | -8.03% | +1.10% | -1.58% | +2.24% |
| Safe-RDO | -9.76% | +0.33% | -3.18% | -3.82% |

Interpretation:

- Safe-RDO is now the best pretrained-GLC mainline checkpoint on DIV2K.
- It improves the previous current lead on DISTS BD, FID BD, KID BD, and
  matched-DISTS bpp saving.
- The only weak point is matched LPIPS, which remains slightly positive
  (`+0.33%`) but is much better than the previous lead (`+1.10%`).
- This is method-faithful evidence that a benefit-cost safe-coarsening teacher
  is better than capacity additions or blind rho/scale tuning.

Decision:

- Promote `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt` to the
  current research lead.
- Next validation target: CLIC2020 test real codec.
- Next improvement direction if CLIC confirms: q-aware Safe-RDO teacher,
  especially to recover low-q rate saving while keeping the high-q DISTS/FID
  gains.

### 2026-06-30 JST - Safe-RDO gate CLIC2020 test result

Run:

- checkpoint:
  `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`
- real codec:
  `experiments/real_codec/clic2020_test_stage_safe_rdo_gate_from_sb03_2000`
- metrics:
  `experiments/real_codec/clic2020_test_stage_safe_rdo_gate_from_sb03_2000_metrics.csv`
- comparison metrics:
  `experiments/real_codec/clic2020_test_stage_safe_rdo_gate_from_sb03_2000_compare_metrics.csv`
- BD:
  `experiments/real_codec/clic2020_test_stage_safe_rdo_gate_from_sb03_2000_compare_bd.md`
- matched-quality:
  `experiments/real_codec/clic2020_test_stage_safe_rdo_gate_from_sb03_2000_compare_matched.md`

Real codec:

- CLIC2020 combined test set, 428 images.
- q0-q3 all passed with `max_abs=0`.
- no control stream was sent (`bpp_control=0`).
- average bpp:
  q0 `0.01858`, q1 `0.02213`, q2 `0.02658`, q3 `0.03056`.
- average encode/decode time:
  q0 `0.657s` / `0.937s`, q1 `0.730s` / `1.014s`,
  q2 `0.780s` / `1.064s`, q3 `0.883s` / `1.160s`.

CLIC2020 point metrics:

| q | bpp | LPIPS | DISTS | FID | KID |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.01858 | 0.1709 | 0.0853 | 6.9813 | 0.0016 |
| 1 | 0.02213 | 0.1504 | 0.0752 | 5.5921 | 0.0011 |
| 2 | 0.02658 | 0.1365 | 0.0686 | 4.8173 | 0.0009 |
| 3 | 0.03056 | 0.1293 | 0.0659 | 4.4838 | 0.0007 |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Previous current lead | -8.37% | +0.68% | -5.50% | -4.39% |
| Safe-RDO | -9.76% | +0.05% | -6.42% | -4.65% |

Matched-quality bpp delta versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| Previous current lead | -7.28% | +1.88% | -2.42% | -1.85% |
| Safe-RDO | -9.70% | +1.08% | -4.00% | -3.43% |

Interpretation:

- Safe-RDO improves the previous current lead on CLIC2020 DISTS BD, FID BD,
  KID BD, matched-DISTS bpp saving, matched-FID bpp saving, and matched-KID
  bpp saving.
- LPIPS remains the weakest metric, but it is nearly neutral in BD-rate
  (`+0.05%`) and improves over the previous lead (`+0.68%`).
- q0 and q1 spend slightly more bpp than the previous lead, but the added bits
  improve all q0/q1 perceptual metrics.  q2 and q3 are both lower-bpp than the
  previous lead while preserving or improving the DISTS/FID curve.
- This confirms the mainline value of the Safe-RDO teacher: learning
  benefit-cost safe coarsening is stronger than simply tuning rho or adding
  residual-refiner capacity.

Decision:

- Promote `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt` to the
  current overall lead for the pretrained-GLC mainline.
- Next implementation should stay on the mainline:
  q-aware Safe-RDO teacher and/or stage-aware residual-variable coding.
- Do not return to blind `rho_target` / `rho_max` / loss-weight sweeps except
  as small diagnostics.

### 2026-06-30 JST - q-aware Safe-RDO fine-tune attempt

Run:

- checkpoint:
  `experiments/stage_safe_rdo_qaware_from_safe_2500/v2_final.pt`
- W&B:
  `stage_safe_rdo_qaware_from_safe_2500`, run id `374jxk5i`
- real codec:
  `experiments/real_codec/kodak8_stage_safe_rdo_qaware_from_safe_2500`
- metrics:
  `experiments/real_codec/kodak8_stage_safe_rdo_qaware_from_safe_2500_metrics.csv`

Design:

- Resume from the current Safe-RDO lead.
- Keep pretrained GLC and residual modules frozen.
- Fine-tune the stage quant gate with q-aware teacher strength:
  - stronger low-q coarsening target:
    `rho_target_by_q = [1.20, 1.19, 1.18, 1.16]`
  - stronger low-q saved-rate weight:
    `gate_rdo_saved_rate_weight_by_q = [2.2, 1.9, 1.5, 1.2]`
  - q-aware RDO sensitivity:
    `lambda_gate_rdo_sens_by_q = [14, 12, 10, 8]`
- Goal:
  recover q0/q1 rate saving while preserving q2/q3 perceptual gains.

Kodak8 real codec:

- q0-q3 all passed with `max_abs=0`.
- no control stream was sent (`bpp_control=0`).
- average bpp:
  q0 `0.02397`, q1 `0.02799`, q2 `0.03273`, q3 `0.03719`.

Kodak8 point metrics:

| q | bpp | LPIPS | DISTS | note |
|---:|---:|---:|---:|---|
| 0 | 0.02397 | 0.2229 | 0.1241 | lower bpp than Safe-RDO, worse quality |
| 1 | 0.02799 | 0.1949 | 0.1115 | slightly lower bpp, worse quality |
| 2 | 0.03273 | 0.1755 | 0.1049 | slightly lower bpp, worse DISTS |
| 3 | 0.03719 | 0.1677 | 0.0982 | higher bpp, similar/worse quality |

Interpretation:

- This q-aware setting is too aggressive.
- It saves a small amount of bpp at q0-q2 but damages perceptual quality enough
  that it is not a candidate lead.
- The failure is useful: pushing rho targets by q without a better
  safe-to-drop estimate can remove information that is still perceptually
  necessary.

Decision:

- Do not promote this run.
- Keep `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt` as the
  current overall lead.
- Next mainline direction should not be a stronger q-aware target sweep.
  Prefer stage-aware residual-variable coding or a better safe-coarsening
  teacher with explicit local damage/benefit estimation.

### 2026-06-30 JST - RDO-weighted stage residual mean continuation

Code change:

- `scripts/train_v2.py`
  - `lambda_mean_pred_safe` now uses `safe_for_control` when an RDO teacher
    target is available.
  - Previously, stage residual safe-mean supervision mostly used the gate's own
    `gate_p_tex` unless the mixed teacher was active.  That made the mean
    predictor supervision less aligned with the Safe-RDO teacher.
  - The new priority is:
    `RDO/mixed safe target -> mixed target -> gate_p_tex -> all ones`.

Reason:

- The next mainline question was whether the current Safe-RDO lead can be
  improved by learning a decoder-computable stage residual mean
  `gp_mu_stage(context)` in the same regions where the RDO teacher says residual
  precision is safe to reduce.
- This tests a more faithful residual-variable mechanism than another
  `rho_target` sweep.

Smoke:

- output:
  `experiments/smoke_stage_safe_rdo_mean_from_safe`
- init:
  `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`
- result:
  10 iterations passed with `rdo_sens`, `safe_mean`, and
  `predictor_unsafe_delta` active.

Full continuation:

- training:
  `experiments/stage_safe_rdo_mean_joint_from_safe_4k`
- W&B:
  `stage_safe_rdo_mean_joint_from_safe_4k`, run id `ser0jkdx`
- init:
  `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`
- data:
  `/dpl/open-images-v6/test/data`
- mode:
  `stage_residual_entropy_quant_gate`
- trained:
  stage residual entropy predictor + stage quant gate
- frozen:
  pretrained GLC, q embedding, global perceptual gate
- no control stream was transmitted.

Kodak8 real codec:

- real codec:
  `experiments/real_codec/kodak8_stage_safe_rdo_mean_joint_from_safe_4k`
- metrics:
  `experiments/real_codec/kodak8_stage_safe_rdo_mean_joint_from_safe_4k_metrics_patch64.csv`
- comparison:
  `experiments/real_codec/kodak8_stage_safe_rdo_mean_joint_from_safe_4k_compare_bd.csv`

Kodak8 BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| SafeRDO | -6.00% | -2.97% | -7.38% | -22.04% |
| JointMean | -7.18% | -1.30% | -5.71% | +16.06% |

Kodak8 interpretation:

- JointMean improves Kodak8 DISTS BD-rate over SafeRDO, but weakens LPIPS,
  FID, and KID.
- Kodak8 is too small for FID/KID to decide the branch, so I evaluated DIV2K
  before rejecting or promoting it.

DIV2K real codec:

- real codec:
  `experiments/real_codec/div2k_stage_safe_rdo_mean_joint_from_safe_4k`
- metrics:
  `experiments/real_codec/div2k_stage_safe_rdo_mean_joint_from_safe_4k_metrics.csv`
- comparison:
  `experiments/real_codec/div2k_stage_safe_rdo_mean_joint_from_safe_4k_compare_bd.csv`

DIV2K BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID |
|---|---:|---:|---:|---:|
| SafeRDO | -10.37% | -0.49% | -5.64% | -4.98% |
| JointMean | -9.95% | -0.49% | -5.45% | -4.52% |

DIV2K matched-quality bpp delta versus local GLC:

| run | DISTS | FID | LPIPS | KID |
|---|---:|---:|---:|---:|
| SafeRDO | -9.76% | -3.18% | +0.33% | -3.82% |
| JointMean | -9.50% | -4.50% | +0.41% | -3.63% |

Decision:

- Do not promote `stage_safe_rdo_mean_joint_from_safe_4k`.
- Keep `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt` as the
  current overall lead.
- The result is informative: RDO-weighted stage residual mean learning can
  repair Kodak DISTS, but it does not move the larger DIV2K curve beyond the
  Safe-RDO gate.  It behaves like a quality/rate redistribution layer rather
  than a new ceiling.
- Next mainline should move beyond decoder-computable mean/gate continuation:
  implement a stronger coded residual/control symbol design with a learned
  entropy model, finite support or escape handling, and exact real-codec
  accounting.

### 2026-06-30 JST - Entropy-family and global-scale codec probes

Reason:

- Before implementing a larger learned residual/control entropy model, I checked
  whether the current Safe-RDO lead is leaving easy real-codec gains in the
  arithmetic-coder distribution family or a global entropy-scale factor.
- These probes do not change reconstruction.  They only change the CDF used by
  the real y-stream arithmetic coder, so any improvement would be an immediate
  serialized-bpp gain.

Checkpoint:

- `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`

Dataset:

- Kodak8, q0-q3, real codec with `max_abs=0` consistency checks.

Average bpp:

| codec CDF | q0 | q1 | q2 | q3 | decision |
|---|---:|---:|---:|---:|---|
| SafeRDO default Gaussian | 0.02429 | 0.02817 | 0.03279 | 0.03704 | current lead |
| Laplace | 0.02712 | 0.03117 | 0.03589 | 0.04019 | reject |
| Logistic | 0.02516 | 0.02906 | 0.03367 | 0.03791 | reject |
| Gaussian scale 0.80 | 0.02513 | 0.02927 | 0.03392 | 0.03812 | reject |
| Gaussian scale 0.90 | 0.02448 | 0.02847 | 0.03307 | 0.03728 | reject |
| Gaussian scale 1.10 | 0.02445 | 0.02826 | 0.03290 | 0.03722 | reject |

Interpretation:

- There is no easy win from replacing the Gaussian CDF with Laplace/logistic or
  from a single global scale factor.
- The previous `StageScaleCalibrator` result and this probe agree: simple
  entropy calibration is not enough to move the curve.
- The next entropy work should be structural:
  decoder-computable per-symbol residual/control priors, finite-support or
  escape coding, and residual/control token design.  A paid control stream also
  needs a decoder-side prior; encoder-only probabilities cannot be used
  uncounted by the arithmetic decoder.

### 2026-06-30 JST - Fixed-prior counted residual-control probes

Reason:

- I checked whether a tiny counted source-side residual/control stream opens an
  obvious ceiling beyond the current Safe-RDO lead.
- This is on-axis conceptually because it sends only a small amount of
  unpredictable control/residual correction and counts every serialized bit.
- The current implementation uses a fixed sparse ternary prior, not yet a
  learned decoder-computable control prior.

Checkpoint:

- `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`

Dataset:

- Kodak8, q0-q3, real codec.
- Metrics use the Kodak patch protocol used in the current comparisons:
  `patch=64`, `split_patch_num=1`.

Runs:

| run | control frac | control delta | groups | avg control bpp | output |
|---|---:|---:|---:|---:|---|
| ResidualControl | 0.0025 | 0.05 | 16 | ~0.00047 | `experiments/real_codec/kodak8_safe_rdo_residual_control_g16_f0025_d005` |
| ResidualControlStrong | 0.0050 | 0.10 | 16 | ~0.00083 | `experiments/real_codec/kodak8_safe_rdo_residual_control_g16_f005_d010` |

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | KID | decision |
|---|---:|---:|---:|---:|---|
| SafeRDO | -6.00% | -2.97% | -7.38% | -22.04% | current lead |
| ResidualControl | -2.45% | -0.44% | -4.30% | +17.12% | reject |
| ResidualControlStrong | -1.58% | +0.33% | -2.80% | +19.67% | reject |

Interpretation:

- The fixed-prior source-side residual/control stream does not beat the
  zero-side Safe-RDO lead after its bit cost is counted.
- Stronger control restores a little local fidelity but the additional control
  rate is not paid back in DISTS/FID BD-rate.
- This does not invalidate counted control as a mainline component.  It says
  the current fixed-prior control path is too weak.
- If control is revisited, it should be stage-wise and paired with a learned
  decoder-computable prior, so the arithmetic decoder can use the same
  probabilities without uncounted encoder-only information.

Decision:

- Do not promote either residual-control probe.
- Keep `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt` as the
  current overall lead.
- Next work should return to a structural implementation:
  stage-aware residual-variable coding and/or learned residual/control entropy
  modeling, not more fixed-prior control sweeps.

### 2026-07-01 JST - Stage-3 selective hard omission and omitted-residual synthesis

Reason:

- The mainline question was whether the current Safe-RDO `rho` map can be
  converted from soft residual coarsening into actual residual omission:
  do not arithmetic-code the selected `y_q` residual symbols, and let the
  decoder reconstruct them without a side map.
- This directly tests the GP-ResLC idea of not transmitting residual
  information that the pretrained GLC decoder can tolerate or recover.

Implementation:

- Added real-codec diagnostic options:
  - `--suppress_yq_stages`
  - `--suppress_rho_threshold`
  - `--omitted_residual_mode`
  - `--omitted_residual_scale`
  - `--omitted_residual_clip`
- Encode removes the selected symbols from the arithmetic-coded `y` stream.
- Decode recomputes the same threshold mask from decoder-available
  `z_hat/q/context/rho`.
- No side map is transmitted.
- All bpp numbers are serialized real-codec bpp.

Important invalidation:

- The first `rhohard_stage3_t120/t125` outputs without `_fixed` in the path are
  invalid for quality judgment.
- The bug was decoder-side: when `--suppress_yq_stages 3` was set, decode
  omitted the whole stage instead of only `rho >= threshold` positions.
- The corrected runs below use `_fixed` paths and were verified by smoke test:
  for a conservative threshold with no selected symbols on `kodim01`, decoded
  output is exactly identical to SafeRDO.

Valid runs:

| run | output |
|---|---|
| stage3 hard, rho>=1.20, zero | `experiments/real_codec/kodak8_safe_rdo_rhohard_stage3_t120_zero_fixed` |
| stage3 hard, rho>=1.25, zero | `experiments/real_codec/kodak8_safe_rdo_rhohard_stage3_t125_zero_fixed` |
| stage3 hard, rho>=1.25, deterministic synthesis | `experiments/real_codec/kodak8_safe_rdo_rhohard_stage3_t125_synth_gclip_s05_fixed` |

Metrics:

- Dataset: Kodak8.
- Metrics: `patch=64`, `split_patch_num=1`, `kid_subset_size=512`.
- Comparison CSV:
  `experiments/real_codec/kodak8_rhohard_stage3_synthesis_compare_metrics.csv`.

BD-rate versus local GLC:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SafeRDO | -6.00% | -2.97% | +1.77% | +0.64% | -7.38% | -22.04% | current anchor |
| stage3 hard, rho>=1.20, zero | +2.49% | +0.89% | +0.60% | +2.24% | -2.01% | +38.68% | reject |
| stage3 hard, rho>=1.25, zero | -5.38% | -3.13% | +1.45% | +0.32% | -7.33% | +10.99% | diagnostic only |
| stage3 hard, rho>=1.25, hash synthesis | -4.93% | -3.05% | +1.65% | +0.48% | -6.65% | +15.85% | reject synthesis |

BD-rate versus SafeRDO:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| stage3 hard, rho>=1.20, zero | +9.87% | +3.19% | -2.34% | -0.06% | +14.92% | +110.85% | reject |
| stage3 hard, rho>=1.25, zero | +0.45% | -0.17% | -0.31% | -0.31% | +0.89% | +44.30% | not a lead |
| stage3 hard, rho>=1.25, hash synthesis | +0.88% | -0.05% | -0.20% | -0.20% | +1.42% | +51.48% | reject |

Interpretation:

- A very conservative stage-3 hard omission can preserve LPIPS slightly better
  than SafeRDO, but it gives back DISTS/FID and is not a new lead.
- A stronger omission threshold saves more `y` bits but moves the main
  perceptual curve in the wrong direction.
- Deterministic pseudo-noise synthesis does not recover the omitted residuals.
  The useful branch is not random hash synthesis; it would need a learned
  decoder-side residual synthesizer or a better safe-to-drop teacher.

Decision:

- Keep the selective hard omission path as a diagnostic and ablation tool.
- Do not promote it over SafeRDO.
- Do not spend more time on scalar threshold sweeps.
- Next mainline should implement a learned safe-to-drop / residual-payoff
  teacher and/or a learned decoder-side omitted-residual synthesis module.

## 2026-07-01 JST - Learned omitted-residual synthesis follow-up

Motivation:

- The previous diagnostic showed that hard omission has real bpp savings but
  loses quality when too many stage-3 residual symbols are removed.
- The mainline hypothesis is therefore not just "drop residuals"; it is
  "drop residuals that can be recovered by a decoder-side generative prior".

Implementation updates:

- Added `StageResidualValueSynthesizer` in `gp_reslc/prior_predictor.py`.
- Added `omitted_residual_mode=learned_value` to the real codec.
- Added checkpoint loading for `stage_residual_value_synthesizer` in
  `scripts/evaluate_real_codec.py`.
- Added `scripts/train_stage3_value_synth.py`, which freezes pretrained GLC and
  the SafeRDO checkpoint and trains only the decoder-side value synthesizer with
  image-space losses.
- Fixed `scripts/train_stage3_symbol_synth.py`: the previous symbol classifier
  accidentally used `y_hat_so_far` after stage 3 during training, while decode
  only has the context before stage 3.  That train/decode mismatch likely
  contributed to the poor real-codec result.

Rejected result:

- Run: `stage3_symbol_synth_t120_weighted_1k`
- W&B: `2npfah7e`
- Real codec output:
  `experiments/real_codec/kodak8_stage3_symbol_synth_t120_weighted_1k`
- BD summary:
  `experiments/real_codec/kodak8_stage3_symbol_synth_compare_bd_vs_glc.md`

Kodak8 / real codec / patch64 split1:

| run | DISTS | LPIPS | FID | decision |
|---|---:|---:|---:|---|
| Stage3SymbolSynthT120 vs GLC | non-overlap / much worse | non-overlap / much worse | +27.66% | reject |
| Stage3SymbolSynthT120 vs SafeRDO | +45.98% | +28.41% | +50.11% | reject |

Interpretation:

- Classification accuracy on rare omitted symbols did not translate to image
  quality.
- The initial training script also had a decoder-context mismatch, so this run
  should not be used as evidence that learned synthesis is impossible.
- The stronger next test is a continuous value synthesizer trained directly by
  reconstruction/perceptual losses under the exact decoder-available context.

Active run:

- Run: `stage3_value_synth_t120_perc_1500`
- W&B: `6x3oe21z`
- Output directory:
  `experiments/stage3_value_synth_t120_perc_1500`
- Mechanism: omit stage-3 residual symbols where `rho >= 1.20`; fill only the
  omitted positions with decoder-computable continuous residual values.
- No side map or seed is sent; serialized bpp should match the corresponding
  hard-omission run.

Smoke finding:

- A 50-iteration smoke checkpoint improved q0 quality at the same bpp over
  hard zero omission:
  `DISTS 0.1281 -> 0.1275`, `LPIPS 0.2261 -> 0.2237`,
  `FID 82.66 -> 82.26`.
- This is not yet a lead, but it is enough evidence to run the longer
  image-loss-trained value synthesizer.

Completed result for `stage3_value_synth_t120_perc_1500`:

- W&B: `6x3oe21z`
- Real codec output:
  `experiments/real_codec/kodak8_stage3_value_synth_t120_perc_1500`
- Metrics:
  `experiments/real_codec/kodak8_stage3_value_synth_t120_perc_1500_metrics_patch64_split1.csv`
- BD summary:
  `experiments/real_codec/kodak8_stage3_value_synth_compare_bd_vs_glc.md`

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | decision |
|---|---:|---:|---:|---|
| stage3 value synth, rho>=1.20 | +2.77% | -0.59% | -2.57% | reject as lead |

BD-rate versus SafeRDO:

| run | DISTS | LPIPS | FID | decision |
|---|---:|---:|---:|---|
| stage3 value synth, rho>=1.20 | +10.09% | +1.81% | +12.32% | reject as lead |

Interpretation:

- Continuous image-loss synthesis clearly improves over hard zero omission at
  the same serialized bpp, but the `rho>=1.20` omission set is too aggressive.
- The mechanism is promising, but the mask must be safer.  The next run moves
  the same value synthesizer to the conservative `rho>=1.25` mask, where hard
  omission was already close to SafeRDO.

Active follow-up:

- Run: `stage3_value_synth_t125_perc_1200`
- W&B: `w1s0ucen`
- Output directory:
  `experiments/stage3_value_synth_t125_perc_1200`

Completed result for `stage3_value_synth_t125_perc_1200`:

- Real codec output:
  `experiments/real_codec/kodak8_stage3_value_synth_t125_perc_1200`
- Metrics:
  `experiments/real_codec/kodak8_stage3_value_synth_t125_perc_1200_metrics_patch64_split1.csv`
- BD summary:
  `experiments/real_codec/kodak8_stage3_value_synth_t125_compare_bd_vs_glc.md`

BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | decision |
|---|---:|---:|---:|---|
| stage3 value synth, rho>=1.25 | -5.31% | -3.25% | -7.07% | close, not lead |

BD-rate versus SafeRDO:

| run | DISTS | LPIPS | FID | decision |
|---|---:|---:|---:|---|
| stage3 value synth, rho>=1.25 | +0.50% | -0.27% | +1.20% | LPIPS improves, DISTS/FID weaker |

Interpretation:

- Conservative value synthesis preserves the hard-omission result and gives the
  best LPIPS among these stage-3 hard-omission variants.
- It still does not beat the SafeRDO lead on DISTS/FID.
- The next candidate keeps the SafeRDO bitstream unchanged and adds a
  decoder-computable stage-3 synthesis residual after decoding, so the test is
  pure no-side quality recovery at SafeRDO bpp.

Active additive follow-up:

- Run: `stage3_value_add_t120_perc_1200`
- W&B: `ll6vhs85`
- Output directory:
  `experiments/stage3_value_add_t120_perc_1200`
- Mechanism: transmit the normal SafeRDO y stream; for stage-3 positions with
  `rho >= 1.20`, add a decoder-side learned residual value after arithmetic
  decoding.  No y symbols are removed and no side stream is sent.

Additive real-codec fix and result:

- Initial `s1/s5` additive evaluations were invalid: `scripts/evaluate_real_codec.py`
  accepted `--synth_yq_stages`, but the normal
  `stage_residual_entropy_quant_gate` branch in `gp_reslc/real_codec.py` did
  not pass these arguments into the actual encode/decode helpers.  Pixel
  comparison against SafeRDO showed exact zero difference even with
  `--synth_value_scale 5.0`.
- Fixed the real-codec path so `synth_yq_stages`,
  `synth_rho_threshold`, and `synth_value_scale` are applied in the
  `stage_residual_quant_gate` helper on both encode and decode.  After the fix,
  `s5` produced large pixel differences and severe quality degradation, proving
  the path is active but that large synthesis amplitude is unsafe.

Same-code SafeRDO anchor:

- Real codec:
  `experiments/real_codec/kodak8_stage_safe_rdo_gate_from_sb03_2000_current`
- Metrics:
  `experiments/real_codec/kodak8_stage_safe_rdo_gate_from_sb03_2000_current_metrics_patch64_split1.csv`

Additive synthesis runs:

| run | output | note |
|---|---|---|
| `s1` | `experiments/real_codec/kodak8_stage3_value_add_t120_perc_1200_s1_fixed` | trained amplitude |
| `s0.5` | `experiments/real_codec/kodak8_stage3_value_add_t120_perc_1200_s05_fixed` | conservative amplitude |
| `s5 q0` | `experiments/real_codec/kodak8_stage3_value_add_t120_perc_1200_s5_q0_fixed` | destructive sanity check |

BD-rate versus same-code SafeRDO, Kodak8 / patch64 split1:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| additive `s1` | +1.44% | -2.26% | -7.62% | -4.58% | -0.71% | -0.63% | LPIPS/PSNR improves, DISTS too loose |
| additive `s0.5` | +0.24% | -1.50% | -3.93% | -2.38% | -1.00% | -0.61% | useful but not a DISTS lead |

Pointwise interpretation:

- Both `s1` and `s0.5` keep serialized bpp identical to SafeRDO because no
  additional symbols or side map are transmitted.
- `s1` improves PSNR, MS-SSIM, and LPIPS at all q points, but worsens DISTS at
  q1-q3.  It is therefore not paper-lead quality recovery.
- `s0.5` reduces the DISTS penalty substantially while preserving LPIPS/PSNR
  gains; q0 even improves all measured metrics relative to SafeRDO.
- This supports the "generative residual synthesis" hypothesis but also shows
  that the current value synthesizer is not DISTS-constrained enough.

Decision:

- Keep additive synthesis as the best current no-side quality-recovery signal.
- Do not claim it as a lead over SafeRDO yet.
- Next implementation should train the value synthesizer with a DISTS-aware
  safety objective against the frozen SafeRDO reconstruction, not just generic
  image-space reconstruction/perceptual loss.
- The target is: keep SafeRDO serialized bpp, improve LPIPS/PSNR/FID, and make
  DISTS non-worse across q.  If achieved, then combine with actual omission or
  counted-control coding to recover both quality and rate.

## 2026-07-01 JST - DISTS-aware additive synthesis training

Implementation:

- Extended `scripts/train_stage3_value_synth.py` with a frozen-baseline safety
  forward pass.
- New optional losses:
  - `lambda_safe_lpips * relu(LPIPS(ours,x) - LPIPS(SafeRDO,x) - margin)`
  - `lambda_safe_dists * relu(DISTS(ours,x) - DISTS(SafeRDO,x) - margin)`
- This directly trains the no-side additive synthesis module to avoid making
  the frozen SafeRDO reconstruction worse on perceptual metrics.

Run A:

- Name: `stage3_value_add_safehinge_dists_t120_2500`
- W&B: `0shvwfi4`
- Output:
  `experiments/stage3_value_add_safehinge_dists_t120_2500`
- Key settings:
  `value_bound=2.0`, `lambda_dists=3`, `lambda_safe_dists=20`,
  `lambda_lpips=2`, `lambda_safe_lpips=5`, additive stage-3, `rho>=1.20`.

Real-codec results versus same-code SafeRDO, Kodak8 / patch64 split1:

| variant | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| final, scale 1.0 | +2.89% | -2.09% | -10.62% | -5.96% | +1.00% | +1.70% | reject: DISTS/FID worse |
| final, scale 0.5 | +0.71% | -1.72% | -5.57% | -3.16% | -0.29% | +0.31% | not better than plain s0.5 |

Comparison to previous plain additive:

| variant | DISTS | LPIPS | PSNR | FID | decision |
|---|---:|---:|---:|---:|---|
| plain additive, scale 0.5 | +0.24% | -1.50% | -3.93% | -1.00% | best current no-side synthesis balance |
| plain additive, scale 1.0 | +1.44% | -2.26% | -7.62% | -0.71% | too much DISTS cost |

Interpretation:

- The DISTS-aware hinge improved fidelity/alignment strongly but did not beat
  the simpler conservative amplitude setting.
- The safe loss was active throughout training, which means the branch is
  fighting a real objective conflict: decoder-side synthesized detail can help
  PSNR/LPIPS but can still disturb DISTS texture/structure.
- The current best no-side additive result is therefore the plain value
  synthesizer with conservative amplitude `s0.5`, not the safe-hinge final.

Run B:

- Name: `stage3_value_add_distsguard_bound05_t120_1800`
- W&B: `ij15qn7q`
- Output:
  `experiments/stage3_value_add_distsguard_bound05_t120_1800`
- Key settings:
  `value_bound=0.5`, `lambda_dists=8`, `lambda_safe_dists=80`.
- Stopped manually after ~700 iterations.

Reason for stopping:

- DISTS safety violations remained frequent.
- `synth_abs_mean` grew instead of shrinking, despite the smaller bound.
- Early logs were clearly worse than the accepted plain `s0.5` balance, so
  continuing was unlikely to produce a useful lead.

Decision:

- Reject bound-only DISTS guard.
- Keep the no-side value synthesis implementation because it gives a real,
  codec-consistent quality recovery signal.
- The next structural improvement should not be another scalar amplitude/loss
  sweep.  It should make synthesis selective:
  - confidence-gated synthesis from decoder-available context,
  - local safe-to-synthesize teacher,
  - or a tiny counted control stream that explicitly marks where synthesis is
    allowed.

## 2026-07-01 JST - Decoder-computable selective value synthesis

Implementation:

- Added `StageResidualSelectiveValueSynthesizer`.
- The module predicts both a stage-3 residual value and a decoder-computable
  sigmoid gate from transmitted/decoded context only.
- No side map or control bits are transmitted.  The serialized payload is
  identical to the SafeRDO anchor; only the decoder-side reconstruction changes.
- Evaluation support was added to `scripts/evaluate_real_codec.py` through the
  checkpoint flag `stage_value_selective`.

Run:

- Name: `stage3_selective_value_add_t120_distsguard_1200`
- W&B: `sdlsddfn`
- Output:
  `experiments/stage3_selective_value_add_t120_distsguard_1200`
- Base checkpoint:
  `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`
- Key settings:
  additive stage-3 synthesis, `rho>=1.20`, `gate_init_prob=0.25`,
  `lambda_gate=0.02`, `lambda_safe_dists=50`, `lambda_safe_lpips=5`.

Training signal:

- The gate learned to be conservative: mean gate dropped from `0.25` to roughly
  `0.07-0.12` late in training.
- This confirmed that the model is not merely adding dense residual texture
  everywhere.
- Some batches still showed DISTS safety violations, so real-codec curve
  evaluation remained necessary.

Kodak8 / patch64 split1, BD-rate versus same-code SafeRDO:

| variant | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| selective final, scale 0.25 | -0.19% | -0.92% | -2.52% | -1.26% | -0.42% | -0.55% | keep as safe no-side synthesis booster |
| selective final, scale 0.5 | +0.04% | -1.74% | -4.93% | -2.48% | -0.21% | -0.74% | useful LPIPS/fidelity booster; DISTS-neutral |
| selective final, scale 1.0 | +1.14% | -2.49% | -9.09% | -4.61% | -0.22% | -0.78% | too much DISTS risk |
| previous plain additive, scale 0.5 | +0.24% | -1.50% | -3.93% | -2.38% | -1.00% | -0.61% | superseded for DISTS-balanced use |

Artifacts:

- Metrics:
  `experiments/real_codec/analysis/kodak8_safe_vs_selective_scales_metrics.csv`
- BD summary:
  `experiments/real_codec/analysis/kodak8_safe_vs_selective_scales_bd.md`

Interpretation:

- Selective no-side synthesis is real and codec-consistent: at `scale=0.25` it
  improves all measured Kodak8 metrics over the same-code SafeRDO anchor without
  adding bits.
- The gain is small, so this should not become the main performance story.
- The result is still important because it validates a decoder-side generative
  residual recovery path: some residual quality can be recovered after reducing
  transmitted precision.
- For large gains, the next mainline should move to actual entropy/rate
  mechanisms: fixed-width `z_hat` coding, stage-aware residual-variable coding,
  learned residual/control entropy, or a tiny counted control stream.

## 2026-07-01 JST - Static entropy coding for z_hat indices

Motivation:

- The real codec previously followed the published GLC assumption and packed
  `z_hat` VQ indices with fixed-width `ceil(log2(16384)) = 14` bits/index.
- Measured Kodak8 and OpenImages distributions are far from uniform, so this is
  a quality-preserving source of real bpp reduction.

Implementation:

- Added `z_entropy_mode in {fixed, static, auto}` to the real codec.
- `static` uses a q-specific CDF learned from training-side data.
- `auto` stores either static arithmetic-coded z indices or fixed-width z
  indices, whichever is shorter for that image.  The z payload is
  self-identifying through a small prefix, and all bytes are counted.
- Existing fixed-width payloads remain backward compatible.
- Added `scripts/estimate_z_index_cdf.py`.

CDF artifact:

- `experiments/z_entropy/openimages_train_2000_crop256_alpha1.pt`
- Source: `/dpl/open-images-v6/train/data`
- 2000 random 256x256 crops, q0-q3, Laplace smoothing `alpha=1.0`.
- Smoothed entropy:
  q0 `10.3904`, q1 `10.8674`, q2 `11.0832`, q3 `10.9667` bits/index.

SafeRDO + z entropy, Kodak8 / patch64 split1:

- Output:
  `experiments/real_codec/kodak8_stage_safe_rdo_zentropy_auto_a1`
- Metrics:
  `experiments/real_codec/kodak8_stage_safe_rdo_zentropy_auto_a1_metrics_patch64_split1.csv`
- BD summary:
  `experiments/real_codec/analysis/kodak8_safe_vs_safe_zentropy_a1_bd.md`

BD-rate versus same-code SafeRDO fixed-z anchor:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| SafeRDO + z entropy auto | -4.13% | -4.11% | -4.03% | -4.05% | -4.14% | -3.91% |

Decode consistency:

- All 32 Kodak8 reconstructions (`4 q x 8 images`) were pixel-identical between
  fixed-z SafeRDO and z-entropy SafeRDO (`max_abs_png_diff = 0`).
- Therefore this gain is pure serialized bpp reduction, not a quality-path
  change.

Composition with selective no-side synthesis:

- Output:
  `experiments/real_codec/kodak8_stage3_selective_value_add_t120_distsguard_1200_s025_zentropy_auto_a1`
- Metrics:
  `experiments/real_codec/kodak8_stage3_selective_value_add_t120_distsguard_1200_s025_zentropy_auto_a1_metrics_patch64_split1.csv`
- BD summary:
  `experiments/real_codec/analysis/kodak8_safe_vs_zentropy_selective_s025_bd.md`

BD-rate versus same-code SafeRDO fixed-z anchor:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| Selective synthesis s0.25 + z entropy auto | -4.32% | -5.01% | -6.52% | -5.29% | -4.53% | -4.40% |

Decision:

- Keep z entropy coding as a fair real-codec booster, provided the CDF is
  learned only from training-side data.
- Do not treat it as a GP-ResLC-specific main contribution: the same coding
  improvement applies to the GLC baseline because it preserves `z_hat` exactly.
- When reporting main GP-ResLC gains, separate:
  1. fixed-z GLC/SafeRDO comparisons,
  2. GLC + z entropy,
  3. GP-ResLC + z entropy.
- Next step: validate on DIV2K and CLIC, and optionally build a larger/more
  stable z CDF from more OpenImages crops.

Fairness check on GLC baseline:

- Output:
  `experiments/real_codec/kodak8_glc_zentropy_auto_a1`
- Metrics:
  `experiments/real_codec/analysis/kodak8_glc_fixed_vs_zentropy_a1_metrics.csv`
- BD summary:
  `experiments/real_codec/analysis/kodak8_glc_fixed_vs_zentropy_a1_bd.md`

GLC fixed-z -> GLC z-entropy auto:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| GLC + z entropy auto | -3.78% | -3.69% | -3.80% | -3.74% | -3.99% | -9.06% |

Interpretation:

- This confirms that z entropy coding is not unique to GP-ResLC.
- It is still useful for a serious codec package, but the main research axis
  must return to `y` residual allocation/coding.

## 2026-07-01 JST - Stage-3 send-control and selective synthesis follow-up

Goal:

- Test a cleaner residual-allocation branch than rho-only tuning:
  transmit only selected stage-3 residual cells under a counted binary send
  mask, and let omitted residuals be handled by the prior/generator.
- Check whether selective synthesis should become the main generator-recovery
  path.

Implementation:

- Added `stage3_send_score_mode` to the real codec evaluator.
- Existing `latent_mse` send-mask selection is kept.
- Added encoder-side image-loss teacher modes:
  - `image_mse_grad`
  - `image_l1_grad`
  - `image_mse_grad_abs`
- The image-gradient teacher uses the source image only at encode time to choose
  a binary mask.  The mask is entropy-coded and counted, so decode remains
  self-contained.

Artifacts:

- Image-gradient send-control:
  `experiments/real_codec/kodak8_stage3_send_img_mse_grad_f075`
- Send-control + selective synthesis:
  `experiments/real_codec/kodak8_stage3_send_f075_selective_s025`
  `experiments/real_codec/kodak8_stage3_send_f090_selective_s025`
  `experiments/real_codec/kodak8_stage3_send_f095_selective_s025`
- Summaries:
  `experiments/real_codec/analysis/kodak8_safe_vs_stage3_send_imggrad_bd.md`
  `experiments/real_codec/analysis/kodak8_safe_vs_send_selective_bd.md`
  `experiments/real_codec/analysis/kodak8_safe_vs_selective_send_f095_bd.md`

Kodak8 / patch64 split1 BD-rate versus same-code SafeRDO:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| stage3 send, latent-MSE, send 75% | +1.09% | +0.95% | -1.31% | +0.08% | +2.75% | +4.77% |
| stage3 send, image-MSE-grad, send 75% | +4.58% | +2.42% | -0.41% | +2.59% | +5.52% | +12.79% |
| send 75% + selective synthesis s0.25 | +0.95% | -0.01% | -3.75% | -0.93% | +3.42% | +11.80% |
| send 90% + selective synthesis s0.25 | +0.56% | -0.59% | -2.49% | -0.62% | +2.09% | -2.24% |
| send 95% + selective synthesis s0.25 | +0.37% | -0.73% | -2.19% | -0.58% | +1.84% | +17.34% |
| selective synthesis s0.25 only | -0.19% | -0.92% | -2.52% | -1.26% | -0.42% | -6.88% |

Decision:

- Reject the current stage-3 send-control branch as a lead.  It saves some
  stage-3 bpp, but DISTS/FID degradation dominates.
- Reject image-MSE-gradient allocation for perceptual compression.  It preserves
  distortion slightly, but it is worse than latent-MSE for DISTS/FID and is not
  aligned with the generative compression objective.
- Keep selective synthesis as the active generator-recovery branch.  It is small
  but consistently safe.
- The next meaningful step is not another `send_frac` sweep.  It is an
  omitted-aware selective synthesizer:

```text
residual transmitted      -> keep original y_q precision
residual not transmitted  -> synthesize only if the decoder-computable gate says safe
otherwise                 -> fall back to GLC/SafeRDO prior mean
```

This is closer to the core GP-ResLC story than generic additive synthesis.

## 2026-07-01 JST - Omitted-aware selective value synthesis training

Rationale:

- The previous selective synthesis checkpoint was trained as an additive
  decoder-side correction.  It improves all Kodak8 metrics at conservative
  scale, but it is not yet the cleanest GP-ResLC mechanism.
- When used directly as `omitted_residual_mode=learned_value`, it improves
  PSNR/LPIPS but still hurts DISTS/FID because it was not trained as an omitted
  residual recovery model:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| omitted learned_value, send 75% | +2.11% | -0.54% | -8.93% | -2.76% | +2.93% | +5.06% |

Decision:

- Train a dedicated omitted-recovery selective synthesizer, with `additive`
  disabled.  During training, selected high-rho stage-3 residual symbols are
  replaced by decoder-computable predicted values, so the training task matches
  the intended real-codec omitted-residual path.

Run:

- Name: `stage3_omitted_selective_value_t120_distsguard_3000`
- W&B: `vgebegpr`
- Base checkpoint:
  `experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt`
- Output:
  `experiments/stage3_omitted_selective_value_t120_distsguard_3000`
- Key settings:
  `rho_threshold=1.20`, `selective=true`, `additive=false`,
  `lambda_dists=8`, `lambda_safe_dists=70`, `lambda_safe_lpips=5`,
  `lambda_gate=0.005`, OpenImages train crops, q0-q3.

Early training read:

- Iteration 0 starts with substantial safety violations because omitted
  positions are reconstructed as near-zero residuals.
- By iterations 25-100 the gate remains active (`~0.23-0.24`) rather than
  collapsing to zero, while the synthesizer learns nonzero omitted residual
  values.
- This run should be evaluated with:

```text
--predictor_param_mode stage_residual_entropy_quant_gate_stage3_send_control
--omitted_residual_mode learned_value
--residual_control_topk_frac {0.75,0.90,0.95}
```

Success criterion:

- At minimum, beat stage3 send-control without learned omitted recovery on
  DISTS/LPIPS at matched real bpp.
- Promotion requires matching or beating the selective-synthesis-only branch
  while reducing serialized y bits.

Follow-up:

- The 1000-step checkpoint was evaluated as
  `experiments/real_codec/kodak8_stage3_omitted_selective_1000_f075`.
- BD-rate versus the same SafeRDO anchor:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| omitted learned old, send 75% | +2.11% | -0.54% | -8.93% | -2.76% | +2.93% | +5.06% |
| omitted selective 1000, send 75% | +1.32% | -0.76% | -8.88% | -3.00% | +3.72% | +39.73% |
| stage3 send only, send 75% | +1.09% | +0.95% | -1.31% | +0.08% | +2.75% | +4.77% |

Interpretation:

- The dedicated omitted-value model recovered LPIPS/PSNR but still damaged
  DISTS/FID, so it is not a lead.
- The issue is partly conceptual and partly implementation-alignment: the
  initial training selected replacement positions by a decoder-computable
  `rho >= 1.20` threshold, but the real-codec branch evaluated with a counted
  latent-MSE stage-3 send mask.  The omitted-cell distribution therefore did
  not match between training and evaluation.
- The training script now supports `--stage3_send_frac` so the learned
  synthesizer is trained on exactly the cells omitted by the real-codec
  counted send-control branch.

New run:

- Name: `stage3_omitted_selective_matchsend_f075_distsguard_3000`
- W&B: `otccokhz`
- Output:
  `experiments/stage3_omitted_selective_matchsend_f075_distsguard_3000`
- Key change:
  `stage3_send_frac=0.75`, `stage3_send_score_mode=latent_mse`,
  `gate_init_prob=0.08`, `value_bound=1.0`, stronger DISTS safety.

This branch tests the real question more cleanly: after the encoder chooses
which stage-3 cells to transmit, can the decoder synthesize the omitted cells
well enough to close the DISTS/FID gap?

Result:

- 1000-step checkpoint:
  `experiments/stage3_omitted_selective_matchsend_f075_distsguard_3000/value_synth_001000.pt`
- Real codec output:
  `experiments/real_codec/kodak8_stage3_omitted_matchsend_1000_f075`
- BD-rate versus SafeRDO:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| match-send omitted selective 1000, send 75% | +3.48% | -0.03% | -11.88% | -3.95% | +0.58% | +33.76% |

Decision:

- Stop this branch as a DISTS-led mainline.
- It partially restores FID relative to plain stage-3 send-control, but it
  substantially worsens DISTS.  This confirms that omitted residual synthesis
  can make images more distributionally natural while disrupting
  DISTS-sensitive local structure.
- Keep the result as evidence that synthesis may remain useful as an auxiliary
  FID/LPIPS branch, but do not use it as the primary rate-saving mechanism.
- Next priority is zero-distortion entropy improvement and stage-aware
  residual-variable coding.

## 2026-07-01 JST - Full Kodak z-entropy and selective synthesis package

Rationale:

- The omitted-send synthesis branches did not close the DISTS gap.
- The strongest reliable bpp reduction available now is zero-distortion
  entropy coding of the GLC `z_hat` VQ indices.
- Pair this with the conservative no-side selective additive synthesis branch
  (`scale=0.25`) because it was the only recent synthesis variant that improved
  all Kodak8 metrics.

Runs:

- Fixed-z SafeRDO anchor:
  `experiments/real_codec/kodak_stage_safe_rdo_current_fixed`
- SafeRDO + z entropy auto:
  `experiments/real_codec/kodak_stage_safe_rdo_zentropy_auto_a1`
- Selective synthesis s0.25 + z entropy auto:
  `experiments/real_codec/kodak_stage3_selective_s025_zentropy_auto_a1`

Average real bpp on Kodak24:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| SafeRDO fixed z | 0.02336 | 0.02716 | 0.03168 | 0.03583 |
| SafeRDO z entropy auto | 0.02203 | 0.02600 | 0.03063 | 0.03475 |
| Selective s0.25 z entropy auto | 0.02203 | 0.02600 | 0.03063 | 0.03475 |

Kodak24 / patch64 split1 BD-rate versus fixed-z SafeRDO:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID |
|---|---:|---:|---:|---:|---:|
| SafeRDO + z entropy auto | -4.38% | -4.28% | -4.17% | -4.20% | -4.34% |
| Selective s0.25 + z entropy auto | -4.52% | -5.00% | -6.87% | -5.72% | -4.92% |

Notes:

- `z_entropy_auto` changes only the serialized `z_hat` index representation.
  Reconstructions are pixel-identical to the fixed-z anchor.
- Selective s0.25 adds no side bits and gives small but consistent quality gains
  on top of the z-entropy left shift.
- KID is intentionally omitted from the table because Kodak is too small for a
  stable KID claim.

Decision:

- Treat `SelectiveS025ZEntropyAutoA1` as the current best conservative package.
- Treat z entropy as a zero-distortion entropy-coding improvement, not the
  whole GP-ResLC novelty.
- Continue mainline research toward stage-aware residual-variable coding and a
  stronger learned residual/control entropy model.

## 2026-07-01 JST - DIV2K validation z-entropy protocol cleanup

Issue:

- An initial DIV2K z-entropy comparison used the old fixed-z anchor
  `experiments/real_codec/div2k_stage_safe_rdo_gate_from_sb03_2000`.
- That anchor was generated before the current real-codec path and used a
  different recorded input path.  Its reconstructions were not pixel-identical
  to the new z-entropy output, so the resulting BD-rate table is not a valid
  pure z-coding comparison and must not be used as evidence.

Corrected runs:

- Current fixed-z SafeRDO anchor:
  `experiments/real_codec/div2k_stage_safe_rdo_current_fixed_20260701`
- SafeRDO + z entropy auto:
  `experiments/real_codec/div2k_stage_safe_rdo_zentropy_auto_a1`
- Metrics/BD table:
  `experiments/real_codec/analysis/div2k_valid_safe_current_vs_zentropy_a1_bd.md`

Consistency checks:

- All 100 DIV2K validation images and all q0-q3 reconstructions are
  byte-identical PNGs between the corrected fixed-z anchor and z-entropy run.
- `avg_bpp_y` is identical at every q.
- Only the serialized `z_hat` representation changes.

Average real bpp on DIV2K validation:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| SafeRDO fixed z current | 0.02106 | 0.02482 | 0.02921 | 0.03334 |
| SafeRDO z entropy auto | 0.01958 | 0.02353 | 0.02803 | 0.03212 |

Corrected DIV2K validation BD-rate versus current fixed-z SafeRDO:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| SafeRDO + z entropy auto | -5.36% | -5.14% | -4.89% | -5.10% | -5.36% | -5.07% |

Decision:

- Keep z entropy as a validated zero-distortion codec improvement on both
  Kodak24 and DIV2K validation.
- Do not present the discarded old-anchor DIV2K table.
- Keep the research narrative centered on y/residual/control coding; z entropy
  is a clean auxiliary improvement, not the full GP-ResLC mechanism.

## 2026-07-01 JST - DIV2K validation selective synthesis s0.25 + z entropy

Purpose:

- Check whether the conservative no-side selective additive synthesis branch
  that worked on Kodak24 also generalizes to DIV2K validation.
- This branch changes reconstruction only; it sends no additional side
  information.  Real bpp is therefore the same as SafeRDO + z entropy auto.

Run:

- `experiments/real_codec/div2k_stage3_selective_s025_zentropy_auto_a1`
- Checkpoint:
  `experiments/stage3_selective_value_add_t120_distsguard_1200/value_synth_final.pt`
- Settings:
  `synth_yq_stages=3`, `synth_rho_threshold=1.20`,
  `synth_value_scale=0.25`, `z_entropy_mode=auto`
- Metrics:
  `experiments/real_codec/analysis/div2k_valid_safe_current_vs_zentropy_selective_s025_bd.md`

Average real bpp:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| SafeRDO fixed z current | 0.02106 | 0.02482 | 0.02921 | 0.03334 |
| SafeRDO z entropy auto | 0.01958 | 0.02353 | 0.02803 | 0.03212 |
| Selective s0.25 z entropy auto | 0.01958 | 0.02353 | 0.02803 | 0.03212 |

DIV2K validation BD-rate versus current fixed-z SafeRDO:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| SafeRDO + z entropy auto | -5.36% | -5.14% | -4.89% | -5.10% | -5.36% | -5.07% |
| Selective s0.25 + z entropy auto | -5.87% | -5.83% | -7.17% | -6.44% | -5.82% | -7.59% |

Decision:

- Promote `SelectiveS025ZEntropyAutoA1` from Kodak-only conservative package
  to the current validated conservative package on Kodak24 + DIV2K validation.
- This is still not the full GP-ResLC mainline: the z-entropy part is a
  zero-distortion codec cleanup, and the selective synthesis part is a small
  decoder-side generator-recovery correction.
- The next mainline step should still attack y/residual/control coding directly.

## 2026-07-01 JST - CLIC2020 test split preparation

Protocol note:

- Built `data_splits/eval/clic2020_test_428` from
  `/dpl/clic/professional/test` and `/dpl/clic/mobile/test`.
- Counts:
  - professional test: 250 images
  - mobile test: 178 images
  - merged CLIC2020 test: 428 images
- Symlink names are prefixed with `pro_` and `mob_` to avoid collisions and to
  keep the subset source explicit.

Current run:

- Current fixed-z SafeRDO anchor completed at
  `experiments/real_codec/clic2020_test_stage_safe_rdo_current_fixed_20260701`.
- The previous CLIC outputs are not reused as anchors until protocol and
  current-code consistency are verified, mirroring the DIV2K cleanup above.

Average real bpp on CLIC2020 test 428:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| SafeRDO fixed z current | 0.01855 | 0.02209 | 0.02648 | 0.03045 |

Average real bpp split:

| q | total | y | z | header | enc s/img | dec s/img |
|---|---:|---:|---:|---:|---:|---:|
| q0 | 0.018548 | 0.014778 | 0.003520 | 0.000249 | 0.655 | 0.934 |
| q1 | 0.022088 | 0.018318 | 0.003520 | 0.000249 | 0.725 | 1.007 |
| q2 | 0.026480 | 0.022711 | 0.003520 | 0.000249 | 0.772 | 1.056 |
| q3 | 0.030451 | 0.026681 | 0.003520 | 0.000249 | 0.888 | 1.163 |

Completed conservative package run:

- Run:
  `experiments/real_codec/clic2020_test_stage3_selective_s025_zentropy_auto_a1`
- Checkpoint:
  `experiments/stage3_selective_value_add_t120_distsguard_1200/value_synth_final.pt`
- Settings:
  `synth_yq_stages=3`, `synth_rho_threshold=1.20`,
  `synth_value_scale=0.25`, `z_entropy_mode=auto`,
  `z_entropy_cdf_path=experiments/z_entropy/openimages_train_2000_crop256_alpha1.pt`

Average real bpp on CLIC2020 test 428:

| run | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| SafeRDO fixed z current | 0.01855 | 0.02209 | 0.02648 | 0.03045 |
| Selective s0.25 z entropy auto | 0.01705 | 0.02078 | 0.02528 | 0.02922 |

Average real bpp split for `SelectiveS025ZEntropyAutoA1`:

| q | total | vs fixed-z | y | z | header | enc s/img | dec s/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| q0 | 0.017055 | -8.05% | 0.014778 | 0.002027 | 0.000249 | 0.746 | 1.016 |
| q1 | 0.020775 | -5.94% | 0.018318 | 0.002208 | 0.000249 | 0.817 | 1.091 |
| q2 | 0.025282 | -4.53% | 0.022711 | 0.002322 | 0.000249 | 0.902 | 1.181 |
| q3 | 0.029219 | -4.05% | 0.026681 | 0.002288 | 0.000249 | 0.989 | 1.270 |

Metrics:

- CSV:
  `experiments/real_codec/analysis/clic2020_test_safe_current_vs_selective_s025_zentropy_metrics.csv`
- BD summary:
  `experiments/real_codec/analysis/clic2020_test_safe_current_vs_selective_s025_zentropy_bd.md`
- FID/KID protocol: 256 x 256 patches with normal tiling plus 128-pixel shift
  (`FID_PATCHES=28650`, `KID_PATCHES=28650`).

CLIC2020 test BD-rate versus current fixed-z SafeRDO:

| branch | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| Selective s0.25 + z entropy auto | -5.87% | -6.58% | -8.12% | -7.42% | -6.29% | -10.17% |

Per-point metric notes:

- q0 and q1 reduce real bpp while keeping DISTS essentially equal and slightly
  improving LPIPS/FID.
- q2 and q3 keep LPIPS/PSNR/MS-SSIM slightly better, while DISTS/FID are nearly
  tied or slightly worse at the same q index.  The BD-rate comparison still
  favors the selective package because the curve is shifted left in real bpp.
- The y stream is identical to the fixed-z anchor in this conservative package;
  the measured gain is dominated by decoder-consistent entropy coding of
  `z_hat`, with the small no-side selective synthesis branch helping quality.

Research interpretation:

- The CLIC run is package validation, not a new mechanism search.
- The current conservative package is now validated on Kodak24, DIV2K
  validation, and full CLIC2020 test 428 under real serialized bpp, but the
  main table should not mix the `z_hat` entropy-coding gain into the GP-ResLC
  method gain.
- `z_hat` entropy coding is a useful codec cleanup and should be reported as an
  auxiliary/appendix result.  It is not the central GP-ResLC mechanism.
- This is a good stopping point for documentation.  The next research step
  should not be more q/rho/loss tuning; it should move back to the GP-ResLC
  mainline mechanisms: stage residual omission diagnostics, same-bpp omitted
  residual synthesis diagnostics, and learned residual/control entropy coding.

## 2026-07-01 JST - z entropy excluded BD-rate audit

Motivation:

- `z_hat` entropy coding is decode-equivalent and improves real bpp, but it is
  not specific to the GP-ResLC residual/generator-recovery idea.
- For the main GP-ResLC claim, the primary comparison should exclude this gain:
  use the fixed-width `z_hat` rate for both anchor and proposal, while keeping
  the selective reconstruction quality.  This isolates the no-side selective
  synthesis effect from the general codec cleanup.

Derived metric files:

- CLIC:
  `experiments/real_codec/analysis/clic2020_test_safe_current_vs_selective_s025_no_zentropy_metrics.csv`
- DIV2K:
  `experiments/real_codec/analysis/div2k_valid_safe_current_vs_selective_s025_no_zentropy_metrics.csv`
- Kodak:
  `experiments/real_codec/analysis/kodak24_safe_vs_selective_s025_no_zentropy_metrics.csv`
- CLIC/DIV2K BD:
  `experiments/real_codec/analysis/no_zentropy_selective_s025_bd_clic_div2k.md`
- Kodak BD:
  `experiments/real_codec/analysis/no_zentropy_selective_s025_bd_kodak24.md`

z entropy excluded BD-rate versus the same fixed-z SafeRDO anchor:

| dataset | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test 428 | +0.18% | -0.65% | -2.42% | -1.66% | -0.17% | -4.89% |
| DIV2K validation | -0.52% | -0.72% | -2.32% | -1.37% | -0.47% | -2.74% |
| Kodak24 | -0.16% | -0.74% | -2.73% | -1.54% | -0.61% | -95.16% |

Interpretation:

- After removing `z_hat` entropy coding, the current selective synthesis branch
  is only a small quality-side improvement.  It should not be claimed as a
  large GP-ResLC rate reduction.
- The large `-5%` to `-10%` BD-rate table is useful as a full codec-package
  result, but it must be clearly separated from the method-mainline result.
- Kodak KID is unstable and should not be used as a primary claim because the
  dataset is small; this is especially visible in the z-excluded derived table.
- Mainline research priority remains unchanged: attack `y`/residual/control
  coding directly.  The next paper-facing gains should come from actual
  residual omission, residual synthesis, or learned residual/control entropy,
  not from `z_hat` coding cleanup.
