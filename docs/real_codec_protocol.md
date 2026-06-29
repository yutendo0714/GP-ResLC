# Real Codec Evaluation Protocol

Last updated: 2026-06-20 JST

## Goal

For paper-facing LIC evaluation, report bitrate from an actual serialized codec path instead of estimated likelihood sums. The implemented real codec keeps the GLC / GP-ResLC inference graph unchanged, but measures

`bpp = 8 * len(serialized_payload_bytes) / (original_height * original_width)`.

Encode/decode wall time is measured around the full payload construction and reconstruction path with CUDA synchronization.

## Payload

Implementation: `gp_reslc/real_codec.py`, runner: `scripts/evaluate_real_codec.py`.

The payload contains:

- compact binary header, counted in bpp;
- `z` as fixed-width VQ codebook indices, matching the public GLC assumption `ceil(log2(codebook_size))` bits/index;
- four `y` arithmetic-coded streams using `torchac`, in the same four-part spatial-prior order as `GLC_Image.test()`;
- per-stream integer support metadata, counted in bpp.

For the Gaussian-coded `y` residuals, the CDF includes explicit lower/upper tail symbols. The observed integer symbols are encoded inside the finite support, while tail mass is preserved so the interval probabilities match the untruncated Gaussian model for observed values.

## Consistency Check

The real decoder reconstructs the same padded image tensor as the previous estimated-bpp forward path. Smoke checks:

- GLC Kodak `kodim01`, q0: `max_abs(real_decode - net.test) = 0.000e+00`.
- GP-ResLC lead checkpoint, Kodak `kodim01`, q0: `max_abs(real_decode - train_forward) = 0.000e+00`.

Thus this changes the measured rate and timing, not the image reconstruction path.

## Usage

GLC:

```bash
.venv/bin/python scripts/evaluate_real_codec.py   --glc_weights pretrained/GLC_image.pth.tar   --input /dpl/kodak   --out experiments/real_codec/kodak_glc   --q_indexes 0 1 2 3
```

GP-ResLC lead checkpoint:

```bash
.venv/bin/python scripts/evaluate_real_codec.py   --glc_weights pretrained/GLC_image.pth.tar   --ckpt experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt   --input /dpl/kodak   --out experiments/real_codec/kodak_gp_reslc_rho116   --q_indexes 0 1 2 3   --predictor_param_mode mean
```

Quality metrics can be computed with the existing grid evaluator; it reads each `q*/bpp.json`, so the resulting CSV uses real codec bpp. Metrics CSVs now include `FID_PATCHES`, `KID_PATCHES`, `FID_PATCH_SIZE`, and `FID_SPLIT_PATCH_NUM` to prevent silent protocol drift:

```bash
.venv/bin/python scripts/evaluate_recon_grid.py   --orig /dpl/kodak   --run glc_real=experiments/real_codec/kodak_glc   --run gp_rho116_real=experiments/real_codec/kodak_gp_reslc_rho116   --q_indexes 0 1 2 3   --patch 64   --split_patch_num 2   --out_json experiments/real_codec/kodak_real_metrics.json   --out_csv experiments/real_codec/kodak_real_metrics.csv
```

## Kodak Real-Codec Result

Protocol: `/dpl/kodak`, original resolution, 24 images. Paper-style 256-patch FID/KID is omitted, matching the GLC paper practice, because only 192 shifted 256-patches are available and KID is unstable. The local diagnostic FID/KID numbers below use the public GLC Kodak workaround, `--patch 64 --split_patch_num 2`, yielding 4,152 patches.

Artifacts:

- GLC payload/recon: `experiments/real_codec/kodak_glc/`
- GP-ResLC payload/recon: `experiments/real_codec/kodak_gp_reslc_rho116/`
- Metrics CSV: `experiments/real_codec/kodak_real_metrics.csv`
- BD-rate: `experiments/real_codec/kodak_real_bd_rate_summary.md`
- Matched metrics: `experiments/real_codec/kodak_real_matched_metric_summary.md`

Average real bpp and time:

| q | GLC bpp | GP-ResLC bpp | delta | GLC y bpp | GP y bpp | y delta | GLC enc/dec s | GP enc/dec s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02620 | 0.02371 | -9.52% | 0.02109 | 0.01860 | -11.82% | 0.099 / 0.103 | 0.086 / 0.104 |
| 1 | 0.03013 | 0.02739 | -9.09% | 0.02502 | 0.02228 | -10.94% | 0.067 / 0.103 | 0.068 / 0.105 |
| 2 | 0.03472 | 0.03197 | -7.93% | 0.02962 | 0.02686 | -9.30% | 0.069 / 0.106 | 0.069 / 0.106 |
| 3 | 0.03897 | 0.03618 | -7.17% | 0.03386 | 0.03107 | -8.25% | 0.071 / 0.109 | 0.071 / 0.109 |

`z` and header bpp are identical for both methods: `z=0.00342`, header `0.00169` on Kodak. Therefore the rate saving comes from the arithmetic-coded `y` stream, which is the intended GP-ResLC mechanism.

Kodak real-bpp BD-rate versus `glc_real`:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| gp_rho116_real | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.14% |

Matched-metric bpp delta:

| metric | points | mean | range |
|---|---:|---:|---:|
| DISTS | 4 | -5.45% | -8.35..-2.29% |
| FID | 4 | -4.40% | -8.55..-0.05% |
| LPIPS | 3 | +0.34% | +0.13..+0.60% |


## DIV2K Real-Codec Result

Protocol: `/dpl/div2k`, original resolution, 100 images (`0801.png`-`0900.png`). The local official-patch metric protocol yields 6,573 shifted 256-patches, matching the GLC supplement count.

Artifacts:

- GLC payload/recon: `experiments/real_codec/div2k_glc/`
- GP-ResLC payload/recon: `experiments/real_codec/div2k_gp_reslc_rho116/`
- Metrics CSV: `experiments/real_codec/div2k_real_metrics.csv`
- BD-rate: `experiments/real_codec/div2k_real_bd_rate_summary.md`
- Matched metrics: `experiments/real_codec/div2k_real_matched_metric_summary.md`

Average real bpp and time:

| q | GLC bpp | GP-ResLC bpp | delta | GLC y bpp | GP y bpp | y delta | GLC enc/dec s | GP enc/dec s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02381 | 0.02133 | -10.39% | 0.02004 | 0.01756 | -12.35% | 0.693 / 0.963 | 0.654 / 0.925 |
| 1 | 0.02764 | 0.02507 | -9.29% | 0.02387 | 0.02130 | -10.76% | 0.747 / 1.038 | 0.718 / 0.992 |
| 2 | 0.03224 | 0.02961 | -8.15% | 0.02847 | 0.02584 | -9.23% | 0.861 / 1.146 | 0.811 / 1.084 |
| 3 | 0.03649 | 0.03388 | -7.16% | 0.03273 | 0.03011 | -7.98% | 0.996 / 1.283 | 0.930 / 1.208 |

DIV2K real-bpp BD-rate versus `glc_real`:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| gp_rho116_real | -10.79% | -0.54% | -1.49% | -0.17% | -5.61% | -6.50% |

Matched-metric bpp delta:

| metric | points | mean |
|---|---:|---:|
| DISTS | 4 | -10.27% |
| FID | 3 | -3.39% |
| LPIPS | 3 | +0.36% |


## CLIC2020 Test Real-Codec Result

Protocol: combined CLIC2020 test at original resolution: `/dpl/clic/professional/test` plus `/dpl/clic/mobile/test`, 428 PNG images. The local official-patch metric protocol yields 28,650 shifted 256-patches, matching the reported GLC/HiFiC CLIC2020-test protocol.

Artifacts:

- Combined input symlink set: `data/clic2020_test_combined/`
- GLC payload/recon: `experiments/real_codec/clic2020_test_glc/`
- GP-ResLC payload/recon: `experiments/real_codec/clic2020_test_gp_reslc_rho116/`
- Metrics CSV: `experiments/real_codec/clic2020_test_real_metrics.csv`
- BD-rate: `experiments/real_codec/clic2020_test_real_bd_rate_summary.md`
- Matched metrics: `experiments/real_codec/clic2020_test_real_matched_metric_summary.md`

Average real bpp and time:

| q | GLC bpp | GP-ResLC bpp | delta | GLC y bpp | GP y bpp | y delta | GLC enc/dec s | GP enc/dec s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02134 | 0.01892 | -11.36% | 0.01757 | 0.01515 | -13.79% | 0.693 / 0.972 | 0.655 / 0.934 |
| 1 | 0.02503 | 0.02244 | -10.32% | 0.02126 | 0.01867 | -12.15% | 0.764 / 1.050 | 0.726 / 1.011 |
| 2 | 0.02958 | 0.02699 | -8.74% | 0.02581 | 0.02322 | -10.02% | 0.866 / 1.151 | 0.798 / 1.084 |
| 3 | 0.03369 | 0.03102 | -7.91% | 0.02992 | 0.02726 | -8.91% | 0.971 / 1.256 | 0.904 / 1.188 |

`z` and header bpp are identical for both methods: `z=0.00352`, header `0.00025` on CLIC2020 test. Therefore the rate saving comes from the arithmetic-coded `y` stream, which is the intended GP-ResLC mechanism.

CLIC2020 test real-bpp BD-rate versus `glc_real`:

| run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| gp_rho116_real | -10.28% | +0.19% | -0.98% | +0.38% | -7.30% | -7.10% |

Matched-metric bpp delta:

| metric | points | mean | range |
|---|---:|---:|---:|
| DISTS | 4 | -10.26% | -11.08..-9.93% |
| FID | 3 | -6.02% | -6.12..-5.84% |
| LPIPS | 3 | +1.24% | +0.36..+1.70% |

## Caveats

- The public GLC image model does not expose `compress()` / `decompress()` methods, so this implementation reconstructs a real codec around the exact public inference graph.
- `z` is fixed-width because the public GLC code evaluates it as fixed `log2(codebook_size)` bits; no learned entropy model for `z` is present in the public image model.
- Kodak header bpp is relatively visible because images are small. For high-resolution CLIC/DIV2K images the same header contributes less per pixel.
- Current GP-ResLC real codec supports `predictor_param_mode` in `mean`, `scale_mean`, and `all`; it intentionally rejects `latent_residual` until that path has a precise decoder-side bitstream design.
