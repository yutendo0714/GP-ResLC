# GP-ResLC Short-Track Method Draft

Last updated: 2026-06-20 JST

## Working Title

Generator-Predictable Residual Suppression for Ultra-Low-Bitrate Perceptual Image Compression

## Short Claim

GP-ResLC reduces perceptual bitrate by avoiding transmission of residual detail that the decoder-side generator can already reconstruct from the transmitted GLC semantic/hyper code. Instead of sending an additional side map, the proposed short-track model predicts a deterministic residual-suppression gate from `z_hat` and the target rate index `q`. The gate coarsens latent residual quantization only in generator-predictable regions.

## Method

Let a pretrained GLC image codec encode an image into a main latent `y` and a hyper/VQ latent `z_hat`. GLC decodes with a generative decoder and entropy-codes `y` conditioned on `z_hat`. GP-ResLC keeps the GLC analysis/synthesis path and entropy model frozen for the current short-track model, and inserts a decoder-recomputable gate:

```text
rho = g_phi(z_hat + e_q),        rho >= 1
q_step' = rho * q_step
y_hat = round(y / q_step') * q_step'
```

Here `e_q` is a learned rate embedding and `rho` is spatially upsampled to the main-latent resolution. Since `rho` depends only on already transmitted information, it costs zero side bits. The monotone constraint `rho >= 1` is important: GP-ResLC never sends extra residual detail relative to GLC; it only suppresses residuals that the generative decoder is expected to absorb perceptually.

The learned gate is trained with a rate-perception objective:

```text
L = lambda_R * R + lambda_P * P(x, x_hat) + lambda_D * D_aux(x, x_hat)
    + lambda_rho * ||mean(rho) - rho_target||^2
    + lambda_send * BCE(normalize(rho), s(x, x_glc))
```

`P` is currently LPIPS-based during training, while DISTS/FID/KID are primary evaluation metrics. `D_aux` is a small stabilizer, not the headline objective. The sendability target `s` is a teacher map: regions where the frozen GLC generator already has low local reconstruction error, low texture variance, and low gradient are encouraged to receive higher `rho`; harder residual regions are protected with lower `rho`.

## Why This Is Not Uniform Re-Quantization

Uniformly increasing quantization would reduce rate but damage all image regions indiscriminately. GP-ResLC predicts `rho` from `z_hat` and `q`, and the measured maps are spatially selective:

| dataset | mean rho | corr(rho, base err) | corr(rho, texture var) | corr(rho, gradient) |
|---|---:|---:|---:|---:|
| Kodak q3 | 1.1716 | -0.234 | -0.243 | -0.213 |
| CLIC-prof-valid q3 | 1.1815 | -0.271 | -0.268 | -0.290 |
| CLIC-mobile-valid q3 | 1.1804 | -0.251 | -0.251 | -0.262 |

High-rho regions have substantially lower local error, texture variance, and gradient magnitude than low-rho regions. This supports the intended mechanism: suppress predictable residuals and keep bits for unpredictable residuals.

## Current Evidence

Paper-facing evidence now uses the implemented real codec: bpp is counted from serialized payload bytes, with fixed-width transmitted `z` indices, compact metadata, and `torchac` arithmetic-coded `y` streams.

BD-rate versus the GLC real codec:

| dataset | run | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | rho1.16 real | -10.28% | +0.19% | -0.98% | +0.38% | -7.30% | -7.10% |
| DIV2K validation | rho1.16 real | -10.79% | -0.54% | -1.49% | -0.17% | -5.61% | -6.50% |
| Kodak | rho1.16 real | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.14% |

Per-q serialized bpp reductions are consistent: CLIC2020 test ranges from -11.36% at q0 to -7.91% at q3, DIV2K from -10.39% to -7.16%, and Kodak from -9.52% to -7.17%. Since `z` and header costs are identical between methods, these reductions come from the arithmetic-coded `y` stream.

A complementary matched-metric interpolation gives the cleanest headline: CLIC2020 test requires 10.26% fewer serialized bits at matched DISTS and 6.02% fewer bits at matched FID. DIV2K requires 10.27% fewer bits at matched DISTS and 3.39% fewer bits at matched FID. Kodak requires 5.45% fewer bits at matched DISTS and 4.40% fewer bits at matched FID. LPIPS remains auxiliary because matched-LPIPS bpp is near neutral to slightly worse.

A secondary official-curve comparison has been generated from graph-extracted GLC paper points. After switching CLIC to the full 428-image professional+mobile test set, local real-codec GLC matches the official CLIC curve closely, including FID. GP-ResLC gives -9.07% DISTS BD-rate and -6.10% FID BD-rate versus official GLC on CLIC2020, and -9.62% / -4.23% on DIV2K.

## Evaluation Policy

Use DISTS and FID as primary perceptual metrics because they match the GLC paper's perceptual evaluation emphasis and better reflect the low-bitrate generative-compression setting. Report KID as auxiliary because it is noisy on small validation sets. Report LPIPS, PSNR, and MS-SSIM honestly as secondary diagnostics. Avoid claiming universal RD superiority for the short-track model; the current contribution is R-P oriented.

## Main Figures

1. Rate-perception curves: real-bpp-DISTS, real-bpp-FID, real-bpp-LPIPS for CLIC2020 test, DIV2K, and Kodak.
2. Rho overlay qualitative grid: Original / GLC / GP-ResLC / rho overlay at q3.
3. Gate-correlation table showing negative correlation between rho and local error/texture/gradient.
4. Ablation table over rho target and teacher variants.

Current figure assets:

- `experiments/paper_assets/clic_q3_rho_overlay_top4.png`
- `experiments/paper_assets/clic_mobile_q3_rho_overlay_top4.png`
- `experiments/paper_assets/kodak_q3_rho_overlay_top4.png`
- `experiments/paper_assets/clic2020_test_real_curves/`
- `experiments/paper_assets/div2k_real_curves/`
- `experiments/paper_assets/kodak_real_curves/`
- `experiments/paper_assets/official_curve_comparison/`

## Limitations and Next Full-Version Direction

The short-track model is not yet the full GP-ResLC R-D-P system. It does not transmit an explicit learned residual code beyond GLC's original bitstream; instead, it suppresses generator-predictable residuals through a zero-side-bit gate. The full version should revisit a stage-aware residual predictor that directly models the unpredictable component of `y`, while preserving the short-track insight that predictable residual detail should not consume bits.
