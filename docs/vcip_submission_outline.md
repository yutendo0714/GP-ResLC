# VCIP Submission Outline

Last updated: 2026-06-20 JST

## Tentative Title

Generator-Predictable Residual Suppression for Ultra-Low-Bitrate Perceptual Image Compression

## One-Sentence Thesis

At ultra-low bitrates, a generative image codec should not spend bits on residual details that its decoder-side generator can already reconstruct; GP-ResLC implements this principle with a zero-side-bit residual-suppression gate predicted from the transmitted hyper/VQ code.

## Abstract Skeleton

Learned generative image codecs can synthesize plausible visual details from compact semantic latents, but existing bit allocation still spends residual bits in regions that are already predictable by the decoder. We propose GP-ResLC, a generator-predictable residual suppression method for ultra-low-bitrate perceptual image compression. Built on the official GLC codec, the current short-track model predicts a deterministic gate `rho(z_hat, q)` from already transmitted information and uses it to coarsen residual quantization only where the generative decoder is expected to absorb the missing detail. The gate is constrained to `rho >= 1`, costs no side bits, and preserves the interpretation that GP-ResLC never sends extra residual detail relative to GLC. We evaluate with actual serialized bitstreams using fixed-width transmitted `z` indices and arithmetic-coded `y` streams. On CLIC2020 test, GP-ResLC reduces real-codec DISTS/FID BD-rate by 10.28% / 7.30% relative to GLC, and matched-metric interpolation shows 10.26% / 6.02% fewer serialized bits at matched DISTS/FID. DIV2K gives 10.79% / 5.61% DISTS/FID BD-rate reductions. Spatial gate analysis shows that high-rho regions have lower baseline error, texture variance, and gradient magnitude, supporting the intended mechanism of suppressing generator-predictable residuals.

## Contributions

1. A zero-side-bit residual-suppression mechanism for generative learned image compression, where `rho(z_hat, q)` is recomputed at the decoder from transmitted information.
2. A monotone gate parameterization (`rho >= 1`) that keeps the short-track method faithful to the claim: suppress predictable residuals rather than add extra bits.
3. A sendability teacher that encourages stronger suppression in low-error, low-texture, low-gradient regions and protects harder residual regions.
4. Real-codec evidence on CLIC2020 test, DIV2K validation, and Kodak, with DISTS/FID-oriented rate-perception gains.
5. Mechanism analysis through rho-error/texture/gradient correlations and qualitative rho overlays.

## Main Result Table

Use `rho1.16` as the paper-facing model. All rates below are real serialized codec bpp, not estimated likelihood bpp.

| dataset | DISTS BD-rate | FID BD-rate | matched-DISTS bpp | matched-FID bpp | note |
|---|---:|---:|---:|---:|---|
| CLIC2020 test | -10.28% | -7.30% | -10.26% | -6.02% | main official-style natural-image result |
| DIV2K validation | -10.79% | -5.61% | -10.27% | -3.39% | supplementary natural-image support |
| Kodak | -4.47% | -1.70% | -5.45% | -4.40% | small set; distribution metrics are noisier |

LPIPS is not the primary claim. It is close in BD-rate on the real codec package but matched-LPIPS bpp is slightly worse for `rho1.16`, especially on CLIC2020 test.

Official graph-extracted curves are now packaged as supplementary positioning. After correcting CLIC to the full 428-image professional+mobile test set, CLIC and DIV2K both support the external comparison; local reproduced GLC closely matches the official CLIC curve including FID.

## Method Figure

Proposed figure:

```text
x -> GLC encoder -> y, z -> z_hat
                         |
                         v
                    rho = g_phi(z_hat + e_q), rho >= 1
                         |
              q_step' = rho * q_step
                         |
              entropy code coarser residual y_hat
                         |
                 frozen GLC generative decoder -> x_hat
```

Caption point: `rho` is not transmitted. It is a deterministic decoder-side function of `z_hat` and `q`.

## Mechanism Figure

Use:

- `experiments/paper_assets/clic_q3_rho_overlay_top4.png`
- `experiments/paper_assets/kodak_q3_rho_overlay_top4.png`

Mechanism sentence:

High-rho regions correlate negatively with baseline local reconstruction error, texture variance, and gradient magnitude, which indicates that GP-ResLC is not applying uniform coarser quantization. It selectively removes bits from visually predictable residual regions.

## Evaluation Framing

Primary:

- DISTS
- FID
- bpp
- matched-metric bpp reduction
- BD-rate for DISTS/FID

Secondary:

- LPIPS
- PSNR
- MS-SSIM
- KID, explicitly marked as auxiliary/noisy for small or patch-count-sensitive sets

Do not frame the short-track paper as universal R-D superiority. Frame it as rate-perception improvement with honest R-D diagnostics.

## Key Caveats

- The short-track model is gate-based residual suppression, not the full residual-predictor GP-ResLC.
- LPIPS and KID are mixed, so DISTS/FID should remain the main R-P claim.
- LPIPS remains near neutral to slightly worse under matched-LPIPS interpolation.
- KID is non-monotonic and patch-count-sensitive; avoid using it as a central claim.
- Official graph-extracted curves are secondary positioning only; use paired local real-codec GLC for the main claim.

## Next Draft Tasks

1. Convert `docs/vcip_method_draft.md` into a formal Method section with equations.
2. Convert this outline into Introduction and Experiments text.
3. Add a compact ablation table: rho1.12, rho1.16, rho1.20, texture-free, edge guard, Alex-LPIPS, baseline distillation.
4. Prepare final figures from `experiments/paper_assets`.
5. Add a limitations paragraph that separates short-track R-P from full GP-ResLC R-D-P.
