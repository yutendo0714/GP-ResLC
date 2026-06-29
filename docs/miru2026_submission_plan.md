# MIRU2026 Submission Plan

Date: 2026-06-22 JST

## Recommendation

Submit the current best real-codec GP-ResLC result to MIRU2026 as a general
paper/poster, not as a new oral-candidate paper.

Reason:

- The oral-candidate deadline has already passed.
- The general-paper deadline is 2026-06-22.
- MIRU is a good intermediate venue for stress-testing the story before a
  larger international submission.
- The current strongest evidence is already real-codec, full-resolution, and
  evaluated on CLIC2020 test, DIV2K, and Kodak.

## Submission Type

Use a 4-page extended abstract, references excluded.

Recommended track/area:

- Area C: Image/video/multimedia processing, especially low-level vision /
  image synthesis / multimedia.
- Area B is also defensible because the method is learned compression, but Area
  C is the clearer fit.

## Main Claim

In ultra-low bitrate generative image compression, GP-ResLC reduces transmitted
bits by suppressing information that the pretrained generative latent decoder
can already recover, and spends bits on the less predictable residual stream.

Safer wording:

> We introduce a decoder-consistent residual-precision control layer on top of
> GLC and show that it reduces arithmetic-coded latent payload while preserving
> perceptual quality under the official-style full-resolution evaluation
> protocol.

Avoid overclaiming:

- Do not claim a complete scratch-trained codec as the main result.
- Do not claim improvement on every metric; LPIPS is near-neutral and sometimes
  slightly worse.
- Do not lead with PSNR/MS-SSIM.
- Do not present center-crop proxy scratch results as paper-facing evidence.

## Lead Checkpoint

Use:

`experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`

W&B:

`a2w5fjt4`

## Paper-Facing Results

Use real arithmetic-coded bpp only.

Local paired real-codec comparison against local GLC:

| Dataset | DISTS BD-rate | FID BD-rate | Matched-DISTS bpp |
|---|---:|---:|---:|
| CLIC2020 test, 428 images | -10.28% | -7.30% | -10.26% |
| DIV2K validation, 100 images | -10.79% | -5.61% | -10.27% |
| Kodak, 24 images | -4.47% | -1.70% | -5.45% |

External positioning against graph-extracted official GLC:

| Dataset | DISTS BD-rate | FID BD-rate |
|---|---:|---:|
| CLIC2020 test | -9.07% | -6.10% |
| DIV2K validation | -9.62% | -4.23% |

Use the local paired real-codec comparison as the primary table, and official
curve positioning as a secondary sanity/positioning statement.

## Evaluation Protocol To State Clearly

- Full-resolution compression and reconstruction.
- bpp is measured from serialized bitstream bytes, not likelihood-estimated bpp.
- CLIC2020 test is professional 250 + mobile 178 = 428 images.
- CLIC2020 FID/KID uses shifted 256x256 patches, 28,650 patches.
- DIV2K validation uses shifted 256x256 patches, 6,573 patches.
- Kodak uses DISTS/LPIPS/PSNR/MS-SSIM as the main stable metrics; FID/KID are
  not emphasized because Kodak has too few 256x256 patches.

## Four-Page Structure

1. Introduction and motivation:
   - Ultra-low bitrate LIC has a rate-perception bottleneck.
   - GLC already uses a perceptual/generative latent space.
   - Still, not all latent information is equally necessary to transmit.
   - GP-ResLC asks whether decoder-predictable information can be suppressed.

2. Method:
   - Start from frozen pretrained GLC.
   - Add a decoder-consistent residual/precision control module.
   - It changes the arithmetic-coded latent payload while keeping decode exact.
   - Emphasize no source-only side map and no estimated-bpp claim.

3. Experiments:
   - CLIC2020 test, DIV2K validation, Kodak.
   - Real codec bpp and encode/decode timing if space permits.
   - Main metrics: DISTS, FID/KID where stable, LPIPS as auxiliary.

4. Discussion:
   - Results show consistent bpp reduction at matched perceptual quality.
   - Scratch/full GP-ResLC remains future work.
   - Limitations: LPIPS not consistently improved; current strongest model is a
     pretrained GLC overlay rather than a fully retrained codec.

## Figures/Tables

Minimum package:

- Table 1: BD-rate vs local real-codec GLC on CLIC2020/DIV2K/Kodak.
- Figure 1: DISTS and FID rate-quality curves for CLIC2020 and DIV2K.
- Figure 2: qualitative grid with original / GLC / GP-ResLC / rho overlay.
- Table 2 or supplement: bpp split showing `z` and header unchanged, `y` stream
  reduced.

Existing assets:

- `experiments/paper_assets/clic2020_test_real_curves/`
- `experiments/paper_assets/div2k_real_curves/`
- `experiments/paper_assets/kodak_real_curves/`
- `experiments/paper_assets/*rho_overlay_top4.png`
- `experiments/paper_assets/official_curve_comparison/`

## Risk Points

- MIRU reviewers may ask whether this is only a rate knob over GLC. Answer with
  real-codec curve-level BD-rate, bpp split, and decoder-consistency.
- The method is currently strongest as a pretrained overlay. Be honest: position
  scratch as future full-design training, not the submitted main method.
- For future international submission, keep MIRU text concise and avoid
  overclaiming unpublished scratch results.

