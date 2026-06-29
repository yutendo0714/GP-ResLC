# Scratch GP-ResLC Design Plan

Last updated: 2026-06-20 JST

This is the high-risk/high-upside branch. The pretrained GLC branch is the safety path; scratch GP-ResLC is the path to remove architectural constraints and make the original claim exact.

## Core Claim

At ultra-low bitrate, the bitstream should contain:

1. a compact semantic/generative code `s`, and
2. only the unpredictable residual `r = y - mu_theta(s)`.

The decoder reconstructs `y_hat = mu_theta(s) + r_hat` and then generates `x_hat`. Any component predictable from `s` should not consume residual bits.

## Why Scratch May Beat Pretrained GLC

Pretrained GLC was not trained to factorize latents into predictable and unpredictable components. Its `z_hat` is a hyper/VQ code for the existing entropy model, not necessarily a semantic code optimized to predict generator-recoverable latent content. Its `y` latent is also not constrained to decompose cleanly as `mu_theta(s) + r`.

A scratch model can enforce this decomposition from the start:

- `s` is trained as a semantic/generative code.
- `mu_theta(s)` is trained to explain as much of `y` as the decoder can recover without residual bits.
- `r` is pressure-minimized by entropy coding.
- the generator is trained to recover perceptual detail from `s` and only use `r` where necessary.

## Stage Plan

### Stage A: Generative Base Autoencoder

Goal: learn a stable perceptual autoencoder before entropy coding.

- Encoder: `E_s(x) -> s_cont`, vector quantized to `s_idx`.
- Decoder/generator: `G_s(s_hat) -> x_base`.
- Loss: LPIPS + DISTS-lite/feature loss + small L1/MSE + optional patch GAN.
- Metrics: FID/KID/DISTS/LPIPS on Kodak/DIV2K subset.
- Risk: codebook collapse or semantic drift.

Exit criterion: `s`-only reconstruction is visually plausible at ultra-low token budget.

### Stage B: Residual Latent Autoencoder

Goal: introduce a continuous latent `y` and a predictor `mu_theta(s)`.

- `E_y(x, s_hat) -> y`.
- `mu_theta(s_hat) -> y_pred`.
- residual `r = y - y_pred`.
- Decoder: `G(y_pred + r_hat, s_hat) -> x_hat`.
- Loss: reconstruction/perceptual + `lambda_pred * smooth_l1(y_pred, stopgrad(y))` + entropy proxy on `r`.

Exit criterion: residual entropy is clearly lower than coding `y` directly at matched perceptual quality.

### Stage C: Entropy Model and Real Codec

Goal: convert residual proxy into an arithmetic-coded bitstream.

- Entropy model: hyperprior for `r`, channel slices, checkerboard/four-part spatial context.
- Semantic code entropy: start fixed-width VQ; later add categorical hyperprior like HVQ-CGIC.
- Real codec: encode `s_idx`, encode residual symbols with finite-support arithmetic coder, decode deterministic `mu_theta(s)` and add residual back.

Exit criterion: real codec bpp is lower than GLC official curve at matched DISTS/FID on Kodak/DIV2K subset.

### Stage D: Joint Rate-Perception Fine-Tune

Goal: maximize official-curve gains.

- Objective: `R_s + R_r + lambda_P * (LPIPS/DISTS proxy) + lambda_D * small distortion + optional GAN`.
- Use multi-rate conditioning to cover q points.
- Evaluate every checkpoint with estimated metrics; promote only winners to real codec.

Exit criterion: CLIC2020 full real-codec official comparison improves DISTS/FID by at least 15-25% BD-rate.

## Practical Model Skeleton

Recommended first scratch backbone:

- Analysis/synthesis transforms: TCM-like ConvNeXt/Swin hybrid blocks, not plain conv only.
- Semantic quantizer: VQ with EMA or entropy-regularized codebook; start with 8x8 or 16x16 grid.
- Residual latent: lower-dimensional than GLC `y`; keep channel grouping explicit.
- Entropy model: four-part spatial context first, then MLIC-like multi-reference slices if needed.
- Generator: lightweight VQGAN-style decoder initially; one-step diffusion is journal/extension unless scratch autoencoder is stable.

## Decision Policy

Scratch is worth running, but not allowed to endanger the VCIP package. Maintain two branches:

- Safety branch: pretrained GLC + latent residual / rho-gate real codec.
- Upside branch: scratch GP-ResLC staged training.

A scratch result becomes VCIP lead only if it beats the pretrained branch on real-codec CLIC2020/DIV2K official-protocol curves, not just estimated bpp.

## Immediate Implementation Tasks

1. Finish pretrained `latent_residual` diagnosis.
2. Add a minimal scratch package under `gp_reslc/scratch/` only after confirming the pretrained branch signal.
3. Start Stage A on OpenImages 256 crops with W&B logging.
4. Use Kodak/DIV2K subset smoke evaluation before any full CLIC run.


## Implementation Status: Stage A Scaffold

Implemented files:

- `gp_reslc/scratch/vq_autoencoder.py`
- `gp_reslc/scratch/__init__.py`
- `scripts/train_scratch_stage_a.py`

Current Stage A model:

- 4x downsample VQ autoencoder: 256x256 input -> 16x16 latent index grid.
- Default `codebook_size=1024`, so fixed semantic-index rate is `16*16*10 / 256^2 = 0.03906 bpp` before entropy coding.
- Loss: L1 + LPIPS + DISTS + VQ, with optional differentiable soft codebook entropy regularization.

Pilot findings:

- Plain VQ collapsed quickly to hard perplexity about 2-3.
- Entropy computed from hard argmin indices is not sufficient because it has no useful gradient through the discrete assignment.
- Soft assignment entropy with `softmax(-dist/tau)`, `tau=0.01`, and `lambda_codebook_entropy=0.5` kept hard perplexity around 30-40 in a 1500-iteration pilot.
- Reconstruction quality is still far from paper-ready; this branch is infrastructure and risk reduction, not a current VCIP lead.

Next Stage A tasks:

1. Add EMA codebook updates or dead-code restart to stabilize usage without relying entirely on soft entropy.
2. Train a longer Stage A pilot and inspect validation panels, not only scalar metrics.
3. Add a lightweight patch discriminator only if LPIPS/DISTS plateau with blurry reconstructions.
4. After semantic reconstructions become plausible, implement Stage B: `y = mu_theta(s) + r`, residual entropy proxy, and then real residual codec.


## Stage A Update: Codebook Restart and Semantic Rate

The latest Stage-A run `scratch_stage_a_vq1024_b64_z128_softent_restart_2k` (W&B `2d7yi3uk`) completed 2000 iterations. Dead-code restart every 200 iterations substantially improved codebook utilization: validation hard perplexity reached `143.7` at it=1500 and final train perplexity was `150.5`, compared with roughly `30-40` in the soft-entropy-only pilot.

However, the 16x16 semantic grid is not rate-compatible with the final codec if used naively: `1024` codes at 16x16 cost `0.03906` fixed bpp before residual coding. This is already near or above GLC q3 total rate. Therefore Stage A now supports `num_down=5`, giving an 8x8 semantic grid and `0.00977` fixed bpp for 256 crops. This lower-rate semantic stream is the better scratch path for the full claim: transmit a cheap generator-conditioning code, then spend remaining bits only on residual information that the generator cannot infer.

Next scratch experiment:

```bash
.venv/bin/python -u scripts/train_scratch_stage_a.py \
  --data /dpl/openimages/train --val /dpl/kodak \
  --out experiments/scratch_stage_a_vq1024_b80_z160_down5_softent_restart_8k \
  --iters 8000 --bs 4 --base_ch 80 --latent_dim 160 --codebook_size 1024 --num_down 5 \
  --vq_beta 0.1 --vq_entropy_tau 0.01 --lambda_codebook_entropy 0.5 \
  --codebook_restart_every 200 --codebook_restart_threshold 0.00001 --codebook_restart_max_fraction 0.05 \
  --lr 0.0002 --lambda_l1 0.5 --lambda_lpips 1.0 --lambda_dists 1.0 --lambda_vq 1.0 \
  --num_workers 4 --log_every 50 --eval_every 500 --save_every 2000 \
  --wandb_project gp-reslc-vcip \
  --wandb_name scratch_stage_a_vq1024_b80_z160_down5_softent_restart_8k \
  --wandb_mode online
```


## Stage B Update: First Hard-Quantized Residual Signal

Stage B is now implemented as a semantic-conditioned residual bottleneck:

- Stage A transmits/fixes the VQ semantic code `s`.
- The decoder predicts `mu_theta(s)` without extra bits.
- A narrow residual latent encodes `r = y - mu_theta(s)`.
- At evaluation, residual symbols are hard-rounded and scored by a Gaussian entropy proxy.
- The residual decoder is zero-initialized, so Stage B starts exactly from the Stage-A base reconstruction.

The first useful run is `scratch_stage_b_down5_r16_q1_lR0p1_pred001_3k` (W&B `8fgx365x`), using Stage-A checkpoint `stage_a_0006000.pt`, `residual_dim=16`, `quant_step=1.0`, `lambda_R=0.1`, and `lambda_pred=0.01`.

Deterministic Kodak center result:

- Stage-A base: semantic bpp `0.00977`, LPIPS `0.45782`, DISTS `0.45264`.
- Stage-B final: semantic bpp `0.00977`, residual bpp `0.02872`, total bpp `0.03848`, LPIPS `0.43485`, DISTS `0.43711`.

Interpretation: the mechanism is now on-axis but not yet strong enough. It proves that hard-quantized unpredictable residual can improve perceptual metrics over the semantic-only generator. It does not yet challenge GLC because the Stage-A generator is weak and the residual proxy is not entropy-calibrated. The next step is a rate/quality sweep around `lambda_R=0.3-0.5` with stronger DISTS weighting, followed by a real residual codec only if the proxy curve becomes competitive.


## Stage B Update: Better Rate-Quality Point

A DISTS-weighted Stage-B run improved the residual tradeoff: `scratch_stage_b_down5_r16_q1_lR0p3_d2_3k` (W&B `2ii44jvx`) uses `lambda_R=0.3`, `lambda_lpips=0.7`, and `lambda_dists=2.0`.

Best deterministic Kodak center checkpoint so far:

- `stage_b_0002000.pt`
- semantic bpp `0.00977`
- residual proxy bpp `0.00861`
- total proxy bpp `0.01838`
- base LPIPS/DISTS `0.45782 / 0.45264`
- ours LPIPS/DISTS `0.45164 / 0.44077`

This is a better proof of the original claim than the previous `lambda_R=0.1` final: the residual is narrow, hard-quantized at eval, and improves DISTS with only about `0.0086` residual bpp. The remaining gap is absolute decoder quality, not the decomposition mechanism itself.


## Stage B Update: residual_dim=8 Pareto Point

The `residual_dim=8` sweep produced the best current Stage-B quality point:

- run: `scratch_stage_b_down5_r8_q1_lR0p3_d2_3k`
- W&B: `r925d692`
- checkpoint: `stage_b_final.pt`
- semantic bpp `0.00977`
- residual proxy bpp `0.01369`
- total proxy bpp `0.02345`
- LPIPS `0.45782 -> 0.44438`
- DISTS `0.45264 -> 0.43024`

The current scratch Pareto has two useful points: r16 at total bpp `0.01838` with DISTS `0.44077`, and r8 at total bpp `0.02345` with DISTS `0.43024`. This supports the decomposition claim, but the absolute quality gap to GLC means the scratch branch should keep improving Stage A/generator quality before any paper-leading claim.


## Stage B Update: Best Current Low-Rate Residual Curve

The best current Stage-B operating point is `scratch_stage_b_down5_r8_q1_lR0p5_d2_3k` (W&B `wwi995cn`). It uses `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.5`, and DISTS-heavy perceptual loss.

Useful deterministic Kodak center points:

- `stage_b_0001000.pt`: total bpp `0.01631`, LPIPS `0.45442`, DISTS `0.44100`.
- `stage_b_0002000.pt`: total bpp `0.01775`, LPIPS `0.45258`, DISTS `0.43456`.
- `stage_b_final.pt`: total bpp `0.01815`, LPIPS `0.44694`, DISTS `0.43709`.
- `stage_b_final.pt` from `lambda_R=0.3`: total bpp `0.02345`, LPIPS `0.44438`, DISTS `0.43024`.

The decomposition mechanism is now empirically real: at hard-quantized evaluation, adding only about `0.0065-0.0137` residual proxy bpp over a `0.00977` semantic code consistently improves DISTS. The bottleneck is the weak scratch generator, not the residual factorization idea.


## Stage A Update: Plain Continuation Is Weak

Continuing the down5 Stage-A VQ-AE from `stage_a_0006000.pt` to 10k iterations with lower lr gave only a tiny deterministic Kodak-center improvement at the best checkpoint: LPIPS/DISTS `0.45782/0.45264 -> 0.45730/0.45180`. The 10k checkpoint improved LPIPS to `0.44929` but worsened DISTS to `0.45899`.

Conclusion: Stage-A needs a changed generator objective, not just more iterations. The next credible lever is adversarial/perceptual fine-tuning or a stronger decoder architecture.


## Stage A Update: Naive PatchGAN Fine-Tuning Failed

A conservative PatchGAN fine-tune (`scratch_stage_a_adv_down5_ladv001_3k`, W&B `7uwfab18`) was stopped early. Validation DISTS worsened from `0.4173` at it=0 to `0.4912` at it=1000, and LPIPS also did not improve. The discriminator became strong quickly, and the current generator objective did not benefit from the adversarial term.

Do not use naive GAN fine-tuning as the next default. If revisited, use delayed adversarial start, much smaller `lambda_adv`, feature matching, and fixed deterministic validation before promoting any checkpoint.


## Stage A Update: DISTS-Heavy Fine-Tune Helps Slightly

A DISTS-heavy fine-tune from Stage-A 6000 (`scratch_stage_a_down5_from6000_dists2_lp05_12k`, W&B `zd8omzv0`) produced a better deterministic Kodak-center base at `stage_a_0008000.pt`: LPIPS/DISTS `0.45221/0.44797` versus the previous `0.45782/0.45264`. This is a small but real improvement and should be used as the next Stage-B source checkpoint.



## Stage B Update: Improved Stage-A Does Not Yet Update Pareto

Using the DISTS-heavy Stage-A `stage_a_0008000.pt` as the source, `scratch_stage_b_from_stageA_d2_8000_r8_q1_lR0p1_d2_3k` (W&B `9gbu1r38`) again improved the base reconstruction with a hard-quantized residual stream. On deterministic Kodak center crops, the final checkpoint reaches total bpp `0.02212`, LPIPS `0.43832`, and DISTS `0.43195` from a base of LPIPS/DISTS `0.45221/0.44797`.

This is conceptually positive but not a Pareto update. The previous r8 `lambda_R=0.5` run remains stronger at the low-rate point: total bpp `0.01775`, DISTS `0.43456`. The r8 `lambda_R=0.3` final remains the better higher-quality scratch point: total bpp `0.02345`, DISTS `0.43024`.

Conclusion: simply improving Stage-A a little and relaxing Stage-B rate pressure does not create a large gain. The scratch bottleneck is still generator capacity/objective, not the residual arithmetic itself.



## Stage-A Latent Refinement: First Strong Scratch Improvement

The attention/refinement branch keeps the original Stage-A decoder intact and inserts an identity-initialized `latent_refine` module before it. This preserves all old checkpoint weights and lets the model learn extra latent-space generation capacity without changing the semantic index rate.

The key run is `scratch_stage_a_down5_attn_refine_from_d2_8000_6k` (W&B `lbzhch1m`). Its DISTS-best checkpoint improves the Stage-A base from LPIPS/DISTS `0.45221/0.44797` to `0.45767/0.43546` at the same semantic bpp `0.00977`. LPIPS worsens, but the DISTS gain is large enough to test as the R-P oriented Stage-B source.

Using that checkpoint, `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d2_3k` (W&B `4a1jwvsw`) updates the scratch Pareto frontier. The final checkpoint reaches total bpp `0.01328`, LPIPS `0.43770`, and DISTS `0.42446`, with only `0.00352` residual proxy bpp beyond the semantic stream.

This is the cleanest current scratch evidence for the paper idea: the generator handles most predictable structure from a cheap code, and a very small residual stream improves the unpredictable part.



## Stage-B Curve Point: lambda_R 0.3

A quality-side Stage-B sweep from the attention-refined Stage-A (`scratch_stage_b_from_attnA_best_r8_q1_lR0p3_d2_3k`, W&B `vo5d3dkz`) gives a second useful scratch point. The final checkpoint reaches total bpp `0.01588`, LPIPS `0.43752`, and DISTS `0.42396`.

This is only a small DISTS improvement over the `lambda_R=0.5` final (`0.01328` bpp, DISTS `0.42446`), so the next scratch improvement should focus on residual representation/progressive residual coding rather than simply reducing `lambda_R`.



## Stage-B Continuation: Current Scratch DISTS Lead

Continuing the `lambda_R=0.5` Stage-B lead to 6000 iterations gives a small but real DISTS update. The checkpoint `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` reaches total bpp `0.01321`, LPIPS `0.43869`, and DISTS `0.42313` on deterministic Kodak center crops.

This is now the best scratch DISTS point. Later checkpoints improve LPIPS slightly but worsen DISTS, so metric-specific checkpoint selection is necessary.



## DIV2K Center-Crop Generalization Check

The current scratch DISTS lead (`stage_b_0004000.pt` from the continued `lambda_R=0.5` run) also improves DIV2K validation center crops. On 100 images from `/dpl/div2k`, total bpp is `0.01364`, LPIPS improves from `0.44078` to `0.42058`, and DISTS improves from `0.42494` to `0.41563`.

This is not an official full-resolution shifted-patch protocol result. It is a sanity check that the semantic-plus-residual decomposition is not only fitting Kodak crops.



## Residual Width Ablation: r4 Is Worse Than r8

The `residual_dim=4` Stage-B sweep (`scratch_stage_b_from_attnA_best_r4_q1_lR0p5_d2_3k`, W&B `2sb82ffg`) did not improve the curve. Its best fixed Kodak point is total bpp `0.01376`, LPIPS `0.43780`, DISTS `0.42853`, which is clearly worse than the r8 lead at total bpp `0.01321`, DISTS `0.42313`.

The narrow model increases residual scale and does not provide a cleaner low-rate point. r8 remains the current preferred residual width.



## Residual Width Ablation: r16 Collapses Under lambda_R 0.5

The `residual_dim=16` sweep (`scratch_stage_b_from_attnA_best_r16_q1_lR0p5_d2_3k`, W&B `kivq0tki`) does not provide a better quality point. The final checkpoint is very low-rate (`0.01258` bpp) but DISTS is `0.43298`, worse than the r8 lead. The random-validation best spends `0.03115` bpp without fixed-eval quality gain.

This suggests the current single residual bottleneck is not capacity-limited in a simple width sense. The next step should be staged/progressive residual coding, not merely changing residual_dim.



## Quantization Ablation: q0.5 Is Worse Than q1.0

The `quant_step=0.5` r8 sweep (`scratch_stage_b_from_attnA_best_r8_q0p5_lR0p5_d2_3k`, W&B `5e0rulf9`) did not improve the scratch curve. Its best checked Kodak DISTS is `0.43199` at `0.01329` bpp, clearly worse than q1.0 r8 continued at `0.42313` DISTS and similar bpp.

Current preferred Stage-B settings remain `residual_dim=8`, `quant_step=1.0`, `lambda_R=0.5` with low-lr continuation.



## Objective Ablation: Stronger DISTS Weight Does Not Help

The `lambda_dists=3.0`, `lambda_lpips=0.5` sweep (`scratch_stage_b_from_attnA_best_r8_q1_lR0p5_d3_lp05_3k`, W&B `hanx5zoe`) is worse than the current lead. Fixed Kodak DISTS is `0.42692` at best, compared with `0.42313` for the original DISTS-heavy setting. The current best objective remains `lambda_dists=2.0`, `lambda_lpips=0.7`, `lambda_R=0.5`.

## Next Sparse-Residual Objective After 12h Sprint

The top-k experiments show a useful split:

- Gate mechanics are solved enough for research: top-k keeps stage 1 open at 2-20% without transmitting a side map.
- Budget alone is not the bottleneck: 2%, 5%, 10%, and 20% gates on the same checkpoint keep Kodak DISTS near `0.42378-0.42380` while bpp rises.
- Stronger global stage-improvement can improve Kodak DISTS (`0.42219` at `0.01504` bpp), but it raises stage0 bpp and worsens LPIPS, and DIV2K does not clearly improve.

Next objective should train the selected positions, not only the full reconstruction. Candidate loss:

1. Decode `x_stage0` and `x_full` with hard top-k active.
2. Build a no-side-info importance target from decoder-available or train-only teacher signals: local `|x - x_stage0|`, gradient-weighted error, and feature residual magnitude.
3. Add a selected-region improvement term: only pixels/features influenced by top-k positions should be rewarded for reducing DISTS/LPIPS proxy error.
4. Add an anti-stage0-leak penalty: hold stage0 bpp or stage0 residual magnitude near the warmup baseline, so the model cannot win by moving all work back into stage0.
5. Sweep `gate_topk_frac={0.05,0.10}` after the selected-region loss is stable, then only promote if DIV2K also improves.

Paper decision remains unchanged: pretrained real-codec rho/stage-quant branches are the VCIP package. Scratch is now a credible complete-design research branch, but not the main quantitative claim yet.


## Stable Bounded Residual Update: Design Decision

The GLC-latent scratch branch now has a cleaner positive result than the earlier generic Stage-B residual experiments. Using a frozen semantic code and frozen GLC/VQGAN generator, the residual branch was constrained to satisfy three properties:

1. `zero_center` correction: no residual payload means exactly no residual correction.
2. `hard_topk`: only selected residual positions can be nonzero.
3. `max_symbol_abs=1`: transmitted values are finite ternary symbols.

The stable entropy proxy with a quadratic tail fallback showed why this matters. If symbols are not bounded, the model can keep top-k sparse positions but grow symbol amplitudes until residual bpp explodes. With ternary symbols, rate stays predictable and the result is method-faithful.

Current adopted center-crop proxy points:

| point | residual bpp | total bpp | main result |
|---|---:|---:|---|
| stable ternary topk0005 | 0.00085 | 0.01061 | improves LPIPS/DISTS/L1/MSE on CLIC2020, DIV2K, Kodak |
| stable ternary topk002 | 0.00338 | 0.01315 | strongest CLIC/DIV2K stable perceptual point; Kodak LPIPS worsens |
| stable small-int2 topk001 | 0.01211 | 0.02188 | improves over no-residual but is dominated by ternary topk002 |

Design conclusion: prefer sending more sparse ternary positions over allowing larger residual amplitudes. This keeps the method simple, codec-friendly, and aligned with the thesis that the bitstream should contain only unpredictable residual information.

Next design step: implement a real finite-support codec for ternary residual symbols and validate full-resolution CLIC/DIV2K/Kodak before adding architectural complexity.


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
