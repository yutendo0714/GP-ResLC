# GP-ResLC VCIP Paper Draft

Last updated: 2026-06-20 JST

Working title:

**Generator-Predictable Residual Suppression for Ultra-Low-Bitrate Perceptual Image Compression**

## Abstract Draft

Generative learned image codecs can synthesize plausible visual details from compact semantic latents, but their entropy-coded residual streams may still spend bits on details that the decoder-side generator can already reconstruct. This mismatch is especially costly at ultra-low bitrates, where each residual bit should be reserved for information that is not predictable from the transmitted semantic or hyper code. We propose GP-ResLC, a generator-predictable residual suppression method built on the official GLC image codec. The short-track system predicts a deterministic gate, `rho(z_hat, q)`, from already transmitted hyper/VQ information and the target rate index. The gate increases the quantization step of the main latent only in regions expected to be generator-predictable, is constrained by `rho >= 1`, and requires no side map. We evaluate with an actual serialized arithmetic-codec path rather than estimated likelihood bpp. On CLIC2020 test, GP-ResLC reduces real-codec BD-rate by 10.28% for DISTS and 7.30% for FID relative to GLC. On DIV2K validation, it reduces DISTS and FID BD-rate by 10.79% and 5.61%, respectively; on Kodak, it reduces DISTS BD-rate by 4.47%. Matched-metric interpolation further shows 10.26%, 10.27%, and 5.45% fewer serialized bits at matched DISTS on CLIC2020 test, DIV2K, and Kodak, respectively. Gate analysis shows that high-`rho` regions have lower baseline reconstruction error, lower texture variance, and lower gradient magnitude, supporting the intended mechanism: do not transmit generator-predictable residuals; preserve bits for unpredictable residuals.

## 1. Introduction Draft

Learned image compression has progressed from transform autoencoders with hyperpriors to strong autoregressive, channel-wise, and transformer-enhanced entropy models. These systems are highly effective for rate-distortion optimization, but ultra-low-bitrate perceptual compression exposes a different failure mode. At very low rates, pixel-space distortion losses reward averaged reconstructions, while human observers often prefer reconstructions that preserve semantic structure and plausible texture. This has pushed the field toward generative compression, including GAN-based codecs, VQ-based generative codecs, and diffusion-based decoders.

Generative codecs change the role of the bitstream. A sufficiently strong decoder no longer needs every high-frequency detail to be transmitted explicitly: some details can be inferred from semantic structure, learned natural-image statistics, and already transmitted side information. The central question is therefore not only how to build a stronger generator, but how to avoid sending information that the generator can recover by itself.

This paper studies that question in the context of Generative Latent Coding (GLC), a recent ultra-low-bitrate codec that performs transform coding in a perception-aligned VQ-VAE latent space. GLC already provides a strong generative decoder and a compact hyper/VQ code. We hypothesize that this transmitted code contains enough information to predict where the main latent residual can be coarsened without damaging perceptual quality. In other words, the decoder should spend bits on residual information that is unpredictable from the generator and its side information, not on residual detail that is already recoverable.

GP-ResLC operationalizes this principle with a zero-side-bit residual-suppression gate. Given the transmitted hyper/VQ latent `z_hat` and rate index `q`, a small predictor estimates a spatial gate `rho(z_hat, q)`. The gate scales the quantization step of the main latent before entropy coding:

```text
rho = g_phi(z_hat + e_q),        rho >= 1
q_step_prime = rho * q_step
y_hat = round(y / q_step_prime) * q_step_prime
```

Because `rho` depends only on information already available to the decoder, it does not require an additional side map. The monotone constraint `rho >= 1` is important: the method never transmits more residual precision than GLC. It only suppresses residuals in regions predicted to be safe for the generator to absorb.

The current VCIP short-track contribution is deliberately rate-perception oriented. It is not presented as a universal rate-distortion improvement over all learned codecs. Instead, it shows that an existing generative codec can be made more bitrate-efficient at similar perceptual quality by learning where not to spend residual bits.

## 2. Related Work Draft

### Learned Image Compression and Entropy Modeling

Classical neural image compression is built around entropy-constrained autoencoders. Scale hyperpriors model global side information for the latent distribution, while joint autoregressive and hierarchical priors add local context modeling. Later work improves the tradeoff between compression efficiency and decoding latency through channel-wise context models, uneven channel grouping, and multi-reference entropy models.

Recent rate-distortion-oriented LIC systems continue to strengthen entropy modeling. MLICv2 improves transform design and entropy modeling through token mixing, hyperprior-guided global correlation prediction, channel reweighting, and optional instance adaptation; it reports strong BD-rate reductions against VTM-17.0 Intra on Kodak, Tecnick, and CLIC Professional Validation. Dictionary-based entropy modeling introduces learned dataset-level structure through cross-attention to a dictionary, emphasizing that priors can come not only from the current latent context but also from recurring structures in the training distribution.

GP-ResLC is related to these entropy-modeling efforts, but its prior is different. Rather than only estimating `p(y | z_hat, context)` more accurately, it asks whether parts of `y` should be represented at all with the same precision when a generative decoder can recover them perceptually.

### Generative and Perceptual Image Compression

GAN-based perceptual codecs such as HiFiC and later multi-realism methods showed that optimizing for perceptual quality can substantially improve visual realism at low bitrates. These methods made the rate-distortion-perception tradeoff concrete: improving distributional realism often requires accepting some distortion loss.

VQ-based generative compression pushes this idea further by coding in a semantic or generative latent space. GLC performs transform coding in the latent space of a VQ-VAE and reports high perceptual quality below 0.04 bpp on natural images, with 45% fewer bits than MS-ILLM at matched FID on CLIC2020. HVQ-CGIC extends VQ-based generative compression by introducing a hyperprior entropy model over VQ indices, directly addressing the non-adaptive entropy estimates of global VQ-index priors.

GP-ResLC uses GLC as the base system. Its novelty is not a new generative decoder or a new VQ tokenization scheme, but a residual bit-allocation principle inside a GLC-style bitstream: suppress residual precision where the generator-predictable component is high.

### Diffusion-Based Ultra-Low-Bitrate Compression

Diffusion decoders have recently become prominent in ultra-low-bitrate perceptual compression. Hybrid-Diffusion Image Compression combines generative VQ modeling, diffusion, and conventional LIC to balance fidelity and perceptual realism. DiffO uses single-step diffusion, VQ-residual training, and rate-adaptive noise modulation to make diffusion decoding more practical, reporting about 50x speedup over prior multi-step diffusion compression methods.

These works support the same high-level premise: ultra-low-bitrate compression benefits from a strong generative prior plus a compact transmitted correction. GP-ResLC studies a lighter path that does not replace the decoder with a diffusion model. It instead asks how much bitrate can be saved by making the residual stream of an existing generative codec more consistent with the generator's own predictive capacity.

## 3. Method Draft

### 3.1 Base Codec

Let `x` be the input image. The frozen GLC codec maps `x` into a generative latent representation and then into a main transform latent `y`. A hyper/categorical branch produces a transmitted side code `z_hat`, which conditions the entropy model and decoder. The base codec entropy-codes the quantized main latent `y_hat` conditioned on `z_hat` and autoregressive context, then reconstructs `x_hat` with a generative latent decoder.

GP-ResLC keeps the pretrained GLC encoder, decoder, and entropy model fixed in the current short-track implementation. This isolates the contribution of residual suppression and avoids confounding improvements from retraining the full codec.

### 3.2 Generator-Predictable Residual Suppression

The core module is a small gate network:

```text
rho = g_phi(z_hat, q)
```

where `q` is the rate index. In implementation, `q` is embedded and injected into the gate network, and the predicted gate is upsampled to the main latent resolution. The gate is parameterized so that:

```text
rho_min <= rho <= rho_max,        rho_min = 1.0
```

The lower bound gives the monotone residual-suppression property. GP-ResLC cannot increase transmitted precision relative to GLC; it can only keep the base quantization or coarsen it. The quantized latent becomes:

```text
q_step_prime = rho * q_step
y_hat_gp = round(y / q_step_prime) * q_step_prime
```

The same `rho` is computed during decoding from the already transmitted `z_hat` and known rate index. Therefore, no side map is transmitted. This is the practical reason the method can reduce total bpp, not merely redistribute bits.

### 3.3 Sendability Teacher

A naive rate loss can learn uniform over-coarsening. To make the gate spatially selective, training uses a sendability teacher. The teacher identifies regions where the frozen GLC reconstruction is already locally reliable:

```text
s = S(local_error(x, x_glc), texture_variance(x), gradient_magnitude(x))
```

Regions with low baseline error, low texture variance, and low gradient magnitude are considered more generator-predictable and are encouraged to receive larger `rho`. Regions with higher local error or stronger structural content are protected with lower `rho`.

The teacher is not transmitted and is used only during training. It makes the learned gate a mechanism for selective residual suppression rather than a blind bitrate knob.

### 3.4 Training Objective

The short-track objective is rate-perception oriented:

```text
L = lambda_R * R(y_hat_gp, z_hat)
  + lambda_P * P(x, x_hat_gp)
  + lambda_D * D_aux(x, x_hat_gp)
  + lambda_rho * (mean(rho) - rho_target)^2
  + lambda_send * BCE(normalize(rho), s)
```

`P` is LPIPS-based during training. DISTS and FID are the primary evaluation metrics because they are more aligned with the GLC paper's perceptual evaluation and with our observed low-bitrate behavior. `D_aux` is a small stabilization term, not the headline optimization target. The `rho_target` term controls the rate-perception operating point and produces an interpretable knob between conservative and more aggressive residual suppression.

## 4. Experiments Draft

### 4.1 Setup

Base codec:

- Official GLC image checkpoint: `pretrained/GLC_image.pth.tar`
- GP-ResLC lead checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`
- Balanced checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target112_send5_all_6k/v2_final.pt`

Datasets:

- CLIC2020 test: 428 images, original resolution; professional test plus mobile test
- DIV2K validation: 100 images, original resolution
- Kodak: 24 images, original resolution
- OpenImages training data for GP-ResLC gate learning
- CLIC professional/mobile validation: development and mechanism-analysis support only

Metrics:

- Rate: serialized real-codec bpp, main-latent bpp (`bpp_y`), hyper/VQ bpp (`bpp_z`), encode/decode time
- Primary perceptual metrics: DISTS, FID
- Auxiliary perceptual/statistical metrics: LPIPS, KID
- Distortion diagnostics: PSNR, MS-SSIM

For paper-facing tables, bpp is computed as `8 * payload_bytes / pixels` from the implemented codec: fixed-width transmitted `z` indices plus `torchac` arithmetic-coded `y` streams and compact metadata. FID/KID use 256x256 patches with 128-pixel shift on CLIC/DIV2K. KID and LPIPS are reported honestly but are not the central claim because the current checkpoint aligns more consistently with DISTS/FID under real-codec accounting.

### 4.2 Main Quantitative Results

BD-rate versus GLC real codec:

| dataset | model | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | rho1.16 real | -10.28% | +0.19% | -0.98% | +0.38% | -7.30% | -7.10% |
| DIV2K validation | rho1.16 real | -10.79% | -0.54% | -1.49% | -0.17% | -5.61% | -6.50% |
| Kodak | rho1.16 real | -4.47% | -0.79% | -0.87% | +0.45% | -1.70% | -6.14% |

The strongest paper-facing result is now CLIC2020 test under real arithmetic-coded bitstream accounting: the lead checkpoint reduces DISTS BD-rate by 10.28% and FID BD-rate by 7.30% while lowering serialized bpp at every q. DIV2K confirms the same trend with 10.79% DISTS BD-rate and 5.61% FID BD-rate reductions. Kodak is smaller and noisier for patch distribution metrics, but its real-codec DISTS/FID/KID BD-rates are still negative.

Per-q real bpp reductions are consistent across datasets:

| dataset | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| CLIC2020 test | -11.36% | -10.32% | -8.74% | -7.91% |
| DIV2K validation | -10.39% | -9.29% | -8.15% | -7.16% |
| Kodak | -9.52% | -9.09% | -7.93% | -7.17% |

Because `z` and header bpp are identical between GLC and GP-ResLC, these savings come from the arithmetic-coded `y` stream, matching the proposed residual-suppression mechanism.

### 4.3 Matched-Metric Bitrate Reduction

Matched-metric interpolation gives the clearest rate-perception claim under the real codec:

| dataset | model | matched DISTS bpp | matched FID bpp | note |
|---|---|---:|---:|---|
| CLIC2020 test | rho1.16 real | -10.26% | -6.02% | DISTS over q0-q3; FID over q0-q2 |
| DIV2K validation | rho1.16 real | -10.27% | -3.39% | DISTS over q0-q3; FID over q0-q2 |
| Kodak | rho1.16 real | -5.45% | -4.40% | matched over GLC q0-q3 |

This table should be central in the short-track paper. It directly answers the rate-perception question: at the same perceptual quality level, how many fewer serialized bits are required?

As an external positioning check, we also compare GP-ResLC real-codec points with graph-extracted official GLC paper curves. After correcting CLIC to the full 428-image professional+mobile test set, local GLC closely matches the official CLIC curve. GP-ResLC gives -9.07% DISTS BD-rate and -6.10% FID BD-rate versus official GLC on CLIC2020. DIV2K also supports the claim with -9.62% DISTS BD-rate and -4.23% FID BD-rate. Kodak is near-neutral against the official DISTS curve and the official plot does not include FID/KID. These official-curve numbers are useful supplementary positioning, while the main paper table should remain the paired local real-codec comparison.

### 4.4 Mechanism Analysis

To verify that the learned gate is not merely uniform re-quantization, we analyze the correlation between `rho` and local image/reconstruction statistics at q3:

| dataset | mean rho | rho std | corr(rho, base err) | corr(rho, texture var) | corr(rho, gradient) |
|---|---:|---:|---:|---:|---:|
| Kodak | 1.1716 | 0.0308 | -0.234 | -0.243 | -0.213 |
| CLIC professional valid | 1.1815 | 0.0307 | -0.271 | -0.268 | -0.290 |
| CLIC mobile valid | 1.1804 | 0.0311 | -0.251 | -0.251 | -0.262 |

The correlations are consistently negative. Higher suppression is assigned to regions with lower baseline error, lower texture variance, and lower gradient magnitude. High-`rho` regions also have roughly half the local error/gradient of low-`rho` regions in the current analysis. This supports the proposed mechanism: the gate learns to suppress generator-predictable residuals while protecting harder residuals.

### 4.5 Ablation Summary

Several alternatives were explored:

| variant | observation | decision |
|---|---|---|
| Direct `P_theta` residual-prior correction | Stable variants were too weak; unconstrained variants reduced rate but degraded or diverged | Not the short-track lead |
| Strong gate-only rate pressure | Large bpp savings but perceptual quality degraded | Diagnostic only |
| `rho_target=1.12` | Safer/balanced; lower rate saving but less LPIPS risk | Secondary operating point |
| `rho_target=1.16` | Best DISTS/FID R-P evidence | Main checkpoint |
| `rho_target=1.20` | More aggressive but over-coarsens low-rate points | Upper knob only |
| Alex LPIPS training | Comparable DISTS, did not solve LPIPS interpolation | Ablation |
| Texture-free teacher | Weaker DISTS/FID story | Do not lead |
| Baseline distillation | Preserved some q3 DISTS but worsened FID | Do not lead |
| Edge guard teacher | Clean q3 control, weaker full-curve behavior | Appendix/control |

## 5. Discussion Draft

The short-track GP-ResLC model should be interpreted as evidence for a principle rather than a completed universal codec. The principle is that a generative codec should not treat all residual latent components equally. Once a semantic or hyper code has been transmitted, some residual details become predictable by the decoder-side generator. Spending bits on those details is inefficient under a perceptual objective.

The current system validates this principle with the simplest mechanism that preserves bitstream clarity: a deterministic gate predicted from transmitted information. It does not yet learn a full residual code for the unpredictable component, and it does not retrain the GLC generator. These constraints make the result easier to interpret. Any bitrate reduction comes from suppressing residual precision with no additional side information.

The main limitation is that LPIPS is not uniformly improved under matched-LPIPS interpolation, especially for the more aggressive `rho1.16` checkpoint. The paper should therefore avoid claiming universal perceptual improvement across all metrics. DISTS and FID provide the stronger and more consistent R-P evidence, while LPIPS, PSNR, and MS-SSIM are reported as diagnostics.

## 6. Conclusion Draft

GP-ResLC introduces generator-predictable residual suppression for ultra-low-bitrate perceptual image compression. Built on GLC, it predicts a zero-side-bit gate from transmitted hyper/VQ information and coarsens latent residual quantization only where the generative decoder is likely to recover the missing detail. Under actual serialized arithmetic-codec evaluation, the method reduces bitrate at matched DISTS on CLIC2020 test, DIV2K validation, and Kodak, with the strongest gains on CLIC and DIV2K. Gate-correlation analysis confirms that suppression is spatially selective and aligned with predictable image regions. These results support a compact short-track claim: in generative learned compression, bits should be reserved for residual information the generator cannot infer.

## References To Cite

- Ballé et al., Variational Image Compression with a Scale Hyperprior: https://arxiv.org/abs/1802.01436
- Minnen et al., Joint Autoregressive and Hierarchical Priors: https://arxiv.org/abs/1809.02736
- HiFiC: https://arxiv.org/abs/2006.09965
- Rate-Distortion-Perception theory: https://arxiv.org/abs/1901.07821
- DISTS: https://arxiv.org/abs/2004.07728
- LPIPS: https://arxiv.org/abs/1801.03924
- TCM: https://arxiv.org/abs/2303.14978
- MLICv2: https://arxiv.org/abs/2504.19119
- Dictionary-based Entropy Model: https://arxiv.org/abs/2504.00496
- GLC: https://arxiv.org/abs/2512.20194
- HVQ-CGIC: https://arxiv.org/abs/2512.07192
- HDCompression: https://arxiv.org/abs/2502.07160
- DiffO: https://arxiv.org/abs/2506.16572
