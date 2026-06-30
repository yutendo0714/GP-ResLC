# Learned Image Compression Survey Notes (2026-06-30)

This note is a working survey for GP-ResLC research. It emphasizes methods that
matter for ultra-low-bitrate generative latent coding: entropy modeling,
hyperprior/autoregressive priors, VQ/generative codecs, perceptual compression,
and rate-distortion-perception optimization.

## Takeaway for GP-ResLC

The strongest research direction is not to replace GLC. The practical path is:

```text
pretrained GLC generative latent codec
+ decoder-available residual prediction
+ residual entropy modeling
+ counted residual/control bits only where needed
+ real serialized codec evaluation
```

Modern LIC gains often come from better conditional probability models rather
than larger decoders. For GP-ResLC this means the next high-value work is
stage-aware residual mean/scale modeling and learned residual/control entropy,
not superficial rho/loss sweeps.

## Classical Neural Image Compression

### Scale Hyperprior

Reference: Ballé et al., "Variational Image Compression with a Scale
Hyperprior", ICLR 2018. arXiv: <https://arxiv.org/abs/1802.01436>

- Idea: transmit a hyper-latent `z` to model spatially varying uncertainty of
  the main latent `y`.
- Novelty: learned side information for entropy modeling.
- Metrics/data: Kodak/Tecnick-style PSNR/MS-SSIM evaluation became standard.
- Difference from older codecs: replaces hand-designed transform and entropy
  model with learned transform plus learned conditional prior.
- Relevance: GLC inherits the same basic split: hyper information plus main
  latent stream. GP-ResLC should improve the main-latent conditional model, not
  ignore the hyperprior.

### Joint Autoregressive and Hierarchical Priors

Reference: Minnen et al., "Joint Autoregressive and Hierarchical Priors for
Learned Image Compression", NeurIPS 2018. arXiv:
<https://arxiv.org/abs/1809.02736>

- Idea: combine hyperprior with autoregressive spatial context.
- Novelty: entropy model gets both global side information and already decoded
  neighboring symbols.
- Cost: serial context can slow decoding.
- Relevance: GLC's four-part context is a practical compromise. GP-ResLC must
  preserve that decode order when adding residual prediction.

### Attention / GMM / Stronger Transform Codecs

Reference: Cheng et al., "Learned Image Compression with Discretized Gaussian
Mixture Likelihoods and Attention Modules", CVPR 2020. arXiv:
<https://arxiv.org/abs/2001.01568>

- Idea: attention-enhanced transforms and mixture likelihoods.
- Novelty: improves both representation and entropy model capacity.
- Metrics: PSNR/MS-SSIM on Kodak, CLIC, Tecnick.
- Relevance: mixture or finite-support residual priors are natural future
  GP-ResLC components after the current Gaussian residual-scale branch.

## Modern Entropy Modeling and Transformer/State-Space LIC

### ELIC

Reference: He et al., "ELIC: Efficient Learned Image Compression with Unevenly
Grouped Space-Channel Contextual Adaptive Coding", CVPR 2022. arXiv:
<https://arxiv.org/abs/2203.10886>

- Idea: uneven channel grouping plus spatial-channel context.
- Novelty: stronger entropy model with efficient grouping.
- Difference from Minnen: less purely serial, more parallel-friendly.
- Relevance: residual symbols in GP-ResLC could be grouped or stage-coded with a
  finite-support/context model instead of only Gaussian scale correction.

### TCM

Reference: Liu et al., "Learned Image Compression with Mixed Transformer-CNN
Architectures", CVPR 2023. arXiv: <https://arxiv.org/abs/2303.14978>

- Idea: mix convolutional locality with transformer global modeling.
- Novelty: stronger analysis/synthesis transforms and prior features.
- Relevance: GP-ResLC should avoid replacing GLC with TCM, but transformer/CNN
  mixed blocks can inspire better residual predictors if current DCVC-style
  blocks saturate.

### MLIC / MLIC++

Reference family: multi-reference / multi-dimensional context learned image
compression. Public code is commonly associated with
<https://github.com/JiangWeibeta/MLIC> (GitHub DNS was unavailable in this
session; re-check before citing as official).

- Idea: use multiple context references and stronger entropy parameter
  estimation.
- Relevance: GP-ResLC's residual stream could borrow multi-reference context,
  but the base codec should remain GLC.

### MambaIC

Reference: "MambaIC: State Space Models for High-Performance Learned Image
Compression", arXiv 2025. arXiv API returned:
<http://arxiv.org/abs/2503.12461>

- Idea: use state-space models for long-range representation/context modeling.
- Relevance: interesting future predictor backbone, but not first priority
  unless the current stage residual predictor lacks receptive field.

## Perceptual and Generative Image Compression

### HiFiC

Reference: Mentzer et al., "High-Fidelity Generative Image Compression",
NeurIPS 2020. Paper/project: <https://hific.github.io/>

- Idea: conditional GAN decoder optimized for perceptual realism at low/mid
  rates.
- Metrics: FID, KID, LPIPS, PSNR/MS-SSIM; CLIC/Kodak-style natural image eval.
- Protocol note: the CLIC2020 428-image test split and 256x256 patch FID/KID
  protocol with 28,650 patches is commonly traced to HiFiC and followed by GLC.
- Relevance: GP-ResLC should report FID/KID/DISTS/LPIPS, not only PSNR.

### MS-ILLM

Reference: Muckley et al., "Improving Statistical Fidelity for Neural Image
Compression with Implicit Local Likelihood Models", 2023/2024 era. Public code
is associated with Meta/NeuralCompression projects.

- Idea: improve perceptual/statistical fidelity via implicit likelihood / local
  discriminator design.
- Relevance: strong generative baseline in GLC comparisons. GP-ResLC claims
  should compare against GLC official curves and MS-ILLM where possible.

### Rate-Distortion-Perception Theory

Reference: Blau and Michaeli, "The Perception-Distortion Tradeoff", CVPR 2018 /
related rate-distortion-perception theory. Project:
<https://arxiv.org/abs/1711.06077>

- Idea: distortion and perceptual realism have a fundamental trade-off.
- Relevance: GP-ResLC's target is not PSNR maximization. It should optimize and
  report perceptual fidelity/realism under actual bpp.

## VQ-Based and Generative Latent Compression

### VQ-VAE / VQGAN

References:

- VQ-VAE: van den Oord et al., "Neural Discrete Representation Learning",
  NeurIPS 2017. <https://arxiv.org/abs/1711.00937>
- VQGAN: Esser et al., "Taming Transformers for High-Resolution Image
  Synthesis", CVPR 2021. <https://arxiv.org/abs/2012.09841>

- Idea: represent images by discrete/tokenized latent codes with a powerful
  decoder/generator.
- Relevance: GLC relies on a VQGAN/VQ-VAE-like generative latent space. GP-ResLC
  should exploit what the decoder can reconstruct from transmitted latent
  context instead of resending predictable details.

### GLC

Reference: Jia et al., "Generative Latent Coding for Ultra-Low Bitrate Image
Compression", CVPR 2024 / arXiv entries. Public repository:
<https://github.com/jzyustc/GLC> (GitHub DNS unavailable during this session;
the repository URL is the known project URL and should be re-checked before
camera-ready citation).

- Idea: train VQ-VAE/VQGAN latent space, then compress the generative latent via
  transform coding, categorical hyper module, and four-part context prior.
- Training protocol from paper/supplement/README:
  - Stage I: autoencoder/VQ latent learning.
  - Stage II: transform coding of learned VQ latent.
  - Stage III: joint pixel-space fine-tuning.
  - Natural image training uses ImageNet for Stage I and OpenImages for Stage
    II/III according to the available public descriptions.
- Metrics: bpp, LPIPS, DISTS, FID, KID; PSNR/MS-SSIM supplementary.
- Evaluation: CLIC2020 test, Kodak, DIV2K validation, MS-COCO 30K in supplement;
  FID/KID are patch-based on high-resolution natural-image sets.
- Relevance: GP-ResLC must keep GLC's codec semantics and compare with real
  serialized bpp, not estimated bpp.

### Fine-tuned VQGAN / Extreme VQ Compression

Reference: "Extreme Image Compression using Fine-tuned VQGANs", arXiv:
<https://arxiv.org/abs/2307.08265>

- Idea: push VQGAN codes to extreme low bitrates with generator priors.
- Relevance: supports the same thesis that generator-recoverable details need
  not be explicitly transmitted.

### DLF / Dual-Branch Generative Latent Fusion

Reference family: DLF / dual-generative latent fusion for extreme compression
reported around 2025.

- Idea: separate semantic and detail latents and fuse them for reconstruction.
- Relevance: DLF sends additional detail/control information. GP-ResLC differs
  by staying inside pretrained GLC and coding only unpredictable residual/control
  with counted bpp.

## Diffusion-Based Perceptual Compression

### HFD / Foundation Diffusion Compression

Reference family: lossy image compression with foundation diffusion models.

- Idea: use a frozen or adapted diffusion model as a strong perceptual prior,
  with a compact conditioning stream.
- Relevance: strong low-bitrate realism, but decoding is often slower and
  protocol differs. Useful comparison on MS-COCO 30K.

### PerCo

Reference: "PerCo (SD): Open Perceptual Compression", arXiv API returned:
<http://arxiv.org/abs/2409.20255>

- Idea: perceptual compression with Stable Diffusion prior.
- Relevance: important generative baseline. GP-ResLC should not inherit its slow
  multi-step decoder unless future work explicitly targets diffusion.

### StableCodec

Reference: "StableCodec: Taming One-Step Diffusion for Extreme Image
Compression", arXiv API returned: <http://arxiv.org/abs/2506.21977>

- Idea: one-step diffusion decoder for extreme low-bitrate compression.
- Relevance: latest strong generative-compression direction. GP-ResLC can cite
  it as evidence that strong generative priors matter, but current project keeps
  GLC pretrained base for manageable real-codec improvements.

### OneDC

Reference: "One-Step Diffusion-Based Image Compression with Semantic
Distillation", arXiv API returned: <http://arxiv.org/abs/2505.16687>

- Idea: one-step diffusion compression with semantic distillation.
- Relevance: semantic guidance and distilled perceptual priors are useful future
  teacher signals for safe-to-drop or residual benefit maps.

## Metrics and Dataset Protocol

Main metrics:

- bpp from serialized bitstream.
- LPIPS: perceptual feature distance.
- DISTS: structure/texture similarity, important for low-bitrate perceptual
  fidelity.
- FID/KID: distribution realism, computed on 256x256 patches for CLIC/DIV2K in
  GLC/HiFiC-style protocol.
- PSNR/MS-SSIM: supplementary distortion metrics.
- BD-rate: compare curves at matched quality, not same quality-index points.

Dataset notes for this project:

- CLIC2020 test should be professional + mobile = 428 images when following the
  HiFiC/GLC natural-image protocol.
- Kodak is useful for quick visual and speed checks, but FID/KID are unstable.
- DIV2K validation is a stronger natural-image transfer check.
- MS-COCO 30K is useful for diffusion-baseline comparison, but not the main GLC
  natural-image protocol.

## Research Implications

Immediate GP-ResLC priorities:

1. Keep `stage_residual_quant_gate` as the current safety lead.
2. Promote `stage_residual_entropy_quant_gate` if the scale-aware residual prior
   beats or matches DISTS while improving LPIPS/FID.
3. Avoid more top-k control sweeps until the teacher/control entropy model is
   better.
4. Implement learned residual/control entropy only with counted streams.
5. Evaluate every promoted branch with real codec on Kodak plus DIV2K, then CLIC.

