# LIC Literature Update For GP-ResLC

Date: 2026-07-01 JST

Purpose: keep the literature scan tied to implementation choices.  The active
project is not to replace GLC, but to improve a pretrained GLC codec by
transmitting only residual/control information that cannot be recovered from
`z_hat`, `q`, context, and the GLC generator.

## Directly Relevant Papers And Repos

| work | area | useful idea for GP-ResLC | public implementation |
|---|---|---|---|
| GLC: Generative Latent Coding for Ultra-Low Bitrate Image Compression | VQ/generative ultra-low bitrate codec | base codec; VQ/VQGAN latent, categorical hyper module, four-part prior, transform coding in generative latent space | https://github.com/jzyustc/GLC |
| RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization | VQ-based R-D optimization | top-k/prefix-style token transmission, masked Transformer entropy model, rate-aware token/control coding | https://github.com/CVL-UESTC/RDVQ |
| ResULIC: Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-Aware Diffusion | residual-guided generative compression | semantic residual and perceptual-fidelity guidance; useful as teacher/guard for residual importance | https://github.com/NJUVISION/ResULIC |
| DiffO: Single-step Diffusion for Image Compression at Ultra-Low Bitrates | one-step diffusion compression | VQ-residual factorization and bitrate-aware generation; supports the "omit residual, synthesize detail" direction | https://github.com/Freemasti/DiffO |
| CADC: Content Adaptive Diffusion-Based Generative Image Compression | diffusion/GIC, content-adaptive allocation | uncertainty/content-adaptive quantization as a teacher for safe-to-drop and benefit-per-bit maps; do not copy it as a mere quant-width trick | verify official repo before use |
| Control-GIC: Controllable Generative Image Compression with Dynamic Granularity Adaptation | VQ/GIC, local granularity control | local information-density based granularity allocation; useful for residual precision/control tokens on top of GLC | verify official repo before use |
| StableCodec | one-step diffusion extreme image compression | structure/detail split and dual-branch fidelity anchor; useful as teacher/upper-bound, not a GLC replacement | verify official repo before use |
| ELIC / MLIC-style context entropy models | learned entropy modeling | grouped/channel/spatial context models for future residual/control entropy coding | reference papers / implementations vary |
| CompressAI entropy models | implementation reference | GaussianConditional/EntropyBottleneck-style actual compress/decompress discipline | https://github.com/InterDigitalInc/CompressAI |
| Learned Image Compression with Dictionary-based Entropy Model | entropy modeling | dictionary/cross-attention prior from training-set structure; useful reference for stronger residual/control priors | CVPR 2025 paper page: https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Learned_Image_Compression_with_Dictionary-based_Entropy_Model_CVPR_2025_paper.html |
| Joint Autoregressive and Hierarchical Priors | classic entropy modeling | canonical hyperprior + autoregressive prior formulation; baseline reference for arithmetic-coded latent probability models | NeurIPS PDF: https://papers.neurips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf |

## Interpretation For Current Experiments

### 1. Residual/control entropy coding is still underdeveloped

RDVQ is the most relevant recent reference for turning local residual/control
decisions into actual coded symbols.  The GP-ResLC base must stay GLC, but RDVQ's
masked-transformer probability model and top-k/prefix transmission discipline
are directly applicable to:

- counted stage send masks,
- tiny control streams,
- residual-symbol priors,
- future finite-support stage residual coding.

### 2. Synthesis is on-axis, but only if it fills omitted information

DiffO and ResULIC both reinforce the same lesson: low-rate generative codecs win
when they separate structure/control from detail synthesis.  For GP-ResLC this
means the useful module is not a generic post-hoc enhancer.  The useful module
is:

```text
transmitted residual cells -> keep exact coded residual
omitted residual cells     -> decoder-computable synthesis only if safe
```

That is why the active run trains on the same omitted-cell distribution used by
the counted stage-3 send-control branch.

### 3. Image-MSE allocation is a bad teacher for our objective

The local image-MSE-gradient send-mask experiment worsened DISTS/FID.  This is
consistent with perceptual-compression literature: pixel/image distortion can
over-protect spatial fidelity while hurting perceptual naturalness.  Future
safe-coarsening teachers should use perceptual/semantic residuals rather than
raw MSE alone.

### 4. Stronger entropy models are the next non-synthesis path

Dictionary-based entropy modeling and classic hyperprior/autoregressive priors
point to a separate, likely complementary path: improve the probability model
for residual/control streams rather than only changing which symbols are sent.
This should be used after the residual variable is cleanly defined, not by
changing the GLC base.

### 5. Content-adaptive allocation must be reframed as safe residual dropping

CADC and Control-GIC are easy to misread as "adaptive quantization is the
contribution."  For GP-ResLC, that framing is too weak.  The useful transfer is:

```text
estimate where residual information is safe to drop or cheap to synthesize,
then code only the residual/control that is not recoverable.
```

Thus their role is to shape safe-coarsening teachers, residual-RDO scores, and
precision/control tokens.  They should not turn the project back into rho-target
or quantization-width tuning.

### 6. One-step/diffusion codecs are teachers, not bases

StableCodec, DiffO, ResULIC, and similar one-step/diffusion codecs support the
idea that very low-rate codecs should transmit structure/control and let a
strong generator recover texture/detail.  GP-ResLC should borrow this separation
but keep the GLC base.  The direct implementation target is omitted-residual
synthesis and residual/control allocation on top of GLC's decoded `z_hat`,
`q`, and four-part context.

## Implementation Consequence

Current priority:

1. finish current CLIC2020 package validation for the conservative
   `SelectiveS025ZEntropyAutoA1` package,
2. run stage residual omission and same-bpp omitted-residual synthesis
   diagnostics,
3. if deterministic/learned synthesis closes DISTS/FID/LPIPS loss, extend it
   with a safe-to-synthesize teacher or tiny counted control stream,
4. in parallel, design the next full residual/control entropy path:
   finite-support or Gaussian/logistic residual symbols, RDVQ-style top-k/escape
   discipline where useful, and actual serialized bpp accounting.

Do not describe the method as adaptive quantization.  The research story remains
predictable/unpredictable residual allocation and real entropy coding on top of
pretrained GLC.
