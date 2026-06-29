# LIC SOTA Survey for GP-ResLC / VCIP 2026

Last updated: 2026-06-21 JST

This note is the working literature map for GP-ResLC. The focus is ultra-low-bitrate learned image compression, generative/perceptual compression, entropy modeling, and rate-distortion-perception optimization. It is intentionally paper-facing: each category ends with implications for the GP-ResLC research direction.

## 1. Classical Neural LIC / Entropy Modeling

| paper | core method | novelty | data / metrics | difference vs prior | relevance to GP-ResLC |
|---|---|---|---|---|---|
| Ballé et al., 2018, Variational Image Compression with a Scale Hyperprior, https://arxiv.org/abs/1802.01436 | transform autoencoder + hyperprior side information | learned side information models spatial scale variation in latents | Kodak-like natural image RD, PSNR/MS-SSIM | moves from factorized latent prior to conditional hyperprior | establishes z as transmitted side information; GP-ResLC asks what residual information remains after z is known |
| Minnen et al., 2018, Joint Autoregressive and Hierarchical Priors, https://arxiv.org/abs/1809.02736 | hyperprior + autoregressive context | combines global side information with local decoded context | Kodak / standard RD metrics | stronger latent entropy model, but slower serial decoding | GLC four-part prior is a parallelized descendant; GP-ResLC must respect context order in real codec |
| Cheng et al., 2020, GMM likelihood + attention, https://arxiv.org/abs/2001.01568 | discretized Gaussian mixture likelihood + attention transforms | more flexible likelihood and stronger transforms | Kodak/high-res, PSNR/MS-SSIM | improves RD through likelihood expressivity | residual distribution may need mixture/logistic tails if GP-ResLC residual is not Gaussian |
| He et al., 2021, Checkerboard Context Model, https://arxiv.org/abs/2103.15306 | checkerboard two-pass context | parallelizes spatial autoregression | Kodak/Tecnick/CLIC, RD and speed | much faster than raster autoregression | validates our four-part real codec implementation strategy |
| He et al., 2022, ELIC, https://arxiv.org/abs/2203.10886 | uneven channel grouping + space-channel context | energy-aware channel grouping and efficient transforms | Kodak, Tecnick, CLIC | high practical RD and speed | residual predictor can borrow uneven grouping if full entropy model is rebuilt |
| Koyuncu et al., 2022/2023, Contextformer/eContextformer, https://arxiv.org/abs/2203.02452, https://arxiv.org/abs/2306.14287 | transformer spatio-channel context | attention over latent context; efficient windowed version | Kodak, CLIC2020, Tecnick | better context modeling with higher complexity | possible scratch entropy backbone, but too heavy for immediate VCIP path |
| Liu et al., 2023, TCM, https://arxiv.org/abs/2303.14978 | mixed Transformer-CNN transform | combines local CNN and non-local transformer modeling | Kodak, Tecnick, CLIC Pro Val | strong RD transform architecture | scratch GP-ResLC should use modern hybrid blocks rather than plain convs |
| Jiang et al., MLIC / MLIC++ / MLICv2, https://arxiv.org/abs/2211.07273, https://arxiv.org/abs/2307.15421, https://arxiv.org/abs/2504.19119 | multi-reference entropy modeling | channel, local, global contexts; MLICv2 adds token mixing, hyperprior-guided global correlation, channel reweighting, instance adaptation | Kodak, Tecnick, CLIC Pro Val, PSNR/MS-SSIM BD-rate | SOTA RD, VTM-beating in distortion regime | residual entropy gains likely need multi-reference context; GP-ResLC complete version should not rely on z-to-mean alone |

Takeaway: RD-oriented LIC wins mostly by increasingly accurate p(y | z, context). GP-ResLC should not compete by only making a stronger context model; its differentiated claim is to redefine the coded target as r = y - mu_theta(z,q), so only generator-unpredictable residuals remain.

## 2. Rate-Distortion-Perception Theory and Perceptual Objectives

| paper | contribution | relevance |
|---|---|---|
| Blau & Michaeli, 2019, Rate-Distortion-Perception Tradeoff, https://arxiv.org/abs/1901.07821 | formalizes that high perceptual quality can require sacrificing distortion or rate | justifies DISTS/FID/KID-centered claims at ultra-low bpp; PSNR loss is not fatal if R-P improves |
| Matsumoto, 2018, R-D-P for information sources, https://arxiv.org/abs/1808.07986 | information-theoretic extension of perception-distortion to source coding | supports framing beyond MSE RD |
| Zhang et al., 2021, Universal R-D-P Representations, https://arxiv.org/abs/2106.10311 | fixed encoder, variable decoder can cover RDP regions under conditions | motivates future decode-time beta steering; not current VCIP core |

Evaluation implication: For GP-ResLC short paper, primary metrics should be bpp-DISTS, bpp-FID, bpp-KID on CLIC2020/DIV2K with real codec. LPIPS, PSNR, and MS-SSIM must be reported but not used as the central claim when they conflict.

## 3. GAN / Perceptual Learned Compression

| paper | core method | novelty / performance | datasets / metrics | relevance |
|---|---|---|---|---|
| Mentzer et al., 2020, HiFiC, https://arxiv.org/abs/2006.09965 | learned codec + GAN/perceptual losses | visually pleasing high-resolution reconstructions, user study, broad bitrate range | Kodak/CLIC-like natural images; FID/KID/LPIPS/MS-SSIM/user study | establishes generative compression evaluation protocol; CLIC2020 428-image patch protocol is inherited by GLC/HiFiC-style comparison |
| MS-ILLM / multi-scale ILLM family | multi-scale adversarial/perceptual codec | strong low-bitrate perceptual baseline in GLC comparisons | CLIC/Kodak, FID/KID/LPIPS/DISTS | official GLC claims 45% fewer bits than MS-ILLM at matched CLIC FID; GP-ResLC should position against GLC, not only older GAN codecs |
| Li et al., 2024, Semantic Ensemble Loss + Latent Refinement, https://arxiv.org/abs/2401.14007 | semantic ensemble perceptual loss and latent refinement | reports large FID-oriented bitrate savings against MS-ILLM on CLIC2024 validation | CLIC2024 val, perceptual metrics | latent refinement is a related way of spending bits where they matter; GP-ResLC should log per-image latent residual statistics |
| Zhu et al., 2025, Fast Training-free Perceptual Image Compression, https://arxiv.org/abs/2506.16102 | post-hoc generative perceptual enhancement for existing codecs | fast training-free realism improvement with time-budgeted decoders | ELIC/VTM/MS-ILLM, FID and perceptual metrics | competitive perceptual route but not a real learned residual codec; useful baseline/caveat for decoding-time realism |

Takeaway: GAN/perceptual methods can improve realism but often do not cleanly explain where bitrate savings come from. GP-ResLC should make the mechanism explicit through bpp_y, residual entropy, and mu_theta/residual distribution analysis.

## 4. VQ-Based and Generative Latent Compression

| paper | core method | novelty / performance | datasets / metrics | relevance |
|---|---|---|---|---|
| GLC, 2025, Generative Latent Coding, https://arxiv.org/abs/2512.20194 | transform coding in VQ-VAE generative latent space + categorical hyper module + code-prediction supervision | high visual quality below 0.04 bpp; reports same CLIC2020 FID as MS-ILLM with 45% fewer bits | CLIC2020 test, DIV2K, Kodak, MS-COCO patches; bpp, FID/KID/LPIPS/DISTS/PSNR/MS-SSIM | base system and official curve anchor; GP-ResLC must beat this, not merely reproduce it |
| HVQ-CGIC, 2025, https://arxiv.org/abs/2512.07192 | hyperprior entropy modeling for VQ-index generative compression | adapts entropy of VQ indices; reports same LPIPS as Control-GIC/CDC/HiFiC with 61.3% fewer Kodak bits | Kodak, LPIPS/RD | VQ generative codecs still leave entropy-modeling gains; scratch GP-ResLC should entropy-code semantic tokens adaptively, not fixed-width forever |
| VQGAN/semantic-token compression family | discrete semantic representation + generative decoder | strong realism at very low bpp but risks semantic drift | MS-COCO, Kodak, CLIC; FID/KID/LPIPS | motivates separating semantic code s from residual code r in complete GP-ResLC |

Takeaway: GLC already moved transform coding into a perceptual/generative latent. The next gain should come from conditional residual coding: if z_hat already gives the generator enough information, do not spend y bits on predictable components.

## 5. Diffusion / Ultra-Low-Bitrate Generative Compression

| paper | core method | novelty / performance | datasets / metrics | relevance |
|---|---|---|---|---|
| PerCo, 2023, https://arxiv.org/abs/2310.10325 | VQ representation + global description + iterative diffusion decoder | realism at 0.1 down to 0.003 bpp; strong FID/KID, weak pixel fidelity | MS-COCO/Kodak-like, FID/KID/LPIPS | shows perception can become weakly bitrate-dependent at ultra-low rates; too slow for practical codec |
| PerCo(SD), 2024, https://arxiv.org/abs/2409.20255 | open Stable Diffusion v2.1 version of PerCo | open alternative to proprietary GLIDE-based PerCo; improved perceptual characteristics at distortion cost | MSCOCO-30k, perceptual metrics | useful open diffusion baseline; GP-ResLC aims to be much lighter and real-codec consistent |
| StableCodec, 2025, https://arxiv.org/abs/2506.21977 | one-step diffusion extreme compression with deep compression latent codec and dual-branch coding | targets <0.05 bpp, reports strong FID/KID/DISTS even near 0.005 bpp with practical speed | CLIC2020, DIV2K, Kodak; FID/KID/DISTS/fidelity | strongest warning: official curve improvement must be large enough to matter; scratch GP-ResLC should consider one-step generative decoder later |
| Diffusion-based residual / hybrid codecs | compact representation + diffusion prior + residual/fidelity branch | improves realism while retaining more fidelity than pure text/semantic diffusion | CLIC, Kodak, COCO; FID/KID/LPIPS/DISTS | complete GP-ResLC can be interpreted as a deterministic residual branch paired with a generative prior |

Takeaway: diffusion codecs dominate ultra-low-bitrate realism but pay complexity and fidelity costs. GP-ResLC's publishable niche is a fast real codec that gets diffusion-like R-P gains by better residual allocation inside a GLC-style latent codec.

## 6. Current GP-ResLC Position

Current rho-gate result:

- It is on-axis but incomplete. It sends less residual precision where a decoder-recomputable gate predicts generator-recoverability.
- It does not yet explicitly learn mu_theta(z_hat,q) as the generator-recoverable latent component.
- It improves official GLC on CLIC2020 by DISTS/FID BD-rate roughly -9.07% / -6.10%, but this margin is not yet large enough for a strong VCIP story if positioned as a major codec advance.

Complete-design direction now underway:

- Implement real codec for predictor_param_mode=latent_residual.
- Encode arithmetic symbols for round(y_scaled - base_mean - mu_theta(z_hat,q)).
- Decode by recomputing mu_theta(z_hat,q) from transmitted z_hat and adding it back.
- Train q-conditioned mu_theta with rate + perceptual loss plus optional Smooth-L1 residual prediction against y_scaled - base_mean.

## 7. Research Hypotheses to Test

1. latent_residual can reduce real bpp_y more than rho-gating because it changes the coded variable, not only quantization strength.
2. Frozen GLC may limit the decomposition because z_hat and y were not trained to split into predictable/unpredictable components. If gains saturate below ~15% official-curve BD-rate, partially unfreeze hyper_dec, y_prior_fusion, and eventually enc/hyper_enc.
3. Scratch GP-ResLC can outperform pretrained adaptation if semantic code s, generator, and residual code r are co-designed. Risk: training instability, codebook collapse, and failure to reproduce GLC-level baseline.
4. Official-curve claims must use real codec bpp on CLIC2020 full test, DIV2K validation, and Kodak. Estimated likelihood bpp is development-only.

## 8. Immediate Action Queue

1. Finish v3_latent_residual_lR10_lp4_mp005_nogate_12k and evaluate all checkpoints on Kodak/DIV2K subset with real codec.
2. If V3 reduces bpp but hurts DISTS/LPIPS, add weak GLC distillation or reduce lambda_R.
3. If V3 does not reduce bpp, increase lambda_R, add residual target pretraining, or use q-specific V1 latent residual runs before all-q V2.
4. If V3 shows signal, run CLIC2020 full real evaluation and official-curve comparison.
5. Begin scratch GP-ResLC skeleton only after the pretrained complete-design branch has a validated or rejected result.


## 9. 2026-06-20 Web-Audited Literature Notes

This section records papers re-checked during the GP-ResLC implementation/evaluation loop. It is intentionally focused on what changes the research plan.

| topic | verified paper/source | key technical point | implication for GP-ResLC |
|---|---|---|---|
| GLC anchor | Generative Latent Coding for Ultra-Low Bitrate Image Compression, arXiv:2512.20194, https://arxiv.org/abs/2512.20194 | GLC performs transform coding in a VQ-VAE generative latent space, adds categorical hyper information, and code-prediction supervision; reports same CLIC2020 FID as MS-ILLM with 45% fewer bits. | Official-curve comparison must be against GLC, not only HiFiC/MS-ILLM. Our CLIC2020 full-test real-codec comparison is the correct anchor. |
| GLC image/video extension | Generative Latent Coding for Ultra-Low Bitrate Image and Video Compression, arXiv:2505.16177, https://arxiv.org/abs/2505.16177 | Generalizes GLC to image/video and reports 65.3% DISTS bitrate saving over PLVC for video. | The image method should keep the residual-suppression story precise; future extension could target video residual predictability, but not for the current paper. |
| RD SOTA entropy modeling | MLICv2, arXiv:2504.19119, https://arxiv.org/abs/2504.19119 | Adds token mixing, hyperprior-guided global correlation, channel reweighting, and instance adaptation; reports strong BD-rate gains versus VTM on Kodak/Tecnick/CLIC Pro Val. | If scratch GP-ResLC is built, the entropy model should be multi-reference/slice-aware. Simple z-to-mean residual prediction is underpowered. |
| diffusion ultra-low bitrate | StableCodec, arXiv:2506.21977, https://arxiv.org/abs/2506.21977 | One-step diffusion codec targets <0.05 bpp, even around 0.005 bpp, with DCL codec and dual-branch coding for fidelity. | This is the high bar for perceptual ultra-low bitrate. GP-ResLC's niche must be real-codec consistency, speed, and residual-allocation interpretability rather than pure diffusion realism. |
| VQ generative compression | HVQ-CGIC, arXiv:2512.07192, https://arxiv.org/abs/2512.07192 | Derives hyperprior entropy modeling for VQ indices and reports same Kodak LPIPS as several generative codecs with 61.3% fewer bits. | Supports the idea that VQ/generative codecs still have significant untapped entropy savings. Scratch GP-ResLC should entropy-code semantic tokens adaptively. |
| perceptual LIC loss/latent refinement | Semantic Ensemble Loss and Latent Refinement, arXiv:2401.14007, https://arxiv.org/abs/2401.14007 | Uses semantic ensemble loss and content-aware latent refinement; reports 62% bitrate saving over MS-ILLM under CLIC2024 FID. | Latent refinement is related to our residual allocation, but paper claims must distinguish test-time optimization from a fixed real codec. |
| diffusion perceptual compression | PerCo, arXiv:2310.10325, https://arxiv.org/abs/2310.10325 | Conditions diffusion decoding on a VQ image representation plus global description; operates down to 0.003 bpp and shows FID/KID weakly dependent on bitrate. | Confirms R-D-P theory in practice. Our short-track should foreground DISTS/FID and be honest about PSNR/LPIPS tradeoffs. |
| open diffusion baseline | PerCo(SD), arXiv:2409.20255, https://arxiv.org/abs/2409.20255 | Stable-Diffusion-based open version of PerCo; improves perceptual characteristics at higher distortion. | Useful comparison class, but too slow/complex for our current real-codec branch. |
| efficient LIC context | ELIC, arXiv:2203.10886, https://arxiv.org/abs/2203.10886 | Uneven grouped space-channel contextual coding gives strong practical RD and speed. | Complete/scratch GP-ResLC should use grouped context; current four-part stage gate is a minimal step in that direction. |
| parallel context | Checkerboard Context Model, arXiv:2103.15306, https://arxiv.org/abs/2103.15306 | Two-pass context enables parallel decoding with near-autoregressive performance and much faster decoding. | Justifies avoiding fully serial raster context in the real codec. |
| hybrid transform | TCM, arXiv:2303.14978, https://arxiv.org/abs/2303.14978 | Mixed Transformer-CNN blocks and channel-wise entropy model perform strongly on Kodak/Tecnick/CLIC Pro Val. | Scratch branch should start from a hybrid transform rather than plain convolution. |

Research conclusion after the audit:

- The current rho1.16 branch is a credible short-track result because it is real-codec, protocol-clean, and improves GLC on CLIC2020/DIV2K DISTS/FID.
- The complete-design branch should not merely unfreeze the old GLC entropy modules. Current experiments show real-codec mismatch. A scratch or staged model must learn the decomposition jointly: semantic/generative code first, then residual code conditioned on that semantic code.
- For a stronger full paper, the target should be a hybrid between GLC/HVQ-CGIC/MLICv2 ideas: adaptive semantic-token entropy, decoder-predictable latent mean, residual entropy coding with grouped context, and an R-P objective verified by serialized bpp.



## 10. 2026-06-21 Additional Ultra-Low-Bitrate Generative Compression Notes

These papers were checked after the scratch GP-ResLC branch started producing positive semantic-plus-residual evidence. They are especially relevant because they split information into semantic/coarse generation and residual/detail channels.

| topic | verified paper/source | key technical point | implication for GP-ResLC |
|---|---|---|---|
| progressive VQ generative residuals | ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization, arXiv:2603.02897, https://arxiv.org/abs/2603.02897 | Uses residual vector quantization (RVQ) to add codewords stage by stage for coarse-to-fine reconstruction and progressive bitstreams; reports large Kodak DISTS/LPIPS bitrate savings against MS-ILLM and faster encode/decode. | Strongly supports the next scratch direction: replace the single residual bottleneck with progressive/RVQ residual stages so early bits carry the most unpredictable correction. |
| explicit semantics + implicit textures | Dual-Representation Image Compression at Ultra-Low Bitrates via Explicit Semantics and Implicit Textures, arXiv:2602.05213, https://arxiv.org/abs/2602.05213 | Conditions a diffusion model on explicit high-level semantics and uses reverse-channel coding for implicit fine details; reports DISTS BD-rate gains on Kodak, DIV2K, and CLIC2020. | Very close in spirit to GP-ResLC's thesis. Our differentiator should be a fast deterministic learned codec with an explicit residual bitstream rather than reverse-channel/diffusion sampling. |
| semantic + pixel diffusion compression | SPRDiff: Exploiting Semantic and Pixel Representations for Ultra-Low Bitrate Image Compression, arXiv:2606.01608, https://arxiv.org/abs/2606.01608 | Uses semantic and distortion/pixel representations plus diffusion guidance to improve ultra-low bitrate R-D-P below 0.03 bpp. | Confirms that pure semantics are insufficient; a residual/pixel-correction channel is needed to avoid semantic drift. Scratch GP-ResLC's semantic code plus residual code is aligned with this trend. |

Updated research implication:

- The scratch result `semantic bpp 0.00977 + residual bpp about 0.0034` is not an isolated trick; it matches the 2026 trend toward explicit semantic/generative information plus compact detail residuals.
- The strongest next architectural move is progressive residual coding: an RVQ-like residual stream, or multiple residual groups with monotonically increasing perceptual/detail roles, with each stage independently entropy modeled and optionally truncatable.
- For VCIP short track, the pretrained real-codec GLC branch remains the safer claim; the scratch branch now provides a method-faithful future-work/ablation story with real positive evidence.


## 2026 Addendum: Diffusion/RDP Ultra-Low-Bitrate Papers

### CADC: Content Adaptive Diffusion-Based Generative Image Compression, arXiv:2602.21591

CADC targets ultra-low-bitrate diffusion-based generative compression and argues that existing diffusion codecs are not content-adaptive enough. Its three main components are uncertainty-guided adaptive quantization, auxiliary-decoder-guided information concentration, and bitrate-free adaptive textual conditioning derived from an auxiliary reconstruction. The relevance to GP-ResLC is direct: CADC treats decoder-side generative prior alignment and uncertainty as the key to deciding what information must be preserved. GP-ResLC can position its residual gate/predictor as a deterministic, codec-friendly alternative to diffusion-time adaptation.

### DCIC: Dual-Constrained Diffusion Image Compression for Operational Rate-Distortion-Perception Optimization, arXiv:2606.13366

DCIC is a very recent RDP-oriented diffusion codec. It constrains diffusion restoration with both distortion and idempotence constraints, using the latter as a practical surrogate for distributional/perception consistency without extra rate. It explicitly exposes RD, RP, and RDP operating points from a single bitstream. This is important for VCIP positioning: GP-ResLC short-track is currently strongest as an R-P real-codec rate-saving method, while the full scratch branch should aim toward R-D-P by preserving a clean stage-0/stage-1 residual decomposition.

### Implication For GP-ResLC

The 2026 diffusion papers strengthen the paper motivation but also raise the bar: ultra-low-bitrate claims increasingly require either explicit RDP control or a convincing content-adaptive mechanism. For GP-ResLC, the strongest differentiator remains practical real-codec residual suppression: no diffusion sampling, no text side-channel, and decoder-computable gates. The scratch experiments show that stage-wise residual coding is plausible, but the fine stage must be trained with a sparse-use objective so it does not collapse or become bit-inefficient.


## 11. 2026-06-21 Web Refresh: What Changes The Next Experiments

Sources checked in this refresh:

- HDCompression: Hybrid-Diffusion Image Compression for Ultra-Low Bitrates, arXiv:2502.07160, https://arxiv.org/abs/2502.07160
- Learned Image Compression with Dictionary-based Entropy Model, arXiv:2504.00496, https://arxiv.org/abs/2504.00496
- HVQ-CGIC: Enabling Hyperprior Entropy Modeling for VQ-Based Controllable Generative Image Compression, arXiv:2512.07192, https://arxiv.org/abs/2512.07192
- ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization, arXiv:2603.02897, https://arxiv.org/abs/2603.02897
- GLC image/video version, arXiv:2505.16177, https://arxiv.org/abs/2505.16177

| paper | new technical signal | consequence for GP-ResLC |
|---|---|---|
| HDCompression | Hybridizes a VQ generative stream, conventional LIC/fidelity stream, and lightweight diffusion correction. The key premise is that neither pure LIC nor pure VQ generation is sufficient at ultra-low bpp. | Confirms our semantic-plus-residual split. GP-ResLC should avoid turning the residual branch into a blind global coarsening knob; it needs content-aware residual usefulness prediction. |
| Dictionary-based Entropy Model | Adds a learnable dictionary and cross-attention prior to capture dataset-level latent structures beyond local hyperprior/autoregressive context. | A future full GP-ResLC entropy model should condition residual coding on learned common structures. This is close to the original idea: common/predictable components should not be sent. |
| HVQ-CGIC | Derives a hyperprior for VQ index entropy and reports very large LPIPS-rate savings versus VQ/GIC baselines. | Scratch Stage-A currently uses fixed semantic bpp. That is a known ceiling; adaptive semantic-token entropy is required for a competitive full paper. |
| ProGIC | Uses RVQ stages for progressive coarse-to-fine generative reconstruction, with reported large Kodak DISTS/LPIPS savings versus MS-ILLM and faster inference. | The scratch progressive residual branch is directionally right, but our fine stage needs a gate objective tied to perceptual payoff and not only L1/DISTS scalar hinges. |
| GLC image/video | Extends generative latent coding to video and reports strong DISTS saving over PLVC. | GP-ResLC should keep its image claim precise: residual precision reduction inside GLC is a real-codec add-on; video extension is future work. |

Updated experiment implication after q1/q0 stage-quant failures:

- A global rho target is too blunt at the lowest rates. It reliably reduces serialized y bits but causes perceptual degradation before it creates a better curve point.
- The next paper-facing improvement should be a sendability predictor: a decoder-computable score trained to identify regions/channels where GLC reconstruction is insensitive to coarsening. Possible teachers are local GLC-vs-coarsened DISTS/LPIPS deltas, residual magnitude, texture/edge statistics, and entropy scale.
- This aligns with HDCompression/CADC/SPDiff-style content adaptation but remains deterministic and real-codec compatible.
- For scratch, the same lesson appears in the selected-region residual experiments: sparse residual positions help only when the selected correction is trained to be useful.


## 2026-06-21 Additional SOTA Refresh

Primary sources checked during the sprint:

- ChWDTA: Channel-wise Wavelet-Domain Transformer Attention and Entropy Modeling for Learned Image Compression, arXiv:2606.00111, https://arxiv.org/abs/2606.00111
- ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization, arXiv:2603.02897, https://arxiv.org/abs/2603.02897
- DiffCR: Towards Efficient Low-rate Image Compression with Frequency-aware Diffusion Prior Refinement, arXiv:2601.10373, https://arxiv.org/abs/2601.10373
- ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation, arXiv:2603.10188, https://arxiv.org/abs/2603.10188
- Learned Image Compression with Dictionary-based Entropy Model, arXiv:2504.00496, https://arxiv.org/abs/2504.00496
- HDCompression: Hybrid-Diffusion Image Compression for Ultra-Low Bitrates, arXiv:2502.07160, https://arxiv.org/abs/2502.07160
- Generative Image Compression by Estimating Gradients of the Rate-variable Feature Distribution, arXiv:2505.20984, https://arxiv.org/abs/2505.20984

### What Changed In The Frontier

1. RD-oriented LIC is still moving through stronger entropy models rather than only larger transforms. ChWDTA and dictionary-based entropy models both point to the same lesson: the prior should expose structure that the entropy model can predict cheaply. ChWDTA uses channel-wise wavelet decomposition in attention and entropy coding; dictionary entropy modeling injects training-set structural memories through cross-attention. For GP-ResLC, this supports the argument that a residual should be sent only when it is not predictable from decoder-side priors/context.

2. Low-rate perceptual compression is increasingly hybrid and progressive. HDCompression, DiffCR, and ProGIC all combine a compact transmitted representation with a strong generative reconstruction prior. The winning theme is not pure hallucination: the codec transmits just enough fidelity or residual information to anchor the generator. This is very close to the GP-ResLC framing, but our contribution should remain codec-grounded: real arithmetic-coded residual bits, no side gate map, and exact bpp accounting.

3. Diffusion compression is becoming faster but still heavy for a practical LIC paper. DiffCR reports a two-step consistency-style decoder by refining a latent diffusion prior without updating the backbone. This is strong evidence for low-rate generative priors, but also leaves room for a lighter GLC-compatible residual-prior approach that avoids diffusion sampling cost.

4. RVQ/progressive GIC is a direct neighbor to scratch GP-ResLC. ProGIC uses residual vector quantization as progressive coarse-to-fine information. Our scratch branch is conceptually aligned, but its current absolute quality is far below pretrained GLC. It is useful as method-faithfulness evidence, while the pretrained real-codec branch remains paper-facing.

5. Local sendability is the unsolved piece. Recent papers improve priors, progressive residuals, or generative refinement, but the exact decision of which residual components can be omitted at ultra-low rate remains under-explored. The q1 target sweeps in this project show that heuristic sendability and global rho targets reduce bpp but overpay in DISTS/LPIPS. A measured local sensitivity predictor is therefore the most plausible next contribution if we want a larger gain while staying on-axis.

### GP-ResLC Implication

For VCIP short-track, the strongest claim should not be framed as a generic new entropy model. The claim should be:

> A generator-aware residual codec can lower real arithmetic-coded bits by coarsening predictable residual components, while preserving perceptual quality under full-resolution real-codec evaluation.

The current rho1.16 pretrained branch already proves this empirically against local and official GLC curves. The faithful stage-quant branch proves feasibility without transmitting a side map but still needs a measured local sensitivity teacher for large gains. Scratch RVQ/progressive residual experiments support the long-term full GP-ResLC direction, but are not yet competitive enough to lead.


## 12. 2026-06-21 Late Sprint Update: What The Experiments Changed

The literature trend and the local experiments now agree on the same point: a simple global residual predictor is not enough. Recent ultra-low-bitrate work increasingly uses content-adaptive or stage-progressive allocation, and the GP-ResLC ablations show why.

Local evidence:

- Pure rho gating remains the paper-facing lead because it gives protocol-clean real-codec gains on CLIC2020 and DIV2K.
- Gate-only DISTS fine-tunes and q0/q1 hinge fine-tunes are stable but mostly redistribute small tradeoffs; they do not create a new official-curve gap.
- Naive latent-residual prediction against frozen GLC destabilizes the four-part prior. The predictor fights GLC's autoregressive context instead of cleanly removing predictable residuals.
- A very small predictor-only mean correction from the rho1.16 lead is safe and exact under real codec, but only shifts Kodak distribution metrics slightly. It does not replace the DISTS lead.

Research interpretation:

1. The short-track claim should stay narrow and strong: decoder-computable residual precision suppression lowers real arithmetic-coded bits for perceptual quality on CLIC2020/DIV2K.
2. The full GP-ResLC model should not be described as solved by the current predictor head. A credible full version needs stage-aware residual prediction because each four-part prior stage has different decoder-side context.
3. A scratch version is still scientifically justified, but the immediate high-probability path is not generic scratch training. It is: semantic/generative code, then stage-wise residual predictor/gate trained with a hard real-codec-compatible objective and sensitivity-aware supervision.
4. Paper positioning should contrast GP-ResLC with diffusion/VQ/RVQ hybrids as a fast deterministic codec: it keeps a real bitstream, no transmitted gate map, and exact encode/decode accounting.

Updated action queue:

1. Keep `rho1.16` as the main VCIP checkpoint unless a new CLIC/DIV2K real-codec run beats its DISTS and FID simultaneously.
2. Use `predonly_b003` as an ablation showing that residual mean prediction has signal but global prediction is insufficient.
3. Prioritize a stage-aware/sensitivity-aware predictor if continuing pretrained work: per-stage mean correction, bounded delta, DISTS/LPIPS hinge, and early real-codec Kodak8/24 checks.
4. For scratch, invest in adaptive semantic entropy and RVQ/progressive residual stages only if the goal is the longer full-paper GP-ResLC rather than short-track risk reduction.


## 13. 2026-06-21 Top-Conference Pivot: Why GLC-Latent Residual Is The Next Main Branch

This update follows the decision to deprioritize the VCIP-short-track cycle and pursue a larger top-conference contribution.

Freshly rechecked primary sources:

- Official GLC repository: https://github.com/jzyustc/GLC
- Official GLC `test_image.py`: https://raw.githubusercontent.com/jzyustc/GLC/main/test_image.py
- Official GLC metric code: https://raw.githubusercontent.com/jzyustc/GLC/main/src/utils/metric_image.py
- TFDS CLIC catalog: https://www.tensorflow.org/datasets/catalog/clic
- HiFiC: High-Fidelity Generative Image Compression, https://arxiv.org/abs/2006.09965
- MS-ILLM: Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models, https://arxiv.org/abs/2301.11189
- DLF: Extreme Image Compression with Dual-generative Latent Fusion, https://arxiv.org/abs/2503.01428
- StableCodec: Taming One-Step Diffusion for Extreme Image Compression, https://arxiv.org/abs/2506.21977
- ResULIC: Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion, https://arxiv.org/abs/2505.08281
- DiffO: Single-step Diffusion for Image Compression at Ultra-Low Bitrates, https://arxiv.org/abs/2506.16572
- One-Step Diffusion for Perceptual Image Compression, https://arxiv.org/abs/2602.01570
- HiDE: Hierarchical Dictionary-Based Entropy Modeling for Learned Image Compression, https://arxiv.org/abs/2603.06766
- Dictionary-based Entropy Model, https://arxiv.org/abs/2504.00496
- ELIC, TCM, MLIC++: https://arxiv.org/abs/2203.10886, https://arxiv.org/abs/2303.14978, https://arxiv.org/abs/2307.15421
- R-D-P theory: https://arxiv.org/abs/1901.07821

Protocol conclusion:

- The CLIC2020 test interpretation remains correct: professional 250 plus mobile 178 gives 428 images. Local shifted 256-patch counting reproduces 28,650 patches, which matches the GLC/HiFiC-style protocol.
- GLC official code uses full-resolution inputs, padding 64, q indices 0-3, and FID/KID patch evaluation through `evaluate_quality`; our real-codec package should keep using this as the protocol anchor.

Research conclusion:

The frontier has converged on a common structure: a compact semantic/generative representation plus a transmitted correction stream. DLF splits semantic and detail generative latents; ResULIC sends semantic residuals into a diffusion model; StableCodec/DiffO/one-step diffusion use strong pretrained priors with fidelity branches; dictionary entropy models use external priors to avoid re-sending common structure. GP-ResLC should therefore stop relying on post-hoc global GLC parameter edits as the full-paper path.

New main branch:

- Use the strong pretrained GLC/VQGAN generator as the synthesis prior.
- Learn a low-rate semantic VQ code `s` and `mu_theta(s)` directly in the GLC/VQGAN latent space.
- Entropy-code only `r = l - mu_theta(s)` or a low-dimensional transform of that residual.
- This keeps the core thesis exact: predictable latent information is generated from `s`; only unpredictable residual information is transmitted.

Why this is stronger than the previous scratch branch:

- The previous scratch Stage-A generator was too weak, so residual coding was testing the idea under a poor generator.
- The new branch isolates the factorization question using a strong generator, letting us measure whether semantic code plus residual can approach or beat GLC at lower bpp.
- If successful, the full paper can later replace fixed semantic bpp with HVQ-style adaptive token entropy and replace the Gaussian residual proxy with a real multi-reference residual entropy model.

Current implementation status:

- Added `gp_reslc/scratch/glc_latent_residual.py` and `scripts/train_glc_latent_residual.py`.
- Started W&B run `glc_latent_residual_predictor_warmup_6k` (`woye1ymw`) to train `mu_theta(s)` before residual coding.


## 13. 2026-06-21 Stable Sparse Residual Positioning Refresh

Additional web-checked sources in this pass:

- UIGC: Unifying Generation and Compression: Ultra-low bitrate Image Coding Via Multi-stage Transformer, arXiv:2403.03736, https://arxiv.org/abs/2403.03736
- HDCompression: Hybrid-Diffusion Image Compression for Ultra-Low Bitrates, arXiv:2502.07160, https://arxiv.org/abs/2502.07160
- PerCo: Towards image compression with perfect realism at ultra-low bitrates, arXiv:2310.10325, https://arxiv.org/abs/2310.10325
- PerCo(SD): Open Perceptual Compression, arXiv:2409.20255, https://arxiv.org/abs/2409.20255
- TCM: Learned Image Compression with Mixed Transformer-CNN Architectures, arXiv:2303.14978, https://arxiv.org/abs/2303.14978
- MLIC++: Linear Complexity Multi-Reference Entropy Modeling for Learned Image Compression, arXiv:2307.15421, https://arxiv.org/abs/2307.15421
- HVQ-CGIC: Enabling Hyperprior Entropy Modeling for VQ-Based Controllable Generative Image Compression, arXiv:2512.07192, https://arxiv.org/abs/2512.07192

### What This Changes

1. UIGC and PerCo-style methods strengthen the core motivation: at <0.05 bpp, the codec must use a learned generative prior instead of trying to transmit all appearance detail. UIGC uses VQ tokenization plus a multi-stage transformer that both estimates token priors and regenerates missing tokens. GP-ResLC should frame `mu_theta(s)` similarly: a decoder-side predictor that reconstructs what the generator can infer before any residual bits are spent.

2. HDCompression and PerCo(SD) show the current frontier trend toward hybrid systems: semantic/generative streams plus fidelity correction streams. GP-ResLC differs by keeping the correction stream deterministic, sparse, finite-valued, and arithmetic-codeable. This is a useful differentiator against diffusion-heavy codecs whose realism comes with sampling complexity and sometimes weaker fidelity.

3. HVQ-CGIC exposes a limitation in the current scratch branch: Stage-A semantic tokens are still fixed-width. The stable residual evidence is encouraging, but a full top-conference version should entropy-code semantic tokens with a hyperprior or context model, otherwise the semantic stream has an avoidable bpp floor.

4. TCM/MLIC++ remind us that strong RD codecs win through better priors/context, not just better decoders. For GP-ResLC, this suggests the residual entropy model should eventually be slice/context-aware. However, the immediate positive result is that even a fixed finite ternary residual stream improves all center-proxy metrics at very low residual bpp. That is a cleaner first-principles claim than adding a heavy context model too early.

### Updated GP-ResLC Research Claim

The strongest current scratch/full-design claim is no longer merely that a residual stream helps. It is more specific:

> A decoder-side generative predictor plus a finite sparse residual stream can improve perceptual quality while sending only the residual symbols the generator cannot infer. In the current center-crop proxy, a stable ternary residual with only ~0.00085 residual bpp improves LPIPS/DISTS/L1/MSE on CLIC2020, DIV2K, and Kodak.

This is closer to the original GP-ResLC thesis than the earlier clamped hard-topk curve. The next paper-facing step is to replace proxy bpp with an actual bounded-symbol arithmetic codec and then move from center-crop proxy to full-resolution CLIC/DIV2K/Kodak evaluation.

### Updated Experiment Priority

1. Implement a real finite-support residual codec for stable ternary symbols. Start with a simple transmitted sparse-index/sign representation or an adaptive categorical model; compare exact serialized bpp to `gaussian_bits_stable`.
2. Run full-resolution CLIC2020/DIV2K/Kodak reconstruction for stable ternary topk0005 and topk002. Only then compute FID/KID shifted 256 patches.
3. Add fixed-validation checkpoint selection. Random Kodak validation batches already selected weak checkpoints in several runs.
4. Explore topk0015 or learned payload budget after the real-codec check. Increasing symbol magnitude is currently dominated; increasing ternary positions is better.
5. Longer-term: adaptive entropy coding for semantic VQ tokens, inspired by HVQ-CGIC, and slice/context-aware residual entropy, inspired by MLIC++/TCM.
