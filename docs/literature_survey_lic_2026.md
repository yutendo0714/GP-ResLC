# Learned Image Compression Survey for GP-ResLC

調査日: 2026-06-18  
目的: VCIP向けに、超低ビットレート画像圧縮で「生成器が自力で復元できる情報は送らず、予測できない残差だけを送る」GP-ResLCの位置付けを明確にする。

## 1. 研究潮流の要約

LICは、Ballé型の変分オートエンコーダ + entropy bottleneckから始まり、hyperprior、autoregressive prior、channel/spatial context、Transformer/RWKV系の広域文脈へ進んできた。歪み系LICはPSNR/MS-SSIMではVVC Intra級またはそれ以上を狙う一方、超低レートではMSE最適化がぼけを生み、知覚品質の主戦場はRate-DistortionからRate-Distortion-Perceptionへ移っている。

GP-ResLCの軸は、GLCのような生成的VQ潜在コーデックに対して、送信済みの意味/ハイパー情報から生成器が復元できる潜在成分を予測し、その予測で説明できない残差だけを符号化すること。これは「より強い生成器を使う」だけでなく、entropy model側を生成事前に整合させる立場であり、GLC/Control-GIC/HVQ-CGIC/OneDC/StableCodecとは違う切り口になる。

## 2. 分野別代表論文

| 分野 | 代表/最新手法 | 手法概要 | 新規性 | 性能・評価 | データセット/指標 | GP-ResLCとの差分 |
|---|---|---|---|---|---|---|
| Hyperprior LIC | Ballé et al., Scale Hyperprior (ICLR 2018) | 解析変換 `g_a`、量子化潜在 `y`、ハイパー潜在 `z`で `p(y|z)` を推定 | side informationとしてhyperpriorをend-to-end学習 | 当時のANN圧縮SOTA、MS-SSIM/PSNRで改善 | Kodak等、PSNR/MS-SSIM/bpp | GP-ResLCもhyperprior構造を使うが、生成器が復元可能な成分を明示的に差し引く |
| Autoregressive + Hyperprior | Minnen et al. (NeurIPS 2018) | masked conv context + hyperprior | 階層priorとautoregressive priorの相補性を示す | 既存DL法比15.8%平均ファイル削減、BPG比8.4%小さい | Kodak、PSNR/MS-SSIM | 文脈は強いが復号逐次性が重い。GP-ResLCはGLCの4分割prior上で最小侵襲に検証 |
| Efficient context LIC | ELIC (CVPR 2022) | uneven channel grouping + space-channel context | energy compactionに合わせた不均一グループ化 | 高速プレビュー/逐次復号を持つ高効率LIC | Kodak/Tecnick/CLIC、PSNR/MS-SSIM | 歪み系SOTAの文脈設計。GP-ResLCでは将来のチャネルslice強化候補 |
| Transformer/CNN LIC | TCM (2023) | Transformer-CNN mixture block + channel-wise entropy model | local CNNとnon-local Transformerを並列融合 | Kodak/Tecnick/CLICでSOTA級RD | PSNR/MS-SSIM/bpp | 変換/entropy backboneの強化。GP-ResLCの主張は低レート生成prior整合 |
| Multi-reference entropy | MLIC++ / MLICv2 | channel/local/global multi-reference entropy, linear attention, instance adaptation | linear-complexity global context、Gumbel annealingによる入力別最適化 | MLIC++はKodakでVTM-17比BD-rate -13.39% PSNR。MLICv2+はKodak/Tecnick/CLICでVTM-17比 -20.46/-24.35/-19.14% | Kodak/Tecnick/CLIC、PSNR/MS-SSIM | `H(y|context)`を下げる方向。GP-ResLCは `H(y|s, generative prior)` を下げたい |
| Dictionary entropy | DCAD/Dictionary-based Entropy Model (2025) | 学習辞書 + cross attention entropy model | 訓練データ由来の典型構造をpriorに注入 | 性能/latencyバランスでSOTA主張 | Kodak等、RD | GP-ResLCの「生成器が予測できる構造」を辞書ではなく復号生成器の事前で表す |
| GAN perceptual | HiFiC (NeurIPS 2020) | learned compression + GAN/perceptual loss | R-D-P理論を実装に落とした高忠実生成圧縮 | ユーザスタディで、2倍以上bitを使う既存法より好まれるケース | Kodak/CLIC等、FID/KID/LPIPS/user study | 生成decoderでrealismを得るが、entropy側は生成予測残差ではない |
| Multi-realism | MS-ILLM / Multi-Realism (2022/2023) | 1つの圧縮表現からrealism-fidelityを復号側で制御 | distortion-realism frontierを単一bitstreamで移動 | 高realism/低distortionの両端でSOTA主張 | FID, KID, LPIPS, PSNR | GP-ResLCの将来β steeringに近い。VCIP版はまずrate削減に集中 |
| Semantic GAN perceptual | EGIC (2023) | semantic segmentation-guided discriminator + output residual prediction | 1モデルでdistortion-perceptionを横断、軽量 | HiFiC/MS-ILLM/DIRACより良好、歪み端ではVTM-20近い | Kodak等、FID/LPIPS/DISTS/PSNR | 出力残差は画像空間のrealism制御。GP-ResLCは符号化潜在の残差entropy |
| VQ generative | GLC (CVPR 2024/TCSVT 2025) | VQ-VAE生成潜在でtransform coding、categorical hyper module、code-prediction supervision | pixel空間でなく生成VQ潜在を圧縮 | 自然画像<0.04 bpp、顔<0.01 bpp。CLIC2020でMS-ILLM同FIDを45%少bit | Kodak/CLIC/FFHQ、FID/KID/LPIPS/DISTS/PSNR | GP-ResLCの直接ベース。GLCの `z_hat -> hyper_dec -> y prior` を生成予測残差へ補正 |
| VQ controllable | Control-GIC (2024) | VQGAN indicesの可変長/多粒度表現、情報密度に応じたgranularity | fine-grained bitrate adaptation | recent SOTA GICより優位主張 | Kodak等、LPIPS/FID/DISTS | VQ index数でrate制御。GP-ResLCは連続latent `y` の条件付き残差rateを削減 |
| VQ hyperprior | HVQ-CGIC (2025) | VQ index entropyにhyperpriorを導入 | 静的global index priorの非適応性を解消 | KodakでControl-GIC/CDC/HiFiCと同LPIPSを平均61.3%少bit | Kodak、LPIPS/bpp | VQ indicesのentropyを改善。GP-ResLCはGLCの連続 `y` entropy改善 |
| RVQ generative | ProGIC (2026) | residual vector quantizationのprogressive bitstream + lightweight backbone | coarse-to-fine progressive GIC | KodakでMS-ILLM比DISTS 57.57%、LPIPS 58.83% bitrate saving、GPUで10倍超高速 | Kodak、DISTS/LPIPS/speed | RVQで段階的に送る。GP-ResLCは送らないでよい成分を予測で引く |
| Extreme compact latent | MRT (2025) | Mixed RWKV-Transformerで1D compact latent、RWKV Compression Model | 2D潜在の空間冗長を避ける | <0.02 bppでDISTS優位、GLC比Kodak 43.75%、CLIC 30.59% bitrate saving | Kodak/CLIC、DISTS | 表現形を1D化。GP-ResLCは既存GLC潜在を残差化する短期実装 |
| Conditional diffusion | CDC/Yang & Mandt (2022/2023) | transform coded content latent + conditional diffusion decoder | textureを復号時生成しcontentだけ送る | GAN系よりFID良好、歪み指標も競争的 | 複数dataset、FID/PSNR/MS-SSIM | 生成器が補完する思想は近いが、GP-ResLCはGLC内のentropy prior改善から着手 |
| Perceptual diffusion | PerCo (2023/2024) | VQ表現 + global descriptionでiterative diffusion復号 | 0.003 bpp級でもrealistic reconstruction | FID/KIDでSOTA visual quality、bitrate依存を弱める | Kodak/MSCOCO等、FID/KID/LPIPS | 多段diffusionで遅い。GP-ResLC VCIP版は拡散を入れずGLC上で検証 |
| Stable Diffusion codec | DiffEIC (2024) | VAE圧縮content variable + frozen SD + control module | SD priorを極低rate復元に利用、space alignment loss | <0.1 bppでSOTAより視覚品質優位主張 | Kodak/CLIC等、FID/LPIPS/DISTS | 強力な外部prior。GP-ResLCはentropy model側の情報削減に焦点 |
| Residual diffusion | ResULIC (ICML 2025) | semantic residual coding + compression-aware diffusion | 残差信号をsemantic retrievalとdiffusion過程に注入 | LPIPS -80.7%、FID -66.3% BD-rate saving主張 | Kodak/MSCOCO等、LPIPS/FID | 「残差」思想は近いが、GP-ResLCは検索/拡散条件ではなく算術符号化latent残差 |
| One-step diffusion | StableCodec (ICCV 2025) | one-step diffusion + deep compression latent codec + dual-branch coding | 多段samplingなしで0.005 bpp級realism/fidelity | CLIC/DIV2K/KodakでFID/KID/DISTS SOTA主張、transform codec級速度 | CLIC2020/DIV2K/Kodak、FID/KID/DISTS | 強力なdecoder設計。GP-ResLCのjournal拡張候補だがVCIPではentropy貢献を分離 |
| One-step diffusion | OneDC (2025) | latent compression + one-step diffusion、hyperpriorをsemantic guidanceに使用、semantic distillation | text promptでなくhyperpriorをsemantic signal化 | multi-step diffusion比40%超bit削減、20倍高速 | Kodak/CLIC等、perceptual metrics | GP-ResLCと最も競合的。差分はhyperpriorをdecoder条件にするだけでなく、`y` priorを生成予測残差化 |
| One-step diffusion | DiffO (2025) | VQ residual training + rate-adaptive noise modulation | structural base code + latent residual、bitrateに応じdenoise strength制御 | 既存diffusion比約50倍高速、ULBで高perceptual quality | Kodak/CLIC等、perceptual metrics | GP-ResLCも低rate×残差だが、GLC entropy stageに閉じた短期検証が可能 |
| Rate-variable diffusion | GIC by feature-distribution gradient (2025) | compression processをSDE forward pathと解釈しreverse netで復元 | Gaussian noise初期化なし、少stepでsmooth rate control | GIC指標群で既存法超え主張 | perceptual/statistical/NR-IQA | rate variable decoder側。GP-ResLCはbitstream内の予測不能成分を減らす |
| Hybrid VQ + diffusion | HDCompression (2025) | generative VQ、conventional LIC、diffusionのdual-stream | diffusionでfidelity補助情報を抽出しVQ latent補正 | ULBで既存LIC/VQ/hybridよりbalanced performance | perceptual + fidelity metrics | 複雑なhybrid。GP-ResLCはGLCに最小追加で効果検証 |

## 3. データセットと評価指標

学習候補:
- OpenImages: 汎用自然画像、今回 `/dpl/openimages/train` を使用候補にする。
- DIV2K/LSDIR: 高解像度fidelity寄りの追加学習候補。現マシンには明示確認できていない。
- FFHQ: 顔の極低rateには有効だがVCIP短期では自然画像優先。

評価候補:
- Kodak: 24枚で高速に回せる。FID/KIDはpatch-FIDにしないと標本が不足する。
- CLIC2020 professional/mobile: GLC論文の主張と近い。FID/KIDは256 patch推奨。
- DIV2K val/Tecnick/MS-COCO 30K: 追加比較・一般化検証。

指標:
- Distortion: PSNR, MS-SSIM。MSEは学習安定用の補助として扱い、headlineにしすぎない。
- Perceptual fidelity: LPIPS, DISTS。低bitrateではGP-ResLCの主評価。
- Realism/distribution: FID, KID。Kodakではpatch-FID/KIDで標本数を補う。
- Rate: total bpp, `bpp_y`, `bpp_z`。本提案の直接効果は `bpp_y`、論文主張はtotal bppと知覚指標で示す。
- Curve: BD-rate over LPIPS/DISTS/FID/KID/PSNR。lower-is-better指標は符号反転して計算する。

## 4. MSEをloss/evalに使うべきか

結論: 使うが、主役にしない。

理由:
- 低bitrate生成圧縮では、MSE最小化はtexture hallucinationを罰し、ぼけた復元を好む。FID/KID/DISTS/LPIPSとは最適点がずれる。
- ただし完全に外すと、semantic driftや構造崩れをFIDだけでは抑えにくい。VCIP短期では `lambda_d * MSE + lambda_lpips * LPIPS` を補助として使い、`lambda_d` の小/中/0 ablationを必ず走らせる。
- 論文の主張は「同等DISTS/LPIPS/FIDで低bpp」、補助表にPSNR/MS-SSIMを出すのが自然。

推奨ablation:
1. `lambda_d=1, lambda_lpips=1`: 安定ベース。
2. `lambda_d=0.1, lambda_lpips=1`: 知覚寄り。
3. `lambda_d=0, lambda_lpips=1`: MSE無しの限界。
4. `lambda_align=0` vs `1`: code-prediction supervisionが本当に効くか確認。

## 5. GP-ResLCのVCIP向け主張

最小投稿ストーリー:
1. GLCは生成VQ潜在で優秀だが、`z_hat`から生成器/decoderが予測できる `y` 成分まで `bit_y` に載せている可能性がある。
2. Pθを `z_hat` から `y` prior paramsへzero-init加算し、生成可能成分をmean側へ寄せる。
3. `bit_z`は固定、変化するのは `bit_y`。同一DISTS/LPIPS/FIDでtotal bppが下がれば「予測不能残差だけ送る」主張が成立する。
4. scaleやquant_step膨張による見かけrate低下を避けるため、`bpp_y`, `bpp_total`, PSNR, LPIPS, DISTS, FID, delta_params norm, mu statsをwandbに記録する。

## 6. Sources

- Ballé et al., Variational Image Compression with a Scale Hyperprior: https://arxiv.org/abs/1802.01436
- Minnen et al., Joint Autoregressive and Hierarchical Priors: https://arxiv.org/abs/1809.02736
- ELIC: https://arxiv.org/abs/2203.10886
- TCM: https://arxiv.org/abs/2303.14978
- MLIC++: https://arxiv.org/abs/2307.15421
- MLICv2: https://arxiv.org/abs/2504.19119
- Dictionary-based Entropy Model: https://arxiv.org/abs/2504.00496
- HiFiC: https://arxiv.org/abs/2006.09965
- MS-ILLM / Multi-Realism: https://arxiv.org/abs/2212.13824
- EGIC: https://arxiv.org/abs/2309.03244
- GLC image/video: https://arxiv.org/abs/2512.20194 and https://arxiv.org/abs/2505.16177
- Control-GIC: https://arxiv.org/abs/2406.00758
- HVQ-CGIC: https://arxiv.org/abs/2512.07192
- ProGIC: https://arxiv.org/abs/2603.02897
- MRT: https://arxiv.org/abs/2511.06717
- CDC: https://arxiv.org/abs/2209.06950
- PerCo: https://arxiv.org/abs/2310.10325 and PerCo(SD): https://arxiv.org/abs/2409.20255
- DiffEIC: https://arxiv.org/abs/2404.18820
- ResULIC: https://arxiv.org/abs/2505.08281
- StableCodec: https://arxiv.org/abs/2506.21977
- OneDC: https://arxiv.org/abs/2505.16687
- DiffO: https://arxiv.org/abs/2506.16572
- HDCompression: https://arxiv.org/abs/2502.07160
- Rate-Distortion-Perception: https://arxiv.org/abs/1901.07821
- LPIPS: https://arxiv.org/abs/1801.03924
- DISTS: https://arxiv.org/abs/2004.07728
- A-DISTS: https://arxiv.org/abs/2110.08521


## 2026-06-19 Latest Check: Generative / VQ / R-P Direction

Recent papers checked during the GP-ResLC experiment loop:

| year | paper | task | key idea | relation to GP-ResLC |
|---:|---|---|---|---|
| 2025 | HDCompression: Hybrid-Diffusion Image Compression for Ultra-Low Bitrates, https://arxiv.org/abs/2502.07160 | ultra-low bitrate generative compression | combines generative VQ modeling, diffusion, and conventional LIC to balance fidelity and perceptual realism | supports the thesis that ultra-low bitrate needs a generative prior plus transmitted correction; GP-ResLC is a lighter residual-suppression variant on top of GLC |
| 2025 | Generative Image Compression by Estimating Gradients of the Rate-variable Feature Distribution, https://arxiv.org/abs/2505.20984 | diffusion-based GIC | treats compression as an SDE-like forward path and learns a reverse process for rate-variable reconstruction | nearby R-P motivation, but changes decoder/generator substantially; GP-ResLC keeps GLC fixed and learns what not to send |
| 2025 | Generative Preprocessing for Image Compression with Pre-trained Diffusion Models, https://arxiv.org/abs/2512.15270 | codec-agnostic R-P preprocessing | distills/fine-tunes diffusion preprocessing under a rate-perception objective, reporting large DISTS BD-rate reductions | strong evidence that DISTS-oriented R-P framing is accepted; GP-ResLC differs by operating inside a learned generative codec rather than preprocessing images |
| 2026 | RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression, https://arxiv.org/abs/2604.10546 | VQ-based ultra-low bitrate GIC | differentiable codebook distribution and entropy-aware autoregressive VQ rate control | points to entropy-constrained tokenization as a strong baseline/future comparison; GP-ResLC can be framed as entropy-aware residual suppression for GLC latents |
| 2022 | Entroformer, https://arxiv.org/abs/2202.05492 | transformer entropy modeling | transformer context model for global dependencies in latent entropy | informs the full-version GP-ResLC direction: replace simple local predictor with stronger context/transformer residual prior after short-track gate evidence is solid |
| 2023 | TCM, https://arxiv.org/abs/2303.14978 | Transformer-CNN LIC | mixed Transformer-CNN transform and entropy modules, evaluated on Kodak/Tecnick/CLIC | useful classical LIC RD baseline family; GP-ResLC short-track should avoid claiming SOTA RD and focus on R-P under GLC |

Implications for VCIP framing:

- The current field is moving toward generative/R-P compression at ultra-low bitrates, often using diffusion or VQ. GP-ResLC should not claim to beat those full systems yet; instead claim a lightweight residual-suppression mechanism for an existing GLC-style generative codec.
- DISTS/FID/KID are legitimate primary R-P metrics. LPIPS should remain reported, but current GP-ResLC results show LPIPS is not always aligned with DISTS/FID under bit suppression.
- The strongest differentiator is decoder-side determinism: the gate is recomputed from transmitted `z_hat`, so it transmits no map and directly operationalizes “do not send predictable residual.”
