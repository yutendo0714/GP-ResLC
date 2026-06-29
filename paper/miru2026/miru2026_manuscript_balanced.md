# MIRU2026 短縮版原稿

作成日: 2026-06-22  
用途: 4ページ制限を意識しつつ，compact版より関連研究と提案手法の説明を少し厚くしたWord/LaTeX転記用本文案。

## 4ページ用の推奨配置

- 図1: GLC概要+GP-ResLC差分の統合手法図 `paper/miru2026/figures/glc_gp_reslc_unified_overview.png`
- 図2: 主結果curve統合図 `paper/miru2026/figures/result_curves_clic_div2k_perceptual_2x4.png`
- 表1: BD-rate主結果
- 表2: CLIC2020のbpp内訳
- 図3: 復元例とrho可視化 `paper/miru2026/figures/clic_q3_rho_overlay_top4.png`

検討用に10-panel版 `paper/miru2026/figures/result_curves_full_10panel.png` も作成済み。ただし本文では小さくなりすぎるため，4ページ原稿ではFID/KID/DISTS/LPIPSの2x4版を推奨する。まだ紙面が厳しい場合は，図3を補足またはポスター用に回す。

---

# 生成潜在表現の予測可能性に基づく超低ビットレート画像圧縮の実符号量削減

著者名・所属・メールアドレスは投稿前に差し替える。

## 概要

超低ビットレート生成型画像圧縮では，復元側が予測できる潜在情報まで送ると符号量が浪費される．本稿ではGLCの生成潜在表現に復号側で再計算可能な残差精度制御を加え，実算術符号化bitstream上で知覚品質を保ったbpp削減を示す．

## 1. はじめに

Learned image compression（LIC）は，画像を潜在表現へ変換し，その潜在表現を確率モデルで符号化することで高い圧縮性能を実現してきた [1,2,3]．特にhyperpriorやautoregressive priorは，潜在変数の分布を精密に推定し，算術符号化に必要なbit数を削減する上で重要である [1,2]．

一方，超低ビットレート領域では，画像の全ての構造やテクスチャを忠実に送ることはできない．この領域では，画素誤差を一様に小さくするだけでは復元画像がぼけやすく，人間の知覚に自然な画像から離れる場合がある．そのため，何をbitstreamで送信し，何を復元側の生成能力に委ねるかが重要になる．

この観点から，HiFiCやGLCに代表される生成型画像圧縮が注目されている [5,6]．GLCは，VQGAN/VQ-VAEにより得られる生成潜在空間上でtransform codingを行い，超低ビットレート自然画像圧縮で高い知覚品質を示した [6,7,12]．しかし，GLCの生成潜在表現の中にも，復号側のpriorや生成decoderから予測しやすい成分と，予測しにくく知覚品質に効く残差成分が混在していると考えられる．前者まで高精度に送ることは，超低ビットレートでは非効率である．

本稿では，公開済みGLC image modelをベースに，復号側で再計算可能な残差precision gateを導入するGP-ResLCを提案する．提案手法はencoder側だけで得られるimportance mapを送らず，既に送信済みのhyper潜在とquality indexから同じgateを復号側で再計算する．これにより，GLCの生成潜在空間を保ったまま，算術符号化される主潜在の符号長を削減する．貢献は，(1) 追加side mapなしの残差precision制御，(2) serialized bitstreamに基づく実bpp評価，(3) CLIC2020 test，DIV2K validation，KodakでのDISTS/FID中心のrate-quality改善である．

## 2. 関連研究と位置づけ

LICでは，入力画像 $x$ をanalysis transformで潜在表現 $y$ へ写像し，量子化後の $\hat{y}$ をentropy modelにより符号化する．復号側では $\hat{y}$ からsynthesis transformにより復元画像 $\hat{x}$ を得る．学習では一般に，符号量を表すrate項と復元誤差を表すdistortion項を組み合わせて最適化する．Balléらはscale hyperpriorを導入し，主潜在の分布を別のhyper潜在から推定した [1]．Minnenらはhyperpriorとautoregressive context modelを組み合わせ，既に復号済みの近傍潜在を利用して分布推定を高精度化した [2]．ChengらはGaussian mixture likelihoodとattention moduleにより，より柔軟なentropy modelを示した [3]．これらは潜在分布を高精度にモデル化することでbitrateを削減する流れである．

BlauとMichaeliは，歪みと知覚品質の間にtrade-offがあることを示した [4]．このrate-distortion-perceptionの観点では，低ビットレート圧縮において，元画像との画素単位の近さだけでなく，復元画像が自然画像としてどれだけもっともらしいかも重要になる．HiFiCはGAN lossを用いた生成型画像圧縮により，低ビットレートで自然な復元を目指した [5]．これは，全ての画像詳細をbitstreamで送るのではなく，復元側の生成能力に一部を委ねる設計と見ることができる．

GLCは，VQGAN/VQ-VAEにより学習した生成潜在空間を符号化対象とし，categorical hyper moduleやcode prediction supervisionを用いて，超低ビットレートで高い知覚品質を実現する [6,7,12]．GLCの強みは，画素空間の細部ではなく，人間知覚に整合しやすい生成潜在表現を符号化する点にある．本研究はGLCを置き換えるのではなく，強いpretrained generative codecとして利用し，その主潜在に残る符号長配分の非効率を狙う．重要なのは，制御信号を復号側で再計算可能な情報に制限し，追加のside mapを送らない点である．これにより，単なるencoder側importance mapではなく，実codecとして整合する残差precision制御を行う．

ここに図1（`glc_gp_reslc_unified_overview.png`）を挿入する。

## 3. 提案手法

### 3.1 GLCにおける主潜在符号化

入力画像を $x$，GLCの主潜在を $y$，hyper潜在を $z$ とする．GLCは量子化された $\hat{z}$ からhyper decoderを通して，$y$ を符号化するためのprior parametersを推定する．また，主潜在 $y$ はfour-part priorにより複数の空間maskに分けて順次符号化され，各stageでは既に復号済みの $\hat{y}$ をcontextとして利用する．

通常のGLCでは，あるstageにおけるprior平均を $\mu_{\mathrm{GLC}}$，scaleを $\sigma_{\mathrm{GLC}}$，量子化stepを $Q_{\mathrm{GLC}}$ とし，主潜在をこのpriorの周りの残差として符号化する．この仕組みにより，hyperpriorとcontext priorが説明できる成分は短い符号で表現される．しかし，priorから予測できるかどうかと，知覚品質に対して高精度に送るべきかどうかは必ずしも一致しない．超低ビットレートでは，復号側の生成priorが自然に補える成分よりも，構造や知覚品質に影響する予測困難な残差にbitを集中すべきである．

### 3.2 復号側で再計算可能なprecision gate

GP-ResLCは，GLCのencoder，decoder，hyperprior，four-part context priorを保持したまま，主潜在 $y$ の符号化直前にprecision gate $\rho_\theta$ を導入する．gateは，bitstreamから復号可能な $\hat{z}$ とquality index $q$ から計算される．したがって，gate map自体は送信しない．

$$
\rho_\theta = f_\theta(\hat{z}, q), \quad \rho_\theta \geq 1
$$

このgateを用いて，符号化時の実効量子化stepを次のように変調する．

$$
Q_{\mathrm{GP}} = Q_{\mathrm{GLC}} \cdot \rho_\theta
$$

$\rho_\theta$ が大きい位置では残差を粗く量子化するため，算術符号化されるsymbolのentropyが下がりやすい．一方，$\rho_\theta$ が1に近い位置ではGLCに近い精度で残差を送る．つまり，precision gateは画像全体を一様に低品質化するrate knobではない．復号側生成priorに任せやすい成分でbitを節約し，必要な残差ではGLCの符号化精度を保つための局所的な符号量配分機構である．


### 3.3 設計制約

本稿の主張では，符号量削減が実bitstream上で成立することが重要である．そのため，GP-ResLCではencoder側でしか計算できない画像依存のimportance mapを送らない．制御信号は，$\hat{z}$，quality index，およびGLCの復号順序で利用可能なcontextから再計算できるものに限定する．この制約により，復号側はpayloadだけからGLCと同じ順序で $\hat{z}$ と $\hat{y}$ を復元し，各stageで同じgateを再計算できる．実装では，公開済みGLC image model [6] を凍結し，補助的なgate moduleを学習する．学習はrate項とDISTS/LPIPSを含む知覚品質維持項に基づく [8,9]．

## 4. 実験

### 4.1 評価設定

ベースラインは，公開済みGLC image modelを同一実装でreal codec化したものとする．提案手法は同じpretrained GLCにGP-ResLCのgate moduleを追加したモデルである．評価は，CLIC2020 test，DIV2K validation，Kodakで行う．CLIC2020 testはprofessional 250枚とmobile 178枚の合計428枚，DIV2Kはvalidation 100枚，Kodakは24枚である．いずれもoriginal resolutionのまま圧縮・復号する．

bppはlikelihood推定値ではなく，serialized payloadのbyte数から測定する．payloadにはheader，固定長符号化された $\hat{z}$ index，およびfour-part prior順序で算術符号化された $y$ streamを含める．bppは $8|\mathrm{payload}|/(HW)$ で計算し，復元画像はpayloadのみから復号する．real decoderの出力は従来forward復元と最大絶対誤差0で一致することを確認した．

評価指標は，DISTS，LPIPS，PSNR，MS-SSIM，FID，KIDである [8,9,10,11]．GLC/HiFiC型の評価に合わせ [5,6]，CLIC2020 testとDIV2Kでは256×256 patchを通常分割に加えて128 pixel shiftでも抽出し，FID/KIDを計算する．このプロトコルでのpatch数は，CLIC2020 testで28,650，DIV2Kで6,573である．Kodakは画像数が少ないため，DISTS，LPIPS，PSNR，MS-SSIMを主に見る．

### 表1: GLC real codecに対するGP-ResLCのBD-rate

負の値は，同一品質においてGP-ResLCが少ない実符号量で到達することを示す。BD-rateは各データセットのrate-quality curveを補間して算出した。

| Dataset | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | -10.28 | +0.19 | -0.98 | +0.38 | -7.30 | -7.10 |
| DIV2K val. | -10.79 | -0.54 | -1.49 | -0.17 | -5.61 | -6.50 |
| Kodak | -4.47 | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |

ここに図2（`result_curves_clic_div2k_perceptual_2x4.png`）を挿入する。

### 4.2 結果

表1に示すように，CLIC2020 testではDISTSで-10.28%，FIDで-7.30%のBD-rate改善が得られた．DIV2KでもDISTSで-10.79%，FIDで-5.61%となり，同様の傾向を示した．Kodakでは画像数が少ないためFID/KIDは参考扱いだが，DISTSで-4.47%の改善を示した．

同一DISTS品質におけるbpp削減率を補間により見ると，CLIC2020 testで平均-10.26%，DIV2Kで-10.27%，Kodakで-5.45%であった．したがって，この改善はquality indexの一点比較ではなく，曲線上の同一知覚品質に対する実符号量削減である．LPIPSはCLIC2020 testでほぼ中立，DIV2K/Kodakでは小幅改善に留まった．この結果は，本手法の主な効果がDISTS/FIDに現れるrate-perception寄りの改善であることを示している．

公式GLC論文図から抽出した曲線との補助比較でも，CLIC2020 testでDISTS/FID BD-rateが-9.07%/-6.10%，DIV2Kで-9.62%/-4.23%となった．ただし，これは図読み取りを含むcross-source比較であるため，本稿の主張は表1の同一実装・同一プロトコル比較に置く．

### 表2: CLIC2020 testにおけるserialized bppの内訳

| Method | q | total bpp | y bpp | z bpp | header bpp |
|---|---:|---:|---:|---:|---:|
| GLC | 0 | 0.02134 | 0.01757 | 0.00352 | 0.00025 |
| GP-ResLC | 0 | 0.01892 | 0.01515 | 0.00352 | 0.00025 |
| GLC | 3 | 0.03369 | 0.02992 | 0.00352 | 0.00025 |
| GP-ResLC | 3 | 0.03102 | 0.02726 | 0.00352 | 0.00025 |

表2に示すように，$z$ streamとheaderはGLCとGP-ResLCで同一であり，削減はarithmetic-coded $y$ streamから生じている．これは，提案手法がpayload外の調整ではなく，実際に送信される主潜在残差の符号長を削減していることを示す．

### 4.3 可視化と機構分析

ここに図3（`clic_q3_rho_overlay_top4.png`）を挿入する。

図3に，CLIC2020 testの復元例とprecision gate $\rho$ の可視化を示す．$\rho$ が高い領域は，GP-ResLCが残差precisionを粗くし，GLCの生成priorに復元を委ねる領域に対応する．一方，構造境界や復元が難しい領域では相対的に低い $\rho$ が割り当てられ，GLCに近い精度で残差が送られる．実際に，q3におけるgate解析では，高い $\rho$ が割り当てられた領域ほどGLC baselineの局所誤差や画像勾配が低い傾向を確認した．これは，予測可能または知覚的に低感度な領域でbitを節約し，予測困難な残差を相対的に保護するという設計と整合する．

## 5. 考察とまとめ

本稿の結果は，超低ビットレート生成型圧縮において，生成潜在表現の予測可能性に基づく符号量配分が有効であることを示している．特に，bpp削減が推定likelihoodではなくserialized payload上で観測され，削減が $z$ やheaderではなく $y$ streamから生じている点は，本手法の主張にとって重要である．

一方で，現時点のGP-ResLCはpretrained GLC上のoverlayであり，VQ-VAE，transform coding，entropy modelを最初から一貫して学習した完全なscratch codecではない．また，PSNR/MS-SSIMやLPIPSの改善は一貫して大きいわけではない．本稿の貢献は，GLCの強い生成潜在空間を利用し，復号側で再計算可能な残差precision制御により，実codec上でDISTS/FIDを中心とする知覚品質を保ったbpp削減が可能であることを示した点にある．今後は，生成潜在空間そのものをGP-ResLCの目的に合わせて学習し，予測可能成分と予測困難残差の分離をより明示的に行う．

## 参考文献

[1] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” Proc. ICLR, 2018.

[2] D. Minnen, J. Ballé, and G. D. Toderici, “Joint autoregressive and hierarchical priors for learned image compression,” Proc. NeurIPS, 2018.

[3] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized Gaussian mixture likelihoods and attention modules,” Proc. CVPR, pp. 7939--7948, 2020.

[4] Y. Blau and T. Michaeli, “Rethinking lossy compression: The rate-distortion-perception tradeoff,” Proc. ICML, 2019.

[5] F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson, “High-fidelity generative image compression,” Proc. NeurIPS, 2020.

[6] Z. Jia, J. Li, B. Li, H. Li, and Y. Lu, “Generative latent coding for ultra-low bitrate image compression,” Proc. CVPR, pp. 26088--26098, 2024.

[7] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” Proc. CVPR, 2021.

[8] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” Proc. CVPR, 2018.

[9] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” IEEE Trans. Pattern Analysis and Machine Intelligence, 2022.

[10] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” Proc. NeurIPS, 2017.

[11] M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying MMD GANs,” Proc. ICLR, 2018.

[12] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” Proc. NeurIPS, 2017.
