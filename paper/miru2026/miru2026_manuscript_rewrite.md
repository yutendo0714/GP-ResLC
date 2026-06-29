# MIRU2026 改稿版原稿

作成日: 2026-06-22  
用途: MIRU2026 Word/LaTeX 転記用の本文案。既存の `miru2026_word_source.md` よりも、投稿論文としての流れを優先して再構成した。

## 投稿規約メモ

- 公式MIRU2026投稿ページ確認済み: https://miru-committee.github.io/miru2026/author/submission/
- 日本語または英語。
- 本文は参考文献を除いて4ページ以下。
- 日本語概要は200文字以下。
- 著者名・所属・メールアドレスを原稿に明記。
- 2026年版テンプレートを使用。
- 本文中では「実codec評価」を独立節にせず、評価設定の前提として記述する。

## 推奨図表配置

4ページに収める優先順位は次の通り。

1. 図1: GLC概要図  
   `paper/miru2026/figures/glc_model_overview_simple.png`
2. 図2: GLCに対するGP-ResLCの差分図  
   `paper/miru2026/figures/gp_reslc_difference_from_glc.png`
3. 表1: GLC real codecに対するBD-rate
4. 図3: CLIC2020/DIV2KのDISTS rate-quality curve  
   `paper/miru2026/figures/clic2020_dists_curve.png`  
   `paper/miru2026/figures/div2k_dists_curve.png`
5. 表2: CLIC2020 testのserialized bpp内訳

余裕があれば，図4として `paper/miru2026/figures/kodak_q3_rho_overlay_top4.png` を入れる。入りきらない場合は図4を落とし，本文で「高rho領域は低誤差・低勾配領域と対応する傾向を確認した」と1文で済ませる。

---

# 生成潜在表現の予測可能性に基づく超低ビットレート画像圧縮の実符号量削減

著者名・所属・メールアドレスは投稿前に差し替える。

## 概要

超低ビットレート生成型画像圧縮では，復元側が予測できる潜在情報まで送ると符号量が浪費される．本稿ではGLCの生成潜在表現に，復号側で再計算可能な残差精度制御を重ね，実算術符号化bitstream上でDISTS/FID品質を保ちながらbppを削減するGP-ResLCを示す．

## 1. はじめに

Learned image compression（LIC）は，画像をニューラルネットワークで潜在表現へ変換し，その潜在表現を確率モデルに基づいて符号化することで，高い圧縮性能を実現してきた [1,2,3]．特にhyperpriorやautoregressive priorは，潜在変数の分布を精密に推定し，算術符号化に必要なbit数を削減する上で中心的な役割を担う [1,2]．この流れにより，PSNRやMS-SSIMを中心とするrate-distortion性能は大きく改善されてきた．

一方，超低ビットレート領域では，入力画像の全ての構造やテクスチャを忠実に送ることはできない．この領域で画素単位の歪みを一様に小さくしようとすると，復元画像はぼけやすく，人間の知覚にとって自然な画像から離れることがある．そのため，低ビットレート画像圧縮では，何をbitstreamで明示的に送信し，何を復元側の生成能力に委ねるかが重要になる．

この観点から，HiFiCやGLCに代表される生成型画像圧縮が注目されている [5,6]．HiFiCはadversarial lossを用いて低ビットレートでの自然なテクスチャ復元を目指した [5]．GLCはさらに，VQGAN/VQ-VAEにより得られる生成潜在空間上でtransform codingを行い，超低ビットレート自然画像圧縮で高い知覚品質を示した [6,7,12]．GLCの重要な点は，画素空間の詳細を全て送るのではなく，生成decoderが画像として復元しやすい潜在空間を符号化対象にしていることである．

本稿の出発点は，このGLCの生成潜在表現の中にも，まだ符号量配分の余地が残っているという観察である．復号側のhyperprior，context prior，生成decoderから予測しやすい潜在成分は，高精度に送らなくても知覚品質への影響が小さい可能性がある．反対に，予測しにくく，画像構造や知覚品質に強く影響する残差成分には，限られたbitを優先して使うべきである．

そこで本稿では，公開済みGLC image modelをベースに，復号側で再計算可能な残差精度制御を導入するGP-ResLCを提案する．提案手法は，encoder側でしか得られない重要度mapをside informationとして送るのではなく，既に送信済みのhyper潜在とquality indexから復号側でも同一に計算できるgateを用いる．これにより，GLCの生成潜在空間を保ったまま，算術符号化される主潜在の符号長を削減する．

本稿の貢献は次の通りである．第一に，GLCの生成潜在表現上で，復号側と整合する残差precision gateを導入し，追加のside mapなしに主潜在の符号長配分を制御する．第二に，推定likelihoodではなく，実際にserialized bitstreamを生成・復号する評価により，bpp削減がarithmetic-coded stream上で生じることを確認する．第三に，CLIC2020 test，DIV2K validation，Kodakにおいて，GLCを同一プロトコルで再評価したpaired baselineに対し，DISTS/FIDを中心とするrate-quality curveを改善する．

## 2. 関連研究

### 2.1 Learned image compressionとentropy modeling

LICの基本形では，入力画像 $x$ をanalysis transformにより潜在表現 $y$ へ写像し，量子化後の $\hat{y}$ をentropy modelにより符号化する．復号側ではsynthesis transformにより復元画像 $\hat{x}$ を得る．この枠組みでは，学習時に符号量を表すrate項と，復元誤差を表すdistortion項を組み合わせて最適化する．

Balléらはscale hyperpriorを導入し，主潜在 $y$ の分布を別のhyper潜在 $z$ から推定することで，潜在表現の符号化効率を高めた [1]．Minnenらはhyperpriorとautoregressive context modelを組み合わせ，既に復号済みの近傍潜在を利用して，より精密な確率モデルを構成した [2]．ChengらはGaussian mixture likelihoodとattention moduleを導入し，より柔軟なentropy modelによる高性能な画像圧縮を示した [3]．これらの研究は，主に潜在分布の近似精度を高めることでbitrateを削減する流れに位置づけられる．

本稿も潜在表現の符号量削減を扱うが，単に確率モデルを複雑化することを目的としない．対象は，GLCが学習した生成潜在表現であり，その中で復元側が予測できる成分と予測しにくい残差成分の符号量配分を制御する点に違いがある．

### 2.2 Rate-distortion-perceptionと生成型画像圧縮

BlauとMichaeliは，画像復元において歪みと知覚品質の間に本質的なtrade-offが存在することを示した [4]．このrate-distortion-perceptionの観点では，特に低ビットレート圧縮において，元画像との画素単位の近さだけでなく，復元画像が自然画像としてどれだけもっともらしいかが重要になる．

HiFiCは，learned compressionにGAN lossを組み合わせ，低ビットレートで自然な復元画像を得る代表的な生成型画像圧縮手法である [5]．この方向では，復元側の生成能力が一部の高周波成分やテクスチャを補うため，画素忠実度の低下を許容しながら知覚品質を高めることができる．VQ modelを用いた知覚品質重視の圧縮も，同様に「送信情報」と「生成復元」の役割分担を設計する問題として捉えられる [7,12]．

ただし，生成型圧縮では，生成モデルが復元できる情報まで符号化してしまうと，低ビットレート領域ではbitの使い方として非効率になる．本稿では，この非効率をGLCの生成潜在空間上で直接扱う．すなわち，生成側に任せられる成分の送信精度を下げ，予測困難な残差にbitを集中することを狙う．

### 2.3 GLCと本研究の位置づけ

GLCは，VQGAN/VQ-VAEにより人間知覚に整合した生成潜在空間を学習し，その潜在表現をtransform codingにより圧縮する [6,7,12]．自然画像codecでは，入力画像は生成潜在表現へ写像され，さらに主潜在 $y$ とhyper潜在 $z$ に基づく符号化が行われる．GLCはcategorical hyper moduleやcode prediction supervisionを用いることで，生成潜在表現の意味的一貫性を保ちながら，超低ビットレートで高い知覚品質を実現している [6]．

ここに図1を挿入する。

本研究は，GLCを置き換える新しい生成decoderを提案するものではない．むしろ，GLCを強いpretrained generative codecとして用い，その内部の主潜在に残る符号長配分の非効率を狙う．既存の重要度map型の制御では，map自体を送る必要がある場合，超低ビットレートではside informationが無視できない．GP-ResLCでは，制御信号を復号側で再計算可能な入力に制限し，追加のside mapを送らずにarithmetic-coded $y$ streamの削減を目指す．この点が，本稿の新規性と実codec上の主張を支える設計である．

## 3. 提案手法

### 3.1 GLCにおける主潜在符号化

入力画像を $x$，GLCの主潜在を $y$，hyper潜在を $z$ とする．GLCはまず $z$ を量子化して $\hat{z}$ を得る．復号側では $\hat{z}$ からhyper decoderを通して，$y$ を符号化するためのprior parametersを推定する．また，GLCの主潜在はfour-part priorにより複数の空間maskに分けて順次符号化され，各stageでは既に復号済みの $\hat{y}$ をcontextとして利用する．

通常のGLCでは，あるstageにおけるprior平均を $\mu_{\mathrm{GLC}}$，scaleを $\sigma_{\mathrm{GLC}}$，量子化stepを $Q_{\mathrm{GLC}}$ とすると，主潜在はこのpriorの周りの残差として符号化される．この仕組みにより，hyperpriorとcontext priorが説明できる成分は短い符号で表現される．しかし，priorが予測できるかどうかと，知覚品質に対して高精度に送るべきかどうかは必ずしも一致しない．

### 3.2 復号側で再計算可能な残差precision gate

GP-ResLCは，GLCのencoder，decoder，hyperprior，four-part context priorを保持したまま，主潜在 $y$ の符号化直前にprecision gate $\rho_\theta$ を導入する．gateは，既にbitstreamから復号可能な $\hat{z}$ とquality index $q$ から計算される．したがって，gate map自体を送信する必要はない．

$$
\rho_\theta = f_\theta(\hat{z}, q), \quad \rho_\theta \geq 1
$$

このgateを用いて，符号化時の実効量子化stepを次のように変調する．

$$
Q_{\mathrm{GP}} = Q_{\mathrm{GLC}} \cdot \rho_\theta
$$

$\rho_\theta$ が大きい位置では，残差が粗く量子化されるため，算術符号化されるsymbolのentropyが下がりやすい．一方，$\rho_\theta$ が1に近い位置では，GLCに近い精度で残差を送る．つまり，precision gateは画像全体を一様に低品質化するのではなく，復号側生成priorに任せやすい成分でbitを節約し，必要な残差ではGLCの符号化精度を保つための機構である．

ここに図2を挿入する。

### 3.3 予測可能情報を送らないための設計制約

本稿の主張では，符号量削減が実際にbitstreamで成立することが重要である．そのため，GP-ResLCではencoder側でしか計算できない画像依存のimportance mapを送らない．制御信号は，$\hat{z}$，quality index，およびGLCの復号順序で利用可能なcontextから再計算できるものに限定する．

この制約により，復号側はpayloadだけからGLCと同じ順序で $\hat{z}$ と $\hat{y}$ を復元し，各stageで同じgateを再計算できる．実装では，GLCの公開済みimage model [6] を凍結し，補助的なgate moduleを学習する．学習はrate項とDISTS/LPIPSを含む知覚品質維持項に基づく [8,9]．PSNRやMS-SSIMは診断指標として報告するが，本稿の主目的はrate-distortion最適化ではなく，rate-perception寄りの実符号量削減である．

## 4. 実験

### 4.1 評価設定

ベースラインは，公開済みGLC image modelを同一実装でreal codec化したものとする．提案手法は同じpretrained GLCを用い，GP-ResLCのgate moduleを追加したモデルである．評価は，CLIC2020 test，DIV2K validation，Kodakで行う．CLIC2020 testはprofessional 250枚とmobile 178枚の合計428枚を用いる．DIV2Kはvalidation 100枚，Kodakは24枚である．いずれもoriginal resolutionのまま圧縮・復号する．

bppは学習時のlikelihood推定値ではなく，実際にserialized payloadを生成して測定する．payloadにはheader，固定長符号化された $\hat{z}$ index，およびfour-part prior順序で算術符号化された $y$ streamを含める．bppは $8|\mathrm{payload}|/(HW)$ で計算し，復元画像はpayloadのみから復号する．smoke checkでは，real decoderの出力が従来のforward復元と最大絶対誤差0で一致することを確認した．

評価指標は，DISTS，LPIPS，PSNR，MS-SSIM，FID，KIDである [8,9,10,11]．GLC/HiFiC型の評価に合わせ [5,6]，CLIC2020 testとDIV2Kでは256×256 patchを通常分割に加えて128 pixel shiftでも抽出し，FID/KIDを計算する．このプロトコルでのpatch数は，CLIC2020 testで28,650，DIV2Kで6,573である．Kodakは画像数が少なくFID/KIDが不安定なため，DISTS，LPIPS，PSNR，MS-SSIMを主に見る．

### 表1: GLC real codecに対するGP-ResLCのBD-rate

負の値は，同一品質においてGP-ResLCが少ない実符号量で到達することを示す。BD-rateはquality indexの一点比較ではなく，各データセットのrate-quality curveを補間して算出した。

| Dataset | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | -10.28 | +0.19 | -0.98 | +0.38 | -7.30 | -7.10 |
| DIV2K val. | -10.79 | -0.54 | -1.49 | -0.17 | -5.61 | -6.50 |
| Kodak | -4.47 | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |

ここに図3を挿入する。

### 4.2 主結果

表1に，GLC real codecに対するGP-ResLCのBD-rateを示す．CLIC2020 testではDISTSで-10.28%，FIDで-7.30%の改善が得られた．DIV2KでもDISTSで-10.79%，FIDで-5.61%となり，同様の傾向を示した．Kodakでは画像数が少ないためFID/KIDは参考扱いだが，DISTSで-4.47%の改善を示した．

同一DISTS品質におけるbpp削減率を補間により見ると，CLIC2020 testで平均-10.26%，DIV2Kで-10.27%，Kodakで-5.45%であった．したがって，この改善は単にGLCのquality indexをずらした一点比較ではなく，曲線上の同一知覚品質に対する実符号量削減である．LPIPSはCLIC2020 testでほぼ中立，DIV2K/Kodakでは小幅改善に留まった．この結果は，本手法の主な効果がDISTS/FIDに現れるrate-perception寄りの改善であることを示している．

公式GLC論文図から抽出した曲線との補助比較でも，CLIC2020 testでDISTS/FID BD-rateが-9.07%/-6.10%，DIV2Kで-9.62%/-4.23%となり，同一実装内のpaired比較と同じ傾向を確認した．ただし，この比較は図読み取りを含むcross-source比較であるため，本稿の主張は表1の同一実装・同一プロトコル比較に置く．

### 表2: CLIC2020 testにおけるserialized bppの内訳

| Method | q | total bpp | y bpp | z bpp | header bpp |
|---|---:|---:|---:|---:|---:|
| GLC | 0 | 0.02134 | 0.01757 | 0.00352 | 0.00025 |
| GP-ResLC | 0 | 0.01892 | 0.01515 | 0.00352 | 0.00025 |
| GLC | 3 | 0.03369 | 0.02992 | 0.00352 | 0.00025 |
| GP-ResLC | 3 | 0.03102 | 0.02726 | 0.00352 | 0.00025 |

### 4.3 符号量削減の要因

表2にCLIC2020 testにおけるserialized bppの内訳を示す．$z$ streamとheaderはGLCとGP-ResLCで同一であり，削減はarithmetic-coded $y$ streamから生じている．これは，提案手法がpayload外の見かけ上の調整ではなく，実際に送信される主潜在残差の符号長を削減していることを示す．

また，gateの解析では，高い $\rho$ が割り当てられた領域ほど，GLC baselineの局所誤差や勾配が低い傾向が確認された．これは，GP-ResLCが予測可能または知覚的に低感度な領域で残差precisionを下げ，難しい領域を相対的に保護するという設計と整合する．

## 5. 考察

本稿の結果は，超低ビットレート生成型画像圧縮において，生成潜在表現の予測可能性に基づく符号量配分が有効であることを示している．特に，bpp削減が推定likelihoodではなくserialized payload上で観測され，その削減が $z$ やheaderではなく $y$ streamから生じている点は，本手法の主張にとって重要である．

一方で，現時点のGP-ResLCはpretrained GLC上のoverlayであり，VQ-VAE，transform coding，entropy modelを最初から一貫して学習した完全なscratch codecではない．これはMIRU投稿時点では正直に制約として述べるべきである．本稿の貢献は，GLCの強い生成潜在空間を利用し，復号側で再計算可能な残差precision制御により，実codec上で知覚品質を保ったbpp削減が可能であることを示した点にある．

また，PSNR/MS-SSIMやLPIPSの改善は一貫して大きいわけではない．これは，本手法が画素忠実度を主目的にしたrate-distortion改善ではなく，DISTS/FIDを中心とするrate-perception改善を狙っているためである．今後は，生成潜在空間そのものをGP-ResLCの目的に合わせて学習するscratch版により，予測可能成分と予測困難残差の分離をより明示的に行う．

## 6. まとめ

本稿では，GLCの生成潜在表現上で復号側と整合する残差precision gateを導入し，超低ビットレート画像圧縮において実符号量を削減するGP-ResLCを示した．CLIC2020 test，DIV2K validation，Kodakのfull-resolution real codec評価により，DISTS/FIDを中心にGLCより少ないbppで同等以上の知覚品質に到達できることを確認した．本結果は，生成器が自力で復元できる情報を送らず，予測困難な残差にbitを集中するという設計方針の有効性を支持する．

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
