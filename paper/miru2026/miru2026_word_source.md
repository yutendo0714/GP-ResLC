# MIRU2026 Word転記用原稿

このMarkdownは、Wordに直接移すための本文・表・添付図リストです。数式はMarkdownプレビューで崩れないように、インライン数式は `$...$`、別行数式は `$$...$$` で記述しています。Wordへ貼る場合は、数式ブロック内のLaTeXをWordの数式入力欄に貼り付けると整形しやすいです。

## 添付する図

### 必須図

**図1: GLCのモデル概要図（自作簡略図）**

- `paper/miru2026/figures/glc_model_overview_simple.png`

キャプション:

> 図1. GLCの自然画像圧縮モデルの概要。GLCはVQGAN/VQ-VAEにより得られる生成潜在空間でtransform codingを行い、hyperpriorとfour-part context priorにより主潜在 $y$ を算術符号化する。本研究はこのGLCをベースラインとして、主潜在 $y$ の符号長配分を再設計する。

**図2: GLCに対するGP-ResLCの差分図**

- `paper/miru2026/figures/gp_reslc_difference_from_glc.png`

キャプション:

> 図2. GLCに対するGP-ResLCの差分。通常のGLCはGLC priorの平均・分散・量子化stepに基づき $y$ を符号化する。GP-ResLCは復号側でも再計算可能なpredictable mean補正 $\Delta\mu(\hat{z},q)$ とprecision gate $\rho(\hat{z},q)$ を追加し、予測可能な成分の送信を抑えてarithmetic-coded $y$ streamを削減する。

**図3: DISTSによるrate-quality curve**

Wordでは次の2枚を横並びにして、1つの図として配置するのがおすすめです。

- `paper/miru2026/figures/clic2020_dists_curve.png`
- `paper/miru2026/figures/div2k_dists_curve.png`

キャプション:

> 図3. DISTSによるrate-quality curve。左: CLIC2020 test、右: DIV2K validation。bppは推定likelihoodではなくserialized bitstreamから測定した。

### 余裕があれば入れる図

**図4: 復元例とprecision gateの可視化**

- `paper/miru2026/figures/kodak_q3_rho_overlay_top4.png`

キャプション:

> 図4. Kodak q3における復元例とprecision gateの可視化。高い $\rho$ は粗い残差precision、すなわち低い $y$ stream bitを意味する。

### 補足候補

FID曲線も入れるなら次の2枚です。ただし4ページ制限を考えると、本文ではDISTS曲線を優先し、FIDは表で示す方が無難です。

- `paper/miru2026/figures/clic2020_fid_curve.png`
- `paper/miru2026/figures/div2k_fid_curve.png`

次の図は参考用です。本文では図1・図2の自作図を優先します。

- `paper/miru2026/figures/gp_reslc_model_overview.png`
- `paper/miru2026/figures/glc_official_pipeline.png`

---

# 生成潜在表現の予測可能性に基づく超低ビットレート画像圧縮の実符号量削減

著者名・所属・メールアドレスはMIRU投稿前に差し替える。

## 概要

超低ビットレート生成型画像圧縮では、全ての潜在情報を同じ精度で送ることは非効率である。本稿では、GLCの生成潜在空間において、復号側priorから予測可能な成分の送信を抑え、予測困難な残差に符号量を集中させるGP-ResLCを提案する。実算術符号化に基づく評価で、CLIC2020 testとDIV2KのDISTS/FID BD-rateを改善した。

## 1. はじめに

画像圧縮では、限られた符号量で復元画像の品質を保つ必要がある。近年のlearned image compression（LIC）は、画像をニューラルネットワークで潜在表現へ変換し、その潜在表現を確率モデルに基づいて符号化することで高いrate-distortion性能を示してきた[1,2,3]。特にhyperpriorやautoregressive priorは、潜在表現の分布を精密に推定し、算術符号化に必要なbit数を削減する上で重要である。

一方、超低ビットレート領域では、画像中の構造やテクスチャを全て忠実に送ることはできない。この領域では、画素単位の歪みを一様に小さくするだけでは、ぼけた復元や不自然な構造が生じやすい。したがって、どの情報を明示的に送信し、どの情報を復元側の生成能力に委ねるかが重要になる。

この観点から、HiFiC[5]やGLC[6]などの生成型画像圧縮が注目されている。特にGLCは、VQGAN/VQ-VAEにより得られる生成潜在空間でtransform codingを行い、超低ビットレートで高い知覚品質を達成している。GLCは、画素空間ではなく生成潜在表現を符号化対象とすることで、復元側decoderの生成能力を積極的に利用する圧縮方式である。

本稿では、このGLCの設計をさらに進める。GLCの生成潜在空間においても、全ての潜在成分を同じ精度で送る必要はない。復号側の生成decoderやentropy priorから予測しやすい成分は、少ないbitでも復元可能である可能性が高い。一方、予測しにくく、知覚品質に影響する残差成分には、限られたbitを優先的に使うべきである。

本稿では、公開済みGLC image modelをベースに、復号側でも再計算可能なprior mean補正とprecision gateを導入するGP-ResLCを提案する。提案手法はencoder側でしか得られないside mapを送らず、GLCの主潜在 $y$ の符号長配分を変更する。実験では、推定likelihoodではなく実際にserialized bitstreamを生成し、DISTS/FIDを中心とするperceptual qualityでGLCより少ないbppを達成することを示す。

本稿の貢献は次の通りである。第一に、GLCの生成潜在表現上で、復号側と整合するprior mean補正およびquantization precision gateを導入し、追加のside informationなしに潜在残差の符号量を削減する。第二に、削減されるbitが $z$ streamやheaderではなく、arithmetic-coded $y$ streamに由来することを示す。第三に、CLIC2020 test、DIV2K validation、Kodakにおいて、同一実装・同一プロトコルで再評価したGLC baselineに対し、DISTS/FIDを中心とするrate-quality curveを改善する。

## 2. 関連研究

### 2.1 Learned image compressionとentropy modeling

LICでは、入力画像 $x$ をencoderにより潜在表現 $y$ へ変換し、量子化された潜在表現 $\hat{y}$ をentropy modelに基づいて符号化する。復号側では $\hat{y}$ からdecoderにより復元画像 $\hat{x}$ を得る。学習では通常、符号量を表すrate項と復元誤差を表すdistortion項を組み合わせて最適化する。

Balléら[1]はscale hyperpriorを導入し、主潜在 $y$ の分布を別のhyper潜在 $z$ から推定することで、潜在表現の符号化効率を高めた。Minnenら[2]はhyperpriorとautoregressive context modelを組み合わせ、既に復号済みの近傍潜在を利用して、より精密に $y$ の確率分布を推定した。Chengら[3]はGaussian mixture likelihoodとattention moduleを導入し、より柔軟なentropy modelを実現した。これらの研究は、潜在表現の確率モデルを高精度化することでbitrateを削減する流れに位置づけられる。

ただし、これらの多くはMSE、MS-SSIM、PSNRなどのdistortion指標を主要な最適化対象とする。中・高ビットレートではこの方針が有効であるが、超低ビットレートでは、画素忠実度を保とうとするほど知覚的には不自然な復元になりやすい。また、学習中のestimated bppはentropy modelのlikelihoodから計算される近似値であり、実際に算術符号化したbitstreamの長さと完全に一致するとは限らない。そのため、本稿では評価時に実bitstreamを生成し、payload bytesに基づいてbppを測定する。

### 2.2 Rate-distortion-perceptionと生成型画像圧縮

BlauとMichaeli[4]は、歪みと知覚品質の間に本質的なtrade-offがあることを示し、rate-distortion-perceptionの観点から画像復元問題を整理した。この視点では、低ビットレート圧縮において、元画像との画素単位の近さだけでなく、復元画像が自然画像分布上でどれだけもっともらしいかも重要になる。

HiFiC[5]はこの方向を画像圧縮に導入した代表的手法である。HiFiCはlearned compressionにGAN lossを組み合わせ、低ビットレートでも自然なテクスチャや構造を復元することを目指す。これは、圧縮側が全ての詳細を送るのではなく、復元側の生成能力に一部の詳細を委ねるという考え方を明確に示している。

GLC[6]は、生成型圧縮をさらに潜在空間符号化として整理した手法である。GLCでは、VQGAN/VQ-VAEによって学習された生成潜在空間上でtransform codingを行う。自然画像codecでは、入力画像を生成潜在表現へ写像し、その潜在表現を低ビットレートで符号化し、復号側の生成decoderで画像を再構成する。さらに、categorical hyper moduleとcode prediction supervisionにより、生成潜在表現の意味的一貫性を保ちながらentropy codingを行う。

ここに図1を挿入する。

本研究は、GLCを単なる比較対象として扱うのではなく、強いpretrained generative codecとして利用する。GLCの利点は、生成潜在空間が既に人間知覚に近い表現を持つ点である。一方、本稿の問題意識は、そのGLC内部の $y$ streamにもまだ符号量配分の余地があるという点にある。すなわち、生成priorや復号済みcontextから予測可能な成分と、予測困難な残差成分を区別し、後者にbitを集中させる。

### 2.3 本研究の位置づけ

既存のentropy modeling研究は、潜在表現の確率分布をより正確に推定することで符号量を削減する。一方、生成型圧縮研究は、復元側の生成能力を利用して、低ビットレートでも自然な画像を得る。本研究はこの二つの流れの間に位置する。すなわち、GLCの生成潜在空間とentropy priorを前提に、復号側で再計算可能な情報を用いて、どの潜在残差をどの精度で送るかを制御する。

重要なのは、提案手法がencoder側だけで計算される重要度mapを送る方式ではない点である。もし重要度map自体を送る必要があると、超低ビットレートではそのside informationが無視できない。GP-ResLCでは、補正とgateを $\hat{z}$、quality index $q$、復号済みcontextから再計算できる形に制限する。これにより、符号量削減の主張を実bitstream上で検証できる。

## 3. 提案手法

### 3.1 GLCにおける符号化対象

入力画像を $x$、GLCのencoderによる主潜在表現を $y$、hyper潜在表現を $z$ とする。GLCはまず $z$ を量子化して $\hat{z}$ を得る。復号側では $\hat{z}$ からhyper decoderにより、主潜在 $y$ を符号化するためのprior parametersを推定する。これらのparametersには、量子化step、scale、meanが含まれる。

GLCの主潜在 $y$ は一度にまとめて符号化されるのではなく、空間的なmaskに基づくfour-part priorにより段階的に符号化される。各stageでは、既に復号済みの部分的な $\hat{y}$ をcontextとして利用し、次の部分のscaleとmeanを更新する。この仕組みにより、GLCはhyperpriorとautoregressive contextの両方を用いて $y$ streamのentropyを下げる。

GP-ResLCは、このGLCの基本構造を変えない。GLCのencoder、decoder、hyperprior、four-part context priorはそのまま用いる。変更するのは、$y$ を算術符号化する直前の「どの平均の周りで残差を見るか」と「どの量子化精度で残差を送るか」である。この設計により、GLCの強い生成潜在空間を維持しつつ、符号量配分だけを再設計する。

ここに図2を挿入する。

### 3.2 予測可能成分と予測困難な残差

本稿の中心仮説は、主潜在 $y$ の中には、復号側の生成priorからある程度予測できる成分と、予測しにくい成分が混在しているというものである。復号側で予測できる成分は、明示的に高精度で送らなくても復元品質への影響が小さい。一方、予測しにくい成分、特に構造や知覚品質に効く成分は、限られたbitを使って送る価値が高い。

この考え方は、画像全体を単純に低品質化することとは異なる。全ての潜在成分を一様に粗く量子化すると、重要な構造まで失われる。GP-ResLCでは、復号側で再計算可能な補助情報に基づいて、予測可能な成分の符号化精度を下げる。これにより、生成priorに任せられる部分ではbitを節約し、必要な残差にはGLCの符号化経路をそのまま利用する。

### 3.3 復号側で再計算可能なprior mean補正

GLCのpriorが予測する平均を $\mu_{\mathrm{GLC}}$、scaleを $\sigma_{\mathrm{GLC}}$ とする。通常のGLCでは、$y$ はこの $\mu_{\mathrm{GLC}}$ の周りの残差として符号化される。GP-ResLCでは、$\hat{z}$ とquality index $q$ から補正 $\Delta\mu_\theta$ を予測し、次の補正後平均を用いる。

$$
\mu_{\mathrm{GP}} =
\mu_{\mathrm{GLC}} + \Delta\mu_\theta(\hat{z}, q)
$$

これにより、符号化対象は次の残差に近づく。

$$
r = y_{\mathrm{scaled}} - \mu_{\mathrm{GP}}
$$

直感的には、$\Delta\mu_\theta$ は「GLCのbase priorだけでは説明しきれないが、$\hat{z}$ とquality条件から復号側でも予測できる成分」をmeanとして吸収する。meanに吸収できた成分は、残差として送る必要が小さくなる。ただし、この補正自体をbitstreamに含めると意味がないため、補正は必ず復号側でも同じ値を再計算できる入力だけから求める。

### 3.4 Precision gateによる残差精度制御

mean補正だけでは、全ての潜在位置で同じ符号化精度を使うことになる。そこでGP-ResLCでは、GLC priorのquantization stepを空間的に変調するprecision gate $\rho_\phi$ を導入する。$\rho_\phi \geq 1$ とし、符号化時の実効量子化stepを次のようにする。

$$
Q_{\mathrm{GP}} =
Q_{\mathrm{GLC}} \cdot \rho_\phi(\hat{z}, q)
$$

$\rho_\phi$ が大きい位置では量子化stepが大きくなり、残差は粗く符号化される。その結果、その位置のarithmetic-coded $y$ streamは短くなる。一方で、$\rho_\phi$ が1に近い位置ではGLCに近い精度を保つ。したがって、precision gateは「どこでbitを節約し、どこでGLCの精度を保つか」を制御する機構である。

このgateもmean補正と同様に、$\hat{z}$ と $q$ から復号側で再計算できる。したがって、gate mapそのものを送る必要はない。本稿では、知覚損失とrate項を組み合わせてこのgateを学習し、予測可能または知覚的に低感度な成分のprecisionを抑える。

## 4. 実験

### 4.1 設定

ベースラインは公開済みGLC image modelを同一実装でreal codec化したものとする。提案手法は同じpretrained GLCを用い、GP-ResLCの補助モジュールのみを追加学習した。評価データセットはCLIC2020 test、DIV2K validation、Kodakである。CLIC2020 testはprofessional 250枚とmobile 178枚の合計428枚を用いる。FID/KIDはGLC/HiFiC型の評価に合わせ、CLIC2020 testとDIV2Kでは256×256 patchを通常分割に加えて128 pixel shiftで抽出する。得られるpatch数はCLIC2020 testで28,650、DIV2Kで6,573である。Kodakは24枚と小さいため、DISTS、LPIPS、PSNR、MS-SSIMを主に見る。

bppは学習時のlikelihood推定値ではなく、実際に符号化されたbitstreamのbyte数から算出する。具体的には、量子化されたhyper潜在 $\hat{z}$、主潜在 $y$ の算術符号化stream、およびheaderを含むserialized payloadを生成し、$8|\mathrm{payload}|/(HW)$ をbppとする。GLCとGP-ResLCはいずれも同じreal codec経路で評価し、復元画像はbitstreamのみから復号する。

### 表1: GLC real codecに対するGP-ResLCのBD-rate

負の値は、同一品質において提案手法が少ない実符号量で到達することを示す。FID/KIDはCLIC2020 testとDIV2Kではshifted 256 patchで評価した。

| Dataset | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | -10.28 | +0.19 | -0.98 | +0.38 | -7.30 | -7.10 |
| DIV2K val. | -10.79 | -0.54 | -1.49 | -0.17 | -5.61 | -6.50 |
| Kodak | -4.47 | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |

### 表2: 公式GLC曲線に対する補助比較

公式論文図から抽出したGLC曲線に対するcross-source比較である。評価実装・図読み取りをまたぐため主張の主表にはしないが、paired real-codec比較と同じ傾向を確認するために示す。

| Dataset | DISTS BD-rate | FID BD-rate | 備考 |
|---|---:|---:|---|
| CLIC2020 test | -9.07 | -6.10 | official graph-extracted GLC vs GP-ResLC real codec |
| DIV2K val. | -9.62 | -4.23 | official graph-extracted GLC vs GP-ResLC real codec |

ここに図3を挿入する。

### 4.2 主結果

表1にreal codec bppによるBD-rateを示す。CLIC2020 testではDISTSで-10.28%、FIDで-7.30%のBD-rate改善が得られた。DIV2KでもDISTS -10.79%、FID -5.61%と同様の傾向を示した。Kodakでは画像数が少ないためFID/KIDは補助的に扱うが、DISTSで-4.47%の改善を示した。LPIPSはCLIC2020 testでほぼ中立、DIV2K/Kodakではわずかに改善した。したがって、本手法の主要な改善はDISTS/FIDに現れ、PSNR/MS-SSIMを主目的とするRD改善ではない。

同一DISTS品質でのbpp削減率を補間により見ると、CLIC2020 testで平均-10.26%、DIV2Kで-10.27%、Kodakで-5.45%であった。これは単にquality indexをずらしただけの一点比較ではなく、曲線上の同一知覚品質に対する実符号量削減を示している。

### 表3: CLIC2020 testにおけるserialized bppの内訳

$z$ streamとheaderはGLCとGP-ResLCで同一であり、削減はarithmetic-coded $y$ streamから生じる。

| Method | q | total bpp | y bpp |
|---|---:|---:|---:|
| GLC | 0 | 0.02134 | 0.01757 |
| GP-ResLC | 0 | 0.01892 | 0.01515 |
| GLC | 3 | 0.03369 | 0.02992 |
| GP-ResLC | 3 | 0.03102 | 0.02726 |

### 4.3 符号量削減の要因

表3にCLIC2020 testのbpp内訳を示す。$z$ streamは両手法で0.00352 bpp、headerは約0.00025 bppで同一である。したがって、総bppの削減は、提案したprior補正およびprecision gateによってarithmetic-coded $y$ streamが短くなったことに由来する。これは本稿の主張である「復号側生成priorから予測可能な成分の送信を抑え、残差に符号量を集中する」方向と整合する。

また、表2に示すように、公式論文図から抽出したGLC曲線とのcross-source比較でも、CLIC2020 testとDIV2Kでpaired real-codec比較と同じ傾向を示した。ただし、これは図読み取りを含む外部比較であるため、主張の中心は表1の同一実装・同一プロトコルでのpaired比較に置く。

ここに図4を挿入する。4ページに収まらない場合、図4は削除して図1、図2、図3を優先する。

## 5. 考察と限界

本稿の結果は、超低ビットレート生成型圧縮において、生成潜在priorの予測可能性を利用した符号量配分が有効であることを示している。特に、実bitstreamに基づく評価で $y$ streamのみが短くなっている点は、推定bppの見かけ上の改善ではなく、実codecとしての削減であることを示す。

一方で、現時点のGP-ResLCはpretrained GLC上のoverlayであり、VQ-VAEからtransform codingまでを一貫して再学習した完全なscratch codecではない。また、改善はDISTS/FIDに強く現れる一方、LPIPSやPSNR/MS-SSIMは中立から小幅改善に留まる。これは、本稿の目的がrate-distortion最適化ではなくrate-perception寄りの実符号量削減であることを反映している。今後は、生成潜在空間そのものをGP-ResLCの目的に合わせて学習し、予測可能成分と予測困難な残差成分の分離をより明示的に行う。

## 6. まとめ

本稿では、GLCの生成潜在表現上で復号側と整合する残差prior補正とprecision gateを導入し、超低ビットレート画像圧縮において実符号量を削減するGP-ResLCを示した。CLIC2020 test、DIV2K、Kodakのfull-resolution real codec評価により、DISTS/FIDを中心にGLCより少ないbppで同等以上の知覚品質に到達できることを確認した。

## 参考文献

[1] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” Proc. ICLR, 2018.

[2] D. Minnen, J. Ballé, and G. D. Toderici, “Joint autoregressive and hierarchical priors for learned image compression,” Proc. NeurIPS, 2018.

[3] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized Gaussian mixture likelihoods and attention modules,” Proc. CVPR, pp. 7939--7948, 2020.

[4] Y. Blau and T. Michaeli, “Rethinking lossy compression: The rate-distortion-perception tradeoff,” Proc. ICML, 2019.

[5] F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson, “High-fidelity generative image compression,” Proc. NeurIPS, 2020.

[6] Z. Jia, J. Li, B. Li, H. Li, and Y. Lu, “Generative latent coding for ultra-low bitrate image compression,” Proc. CVPR, pp. 26088--26098, 2024.

[7] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” Proc. CVPR, 2018.

[8] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” IEEE Trans. Pattern Analysis and Machine Intelligence, 2022.

[9] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” Proc. NeurIPS, 2017.
