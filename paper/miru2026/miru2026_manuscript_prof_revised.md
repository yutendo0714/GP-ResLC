# MIRU2026 教授コメント反映版原稿

作成日: 2026-06-22  
用途: `miru2026_manuscript_balanced.md` をベースに，教授コメントを反映して表現・引用・損失説明・限界記述を修正した提出候補。

## 4ページ用の推奨配置

- 図1: GLC概要+GP-ResLC差分の統合手法図 `paper/miru2026/figures/glc_gp_reslc_unified_overview.png`
- 図2: 主結果curve統合図 `paper/miru2026/figures/result_curves_clic_div2k_perceptual_2x4_readable.png`
- 表1: BD-rate主結果
- 表2: CLIC2020 testのbpp内訳
- 図3: 復元例とrho可視化 `paper/miru2026/figures/clic_q3_rho_overlay_top4.png`

図2は本文で大きく使う。外部図読み取りに基づく補助比較は本文から外し，必要なら発表時または補足説明に回す。

---

# 復号側で再計算可能な量子化倍率による超低ビットレート生成型画像圧縮

著者名・所属・メールアドレスは投稿前に差し替える。

## 概要

画像圧縮は，画像を少ないビット列で表し，復元時の劣化をできるだけ小さくする問題である。近年は，ニューラルネットワークで画像を潜在表現へ変換し，その潜在表現の確率を学習して符号化するlearned image compressionが発展している。特に低ビットレートでは，画素値を忠実に近づけるだけでなく，人間にとって自然に見える復元画像を得ることが重要になる。生成型画像圧縮では，復号側の生成器が画像らしい構造やテクスチャを補完できるため，すべての潜在情報を同じ精度で送信することは非効率である。本稿では，訓練済みGLCを基盤とし，復号側で再計算可能な量子化倍率を導入するGP-ResLCを提案する。提案手法は，すでに送信済みのhyper潜在と品質インデックスから主潜在と同じサイズの倍率を推定するため，重要度マップを追加の補助情報として送信しない。倍率が大きい位置では主潜在残差を粗く量子化し，倍率が1に近い位置ではGLCに近い精度を保つ。これにより，GLCのVQGAN/VQ-VAE復号器による生成潜在復元能力を保ったまま，算術符号化される主潜在表現の符号長を削減する。CLIC2020 test，DIV2K validation，Kodakで評価し，DISTS/FIDを中心とする知覚品質を保ったbpp削減を確認した。

## 1. はじめに

画像圧縮の目的は，画像をできるだけ短いビット列で表し，復号後の画像を元画像に近づけることである。Learned image compression（LIC）では，この処理をニューラルネットワークで学習する。具体的には，符号化器が画像を特徴量である潜在表現へ変換し，その潜在表現を量子化した上で，確率モデルに基づいて符号化する。確率モデルが「出現しやすい」と判断した値には短い符号が割り当てられ，「出現しにくい」と判断した値には長い符号が必要になる。そのため，潜在表現の分布を正確に推定する確率モデリングが圧縮性能を大きく左右する。

この文脈で，Hyperpriorは，主潜在表現とは別のhyper潜在から分布パラメータを推定することで符号長を削減する [1]。Autoregressive priorは，すでに復号済みの近傍潜在を文脈情報として利用し，さらに精密な分布推定を行う [2]。また，Gaussian mixture likelihoodやattention moduleを導入したモデルは，より柔軟な潜在分布の確率モデリングを実現した [3]。これらは，いずれも潜在表現をどれだけ短く正確に符号化できるかを改善する流れである。

一方，ビット数が極端に限られる低ビットレート領域では，画像の全ての構造やテクスチャを忠実に送ることはできない。この領域では，画素誤差を一様に小さくするだけでは復元画像がぼけやすく，人間の知覚に自然な画像から離れる場合がある。Rate-distortion-perception trade-offの観点からも，低ビットレート圧縮では画素単位の歪みだけでなく，復元画像が自然画像としてどれだけもっともらしいかが重要である [4]。

この観点から，HiFiCのような生成型画像圧縮が注目されている [5]。生成型画像圧縮では，全ての詳細をビット列で送るのではなく，復号側の生成器が画像らしい構造やテクスチャを補完する。GLCは，VQGAN [7] やVQ-VAE [8] に基づく生成潜在空間上で変換符号化を行い，超低ビットレート自然画像圧縮で高い知覚品質を示した [6]。しかし，訓練済みGLCをそのまま用いる場合，主潜在表現 $y$ のすべての位置で同じ設計の量子化精度を使うため，生成器が補いやすい残差にも符号長が割かれる可能性がある。一方で，priorが予測しにくい残差は符号長が大きくなりやすく，限られた符号量では十分な量子化精度を保ちにくい。したがって，単に予測困難な残差へビットを割り当てるのではなく，生成器が補える領域の残差精度を下げ，構造や知覚品質に効く領域を相対的に保護することが重要である。

本稿では，提案手法GP-ResLCとして，訓練済みGLCに復号側で再計算可能な量子化倍率を導入する。提案手法は符号化器側でのみ計算可能な重要度マップを送らず，すでに送信済みのhyper潜在 $\hat{z}$ と品質インデックス $q$ から同じ倍率を復号側で再計算する。これにより，GLCの生成潜在空間を保ったまま，算術符号化される主潜在表現の符号長を削減する。本稿の貢献は，(1) 追加の重要度マップなしに，復号済み情報から重要度に応じた量子化倍率を生成するネットワークを構成したこと，(2) 複数のベンチマークにおいて，訓練済みGLCに対する知覚品質中心の符号量-品質曲線改善を示したことである。

## 2. 関連研究と位置づけ

LICでは，入力画像 $\mathbf{x}$ を解析変換で潜在表現 $\mathbf{y}$ へ写像し，量子化後の $\hat{\mathbf{y}}$ をエントロピー符号化モデルにより符号化する。復号側では $\hat{\mathbf{y}}$ から合成変換により復元画像 $\hat{\mathbf{x}}$ を得る。Hyperprior [1]，autoregressive prior [2]，Gaussian mixture likelihood [3] は，いずれも潜在分布の推定精度を上げることで算術符号化に必要なbit数を減らす流れである。

生成型圧縮では，すべての画像詳細をビット列で送るのではなく，復元側の生成能力に一部を委ねる。HiFiCはGAN損失を用いた生成型画像圧縮により，低ビットレートで自然な復元を目指した [5]。GLCは，VQGAN [7] やVQ-VAE [8] により学習した生成潜在空間を符号化対象とし，categorical hyper moduleやコード予測の補助教師信号を用いて，超低ビットレートで高い知覚品質を実現する [6]。GLCの強みは，画素空間の細部ではなく，人間知覚に整合しやすい生成潜在表現を符号化する点にある。

本研究はGLCを置き換える新しい生成復号器を提案するものではない。むしろ，訓練済みGLCを強い生成型圧縮モデルとして利用し，その潜在表現のビット割り当てを見直す。空間適応量子化や重要度マップに基づくビット割り当ては既存にも多いが，符号化器側の画像依存マップを送る方式では，超低ビットレートで補助情報の負担が無視できない場合がある。GP-ResLCでは，制御信号を復号側で再計算可能な $\hat{\mathbf{z}}$ と $q$ に制限する。したがって，追加の重要度マップなしでGLCの復号処理と整合する量子化倍率制御を行える。

図1に，GLC圧縮モデルと提案手法GP-ResLCの差分を示す。上段はGLCの主潜在表現 $\mathbf{y}$ とhyper潜在 $\hat{\mathbf{z}}$ の符号化経路，下段は訓練済みGLCに追加する復号側再計算可能な量子化倍率を示す。GP-ResLCは $\hat{\mathbf{z}}$ と品質インデックスから倍率を推定するため，追加の重要度マップなしで算術符号化された主潜在表現 $\mathbf{y}$ のビット列の符号長を削減する。

**図1: GLC圧縮モデルとGP-ResLCの概要。** GLCは生成潜在を変換符号化し，hyperpriorと4分割文脈priorにより主潜在を符号化する。GP-ResLCは訓練済みGLC基盤モデルを利用し，復号側で再計算可能な量子化倍率により主潜在の量子化精度を位置ごとに制御する。

## 3. 提案手法

### 3.1 GLCにおける主潜在符号化

入力画像を $\mathbf{x}$，GLCの主潜在表現を $\mathbf{y}$，hyper潜在を $\mathbf{z}$ とする。GLCは量子化された $\hat{\mathbf{z}}$ からhyper復号器を通して，$\mathbf{y}$ を符号化するための確率分布パラメータを推定する。主潜在表現 $\mathbf{y}$ は4分割priorにより4つの互いに重ならない空間マスクに分割され，順番に符号化・復号される。第1部分はhyperprior由来の共通パラメータから符号化され，第2部分以降はすでに復号済みの $\hat{\mathbf{y}}$ の一部を文脈情報として用いてスケールと平均を更新する。これにより，GLCはhyperpriorと自己回帰的な文脈情報の両方を使って $\mathbf{y}$ の確率分布を推定する。

通常のGLCでは，ある段階におけるprior平均を $\boldsymbol{\mu}_{\mathrm{GLC}}$，スケールを $\boldsymbol{\sigma}_{\mathrm{GLC}}$，量子化ステップを $\mathbf{Q}_{\mathrm{GLC}}$ とし，主潜在表現をこのpriorの周りの残差として符号化する。Priorがよく予測できる値は短く符号化される一方，priorが予測しにくい値は高い符号長を要する。超低ビットレートでは，このような難しい残差にすでに多くの符号長が必要になるだけでなく，量子化ステップを十分細かくできないため，復元時には粗い残差としてしか送れない場合がある。したがって問題は，priorで予測困難な残差に単純に多くのビットを割くことではなく，限られた符号長の中で，どの残差精度を下げても知覚品質を大きく損なわないかを判断することにある。

### 3.2 復号側で再計算可能な量子化倍率

提案手法GP-ResLCでは，GLCの符号化器，復号器，hyperprior，4分割文脈priorを保持したまま，主潜在表現 $\mathbf{y}$ の符号化直前に量子化倍率 $\boldsymbol{\rho}_\theta$ を導入する。倍率は，ビット列から復号可能な $\hat{\mathbf{z}}$ と品質インデックス $q$ から計算される。

$$
\boldsymbol{\rho}_\theta = f_\theta(\hat{\mathbf{z}}, q), \quad
\boldsymbol{\rho}_\theta \in [1,\rho_{\max}]^{C_y \times H_y \times W_y}
$$

ここで $\boldsymbol{\rho}_\theta$ は $\mathbf{y}$ と同じサイズで定義される量子化倍率であり，各要素は1以上である。実装では1チャネルの空間マップを推定し，チャネル方向に複製することで同じサイズの倍率として用いる。この倍率により，GLCの量子化ステップを要素ごとに変調する。

$$
\mathbf{Q}_{\mathrm{GP}} =
\mathbf{Q}_{\mathrm{GLC}} \odot \boldsymbol{\rho}_\theta
$$

$\boldsymbol{\rho}_\theta$ が大きい位置では量子化ステップが大きくなり，対応する残差値は粗く量子化されるため，算術符号化される主潜在表現の符号長が短くなりやすい。一方，$\boldsymbol{\rho}_\theta$ が1に近い位置ではGLCに近い精度で残差を送る。これは単に画像全体の品質を一様に下げる調整ではなく，復号側で再計算可能な特徴に基づいて，知覚的に低感度または生成器が補いやすい領域の残差精度を下げる局所的な量子化倍率制御である。

### 3.3 学習目的と設計制約

GP-ResLCでは符号化器側でしか計算できない画像依存の重要度マップを送らない。制御信号は，$\hat{\mathbf{z}}$，品質インデックス，およびGLCの復号順序で利用可能な文脈情報から再計算できるものに限定する。この制約により，復号側は通常の復号過程で得られる情報だけからGLCと同じ順序で $\hat{\mathbf{z}}$ と $\hat{\mathbf{y}}$ を復元し，同じ $\boldsymbol{\rho}_\theta$ を再計算できる。

実装では，訓練済みGLC [6] を凍結し，補助的な量子化倍率ネットワークを学習する。主結果で用いた学習目的は次式で表される。

$$
\mathcal{L}
= \lambda_R R_y
+ \mathcal{L}_{\mathrm{quality}}
+ \lambda_{\rho}\mathcal{L}_{\rho}
+ \lambda_{\mathrm{send}}\mathcal{L}_{\mathrm{send}}
+ \lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}
$$

$$
\mathcal{L}_{\mathrm{quality}}
= \lambda_{\mathrm{MSE}}\mathcal{L}_{\mathrm{MSE}}(\mathbf{x},\hat{\mathbf{x}})
+ \lambda_{\mathrm{LPIPS}}\mathcal{L}_{\mathrm{LPIPS}}(\mathbf{x},\hat{\mathbf{x}})
$$

ここで $R_y$ は，GLCのエントロピー符号化モデルから得られる学習時の主潜在表現の推定符号量であり，$-\sum \log_2 p(\hat{\mathbf{y}}|\hat{\mathbf{z}},q)$ に対応する。$\mathcal{L}_{\mathrm{MSE}}$ は画素空間の平均二乗誤差，$\mathcal{L}_{\mathrm{LPIPS}}$ はVGG-LPIPS [12] である。$\mathcal{L}_{\rho}=\max(0,\rho_{\mathrm{target}}-\mathrm{mean}(\boldsymbol{\rho}_\theta))$ は平均倍率を目標値へ保つ項，$\mathcal{L}_{\mathrm{send}}$ は低誤差・低勾配領域で倍率を大きくしやすくする学習時のみ用いる送信しやすさの教師信号である。$\mathcal{L}_{\mathrm{align}}$ はGLCのVQコード予測の補助教師信号に基づく補助的なコード整合項であり，提案の主機構ではなく訓練済みGLCの生成潜在表現から大きく外れないために用いる。主結果では $\lambda_R=10$，$\lambda_{\mathrm{MSE}}=0.08$，$\lambda_{\mathrm{LPIPS}}=4$，$\lambda_{\rho}=2$，$\lambda_{\mathrm{send}}=5$，$\lambda_{\mathrm{align}}=1$，$\rho_{\mathrm{target}}=1.16$ とした。DISTS [13] は主評価指標として用いるが，主結果の学習損失には直接入れていない。DISTS損失を加えた追加学習も検証したが，CLIC2020 testとDIV2Kで主結果を上回らなかったため，本稿の主結果には採用しない。

## 4. 実験

### 4.1 評価設定

ベースラインは，訓練済みGLC [6] とする。提案手法は同じ訓練済みGLCにGP-ResLCの量子化倍率ネットワークを追加したモデルである。評価は，CLIC2020 test [9]，DIV2K validation [10]，Kodak [11]で行う。CLIC2020 testはprofessional 250枚とmobile 178枚の合計428枚，DIV2Kはvalidation 100枚，Kodakは24枚である。

bppは，ヘッダ，固定長符号化された $\hat{\mathbf{z}}$ のインデックス，および4分割prior順序で算術符号化された主潜在表現 $\mathbf{y}$ のビット列を含む全体のビット列長から算出する。

評価指標は，PSNR，MS-SSIM [14]，LPIPS [12]，DISTS [13]，FID [15]，KID [16]である。GLC/HiFiC型の評価に合わせ [5,6]，CLIC2020 testとDIV2Kでは256×256 パッチを通常分割に加えて128 pixel shiftでも抽出し，FID/KIDを計算する。このプロトコルでのパッチ数は，CLIC2020 testで28,650，DIV2Kで6,573である。Kodakは画像数が少ないため，DISTS，LPIPS，PSNR，MS-SSIMを主に見る。

### 表1: 訓練済みGLCに対するGP-ResLCのBD-rate

負の値は，Bj{\o}ntegaard法 [17] に基づき，同一品質に到達するための平均bppがGP-ResLCで小さいことを示す。したがって表1は品質インデックスごとの一点比較ではなく，符号量-品質曲線間の平均符号量差である。

| Dataset | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | -10.28 | +0.19 | -0.98 | +0.38 | -7.30 | -7.10 |
| DIV2K val. | -10.79 | -0.54 | -1.49 | -0.17 | -5.61 | -6.50 |
| Kodak | -4.47 | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |

図2に，CLIC2020 testとDIV2K validationにおける符号量-品質曲線を示す。各行がデータセット，各列がFID，KID，DISTS，LPIPSである。横軸はbppであり，縦軸の各指標は低いほど良い。GLCとGP-ResLCの曲線を比較することで，品質インデックスの一点比較ではなく，曲線全体として同等知覚品質に必要なbppが下がるかを確認できる。

**図2: CLIC2020 testとDIV2K validationにおける知覚品質指標の符号量-品質曲線。** FID，KID，DISTS，LPIPSはいずれも低いほど良い。

### 4.2 結果

表1に示すように，CLIC2020 testではDISTSで-10.28%，FIDで-7.30%のBD-rate改善が得られた。DIV2KでもDISTSで-10.79%，FIDで-5.61%となり，同様の傾向を示した。Kodakでは画像数が少ないためFID/KIDは参考扱いだが，DISTSで-4.47%の改善を示した。LPIPSはCLIC2020 testでほぼ中立，DIV2K/Kodakでは小幅改善に留まった。この結果は，本手法の主な効果がDISTS/FIDに現れる符号量-知覚品質寄りの改善であることを示している。

なお，DISTSについて各GLC評価点と同じ品質に対応するGP-ResLCのbppを曲線上で確認しても，CLIC2020 testで平均-10.26%，DIV2Kで-10.27%，Kodakで-5.45%の削減となり，表1のBD-rateと同じ傾向を示した。

### 表2: CLIC2020 testにおけるbppの内訳

| 手法 | q | 合計bpp | y bpp | z bpp | ヘッダ bpp |
|---|---:|---:|---:|---:|---:|
| GLC | 0 | 0.02134 | 0.01757 | 0.00352 | 0.00025 |
| GP-ResLC | 0 | 0.01892 | 0.01515 | 0.00352 | 0.00025 |
| GLC | 3 | 0.03369 | 0.02992 | 0.00352 | 0.00025 |
| GP-ResLC | 3 | 0.03102 | 0.02726 | 0.00352 | 0.00025 |

表2に示すように，$\hat{\mathbf{z}}$ の符号化部分とヘッダはGLCとGP-ResLCで同一であり，削減は算術符号化された主潜在表現 $\mathbf{y}$ のビット列から生じている。これは，提案手法による削減が主潜在表現の符号化部分から生じていることを示す。

### 4.3 可視化と機構分析

**図3: 復元例と量子化倍率 $\boldsymbol{\rho}_\theta$ の可視化。** 高い倍率は粗い残差量子化を表し，GLCの生成復号器に復元を委ねる領域に対応する。実際の符号長削減は表2の主潜在表現 $\mathbf{y}$ のbpp内訳で確認する。

$\boldsymbol{\rho}_\theta$ が高い領域は，GP-ResLCが残差をより粗く量子化し，GLCの生成復号器に復元を委ねる領域に対応する。一方，構造境界や復元が難しい領域では相対的に低い倍率が割り当てられ，GLCに近い精度で残差が送られる。図3は主にこの空間的な倍率分布を示すものであり，実際の符号長削減は表2の主潜在表現 $y$ のbpp内訳で確認できる。補助解析として，q3では高い倍率が割り当てられた領域ほどGLC ベースラインの局所誤差や画像勾配が低い傾向を確認しており，知覚的に低感度な領域で符号長を削減するという設計と整合する。

## 5. 考察とまとめ

本稿の結果は，超低ビットレート生成型圧縮において，復号側で再計算可能な情報に基づく量子化倍率制御が有効であることを示している。特に，$\hat{\mathbf{z}}$ やヘッダは変えず，削減が主潜在表現 $\mathbf{y}$ のビット列から生じている点は，本手法の設計と整合する。

一方で，現時点のGP-ResLCは訓練済みGLCに追加したモジュールであり，VQ-VAE，変換符号化，エントロピー符号化モデルを最初から一貫して学習した圧縮モデルではない。また，PSNR/MS-SSIMやLPIPSの改善は一貫して大きいわけではない。さらに，量子化倍率ネットワークは符号化器側と復号器側で共有され，復号時にも $\boldsymbol{\rho}_\theta$ を計算する必要があるため，わずかな計算量増加が生じる。ただし，追加の重要度マップは送信せず，評価では符号化/復号時間がGLCと同程度であることを確認している。

本稿では，訓練済みGLCの生成潜在表現上で復号側と整合する量子化倍率を導入し，超低ビットレート画像圧縮において符号長を削減するGP-ResLCを示した。CLIC2020 test，DIV2K validation，Kodakで評価し，DISTS/FIDを中心にGLCより少ないbppで同等以上の知覚品質に到達できることを確認した。今後は，生成潜在空間そのものをGP-ResLCの目的に合わせて学習し，生成器が補える成分と送るべき残差成分の分離をより明示的に行う。

## 参考文献

[1] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a スケール hyperprior,” Proc. ICLR, 2018.

[2] D. Minnen, J. Ballé, and G. D. Toderici, “Joint autoregressive and hierarchical priors for learned image compression,” Proc. NeurIPS, 2018.

[3] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized Gaussian mixture likelihoods and attention modules,” Proc. CVPR, pp. 7939--7948, 2020.

[4] Y. Blau and T. Michaeli, “Rethinking 損失y compression: The rate-distortion-perception tradeoff,” Proc. ICML, 2019.

[5] F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson, “High-fidelity generative image compression,” Proc. NeurIPS, 2020.

[6] Z. Jia, J. Li, B. Li, H. Li, and Y. Lu, “Generative latent coding for ultra-low bitrate image compression,” Proc. CVPR, pp. 26088--26098, 2024.

[7] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” Proc. CVPR, 2021.

[8] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” Proc. NeurIPS, 2017.

[9] Challenge on Learned Image Compression, “CLIC 2020,” https://www.compression.cc/ （accessed 2026-06-22）.

[10] E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: Dataset and study,” Proc. CVPR Workshops, 2017.

[11] Eastman Kodak Company, “Kodak 損失less true color image suite,” http://r0k.us/graphics/kodak/ （accessed 2026-06-22）.

[12] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” Proc. CVPR, 2018.

[13] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” IEEE Trans. Pattern Analysis and Machine Intelligence, 2022.

[14] Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiスケール structural similarity for image quality assessment,” Proc. Asilomar Conference on Signals, Systems and Computers, 2003.

[15] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-スケール update rule converge to a local Nash equilibrium,” Proc. NeurIPS, 2017.

[16] M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying MMD GANs,” Proc. ICLR, 2018.

[17] G. Bj{\o}ntegaard, “Calculation of average PSNR differences between RD-curves,” ITU-T VCEG-M33, 2001.
