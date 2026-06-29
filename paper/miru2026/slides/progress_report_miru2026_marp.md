---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section { font-family: "Noto Sans CJK JP", "Yu Gothic", "Hiragino Sans", sans-serif; }
  h1 { font-size: 34px; }
  h2 { font-size: 28px; }
  h3 { font-size: 22px; }
  p, li { font-size: 23px; line-height: 1.35; }
  table { font-size: 20px; }
  .small { font-size: 18px; }
  .tiny { font-size: 15px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; align-items: center; }
  .cols3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
  .key { color: #0b6b50; font-weight: 700; }
  .warn { color: #9a4d00; font-weight: 700; }
---

# 研究進捗報告
## GP-ResLC: 生成潜在表現の復元可能性を利用した超低ビットレート画像圧縮

- MIRU2026投稿予定内容の整理
- ベース: 訓練済みGLC自然画像モデル
- 主張: <span class="key">生成復号器が補える成分は粗く送り，知覚品質に効く残差を相対的に守る</span>

<div class="small">2026-06-24</div>

---

# Outline

1. 背景: learned image compressionと低ビットレート
2. 関連研究: 生成型圧縮とGLC
3. 提案手法: GP-ResLC
4. 学習目的と評価設定
5. 結果: BD-rate / 曲線 / ビット列内訳
6. まとめとFuture Work

---

# 背景: 画像圧縮をニューラルで行う

<div class="cols">
<div>

## Learned Image Compression（LIC）

- 画像を潜在表現へ変換
- 潜在表現を量子化して符号化
- 確率モデルで「短く送れる値」を推定
- Hyperprior / 文脈priorで符号長を削減


</div>
<div>

## 超低ビットレートでは

- 全ての構造・テクスチャは送れない
- 画素誤差だけを小さくするとぼけやすい
- 見た目の自然さも重要になる
- 何を送るか，何を生成器に任せるかが鍵

</div>
</div>

<div class="small">代表例: Ballé et al., Minnen et al., Cheng et al.</div>

---

# 関連研究: 生成型圧縮 / GLC

- HiFiC: GAN損失で低ビットレートの自然な復元を狙う
- GLC: VQGAN/VQ-VAEの<span class="key">生成潜在空間</span>を変換符号化
- GLCの強み
  - 画素空間ではなく，人間知覚に近い潜在表現を符号化
  - hyperprior + 4分割文脈priorで主潜在表現 $y$ を符号化
  - 超低ビットレートでDISTS/FID/KIDが強い

---

# 着眼点

## GLCの生成潜在表現にも「粗く送れる場所」があるのでは？

- 生成復号器が自然に補える成分まで高精度に送るのは非効率
- 一方で，構造や知覚品質に効く残差は粗くしすぎたくない
- 重要度マップを別に送ると，超低ビットレートでは補助情報が重い

<div class="key">
狙い: 追加の重要度マップなしで，主潜在表現のビット割り当てを見直す
</div>

---

# 提案: GP-ResLC

![w:980 center](../figures/glc_gp_reslc_unified_overview.png)

訓練済みGLCは固定し，復号側でも再計算できる量子化スケールを追加する。  
削減対象は主に算術符号化された主潜在表現 $y$ のビット列。

---

# 提案手法の要点

## 復号側で再計算可能な量子化スケール

$$
\boldsymbol{\rho}_\theta(\hat{z}, q) \ge 1
$$

$$
\mathbf{Q}_{\mathrm{GP}} = \mathbf{Q}_{\mathrm{GLC}} \odot \boldsymbol{\rho}_\theta(\hat{z}, q)
$$

- $\hat{z}$ と品質インデックス $q$ からスケールを計算
- スケール用の重要度マップは送信しない
- $\rho$ が大きい領域: 残差を粗く量子化し，符号長を削減
- $\rho$ が1に近い領域: GLCに近い精度で残差を保持

---

# 実験設定

<div class="cols">
<div>

## データセット

- CLIC2020 test: 428 images
  - professional 250 + mobile 178
- DIV2K validation: 100 images
- Kodak: 24 images

</div>
<div>

## 評価指標

- 主指標: FID / KID / DISTS / LPIPS
- 補助指標: PSNR / MS-SSIM
- CLIC/DIV2K: shifted 256×256パッチ
  - CLIC: 28,650パッチ
  - DIV2K: 6,573パッチ

</div>
</div>

**bpp:** ヘッダ，$\hat{z}$ のインデックス，主潜在表現 $y$ のビット列を含む全体のビット列長から算出。

---

# 学習目的

$$
\mathcal{L}
= \lambda_R R_y
+ \mathcal{L}_{\mathrm{quality}}
+ \lambda_\rho \mathcal{L}_\rho
+ \lambda_{\mathrm{send}} \mathcal{L}_{\mathrm{send}}
+ \lambda_{\mathrm{align}} \mathcal{L}_{\mathrm{align}}
$$

$$
\mathcal{L}_{\mathrm{quality}}
= \lambda_{\mathrm{MSE}}\mathcal{L}_{\mathrm{MSE}}
+ \lambda_{\mathrm{LPIPS}}\mathcal{L}_{\mathrm{LPIPS}}
$$

- $R_y$: 主潜在表現の推定符号量
- $\mathcal{L}_\rho$: 平均スケールの制御
- $\mathcal{L}_{\mathrm{send}}$: 低誤差・低勾配領域を粗くしやすくする教師信号
- $\mathcal{L}_{\mathrm{align}}$: GLCの生成潜在から外れすぎないための補助項

**補足:** DISTSは主評価指標として用いるが，主結果の学習損失には直接入れていない。  
**主設定:** $\lambda_R=10$, $\lambda_{\mathrm{MSE}}=0.08$, $\lambda_{\mathrm{LPIPS}}=4$, $\lambda_\rho=2$, $\lambda_{\mathrm{send}}=5$, $\rho_{\mathrm{target}}=1.16$

---

# 主結果: BD-rate vs 訓練済みGLC

| データセット | DISTS | LPIPS | PSNR | MS-SSIM | FID | KID |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test | **-10.28** | +0.19 | -0.98 | +0.38 | **-7.30** | **-7.10** |
| DIV2K val. | **-10.79** | -0.54 | -1.49 | -0.17 | **-5.61** | **-6.50** |
| Kodak | **-4.47** | -0.79 | -0.87 | +0.45 | -1.70 | -6.14 |

<div class="small">
負の値は，同一品質に到達するための平均bppが小さいことを示す。主な改善はDISTS/FID/KID側に出ている。
</div>

---

# 符号量-品質曲線

![w:1100 center](../figures/result_curves_clic_div2k_perceptual_2x4_readable.png)

<div class="small">
CLIC2020 test / DIV2K validationで，FID・KID・DISTS・LPIPSを比較。曲線全体として同等品質に必要なbppが下がるかを確認。
</div>

---

# どこでビットが減ったのか？

## CLIC2020 test bpp内訳

| 手法 | q | 合計bpp | y bpp | z bpp | ヘッダbpp |
|---|---:|---:|---:|---:|---:|
| GLC | 0 | 0.02134 | 0.01757 | 0.00352 | 0.00025 |
| GP-ResLC | 0 | **0.01892** | **0.01515** | 0.00352 | 0.00025 |
| GLC | 3 | 0.03369 | 0.02992 | 0.00352 | 0.00025 |
| GP-ResLC | 3 | **0.03102** | **0.02726** | 0.00352 | 0.00025 |

**確認:** $\hat{z}$ とヘッダは同一。削減は主潜在表現 $y$ のビット列から生じている。

---

# 可視化: $\rho$ と復元例

![w:760 center](../figures/clic_q3_rho_overlay_top4.png)

高い $\rho$ は残差を粗く量子化する領域。  
生成復号器が補いやすい/知覚的に低感度な領域でビットを節約し，難しい領域を相対的に保護する。

---

# 現状まとめ

## MIRU投稿での主張

- GLCの生成潜在空間上で，復号側と整合する量子化スケールを導入
- 追加の重要度マップなしで，主潜在表現 $y$ のビット列を削減
- CLIC2020 / DIV2K / KodakでDISTS/FID/KID中心にBD-rate改善

## 注意点

- LPIPS/PSNR/MS-SSIMは大幅改善ではない
- 現状は訓練済みGLCに追加したモジュール
- 完全なGP-ResLCは，VQ潜在・変換符号化・量子化制御を一貫学習する必要がある

---

# Future Work / 今後

1. **完全版GP-ResLC**
   - VQ潜在・変換符号化・残差分離を一貫学習

2. **Hyper潜在の改善**
   - 現状 $\hat{z}$ はインデックス符号化
   - 将来的には $\hat{z}$ のエントロピー符号化も検討

3. **Ablation / 解析強化**
   - スケールなし / スケール強度 / データセット汎化
   - $\rho$ と誤差・テクスチャ・勾配の関係をより体系的に分析

4. **次の投稿に向けて**
   - 訓練済みGLC依存から完全版へ拡張
   - 長期学習で性能上限を押し上げる


---

# References

<div class="tiny">

[1] Ballé et al., “Variational image compression with a scale hyperprior,” ICLR, 2018.  
[2] Minnen et al., “Joint autoregressive and hierarchical priors for learned image compression,” NeurIPS, 2018.  
[3] Cheng et al., “Learned image compression with discretized Gaussian mixture likelihoods and attention modules,” CVPR, 2020.  
[4] Blau and Michaeli, “Rethinking lossy compression: The rate-distortion-perception tradeoff,” ICML, 2019.  
[5] Mentzer et al., “High-fidelity generative image compression,” NeurIPS, 2020.  
[6] Jia et al., “Generative latent coding for ultra-low bitrate image compression,” CVPR, 2024.  
[7] Esser et al., “Taming transformers for high-resolution image synthesis,” CVPR, 2021.  
[8] Zhang et al., “The unreasonable effectiveness of deep features as a perceptual metric,” CVPR, 2018.  
[9] Ding et al., “Image quality assessment: Unifying structure and texture similarity,” TPAMI, 2022.  
[10] Heusel et al., “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” NeurIPS, 2017.  
[11] Bińkowski et al., “Demystifying MMD GANs,” ICLR, 2018.  
[12] van den Oord et al., “Neural discrete representation learning,” NeurIPS, 2017.

</div>
