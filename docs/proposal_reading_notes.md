# GP-ResLC Proposal Reading Notes

読了対象:
- `proposal_design/GP-ResLC_proposal_design.md`
- `proposal_design/RUNBOOK.md`
- `proposal_design/prior_predictor.py`
- `proposal_design/train_v1.py`, `test_v1.py`, `train_v2.py`, `test_v2.py`
- GLC本体 `src/models/image_model.py`, `src/models/common_model.py`, `test_image.py`

## 理解した提案手法

短期VCIP版は、GLCの公式pretrained image codecを凍結し、追加の `P_theta` だけを学習する。GLCの画像圧縮フローは次の通り。

1. `x -> vqgan.encoder -> y_ori`
2. `y_ori -> enc(q_enc) -> y`
3. `y -> hyper_enc -> z`
4. `z -> z_vq indices -> z_hat`
5. `z_hat -> hyper_dec -> y_prior_fusion -> params=(quant_step, scales, means)`
6. `forward_four_part_prior(y, params, ...)` で4分割checkerboard的に `y` を量子化/レート推定
7. `y_hat -> dec(q_dec) -> vqgan.generator -> x_hat`

GP-ResLC V1は、5の後に `PriorPredictor(z_hat)` を挿入して `delta_params` を足す。zero-init gateにより、初期状態ではbaseline GLCと厳密一致する。学習で `means/scales/quant_step` を補正し、生成器/decoderが `z_hat` から予測できる `y` 成分をprior側に寄せ、残る予測不能成分の `bit_y` を下げる。

## 実装整理

`proposal_design/` は外部ハンドオフとして残し、実行用コードを以下へコピーした。

- `gp_reslc/prior_predictor.py`: Pθと `train_forward`
- `gp_reslc/perceptual_gate.py`: V2用の知覚ゲート
- `scripts/train_v1.py`, `scripts/test_v1.py`
- `scripts/train_v2.py`, `scripts/test_v2.py`
- `scripts/eval_metrics.py`, `scripts/make_curves.py`, `scripts/make_comparison.py`, `scripts/run_ablation.py`
- `scripts/smoke_gp_reslc.py`: zero-init同一性・CUDA・checkpointロード確認
- `configs/*.example.json`: 比較/アブレーション設定例

`src/` は公式GLC実装として極力触らない。研究コードは横に置き、差分の意味を明確にする。

## 重要な検証ポイント

1. Zero-init equivalence  
   未学習Pθの `use_predictor=True` が `use_predictor=False` と完全一致すること。これは `scripts/smoke_gp_reslc.py` で確認済み。

2. `bit_y`だけでなくtotal bpp  
   `bit_z` はGLC実装上 `H_z * W_z * log2(codebook_size)` で固定。主張の直接対象は `bit_y` だが、論文表ではtotal bppを必ず使う。

3. 見かけのrate低下の排除  
   `delta_params` は `quant_step/scales/means` 全部に作用できる。単にscaleを大きくして推定rateを下げ、DISTS/LPIPSが悪化する可能性がある。wandbに `delta_abs`, `mu_mean/std`, `bpp_y`, `bpp_total`, PSNR, LPIPSを記録するよう補強した。

4. MSEの扱い  
   MSEは構造保持の補助として使うが、低bitrate生成圧縮の主指標ではない。`lambda_d` ablationを行い、DISTS/LPIPS/FIDで主張する。

5. CodePredictionLossの形状  
   `CodePredictionLoss(latent_size=16x16)` 前提のため、学習cropは256x256固定が安全。smokeは通ったが、実学習の最初の数iterでも `l_align` のfinite性を確認する。

6. Training autograd and inplace ops  
   GLC inference commonly instantiates `GLC_Image(inplace=True)`, but differentiable training through `forward_four_part_prior` fails because inplace activations modify split views. `scripts/train_v1.py` and `scripts/train_v2.py` now use `GLC_Image(inplace=False)`. Keep inference scripts unchanged unless gradients are needed.

## VCIP向けの最短実験順

1. V0: GLC pretrainedでKodak/CLICを再現。
2. V1-q2 smoke training: OpenImages小subset、`q_index=2`, 500〜2000 iterで `ab/delta_bpp_y` を見る。
3. V1 all-q: q=0..3を別々に学習し、Kodak/CLICで曲線。
4. If weak: `--unfreeze_fusion`, `lambda_align=0`, `lambda_d=0.1`, predictor容量増の順に試す。
5. V2はV1で利得が見えてから。q条件化 + gateは論文の第2貢献候補。


## Early empirical lesson

The unconstrained `delta_params` path can reduce bpp quickly, but the reduction appears to come from changing quantization/rate-model parameters rather than a clean generative residual mean prediction. To keep the VCIP claim defensible, `train_forward` now supports:

- `--predictor_param_mode mean`: only means correction.
- `--predictor_param_mode scale_mean`: scales + means correction, quant_step frozen. This is now default.
- `--predictor_param_mode all`: diagnostic/legacy mode, includes quant_step and can produce degenerate rate-quality tradeoffs.

For a paper claim, prefer `scale_mean` or a regularized variant. `all` should only be used as an ablation showing why quant_step must be controlled.


## Updated reading after 2026-06-19 experiments

The original V1 assumption was that a global `z_hat -> mu_y` predictor could directly improve the entropy mean. Experiments showed this is too naive for GLC:

- GLC's four-part spatial prior overwrites/refines means after each mask using `y_hat_so_far`.
- If `mu_y` is subtracted before the four-part prior, the spatial prior sees residual-domain contexts it was not trained on.
- If `mu_y` is added to every stage mean, the spatial context stays in-distribution, but the simple target `y*q_enc-base_mean` still increases rate.

The most reliable short-track mechanism is now the V2 quantization gate:

- `rho(z_hat, q)` is deterministic at decoder and costs zero bits.
- Zero initialization gives exact GLC equivalence.
- Increasing `rho` coarsens latent residual quantization where the generator can perceptually absorb detail loss.
- This is a rate-perception control mechanism rather than a pure entropy-mean correction.

Implementation detail: exact identity matters. Even `rho=1.0003` can change quantized latents and produce large reconstruction diffs near quantization boundaries. The current `PerceptualGate` uses centered sigmoid so zero logits give exactly `rho=1`.

Paper positioning should be precise:

- Do not claim current V1 P_theta alone solves residual entropy modeling.
- Claim the working short-track method is a generator-predictability gate over residual transmission strength.
- Keep V1 P_theta as an ablation/negative result motivating stage-aware future work for the full GP-ResLC R-D-P paper.


## Updated empirical position after 12k V2 run

The working mechanism for the VCIP short paper is now concrete:

- The transmitted GLC hyper/VQ code z_hat lets the decoder infer a spatially varying residual-transmission gate rho(z_hat, q).
- rho costs no side bits because it is computed at both encoder-side training/inference and decoder-side reconstruction from transmitted z_hat and q.
- rho>1 increases the effective quantization step for y residual coding, so fewer bits are spent where the generator can perceptually absorb missing latent detail.
- This is not yet the full original P_theta residual-mean story, but it preserves the core research axis: do not transmit generator-recoverable detail.

Best current evidence:

- W&B run zbuykb7n trained the balanced V2 gate for 12k iterations.
- The best checkpoint is v2_6000, not final.
- Kodak q3 reaches DISTS 0.0981 at 0.0319 bpp.
- GLC baseline q2 reaches DISTS 0.0983 at 0.032781 bpp.
- The 4-point DISTS BD-rate is -3.94% versus official GLC.

Limits that must be stated honestly:

- LPIPS BD-rate is still slightly positive, so the current model is not a universal perceptual win.
- PSNR is worse, as expected for a rate-perception-oriented low-bitrate method.
- The result should be presented as a short-track R-P contribution, not as the final R-D-P GP-ResLC story.
- Future full GP-ResLC should reintroduce P_theta as stage-aware residual prediction inside the four-part spatial prior, instead of a global z_hat-to-y mean correction.
