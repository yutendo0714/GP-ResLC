# prior_predictor.py
# ============================================================================
#  生成事前予測残差エントロピーモデル — GLC への最小侵襲インテグレーション
# ============================================================================
#
#  狙い（VCIP の単一貢献）:
#    GLC のエントロピー段では、y を符号化する prior (means/scales) を
#    「y の要約 z をVQ符号化したもの → hyper_dec」から作る。
#    本貢献は、その prior の平均を「意味コード s から生成器が復元できる潜在の
#    予測 μ_θ(s)」へと“生成事前整合”させ、残差 y - μ_θ(s) のみを符号化する。
#    = 「生成器が自力で復元できる成分は送らない」を、エントロピー prior に実装。
#
#  設計の肝:
#    出力を ZERO-INIT GATE（ControlNet 流の zero-conv）で GLC の prior に加算する。
#    → 学習開始時は厳密に GLC と同一（無歪み）。ゲートを学習すると生成事前整合分だけ改善。
#    → ゲート off = GLC（ベースライン）, ゲート on = Ours、が自然な A/B になる。
#
#  GLC の該当 API（src/models/）:
#    image_model.GLC_Image.test():
#        z_hat   = self.z_vq.get_quan_feat(index, ...)         # 意味コード s ≒ z の量子化特徴
#        params  = self.hyper_dec(z_hat)
#        params  = self.y_prior_fusion(params)                 # ← ここに加算注入する
#        ... = self.forward_four_part_prior(y, params, ...)    # params から means/scales を生成
#    common_model.separate_prior(params): params.chunk(3,1) → (quant_step, scales, means)
#    common_model.get_y_gaussian_bits(y_q, scales) でレート算出
#
#  ※ GLC 公式リポジトリは推論コード＋重みのみ公開（学習コードは非公開）。
#    Phase 1 では「GLC 全体を凍結し、本モジュール（と任意で fusion 微調整）だけ学習」する。
# ============================================================================

import torch
from torch import nn

try:
    # GLC のブロックを流用（リポジトリ直下から import する想定）
    from src.models.layers import DepthConvBlock, ResidualBlockUpsample
except Exception:  # スタンドアロン確認用フォールバック
    DepthConvBlock = None
    ResidualBlockUpsample = None


def zero_module(m: nn.Module) -> nn.Module:
    """重み・バイアスを 0 初期化（ControlNet 流ゲート）。"""
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class PriorPredictor(nn.Module):
    """
    生成事前予測器 P_θ。
      入力 : z_hat（意味コード s の量子化特徴, N×h×w, y の 1/4 解像度）
      出力 : GLC の prior params（3N チャネル: quant_step|scales|means）への
             ゼロ初期化加算補正 Δparams。特に means 成分を μ_θ(s) へ近づける。
    GLC の hyper_dec と同じ ×4 アップサンプルで y 解像度へ整合させる。
    """

    def __init__(self, N: int = 256):
        super().__init__()
        assert ResidualBlockUpsample is not None, "GLC リポジトリ直下で import してください。"

        # hyper_dec と同形のアップサンプラ（z_hat → y 解像度の意味特徴）
        self.up = nn.Sequential(
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockUpsample(N, N, 2),
            DepthConvBlock(N, N),
        )
        # 生成事前整合のための予測ヘッド（μ_θ の素）
        self.pred = nn.Sequential(
            DepthConvBlock(N, N),
            DepthConvBlock(N, N),
        )
        # GLC の prior(3N) への補正を出すゲート（zero-init → 開始時は GLC と一致）
        self.gate = zero_module(nn.Conv2d(N, N * 3, 1))
        # μ_θ 自体（生成器整合損失に使う; means チャネルへ写像）
        self.to_mu = nn.Conv2d(N, N, 1)

    def forward(self, z_hat: torch.Tensor):
        feat = self.up(z_hat)        # N × H' × W'（y 解像度）
        sem = self.pred(feat)        # 意味整合特徴
        delta_params = self.gate(sem)  # 3N: GLC prior への加算補正（zero-init）
        mu_pred = self.to_mu(sem)      # μ_θ(s): 生成器が復元する潜在の予測
        return delta_params, mu_pred, feat



def forward_four_part_prior_with_latent_mean(net, y, common_params, latent_mean_scaled):
    """GLC four-part prior with an explicit generator-predictable latent mean.

    `latent_mean_scaled` is in the same quantized/scaled y domain as the prior
    means. At each autoregressive stage we code only
    y_scaled - (base_mean + latent_mean_scaled), while y_hat_so_far still
    contains the added latent mean. This keeps GLC's spatial prior on its normal
    reconstructed-latent distribution instead of feeding it centered residuals.
    """
    q_enc, q_dec, scales, means = net.separate_prior(common_params)
    if net.y_spatial_prior_reduction is not None:
        common_params_reduced = net.y_spatial_prior_reduction(common_params)
    else:
        common_params_reduced = common_params

    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = net.get_mask_four_parts(B, C, H, W, dtype, device)

    y_scaled = y * q_enc

    y_res_0, y_q_0, y_hat_0, s_hat_0 = net.process_with_mask(
        y_scaled, scales, means + latent_mean_scaled, mask_0)

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    y_res_1, y_q_1, y_hat_1, s_hat_1 = net.process_with_mask(
        y_scaled, scales, means + latent_mean_scaled, mask_1)

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    y_res_2, y_q_2, y_hat_2, s_hat_2 = net.process_with_mask(
        y_scaled, scales, means + latent_mean_scaled, mask_2)

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    y_res_3, y_q_3, y_hat_3, s_hat_3 = net.process_with_mask(
        y_scaled, scales, means + latent_mean_scaled, mask_3)

    y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
    y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
    y_hat = y_hat_so_far + y_hat_3
    scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
    y_hat = y_hat * q_dec
    return y_res, y_q, y_hat, scales_hat

# ============================================================================
#  GLC_Image への統合（image_model.py を以下のように改変する）
# ============================================================================
#
#  (1) __init__ の末尾に追加:
#        from .prior_predictor import PriorPredictor
#        self.prior_predictor = PriorPredictor(N)
#
#  (2) test() / 学習 forward の prior 生成部を差し替え:
#
#        params = self.hyper_dec(z_hat)
#        params = self.y_prior_fusion(params)
#        # --- 追加: 生成事前予測の加算注入（zero-init なので初期は GLC と同一）---
#        delta_params, mu_pred, _ = self.prior_predictor(z_hat)
#        params = params + delta_params
#        # ----------------------------------------------------------------
#        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
#            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
#            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
#            y_spatial_prior_reduction=self.y_spatial_prior_reduction)
#
#     これだけで means が μ_θ(s) 方向に補正され、four_part_prior が
#     残差 y - μ_θ を符号化する。bit_z（意味コードのビット）は不変なので、
#     bit_y の減少がそのまま「事前予測残差の利得」になる（クリーンな A/B）。
#
#  (3) 学習信号（μ_θ を“生成器が復元できる予測”へ整合させる鍵）:
#     GLC は CodePredictionLoss を同梱（self.code_pred_loss / code_pred_pix_loss）。
#     これを μ_pred 経路に適用し、μ_θ(s) を生成器整合へ導く。
#     総損失（GLC 凍結・本モジュールのみ学習）の例:
#
#        L = lambda_R * R_y                       # = get_y_gaussian_bits(y_q, scales_hat).mean()
#          + lambda_align * L_codepred(mu_pred)   # μ_θ を生成器の復元可能潜在へ整合
#          + lambda_d * (MSE(x, x_hat) + LPIPS(x, x_hat))   # 任意: 仕上げ
#
#     bit_z は固定（z_vq は凍結）なので R_z は最適化対象外でよい。
#
# ============================================================================
#  学習ループの骨子（Phase 1, GLC 凍結 + 本モジュールのみ学習）
# ============================================================================
"""
net = GLC_Image(inplace=True)
net.load_state_dict(get_state_dict(GLC_WEIGHTS), strict=False)  # 既存重みをロード
net.prior_predictor = PriorPredictor(net.N)                     # 新規モジュール

# GLC 本体を凍結、新規モジュールのみ学習
for p in net.parameters():
    p.requires_grad_(False)
for p in net.prior_predictor.parameters():
    p.requires_grad_(True)
# 任意: y_prior_fusion も微調整したい場合は requires_grad_(True)

opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

for x in loader:                       # x: OpenImages のサブセット, 256x256 クロップ
    out = net.train_forward(x, q_index)   # ← test() を微分可能な学習版にしたメソッド
    R_y     = out["bit_y"] / num_pixels
    L_align = net.code_pred_loss(out["mu_pred"], out["code_target"])
    L_dist  = mse(x, out["x_hat"]) + lpips(x, out["x_hat"])
    loss = LAMBDA_R * R_y + LAMBDA_ALIGN * L_align + LAMBDA_D * L_dist
    loss.backward(); opt.step(); opt.zero_grad()

# 評価（A/B）: net.prior_predictor.gate を 0 にすれば GLC ベースライン、
#             学習済みゲートで Ours。同一 q_index で bpp と DISTS/FID を比較。
"""


# ============================================================================
#  train_forward: GLC を「微分可能な学習モード」で通す（test() の学習版）
# ----------------------------------------------------------------------------
#  GLC の quant() は STE 丸め（round - x を detach して加算）なので、
#  test() の本体はそのまま微分可能。違いは:
#    - torch.no_grad を外す
#    - prior_predictor で prior に delta_params を加算注入（use_predictor で A/B）
#    - bit_y を .item() せず微分可能なまま返す
# ============================================================================
import math as _math


def train_forward(net, x, q_index, use_predictor: bool = True, gate=None, q_shift=None,
                  predictor_param_mode: str = "scale_mean", predictor_delta_bound: float = 0.0):
    """
    Args:
        net          : GLC_Image インスタンス（net.prior_predictor を attach 済み）
        x            : 入力画像 [-1,1], (B,3,256,256)
        q_index      : 品質インデックス（GLC は 0..3）
        use_predictor: True=Ours（P_θ on）, False=ベースライン（厳密に GLC）
        predictor_param_mode: "latent_residual" は μ(z_hat) を y から明示的に引いて residual を符号化,
                              "scale_mean" は scales/means 補正, "mean" は means のみ, "all" は quant_step/scales/means 全補正
        predictor_delta_bound: >0 のとき delta を bound*tanh(delta/bound) で有界化
    Returns dict:
        bit_y   : 微分可能なビット数（バッチ合計）
        bit_z   : 意味コードのビット（定数, P_θ では不変）
        x_hat   : 再構成（微分可能, [-1,1]）
        mu_pred : μ_θ(s)（use_predictor=False なら None）
        y       : 符号化対象の連続潜在
    """
    curr_q_enc = net.q_enc[q_index:q_index + 1]
    curr_q_dec = net.q_dec[q_index:q_index + 1]

    y_ori = net.vqgan.encoder(x)                       # 凍結
    y = net.enc(y_ori, curr_q_enc)                     # 凍結（解析変換 g_a）
    z = net.hyper_enc(y)                               # 凍結
    index = net.z_vq.get_indices(z)
    z_hat = net.z_vq.get_quan_feat(
        index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))

    params = net.hyper_dec(z_hat)
    params = net.y_prior_fusion(params)                # GLC 側は z_hat をそのまま使う
    params_base = params

    # P_θ / gate 用の q 条件付け（z_hat を q シフト。hyper_dec には影響させない）
    z_cond = z_hat if q_shift is None else z_hat + q_shift

    mu_pred = None
    delta_params = None
    latent_pred_scaled = None
    latent_pred = None
    if use_predictor:
        delta_params, mu_pred, _ = net.prior_predictor(z_cond)
        if predictor_delta_bound and predictor_delta_bound > 0:
            delta_params = predictor_delta_bound * torch.tanh(delta_params / predictor_delta_bound)
        if predictor_param_mode == "latent_residual":
            masked = torch.zeros_like(delta_params)
            masked[:, 2 * net.N:] = delta_params[:, 2 * net.N:]
            delta_params = masked
            _, q_dec_base, _, _ = net.separate_prior(params)
            latent_pred_scaled = delta_params[:, 2 * net.N:]
            latent_pred = latent_pred_scaled * q_dec_base
        elif predictor_param_mode == "mean":
            masked = torch.zeros_like(delta_params)
            masked[:, 2 * net.N:] = delta_params[:, 2 * net.N:]
            delta_params = masked
        elif predictor_param_mode == "scale_mean":
            masked = torch.zeros_like(delta_params)
            masked[:, net.N:] = delta_params[:, net.N:]
            delta_params = masked
        elif predictor_param_mode != "all":
            raise ValueError(f"unknown predictor_param_mode: {predictor_param_mode}")
        if predictor_param_mode != "latent_residual":
            params = params + delta_params             # モジュール①（zero-init ゲート, off で GLC厳密一致）

    gate_rho = None
    gate_p_tex = None
    if gate is not None:                                # モジュール②（任意, V2）
        gate_rho, gate_p_tex = gate(z_cond)
        params = torch.cat((params[:, :net.N] * gate_rho, params[:, net.N:]), dim=1)
    params_after = params

    if latent_pred_scaled is not None:
        y_res, y_q, y_hat, scales_hat = forward_four_part_prior_with_latent_mean(
            net, y, params, latent_pred_scaled)
    else:
        y_res, y_q, y_hat, scales_hat = net.forward_four_part_prior(
            y, params,
            net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
            net.y_spatial_prior_adaptor_3, net.y_spatial_prior,
            y_spatial_prior_reduction=net.y_spatial_prior_reduction)

    y_hat = net.dec(y_hat, curr_q_dec)                 # 合成変換 g_s
    x_hat = net.vqgan.generator(y_hat)                 # 生成デコーダ

    bit_y = net.get_y_gaussian_bits(y_q, scales_hat).sum()          # 微分可能（.item() しない）
    bit_z = z_hat.shape[-2] * z_hat.shape[-1] * _math.log2(net.codebook_size)

    return {"bit_y": bit_y, "bit_z": bit_z, "x_hat": x_hat,
            "mu_pred": mu_pred, "delta_params": delta_params, "y": y,
            "params_base": params_base, "params_after": params_after,
            "latent_pred_scaled": latent_pred_scaled, "latent_pred": latent_pred,
            "gate_rho": gate_rho, "gate_p_tex": gate_p_tex,
            "y_res": y_res, "y_q": y_q, "scales_hat": scales_hat}
