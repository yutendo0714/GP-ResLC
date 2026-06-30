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
import torch.nn.functional as F

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


class StageResidualPredictor(nn.Module):
    """Decoder-recomputable mean correction for each four-part prior stage.

    Stage 0 can only use the hyperprior-derived common parameters. Stages 1-3
    can also use y_hat_so_far, which is already decoded at that point. Therefore
    these corrections can be used in the real arithmetic codec without sending
    extra side information.
    """

    def __init__(self, N: int = 256):
        super().__init__()
        assert DepthConvBlock is not None, "GLC リポジトリ直下で import してください。"
        self.stage0 = nn.Sequential(
            DepthConvBlock(N, N),
            zero_module(nn.Conv2d(N, N, 1)),
        )
        self.stages = nn.ModuleList([
            nn.Sequential(
                DepthConvBlock(N * 2, N),
                DepthConvBlock(N, N),
                zero_module(nn.Conv2d(N, N, 1)),
            )
            for _ in range(3)
        ])

    def forward_stage(self, stage_idx: int, common_params: torch.Tensor,
                      y_hat_so_far: torch.Tensor | None = None) -> torch.Tensor:
        if stage_idx == 0:
            return self.stage0(common_params)
        if y_hat_so_far is None:
            raise ValueError("y_hat_so_far is required for stages 1-3")
        return self.stages[stage_idx - 1](torch.cat((y_hat_so_far, common_params), dim=1))


class StageResidualEntropyPredictor(nn.Module):
    """Decoder-recomputable residual mean and scale correction per GLC stage.

    This is the entropy-model extension of :class:`StageResidualPredictor`.
    It keeps the same four-part decoding order, but predicts both

        residual_mean_delta, residual_scale_multiplier

    from decoder-available signals. The final convs are zero initialized, so
    the initial state is exactly delta=0 and scale_multiplier=1.
    """

    def __init__(self, N: int = 256, scale_log_bound: float = 0.7):
        super().__init__()
        assert DepthConvBlock is not None, "GLC リポジトリ直下で import してください。"
        self.N = int(N)
        self.scale_log_bound = float(scale_log_bound)
        self.stage0 = nn.Sequential(
            DepthConvBlock(N, N),
            zero_module(nn.Conv2d(N, N * 2, 1)),
        )
        self.stages = nn.ModuleList([
            nn.Sequential(
                DepthConvBlock(N * 2, N),
                DepthConvBlock(N, N),
                zero_module(nn.Conv2d(N, N * 2, 1)),
            )
            for _ in range(3)
        ])

    def _split(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta, log_scale_raw = raw.chunk(2, dim=1)
        bound = max(float(self.scale_log_bound), 1e-6)
        log_scale = bound * torch.tanh(log_scale_raw / bound)
        scale_mul = torch.exp(log_scale).clamp(0.25, 4.0)
        return delta, scale_mul

    def forward_stage(self, stage_idx: int, common_params: torch.Tensor,
                      y_hat_so_far: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if stage_idx == 0:
            return self._split(self.stage0(common_params))
        if y_hat_so_far is None:
            raise ValueError("y_hat_so_far is required for stages 1-3")
        return self._split(self.stages[stage_idx - 1](torch.cat((y_hat_so_far, common_params), dim=1)))


class StageQuantGate(nn.Module):
    """Stage-aware decoder-recomputable quantization gate.

    The gate predicts rho >= 1 for each four-part prior stage. Larger rho means
    a larger quant_step, so predictable residual locations are sent with lower
    precision without transmitting an additional mask.
    """

    def __init__(self, N: int = 256, rho_max: float = 1.5, softplus_shift: float = 2.0,
                 softplus_tau: float = 1.0):
        super().__init__()
        assert DepthConvBlock is not None, "GLC リポジトリ直下で import してください。"
        self.rho_max = rho_max
        self.softplus_shift = softplus_shift
        self.softplus_tau = softplus_tau
        self.stage0 = nn.Sequential(
            DepthConvBlock(N, N),
            zero_module(nn.Conv2d(N, 1, 1)),
        )
        self.stages = nn.ModuleList([
            nn.Sequential(
                DepthConvBlock(N * 2, N),
                DepthConvBlock(N, N),
                zero_module(nn.Conv2d(N, 1, 1)),
            )
            for _ in range(3)
        ])

    def _rho_from_raw(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p_tex = torch.sigmoid(raw)
        shift = raw.new_tensor(self.softplus_shift)
        tau = max(float(self.softplus_tau), 1e-6)
        excess = F.softplus(raw + shift) - F.softplus(shift)
        excess = excess.clamp_min(0.0)
        rho = 1.0 + (self.rho_max - 1.0) * (1.0 - torch.exp(-excess / tau))
        return rho, p_tex

    def forward_stage(self, stage_idx: int, common_params: torch.Tensor,
                      y_hat_so_far: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if stage_idx == 0:
            raw = self.stage0(common_params)
        else:
            if y_hat_so_far is None:
                raise ValueError("y_hat_so_far is required for stages 1-3")
            raw = self.stages[stage_idx - 1](torch.cat((y_hat_so_far, common_params), dim=1))
        return self._rho_from_raw(raw)


class StageTinyControlEncoder(nn.Module):
    """Encoder-only sparse control symbols for stage precision repair.

    The decoder cannot recompute this map, so the binary symbols must be sent
    and counted. The map is intentionally predicted at z resolution and then
    nearest-upsampled to y resolution, keeping the payload in the
    sub-millibpp-to-few-millibpp regime on natural images.

    A symbol value of 1 means "protect this coarse region": reduce the
    decoder-computable rho back toward 1.0. This lets the base stage gate stay
    aggressive while a tiny paid stream fixes unpredictable unsafe regions.
    """

    def __init__(
        self,
        N: int = 256,
        num_q: int = 4,
        init_prob: float = 0.05,
        threshold: float = 0.5,
        hard_mode: str = "threshold",
        topk_frac: float = 0.06,
    ):
        super().__init__()
        assert DepthConvBlock is not None, "GLC リポジトリ直下で import してください。"
        init_prob = min(max(float(init_prob), 1e-4), 1.0 - 1e-4)
        init_logit = torch.logit(torch.tensor(init_prob)).item()
        self.threshold = float(threshold)
        self.hard_mode = str(hard_mode)
        self.topk_frac = float(topk_frac)
        self.body = nn.Sequential(
            DepthConvBlock(N * 2, N),
            DepthConvBlock(N, N),
        )
        self.to_logits = nn.Conv2d(N, 4, 1)
        nn.init.zeros_(self.to_logits.weight)
        nn.init.constant_(self.to_logits.bias, init_logit)
        self.q_bias = nn.Parameter(torch.zeros(num_q, 4, 1, 1))

    def forward(
        self,
        y: torch.Tensor,
        common_params: torch.Tensor,
        z_hw: tuple[int, int],
        q_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.body(torch.cat((y, common_params), dim=1))
        logits_y = self.to_logits(feat)
        logits = F.adaptive_avg_pool2d(logits_y, z_hw)
        q_idx = max(0, min(int(q_index), self.q_bias.shape[0] - 1))
        logits = logits + self.q_bias[q_idx:q_idx + 1]
        prob = torch.sigmoid(logits)
        if self.hard_mode == "topk":
            flat = prob.flatten(1)
            k = int(round(max(0.0, min(self.topk_frac, 1.0)) * flat.shape[1]))
            if k <= 0:
                hard = torch.zeros_like(prob)
            else:
                k = min(k, flat.shape[1])
                idx = torch.topk(flat, k=k, dim=1).indices
                hard_flat = torch.zeros_like(flat)
                hard_flat.scatter_(1, idx, 1.0)
                hard = hard_flat.reshape_as(prob)
        elif self.hard_mode == "threshold":
            hard = (prob > self.threshold).to(prob.dtype)
        else:
            raise ValueError(f"unknown tiny-control hard_mode: {self.hard_mode}")
        symbols = prob + (hard - prob).detach()
        return symbols, prob, logits


def bernoulli_nll_bits(symbols: torch.Tensor, prob_one: float) -> torch.Tensor:
    """Differentiable fixed-prior Bernoulli code length for control symbols."""
    p1 = min(max(float(prob_one), 1e-5), 1.0 - 1e-5)
    p0 = 1.0 - p1
    c = symbols.clamp(0.0, 1.0)
    return -(c * torch.log2(c.new_tensor(p1)) + (1.0 - c) * torch.log2(c.new_tensor(p0))).sum()


def _control_stage_to_y(control_maps: torch.Tensor, stage_idx: int, h: int, w: int) -> torch.Tensor:
    ctrl = control_maps[:, stage_idx:stage_idx + 1]
    return F.interpolate(ctrl, size=(h, w), mode="nearest")


def _apply_stage_q_condition(common_params: torch.Tensor, q_shift: torch.Tensor | None) -> torch.Tensor:
    if q_shift is None:
        return common_params
    return common_params + q_shift.to(device=common_params.device, dtype=common_params.dtype)


def _stage_delta_and_scales(
    stage_predictor,
    stage_idx: int,
    common_params: torch.Tensor,
    y_hat_so_far: torch.Tensor | None,
    base_scales: torch.Tensor,
    predictor_delta_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stage_idx == 0:
        pred = stage_predictor.forward_stage(0, common_params)
    else:
        pred = stage_predictor.forward_stage(stage_idx, common_params, y_hat_so_far)
    if isinstance(pred, tuple):
        delta, scale_mul = pred
        scales = (base_scales * scale_mul).clamp_min(1e-5)
    else:
        delta = pred
        scales = base_scales
    return _bound_delta(delta, predictor_delta_bound), scales


def forward_four_part_prior_with_stage_quant_gate(net, y, common_params, stage_gate,
                                                  q_shift: torch.Tensor | None = None):
    """GLC four-part prior with stage-aware quant_step modulation."""
    quant_step, scales, means = common_params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    if net.y_spatial_prior_reduction is not None:
        common_params_reduced = net.y_spatial_prior_reduction(common_params)
    else:
        common_params_reduced = common_params
    common_params_reduced = _apply_stage_q_condition(common_params_reduced, q_shift)

    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = net.get_mask_four_parts(B, C, H, W, dtype, device)

    rho_0, p_0 = stage_gate.forward_stage(0, common_params_reduced)
    q_enc_0 = 1.0 / (quant_step * rho_0)
    q_dec_map = quant_step * rho_0 * mask_0
    y_res_0, y_q_0, y_hat_0, s_hat_0 = net.process_with_mask(
        y * q_enc_0, scales, means, mask_0)

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    rho_1, p_1 = stage_gate.forward_stage(1, common_params_reduced, y_hat_so_far)
    q_enc_1 = 1.0 / (quant_step * rho_1)
    q_dec_map = q_dec_map + quant_step * rho_1 * mask_1
    y_res_1, y_q_1, y_hat_1, s_hat_1 = net.process_with_mask(
        y * q_enc_1, scales, means, mask_1)

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    rho_2, p_2 = stage_gate.forward_stage(2, common_params_reduced, y_hat_so_far)
    q_enc_2 = 1.0 / (quant_step * rho_2)
    q_dec_map = q_dec_map + quant_step * rho_2 * mask_2
    y_res_2, y_q_2, y_hat_2, s_hat_2 = net.process_with_mask(
        y * q_enc_2, scales, means, mask_2)

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    rho_3, p_3 = stage_gate.forward_stage(3, common_params_reduced, y_hat_so_far)
    q_enc_3 = 1.0 / (quant_step * rho_3)
    q_dec_map = q_dec_map + quant_step * rho_3 * mask_3
    y_res_3, y_q_3, y_hat_3, s_hat_3 = net.process_with_mask(
        y * q_enc_3, scales, means, mask_3)

    y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
    y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
    y_hat = y_hat_so_far + y_hat_3
    scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
    rho_map = rho_0 * mask_0 + rho_1 * mask_1 + rho_2 * mask_2 + rho_3 * mask_3
    p_map = p_0 * mask_0 + p_1 * mask_1 + p_2 * mask_2 + p_3 * mask_3
    y_hat = y_hat * q_dec_map
    return y_res, y_q, y_hat, scales_hat, rho_map, p_map


def forward_four_part_prior_with_stage_residual_quant_gate(
    net,
    y,
    common_params,
    stage_predictor,
    stage_gate,
    predictor_delta_bound: float = 0.0,
    q_shift: torch.Tensor | None = None,
):
    """GLC four-part prior with stage-aware residual means and quant gates.

    This mode is the closest pretrained-GLC implementation of the main
    GP-ResLC thesis so far:

      y_stage = base_mean_stage + gp_mu_stage(context) + residual_stage

    while the residual precision can still be reduced by a decoder-computable
    stage gate. Both `gp_mu_stage` and `rho_stage` use only information that is
    available at the corresponding GLC four-part decoding stage.
    """
    quant_step, scales, means = common_params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    if net.y_spatial_prior_reduction is not None:
        common_params_reduced = net.y_spatial_prior_reduction(common_params)
    else:
        common_params_reduced = common_params
    common_params_reduced = _apply_stage_q_condition(common_params_reduced, q_shift)

    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = net.get_mask_four_parts(B, C, H, W, dtype, device)

    stage_pred_losses = []
    stage_pred_abs_vals = []
    stage_target_abs_vals = []

    def add_stage_stats(delta, target, mask):
        active = mask > 0.5
        delta_active = delta[active]
        target_active = target.detach()[active]
        stage_pred_losses.append(F.smooth_l1_loss(delta_active, target_active))
        stage_pred_abs_vals.append(delta_active.detach().abs().mean())
        stage_target_abs_vals.append(target_active.detach().abs().mean())

    rho_0, p_0 = stage_gate.forward_stage(0, common_params_reduced)
    q_enc_0 = 1.0 / (quant_step * rho_0)
    y_scaled_0 = y * q_enc_0
    q_dec_map = quant_step * rho_0 * mask_0
    delta_0, scales_0 = _stage_delta_and_scales(
        stage_predictor, 0, common_params_reduced, None, scales, predictor_delta_bound)
    add_stage_stats(delta_0, y_scaled_0 - means, mask_0)
    y_res_0, y_q_0, y_hat_0, s_hat_0 = net.process_with_mask(
        y_scaled_0, scales_0, means + delta_0, mask_0)

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    rho_1, p_1 = stage_gate.forward_stage(1, common_params_reduced, y_hat_so_far)
    q_enc_1 = 1.0 / (quant_step * rho_1)
    y_scaled_1 = y * q_enc_1
    q_dec_map = q_dec_map + quant_step * rho_1 * mask_1
    delta_1, scales_1 = _stage_delta_and_scales(
        stage_predictor, 1, common_params_reduced, y_hat_so_far, scales, predictor_delta_bound)
    add_stage_stats(delta_1, y_scaled_1 - means, mask_1)
    y_res_1, y_q_1, y_hat_1, s_hat_1 = net.process_with_mask(
        y_scaled_1, scales_1, means + delta_1, mask_1)

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    rho_2, p_2 = stage_gate.forward_stage(2, common_params_reduced, y_hat_so_far)
    q_enc_2 = 1.0 / (quant_step * rho_2)
    y_scaled_2 = y * q_enc_2
    q_dec_map = q_dec_map + quant_step * rho_2 * mask_2
    delta_2, scales_2 = _stage_delta_and_scales(
        stage_predictor, 2, common_params_reduced, y_hat_so_far, scales, predictor_delta_bound)
    add_stage_stats(delta_2, y_scaled_2 - means, mask_2)
    y_res_2, y_q_2, y_hat_2, s_hat_2 = net.process_with_mask(
        y_scaled_2, scales_2, means + delta_2, mask_2)

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    rho_3, p_3 = stage_gate.forward_stage(3, common_params_reduced, y_hat_so_far)
    q_enc_3 = 1.0 / (quant_step * rho_3)
    y_scaled_3 = y * q_enc_3
    q_dec_map = q_dec_map + quant_step * rho_3 * mask_3
    delta_3, scales_3 = _stage_delta_and_scales(
        stage_predictor, 3, common_params_reduced, y_hat_so_far, scales, predictor_delta_bound)
    add_stage_stats(delta_3, y_scaled_3 - means, mask_3)
    y_res_3, y_q_3, y_hat_3, s_hat_3 = net.process_with_mask(
        y_scaled_3, scales_3, means + delta_3, mask_3)

    y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
    y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
    y_hat = y_hat_so_far + y_hat_3
    scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
    rho_map = rho_0 * mask_0 + rho_1 * mask_1 + rho_2 * mask_2 + rho_3 * mask_3
    p_map = p_0 * mask_0 + p_1 * mask_1 + p_2 * mask_2 + p_3 * mask_3
    y_hat = y_hat * q_dec_map
    stage_delta_abs = torch.stack(stage_pred_abs_vals).mean()
    stage_target_abs = torch.stack(stage_target_abs_vals).mean()
    stage_mean_pred_loss = torch.stack(stage_pred_losses).mean()
    return (
        y_res,
        y_q,
        y_hat,
        scales_hat,
        rho_map,
        p_map,
        stage_delta_abs,
        stage_target_abs,
        stage_mean_pred_loss,
    )


def forward_four_part_prior_with_stage_residual_quant_gate_control(
    net,
    y,
    common_params,
    stage_predictor,
    stage_gate,
    control_maps,
    predictor_delta_bound: float = 0.0,
    q_shift: torch.Tensor | None = None,
):
    """Stage residual/quant gate with a tiny counted protection stream.

    `control_maps` has shape Bx4xHzxWz. It is transmitted in the real codec.
    At each stage, control=1 moves the effective rho toward 1.0:

        rho_eff = 1 + (rho_base - 1) * (1 - control)

    so the paid stream is used only to protect regions where decoder-only
    coarsening would damage quality.
    """
    quant_step, scales, means = common_params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    if net.y_spatial_prior_reduction is not None:
        common_params_reduced = net.y_spatial_prior_reduction(common_params)
    else:
        common_params_reduced = common_params
    common_params_reduced = _apply_stage_q_condition(common_params_reduced, q_shift)

    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = net.get_mask_four_parts(B, C, H, W, dtype, device)

    stage_pred_losses = []
    stage_pred_abs_vals = []
    stage_target_abs_vals = []

    def add_stage_stats(delta, target, mask):
        active = mask > 0.5
        delta_active = delta[active]
        target_active = target.detach()[active]
        stage_pred_losses.append(F.smooth_l1_loss(delta_active, target_active))
        stage_pred_abs_vals.append(delta_active.detach().abs().mean())
        stage_target_abs_vals.append(target_active.detach().abs().mean())

    def protect_rho(rho, stage_idx):
        ctrl = _control_stage_to_y(control_maps, stage_idx, H, W).to(dtype=rho.dtype, device=rho.device)
        return 1.0 + (rho - 1.0) * (1.0 - ctrl)

    rho_0, p_0 = stage_gate.forward_stage(0, common_params_reduced)
    rho_0 = protect_rho(rho_0, 0)
    q_enc_0 = 1.0 / (quant_step * rho_0)
    y_scaled_0 = y * q_enc_0
    q_dec_map = quant_step * rho_0 * mask_0
    delta_0, scales_0 = _stage_delta_and_scales(
        stage_predictor, 0, common_params_reduced, None, scales, predictor_delta_bound)
    add_stage_stats(delta_0, y_scaled_0 - means, mask_0)
    y_res_0, y_q_0, y_hat_0, s_hat_0 = net.process_with_mask(
        y_scaled_0, scales_0, means + delta_0, mask_0)

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    rho_1, p_1 = stage_gate.forward_stage(1, common_params_reduced, y_hat_so_far)
    rho_1 = protect_rho(rho_1, 1)
    q_enc_1 = 1.0 / (quant_step * rho_1)
    y_scaled_1 = y * q_enc_1
    q_dec_map = q_dec_map + quant_step * rho_1 * mask_1
    delta_1, scales_1 = _stage_delta_and_scales(
        stage_predictor, 1, common_params_reduced, y_hat_so_far, scales, predictor_delta_bound)
    add_stage_stats(delta_1, y_scaled_1 - means, mask_1)
    y_res_1, y_q_1, y_hat_1, s_hat_1 = net.process_with_mask(
        y_scaled_1, scales_1, means + delta_1, mask_1)

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    rho_2, p_2 = stage_gate.forward_stage(2, common_params_reduced, y_hat_so_far)
    rho_2 = protect_rho(rho_2, 2)
    q_enc_2 = 1.0 / (quant_step * rho_2)
    y_scaled_2 = y * q_enc_2
    q_dec_map = q_dec_map + quant_step * rho_2 * mask_2
    delta_2, scales_2 = _stage_delta_and_scales(
        stage_predictor, 2, common_params_reduced, y_hat_so_far, scales, predictor_delta_bound)
    add_stage_stats(delta_2, y_scaled_2 - means, mask_2)
    y_res_2, y_q_2, y_hat_2, s_hat_2 = net.process_with_mask(
        y_scaled_2, scales_2, means + delta_2, mask_2)

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    rho_3, p_3 = stage_gate.forward_stage(3, common_params_reduced, y_hat_so_far)
    rho_3 = protect_rho(rho_3, 3)
    q_enc_3 = 1.0 / (quant_step * rho_3)
    y_scaled_3 = y * q_enc_3
    q_dec_map = q_dec_map + quant_step * rho_3 * mask_3
    delta_3, scales_3 = _stage_delta_and_scales(
        stage_predictor, 3, common_params_reduced, y_hat_so_far, scales, predictor_delta_bound)
    add_stage_stats(delta_3, y_scaled_3 - means, mask_3)
    y_res_3, y_q_3, y_hat_3, s_hat_3 = net.process_with_mask(
        y_scaled_3, scales_3, means + delta_3, mask_3)

    y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
    y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
    y_hat = y_hat_so_far + y_hat_3
    scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
    rho_map = rho_0 * mask_0 + rho_1 * mask_1 + rho_2 * mask_2 + rho_3 * mask_3
    p_map = p_0 * mask_0 + p_1 * mask_1 + p_2 * mask_2 + p_3 * mask_3
    y_hat = y_hat * q_dec_map
    stage_delta_abs = torch.stack(stage_pred_abs_vals).mean()
    stage_target_abs = torch.stack(stage_target_abs_vals).mean()
    stage_mean_pred_loss = torch.stack(stage_pred_losses).mean()
    return (
        y_res,
        y_q,
        y_hat,
        scales_hat,
        rho_map,
        p_map,
        stage_delta_abs,
        stage_target_abs,
        stage_mean_pred_loss,
    )


def _bound_delta(delta: torch.Tensor, bound: float) -> torch.Tensor:
    if bound and bound > 0:
        return bound * torch.tanh(delta / bound)
    return delta


def forward_four_part_prior_with_stage_residual(net, y, common_params, stage_predictor,
                                                predictor_delta_bound: float = 0.0,
                                                q_shift: torch.Tensor | None = None):
    """GLC four-part prior with stage-aware decoder-recomputable residual means."""
    q_enc, q_dec, scales, means = net.separate_prior(common_params)
    if net.y_spatial_prior_reduction is not None:
        common_params_reduced = net.y_spatial_prior_reduction(common_params)
    else:
        common_params_reduced = common_params
    common_params_reduced = _apply_stage_q_condition(common_params_reduced, q_shift)

    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = net.get_mask_four_parts(B, C, H, W, dtype, device)
    y_scaled = y * q_enc
    stage_pred_losses = []
    stage_pred_abs_vals = []
    stage_target_abs_vals = []

    def add_stage_stats(delta, target, mask):
        active = mask > 0.5
        delta_active = delta[active]
        target_active = target.detach()[active]
        stage_pred_losses.append(F.smooth_l1_loss(delta_active, target_active))
        stage_pred_abs_vals.append(delta_active.detach().abs().mean())
        stage_target_abs_vals.append(target_active.detach().abs().mean())

    delta_0 = _bound_delta(stage_predictor.forward_stage(0, common_params_reduced), predictor_delta_bound)
    add_stage_stats(delta_0, y_scaled - means, mask_0)
    y_res_0, y_q_0, y_hat_0, s_hat_0 = net.process_with_mask(
        y_scaled, scales, means + delta_0, mask_0)

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    delta_1 = _bound_delta(stage_predictor.forward_stage(1, common_params_reduced, y_hat_so_far), predictor_delta_bound)
    add_stage_stats(delta_1, y_scaled - means, mask_1)
    y_res_1, y_q_1, y_hat_1, s_hat_1 = net.process_with_mask(
        y_scaled, scales, means + delta_1, mask_1)

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    delta_2 = _bound_delta(stage_predictor.forward_stage(2, common_params_reduced, y_hat_so_far), predictor_delta_bound)
    add_stage_stats(delta_2, y_scaled - means, mask_2)
    y_res_2, y_q_2, y_hat_2, s_hat_2 = net.process_with_mask(
        y_scaled, scales, means + delta_2, mask_2)

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params_reduced), dim=1)
    scales, means = net.y_spatial_prior(net.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    delta_3 = _bound_delta(stage_predictor.forward_stage(3, common_params_reduced, y_hat_so_far), predictor_delta_bound)
    add_stage_stats(delta_3, y_scaled - means, mask_3)
    y_res_3, y_q_3, y_hat_3, s_hat_3 = net.process_with_mask(
        y_scaled, scales, means + delta_3, mask_3)

    y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
    y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
    y_hat = y_hat_so_far + y_hat_3
    scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
    y_hat = y_hat * q_dec
    stage_delta_abs = torch.stack(stage_pred_abs_vals).mean()
    stage_target_abs = torch.stack(stage_target_abs_vals).mean()
    stage_mean_pred_loss = torch.stack(stage_pred_losses).mean()
    return y_res, y_q, y_hat, scales_hat, stage_delta_abs, stage_target_abs, stage_mean_pred_loss



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
    stage_delta_abs = None
    stage_target_abs = None
    stage_mean_pred_loss = None
    use_stage_residual = use_predictor and predictor_param_mode == "stage_latent_residual"
    use_stage_quant_gate = use_predictor and predictor_param_mode == "stage_quant_gate"
    use_stage_residual_quant_gate = (
        use_predictor and predictor_param_mode == "stage_residual_quant_gate")
    use_stage_residual_entropy_quant_gate = (
        use_predictor and predictor_param_mode == "stage_residual_entropy_quant_gate")
    use_stage_residual_entropy_quant_gate_control = (
        use_predictor and predictor_param_mode == "stage_residual_entropy_quant_gate_control")
    use_stage_residual_quant_gate_control = (
        use_predictor and predictor_param_mode == "stage_residual_quant_gate_control")
    if (use_predictor and not use_stage_residual and not use_stage_quant_gate
            and not use_stage_residual_quant_gate
            and not use_stage_residual_entropy_quant_gate
            and not use_stage_residual_entropy_quant_gate_control
            and not use_stage_residual_quant_gate_control):
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

    bit_control = y.new_tensor(0.0)
    control_symbols = None
    control_prob = None
    if use_stage_residual_quant_gate_control or use_stage_residual_entropy_quant_gate_control:
        if not hasattr(net, "tiny_control_encoder") or net.tiny_control_encoder is None:
            raise ValueError(f"{predictor_param_mode} requires net.tiny_control_encoder")
        if net.y_spatial_prior_reduction is not None:
            common_for_control = net.y_spatial_prior_reduction(params)
        else:
            common_for_control = params
        common_for_control = _apply_stage_q_condition(common_for_control, q_shift)
        control_symbols, control_prob, _ = net.tiny_control_encoder(
            y, common_for_control, (z_hat.shape[-2], z_hat.shape[-1]), int(q_index))
        bit_control = bernoulli_nll_bits(
            control_symbols, getattr(net, "tiny_control_prob_one", 0.08))
        (y_res, y_q, y_hat, scales_hat, gate_rho, gate_p_tex, stage_delta_abs,
         stage_target_abs, stage_mean_pred_loss) = forward_four_part_prior_with_stage_residual_quant_gate_control(
            net, y, params, net.stage_residual_predictor, net.stage_quant_gate,
            control_symbols, predictor_delta_bound, q_shift=q_shift)
    elif use_stage_residual_quant_gate or use_stage_residual_entropy_quant_gate:
        (y_res, y_q, y_hat, scales_hat, gate_rho, gate_p_tex, stage_delta_abs,
         stage_target_abs, stage_mean_pred_loss) = forward_four_part_prior_with_stage_residual_quant_gate(
            net, y, params, net.stage_residual_predictor, net.stage_quant_gate,
            predictor_delta_bound, q_shift=q_shift)
    elif use_stage_quant_gate:
        y_res, y_q, y_hat, scales_hat, gate_rho, gate_p_tex = forward_four_part_prior_with_stage_quant_gate(
            net, y, params, net.stage_quant_gate, q_shift=q_shift)
    elif use_stage_residual:
        (y_res, y_q, y_hat, scales_hat, stage_delta_abs,
         stage_target_abs, stage_mean_pred_loss) = forward_four_part_prior_with_stage_residual(
            net, y, params, net.stage_residual_predictor, predictor_delta_bound, q_shift=q_shift)
    elif latent_pred_scaled is not None:
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
            "control_symbols": control_symbols, "control_prob": control_prob,
            "bit_control": bit_control,
            "stage_delta_abs": stage_delta_abs,
            "stage_target_abs": stage_target_abs,
            "stage_mean_pred_loss": stage_mean_pred_loss,
            "y_res": y_res, "y_q": y_q, "scales_hat": scales_hat}
