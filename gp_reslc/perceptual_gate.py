# perceptual_gate.py
# ============================================================================
#  モジュール② 知覚重要度ゲート（VCIP Phase V2 / 任意）
# ----------------------------------------------------------------------------
#  追加ビットなしで復号側が s(=z_hat) から決定論的に再計算できる知覚マップ M を作り、
#  GLC の prior の quant_step チャネルを空間変調して bit を配分する。
#
#  GLC の量子化機構（src/models/common_model.py: separate_prior）:
#     quant_step, scales, means = params.chunk(3, 1)
#     quant_step = quant_step.clamp_min(0.5); q_enc = 1/quant_step
#     y = y * q_enc   →  quant_step が大きいほど粗い量子化＝低ビット・高歪み
#  したがって「テクスチャ域（生成器が補完できる）を粗く」するには quant_step を大きくする。
#
#  M は z_hat（= 送信済み意味コードの量子化特徴, 凍結 z_vq）から計算するため、
#  エンコーダ・デコーダで同一 → 追加ビット不要・ビット完全一致。
#
#  ★ V2/任意。まず V1（prior_predictor 単体）で主張①を確定してから着手すること。
#    A-DISTS の分散指数（局所分散/局所平均）で warm-start すると収束が安定。
# ============================================================================

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from src.models.layers import DepthConvBlock, ResidualBlockUpsample
except Exception:
    DepthConvBlock = None
    ResidualBlockUpsample = None


class PerceptualGate(nn.Module):
    """
    z_hat (N×h×w) → テクスチャ確率 p_tex (1×H'×W', y 解像度) → quant_step 倍率 ρ。
    rho_min=1 can forbid bit-increasing rho<1 while preserving zero-init identity.
    """

    def __init__(self, N: int = 256, rho_max: float = 2.0, rho_min: float = 0.5,
                 rho_mode: str = "hard", softplus_shift: float = 2.0,
                 softplus_tau: float = 1.0, rho_init: float = 1.0):
        super().__init__()
        assert ResidualBlockUpsample is not None, "GLC リポジトリ直下で import してください。"
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.rho_mode = rho_mode
        self.softplus_shift = softplus_shift
        self.softplus_tau = softplus_tau
        self.rho_init = rho_init
        self.up = nn.Sequential(
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockUpsample(N, N, 2),
            DepthConvBlock(N, N),
        )
        self.head = nn.Conv2d(N, 1, 1)  # → テクスチャ logit
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # exact-identity rho at start
        if rho_init > 1.0:
            nn.init.constant_(self.head.bias, self._raw_for_rho_init(rho_init))

    def _raw_for_rho_init(self, rho_init: float) -> float:
        if self.rho_max <= 1.0:
            return 0.0
        eps = 1e-6
        target = min(max(float(rho_init), 1.0 + eps), float(self.rho_max) - eps)
        frac = (target - 1.0) / (float(self.rho_max) - 1.0)
        frac = min(max(frac, eps), 1.0 - eps)
        if self.rho_mode == "softplus":
            tau = max(float(self.softplus_tau), eps)
            shift = float(self.softplus_shift)
            base = math.log1p(math.exp(shift))
            excess = -tau * math.log(max(1.0 - frac, eps))
            y = base + excess
            inv_softplus = y + math.log(-math.expm1(-y))
            return inv_softplus - shift
        p_tex = 0.5 * (1.0 + frac)
        p_tex = min(max(p_tex, eps), 1.0 - eps)
        return math.log(p_tex / (1.0 - p_tex))

    def forward(self, z_hat):
        feat = self.up(z_hat)                 # N×H'×W'（y 解像度）
        raw = self.head(feat)
        p_tex = torch.sigmoid(raw)  # logging/visualization map

        if self.rho_mode == "softplus":
            # Exact identity at zero-init with a positive-side gradient. This avoids
            # the dead boundary seen when a hard monotone clamp is trained with a
            # weak rate term. rho is still constrained to [1, rho_max).
            shift = raw.new_tensor(self.softplus_shift)
            tau = max(float(self.softplus_tau), 1e-6)
            excess = F.softplus(raw + shift) - F.softplus(shift)
            excess = excess.clamp_min(0.0)
            rho = 1.0 + (self.rho_max - 1.0) * (1.0 - torch.exp(-excess / tau))
            return rho, p_tex

        rho = 1.0 + (self.rho_max - 1.0) * (2.0 * p_tex - 1.0)
        rho = rho.clamp_min(self.rho_min)
        return rho, p_tex


# ============================================================================
#  GLC_Image / train_forward への統合（prior_predictor の直後に適用）
# ============================================================================
#  __init__:
#     self.perceptual_gate = PerceptualGate(N, rho_max=2.0)
#
#  train_forward / test の prior 生成部（prior_predictor の加算注入の直後）:
#     params = params + delta_params                      # モジュール①
#     rho, p_tex = self.perceptual_gate(z_hat)            # モジュール②
#     params = params.clone()
#     params[:, :self.N] = params[:, :self.N] * rho       # quant_step を空間変調
#     ... = self.forward_four_part_prior(y, params, ...)
#
#  学習: 既存の R-D 損失（λ_R·bpp_y + λ_d·(MSE+LPIPS)）に乗せるだけで、
#        「知覚損失を最も減らす場所へビットを寄せる」配分を自律学習する。
#        rho_max は 1.5〜3.0 で探索。warm-start する場合は A-DISTS 分散指数を
#        p_tex の弱教師（BCE）として数千 iter 付与してから外す。
#
#  検証（アブレーション）: perceptual_gate を恒等（rho=1）に固定した版と比較し、
#        同 bpp で DISTS/LPIPS が改善するか（主張②）を確認する。
# ============================================================================
