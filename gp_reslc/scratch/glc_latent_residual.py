"""GLC-latent residual decomposition for the scratch GP-ResLC branch.

This branch uses a frozen, pretrained GLC/VQGAN generator as the synthesis
prior, while learning a cheap semantic code and a low-dimensional residual
latent from scratch. The transmitted information is:

1. a fixed-rate semantic VQ index grid from Stage A, and
2. an entropy-modeled residual that corrects only the part of the GLC latent
   that cannot be predicted from that semantic code.

The module is intentionally small and uses the existing Stage-A VQ encoder as
the semantic source. It is a research scaffold for the full-design path, not a
replacement for the current real-codec GLC branch yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .vq_autoencoder import ResBlock, _norm_groups
from .residual_autoencoder import gaussian_bits


def gaussian_bits_stable(symbols: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Gaussian bits with a quadratic tail fallback instead of flat probability clamping."""
    safe_scales = scales.clamp_min(1e-4)
    inv_std = 1.0 / safe_scales
    upper = (symbols + 0.5) * inv_std
    lower = (symbols - 0.5) * inv_std
    normalizer = math.sqrt(0.5)
    probs = 0.5 * (torch.erf(upper * normalizer) - torch.erf(lower * normalizer))
    cdf_bits = -torch.log2(probs.clamp_min(1e-12))
    tail_start = (symbols.abs() - 0.5).clamp_min(0.0) * inv_std
    tail_bits = (0.5 * tail_start.pow(2) + torch.log(safe_scales * math.sqrt(2.0 * math.pi))) / math.log(2.0)
    tail_bits = tail_bits.clamp_min(0.0)
    return torch.where(probs > 1e-12, cdf_bits, tail_bits)


class _UpsamplePredictor(nn.Module):
    def __init__(self, semantic_dim: int, hidden_dim: int, target_dim: int):
        super().__init__()
        self.in_conv = nn.Conv2d(semantic_dim, hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden_dim, target_dim, 3, padding=1)

    def forward(self, z_s: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.in_conv(z_s)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.out_conv(self.blocks(x))


class _ResidualEncoder(nn.Module):
    def __init__(self, semantic_dim: int, target_dim: int, residual_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(target_dim * 2 + semantic_dim, hidden_dim, 3, padding=1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, residual_dim, 3, padding=1),
        )

    def forward(self, target_latent: torch.Tensor, mu: torch.Tensor, z_up: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([target_latent, mu, z_up], dim=1))


class _ResidualDecoder(nn.Module):
    def __init__(self, semantic_dim: int, target_dim: int, residual_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(target_dim + semantic_dim + residual_dim, hidden_dim, 3, padding=1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, target_dim, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q_residual: torch.Tensor, mu: torch.Tensor, z_up: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q_residual, mu, z_up], dim=1))


class _ScaleNet(nn.Module):
    def __init__(self, semantic_dim: int, target_dim: int, residual_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(semantic_dim + target_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, residual_dim, 3, padding=1),
        )

    def forward(self, z_up: torch.Tensor, mu: torch.Tensor, scale_floor: float) -> torch.Tensor:
        return F.softplus(self.net(torch.cat([z_up, mu], dim=1))) + float(scale_floor)


class _ResidualSelectorNet(nn.Module):
    """Encoder-side residual-value predictor for sparse symbol selection.

    The selector is not decoded and sends no side map. It is used only by the
    encoder to rank candidate residual symbols before the fixed top-k budget is
    arithmetic-coded.
    """

    def __init__(self, semantic_dim: int, target_dim: int, residual_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(target_dim * 2 + semantic_dim + residual_dim * 2, hidden_dim, 3, padding=1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, residual_dim, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        target_latent: torch.Tensor,
        mu: torch.Tensor,
        z_up: torch.Tensor,
        symbols: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([target_latent, mu, z_up, symbols, scales], dim=1)
        return self.net(x)


class _DeltaScaleNet(nn.Module):
    def __init__(self, semantic_dim: int, target_dim: int, residual_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(semantic_dim + target_dim + residual_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        q_residual: torch.Tensor,
        mu: torch.Tensor,
        z_up: torch.Tensor,
        scale_min: float,
        scale_max: float,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.net(torch.cat([q_residual, mu, z_up], dim=1)))
        return float(scale_min) + (float(scale_max) - float(scale_min)) * gate


@dataclass
class GLCLatentResidualOutput:
    latent_hat: torch.Tensor
    mu: torch.Tensor
    residual_latent: torch.Tensor
    q_symbols: torch.Tensor
    residual_delta: torch.Tensor
    scales: torch.Tensor
    residual_bpp: torch.Tensor
    pred_loss: torch.Tensor
    latent_loss: torch.Tensor


class GLCLatentResidualBottleneck(nn.Module):
    """Predict GLC/VQGAN latent from semantic code and entropy-code residuals."""

    def __init__(
        self,
        semantic_dim: int,
        target_dim: int = 256,
        residual_dim: int = 24,
        hidden_dim: int = 256,
        quant_step: float = 0.5,
        scale_floor: float = 0.11,
    ):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.target_dim = int(target_dim)
        self.residual_dim = int(residual_dim)
        self.hidden_dim = int(hidden_dim)
        self.quant_step = float(quant_step)
        self.scale_floor = float(scale_floor)

        self.predictor = _UpsamplePredictor(self.semantic_dim, self.hidden_dim, self.target_dim)
        self.residual_encoder = _ResidualEncoder(self.semantic_dim, self.target_dim, self.residual_dim, self.hidden_dim)
        self.residual_decoder = _ResidualDecoder(self.semantic_dim, self.target_dim, self.residual_dim, self.hidden_dim)
        self.residual_decoder_stage1 = _ResidualDecoder(self.semantic_dim, self.target_dim, self.residual_dim, self.hidden_dim)
        self.residual_decoder_stage2 = _ResidualDecoder(self.semantic_dim, self.target_dim, self.residual_dim, self.hidden_dim)
        self.scale_net = _ScaleNet(self.semantic_dim, self.target_dim, self.residual_dim, self.hidden_dim)
        self.selector_net = _ResidualSelectorNet(self.semantic_dim, self.target_dim, self.residual_dim, self.hidden_dim)
        self.delta_scale_net = _DeltaScaleNet(self.semantic_dim, self.target_dim, self.residual_dim, self.hidden_dim)

    def forward(
        self,
        z_s: torch.Tensor,
        target_latent: torch.Tensor,
        *,
        use_residual: bool = True,
        quant_mode: str = "noise",
        delta_gate_mode: str = "none",
        force_topk_frac: float = 0.0,
        hard_topk: bool = False,
        entropy_mode: str = "clamped",
        max_symbol_abs: float = 0.0,
        delta_scale: float = 1.0,
        adaptive_delta_scale: bool = False,
        delta_scale_min: float = 0.0,
        delta_scale_max: float = 1.0,
        progressive_residual: bool = False,
        stage1_channels: int = 0,
        stage1_delta_scale: float = 1.0,
        stage2_delta_scale: float = 1.0,
        progressive_stage_topk: bool = False,
        stage1_topk_frac: float = 0.0,
        stage2_topk_frac: float = 0.0,
        topk_score_mode: str = "abs",
    ) -> dict[str, torch.Tensor]:
        target_size = (target_latent.shape[-2], target_latent.shape[-1])
        z_up = F.interpolate(z_s, size=target_size, mode="bilinear", align_corners=False)
        mu = self.predictor(z_s, target_size)
        scales = self.scale_net(z_up, mu.detach(), self.scale_floor)
        stage1_split = max(1, min(self.residual_dim - 1, int(stage1_channels) if stage1_channels > 0 else self.residual_dim // 2))

        if use_residual:
            residual_latent = self.residual_encoder(target_latent, mu.detach(), z_up)
            symbols = residual_latent / self.quant_step
            rounded_symbols = symbols.round()
            if topk_score_mode == "abs":
                topk_scores = symbols.detach().abs()
            elif topk_score_mode == "latent_error":
                spatial_error = (target_latent.detach() - mu.detach()).abs().mean(dim=1, keepdim=True)
                topk_scores = symbols.detach().abs() * spatial_error
            elif topk_score_mode == "latent_error_sq":
                spatial_error = (target_latent.detach() - mu.detach()).abs().mean(dim=1, keepdim=True)
                topk_scores = symbols.detach().abs() * spatial_error.pow(2)
            elif topk_score_mode == "learned_selector":
                topk_scores = self.selector_net(
                    target_latent.detach(),
                    mu.detach(),
                    z_up.detach(),
                    symbols.detach(),
                    scales.detach(),
                )
            elif topk_score_mode in {"latent_grad", "latent_grad_improve"}:
                if progressive_residual:
                    raise ValueError("latent_grad top-k is currently implemented for the single residual decoder only")
                with torch.enable_grad():
                    probe_symbols = torch.zeros_like(symbols.detach(), requires_grad=True)
                    probe_residual = probe_symbols * self.quant_step
                    probe_delta = self.residual_decoder(probe_residual, mu.detach(), z_up.detach())
                    if delta_gate_mode == "zero_center":
                        probe_delta = probe_delta - self.residual_decoder(torch.zeros_like(probe_residual), mu.detach(), z_up.detach())
                    elif delta_gate_mode in {"payload_ste", "payload_hard"}:
                        probe_activity = (probe_symbols.abs().sum(dim=1, keepdim=True) > 0).to(probe_delta.dtype)
                        probe_delta = probe_delta * probe_activity
                    elif delta_gate_mode != "none":
                        raise ValueError(f"unknown delta_gate_mode: {delta_gate_mode}")
                    probe_loss = F.smooth_l1_loss(mu.detach() + probe_delta, target_latent.detach())
                    probe_grad = torch.autograd.grad(probe_loss, probe_symbols, retain_graph=False, create_graph=False)[0]
                candidate_symbols = rounded_symbols.detach()
                forced_sign = torch.where(symbols.detach() >= 0, torch.ones_like(symbols), -torch.ones_like(symbols))
                candidate_symbols = torch.where(candidate_symbols == 0, forced_sign, candidate_symbols)
                first_order_delta = probe_grad.detach() * candidate_symbols
                if topk_score_mode == "latent_grad_improve":
                    topk_scores = (-first_order_delta).clamp_min(0.0)
                else:
                    topk_scores = first_order_delta.abs()
            else:
                raise ValueError(f"unknown topk_score_mode: {topk_score_mode}")
            selector_scores = topk_scores
            topk_mask = None
            if progressive_residual and progressive_stage_topk:
                s1_frac = float(stage1_topk_frac) if stage1_topk_frac > 0 else float(force_topk_frac)
                s2_frac = float(stage2_topk_frac) if stage2_topk_frac > 0 else float(force_topk_frac)
                topk_mask = torch.zeros_like(symbols, dtype=torch.bool)
                selected_any = False
                for c0, c1, frac in ((0, stage1_split, s1_frac), (stage1_split, self.residual_dim, s2_frac)):
                    if frac <= 0:
                        continue
                    part_symbols = symbols[:, c0:c1]
                    flat_scores = topk_scores[:, c0:c1].flatten(1)
                    k = max(1, min(flat_scores.shape[1], int(flat_scores.shape[1] * frac)))
                    topk_idx = flat_scores.topk(k, dim=1).indices
                    part_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
                    part_mask.scatter_(1, topk_idx, True)
                    part_mask = part_mask.view_as(part_symbols)
                    forced_sign = torch.where(part_symbols >= 0, torch.ones_like(part_symbols), -torch.ones_like(part_symbols))
                    rounded_part = rounded_symbols[:, c0:c1]
                    rounded_symbols[:, c0:c1] = torch.where(part_mask & (rounded_part == 0), forced_sign, rounded_part)
                    topk_mask[:, c0:c1] = part_mask
                    selected_any = True
                if not selected_any:
                    topk_mask = None
            elif force_topk_frac > 0:
                flat_scores = topk_scores.flatten(1)
                k = max(1, min(flat_scores.shape[1], int(flat_scores.shape[1] * float(force_topk_frac))))
                topk_idx = flat_scores.topk(k, dim=1).indices
                topk_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
                topk_mask.scatter_(1, topk_idx, True)
                topk_mask = topk_mask.view_as(symbols)
                forced_sign = torch.where(symbols >= 0, torch.ones_like(symbols), -torch.ones_like(symbols))
                rounded_symbols = torch.where(topk_mask & (rounded_symbols == 0), forced_sign, rounded_symbols)
            if hard_topk and topk_mask is not None:
                rounded_symbols = torch.where(topk_mask, rounded_symbols, torch.zeros_like(rounded_symbols))
            if max_symbol_abs > 0:
                rounded_symbols = rounded_symbols.clamp(-float(max_symbol_abs), float(max_symbol_abs))
            if self.training and quant_mode == "noise":
                q_symbols = symbols + torch.empty_like(symbols).uniform_(-0.5, 0.5)
            elif self.training and quant_mode == "ste":
                q_symbols = symbols + (rounded_symbols - symbols).detach()
            else:
                q_symbols = rounded_symbols
            q_residual = q_symbols * self.quant_step
            adaptive_scale = torch.ones(
                target_latent.shape[0], 1, *target_size, dtype=target_latent.dtype, device=target_latent.device
            )
            delta_activity = (rounded_symbols.detach().abs().sum(dim=1, keepdim=True) > 0).to(target_latent.dtype)
            if progressive_residual:
                c1 = stage1_split
                q_stage1 = torch.zeros_like(q_residual)
                q_stage2 = torch.zeros_like(q_residual)
                q_stage1[:, :c1] = q_residual[:, :c1]
                q_stage2[:, c1:] = q_residual[:, c1:]
                zero_residual = torch.zeros_like(q_residual)
                delta_stage1 = self.residual_decoder_stage1(q_stage1, mu, z_up)
                delta_stage2 = self.residual_decoder_stage2(q_stage2, mu, z_up)
                if delta_gate_mode == "zero_center":
                    delta_stage1 = delta_stage1 - self.residual_decoder_stage1(zero_residual, mu, z_up)
                    delta_stage2 = delta_stage2 - self.residual_decoder_stage2(zero_residual, mu, z_up)
                elif delta_gate_mode in {"payload_ste", "payload_hard"}:
                    hard_activity = delta_activity
                    if delta_gate_mode == "payload_ste" and self.training:
                        soft_activity = symbols.abs().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
                        hard_activity = soft_activity + (delta_activity - soft_activity).detach()
                    delta_stage1 = delta_stage1 * hard_activity
                    delta_stage2 = delta_stage2 * hard_activity
                elif delta_gate_mode != "none":
                    raise ValueError(f"unknown delta_gate_mode: {delta_gate_mode}")
                stage1_delta_scaled = delta_stage1 * float(stage1_delta_scale)
                stage2_delta_scaled = delta_stage2 * float(stage2_delta_scale)
                residual_delta = stage1_delta_scaled + stage2_delta_scaled
                latent_stage1_hat = mu + stage1_delta_scaled
                stage1_delta_abs_mean = stage1_delta_scaled.detach().abs().mean()
                stage2_delta_abs_mean = stage2_delta_scaled.detach().abs().mean()
            else:
                residual_delta = self.residual_decoder(q_residual, mu, z_up)
                if delta_gate_mode == "zero_center":
                    residual_delta = residual_delta - self.residual_decoder(torch.zeros_like(q_residual), mu, z_up)
                elif delta_gate_mode != "none":
                    hard_activity = delta_activity
                    if delta_gate_mode == "payload_ste" and self.training:
                        soft_activity = symbols.abs().sum(dim=1, keepdim=True).clamp(0.0, 1.0)
                        hard_activity = soft_activity + (delta_activity - soft_activity).detach()
                    elif delta_gate_mode not in {"payload_ste", "payload_hard"}:
                        raise ValueError(f"unknown delta_gate_mode: {delta_gate_mode}")
                    residual_delta = residual_delta * hard_activity
                latent_stage1_hat = mu + residual_delta
                stage1_delta_abs_mean = residual_delta.detach().abs().mean()
                stage2_delta_abs_mean = torch.zeros((), dtype=target_latent.dtype, device=target_latent.device)
            if entropy_mode == "clamped":
                bits = gaussian_bits(q_symbols, scales)
            elif entropy_mode == "stable":
                bits = gaussian_bits_stable(q_symbols, scales)
            else:
                raise ValueError(f"unknown entropy_mode: {entropy_mode}")
            num_pixels = target_latent.shape[0] * (target_latent.shape[-2] * 16) * (target_latent.shape[-1] * 16)
            residual_bpp = bits.sum() / max(1, num_pixels)
        else:
            residual_latent = torch.zeros(
                target_latent.shape[0],
                self.residual_dim,
                *target_size,
                dtype=target_latent.dtype,
                device=target_latent.device,
            )
            q_symbols = residual_latent
            rounded_symbols = residual_latent
            residual_delta = torch.zeros_like(mu)
            latent_stage1_hat = mu
            stage1_delta_abs_mean = target_latent.new_tensor(0.0)
            stage2_delta_abs_mean = target_latent.new_tensor(0.0)
            delta_activity = torch.zeros_like(mu[:, :1])
            adaptive_scale = torch.ones_like(delta_activity)
            residual_bpp = target_latent.new_tensor(0.0)
            selector_scores = torch.zeros_like(residual_latent)
            topk_mask = None

        if adaptive_delta_scale and use_residual:
            adaptive_scale = self.delta_scale_net(
                q_residual,
                mu,
                z_up,
                scale_min=delta_scale_min,
                scale_max=delta_scale_max,
            )
            residual_delta = residual_delta * adaptive_scale
        residual_delta = residual_delta * float(delta_scale)
        latent_hat = mu + residual_delta
        pred_loss = F.smooth_l1_loss(mu, target_latent.detach())
        latent_loss = F.smooth_l1_loss(latent_hat, target_latent.detach())
        return {
            "latent_hat": latent_hat,
            "latent_stage1_hat": latent_stage1_hat,
            "mu": mu,
            "residual_latent": residual_latent,
            "q_symbols": q_symbols,
            "residual_delta": residual_delta,
            "scales": scales,
            "residual_bpp": residual_bpp,
            "pred_loss": pred_loss,
            "latent_loss": latent_loss,
            "residual_abs_mean": q_symbols.detach().abs().mean(),
            "residual_std": q_symbols.detach().std(),
            "rounded_abs_mean": rounded_symbols.detach().abs().mean(),
            "rounded_max_abs": rounded_symbols.detach().abs().max(),
            "rounded_nonzero_frac": (rounded_symbols.detach().abs() > 0).float().mean(),
            "rounded_symbols": rounded_symbols.detach(),
            "topk_mask": (topk_mask.detach().float() if topk_mask is not None else torch.zeros_like(rounded_symbols)),
            "selector_scores": selector_scores,
            "selector_scores_mean": selector_scores.detach().mean(),
            "selector_scores_std": selector_scores.detach().std(),
            "stage1_rounded_nonzero_frac": (rounded_symbols.detach()[:, :stage1_split].abs() > 0).float().mean(),
            "stage2_rounded_nonzero_frac": (rounded_symbols.detach()[:, stage1_split:].abs() > 0).float().mean(),
            "topk_score_mode_latent_error": target_latent.new_tensor(1.0 if topk_score_mode == "latent_error" else 0.0),
            "topk_score_mode_latent_error_sq": target_latent.new_tensor(1.0 if topk_score_mode == "latent_error_sq" else 0.0),
            "topk_score_mode_latent_grad": target_latent.new_tensor(1.0 if topk_score_mode == "latent_grad" else 0.0),
            "topk_score_mode_latent_grad_improve": target_latent.new_tensor(1.0 if topk_score_mode == "latent_grad_improve" else 0.0),
            "topk_score_mode_learned_selector": target_latent.new_tensor(1.0 if topk_score_mode == "learned_selector" else 0.0),
            "delta_active_frac": delta_activity.detach().mean(),
            "delta_activity": delta_activity,
            "adaptive_delta_scale": adaptive_scale.detach(),
            "adaptive_delta_scale_mean": adaptive_scale.detach().mean(),
            "adaptive_delta_scale_min": adaptive_scale.detach().amin(),
            "adaptive_delta_scale_max": adaptive_scale.detach().amax(),
            "scale_mean": scales.detach().mean(),
            "mu_abs_mean": mu.detach().abs().mean(),
            "delta_abs_mean": residual_delta.detach().abs().mean(),
            "stage1_delta_abs_mean": stage1_delta_abs_mean,
            "stage2_delta_abs_mean": stage2_delta_abs_mean,
        }

    @staticmethod
    def semantic_bpp(codebook_size: int, grid_h: int, grid_w: int, image_h: int, image_w: int) -> float:
        return (grid_h * grid_w * math.log2(codebook_size)) / max(1, image_h * image_w)
