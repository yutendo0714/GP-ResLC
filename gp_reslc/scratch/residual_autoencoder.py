"""Stage-B residual decomposition for scratch GP-ResLC.

This module keeps the Stage-A semantic code external. Given a decoder-side
semantic latent z_s, it predicts a latent mean mu(z_s), encodes only the
unpredictable residual r = y - mu(z_s), and decodes an image correction from
mu(z_s) + r_hat. The first version uses a differentiable Gaussian entropy proxy;
real arithmetic coding should be added after the proxy shows a useful trend.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .vq_autoencoder import Down, ResBlock, Up, _norm_groups


def gaussian_bits(symbols: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Bits for unit-quantized symbols under a zero-mean Gaussian model."""
    inv_std = 1.0 / scales.clamp_min(1e-4)
    upper = (symbols + 0.5) * inv_std
    lower = (symbols - 0.5) * inv_std
    normalizer = math.sqrt(0.5)
    probs = 0.5 * (torch.erf(upper * normalizer) - torch.erf(lower * normalizer))
    return -torch.log2(probs.clamp_min(1e-9))


class ResidualEncoder(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int, base_ch: int, num_down: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, base_ch, 3, padding=1), ResBlock(base_ch)]
        ch = base_ch
        for i in range(num_down):
            if i == num_down - 1:
                out_ch = latent_dim
            else:
                out_ch = base_ch * min(i + 1, 4)
            layers.append(Down(ch, out_ch))
            ch = out_ch
        layers.extend([ResBlock(latent_dim), nn.Conv2d(latent_dim, latent_dim, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDecoder(nn.Module):
    def __init__(self, semantic_dim: int, residual_dim: int, base_ch: int, num_down: int):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(semantic_dim + residual_dim, residual_dim, 3, padding=1),
            ResBlock(residual_dim),
        ]
        ch = residual_dim
        for i in range(num_down):
            if i == num_down - 1:
                out_ch = base_ch
            else:
                out_ch = base_ch * min(num_down - i, 4)
            layers.append(Up(ch, out_ch))
            ch = out_ch
        layers.extend([nn.GroupNorm(_norm_groups(base_ch), base_ch), nn.SiLU(inplace=True)])
        final = nn.Conv2d(base_ch, 3, 3, padding=1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, y_hat: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y_hat, z_s], dim=1))


class ScratchResidualBottleneck(nn.Module):
    """Predictable/residual latent split conditioned on Stage-A semantic z_s."""

    def __init__(
        self,
        semantic_dim: int,
        residual_dim: int = 32,
        base_ch: int = 96,
        num_down: int = 5,
        quant_step: float = 0.25,
        scale_floor: float = 0.11,
    ):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.residual_dim = int(residual_dim)
        self.num_down = int(num_down)
        self.quant_step = float(quant_step)
        self.scale_floor = float(scale_floor)
        self.encoder = ResidualEncoder(9, self.residual_dim, base_ch, num_down)
        self.predictor = nn.Sequential(
            nn.Conv2d(self.semantic_dim, self.residual_dim, 3, padding=1),
            ResBlock(self.residual_dim),
            ResBlock(self.residual_dim),
            nn.Conv2d(self.residual_dim, self.residual_dim, 3, padding=1),
        )
        self.scale_net = nn.Sequential(
            nn.Conv2d(self.semantic_dim, self.residual_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.residual_dim, self.residual_dim, 3, padding=1),
        )
        self.decoder = ResidualDecoder(self.semantic_dim, self.residual_dim, base_ch, num_down)

    def forward(self, x: torch.Tensor, base: torch.Tensor, z_s: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_in = torch.cat([x, base, x - base], dim=1)
        y = self.encoder(enc_in)
        mu = self.predictor(z_s)
        scales = F.softplus(self.scale_net(z_s)) + self.scale_floor
        residual_symbols = (y - mu) / self.quant_step
        if self.training:
            q_symbols = residual_symbols + torch.empty_like(residual_symbols).uniform_(-0.5, 0.5)
        else:
            q_symbols = residual_symbols.round()
        y_hat = mu + q_symbols * self.quant_step
        correction = self.decoder(y_hat, z_s)
        base_safe = base.clamp(1e-4, 1.0 - 1e-4)
        x_hat = torch.sigmoid(torch.logit(base_safe) + correction)
        bits = gaussian_bits(q_symbols, scales)
        num_pixels = x.shape[0] * x.shape[-2] * x.shape[-1]
        residual_bpp = bits.sum() / max(1, num_pixels)
        pred_loss = F.smooth_l1_loss(mu, y.detach())
        return {
            "x_hat": x_hat,
            "stage0_x_hat": x_hat,
            "y": y,
            "mu": mu,
            "residual_symbols": residual_symbols,
            "q_symbols": q_symbols,
            "y_hat": y_hat,
            "scales": scales,
            "bits": bits,
            "residual_bpp": residual_bpp,
            "pred_loss": pred_loss,
            "residual_abs_mean": residual_symbols.detach().abs().mean(),
            "residual_std": residual_symbols.detach().std(),
            "scale_mean": scales.detach().mean(),
        }



class ScratchProgressiveResidualBottleneck(nn.Module):
    """Progressive latent residual coder conditioned on Stage-A semantics.

    Stage 0 is intentionally compatible with ScratchResidualBottleneck: it uses
    the same encoder, predictor, scale_net, and decoder names. This lets us
    initialize from the best single-stage checkpoint, then add finer residual
    refinement stages without discarding the working decomposition.
    """

    def __init__(
        self,
        semantic_dim: int,
        residual_dim: int = 32,
        base_ch: int = 96,
        num_down: int = 5,
        quant_step: float = 1.0,
        progressive_stages: int = 2,
        stage_quant_steps: tuple[float, ...] | None = None,
        scale_floor: float = 0.11,
        gate_extra_stages: bool = False,
        gate_threshold: float = 0.2,
        gate_init_bias: float = -2.0,
        gate_soft_train: bool = False,
        stage_correction_decoder: bool = False,
        gate_topk_frac: float = 0.0,
    ):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.residual_dim = int(residual_dim)
        self.num_down = int(num_down)
        self.quant_step = float(quant_step)
        self.progressive_stages = max(1, int(progressive_stages))
        self.scale_floor = float(scale_floor)
        self.gate_extra_stages = bool(gate_extra_stages)
        self.gate_threshold = float(gate_threshold)
        self.gate_init_bias = float(gate_init_bias)
        self.gate_soft_train = bool(gate_soft_train)
        self.stage_correction_decoder = bool(stage_correction_decoder)
        self.gate_topk_frac = float(gate_topk_frac)
        if stage_quant_steps is None:
            steps = [self.quant_step * (0.5 ** i) for i in range(self.progressive_stages)]
        else:
            steps = [float(s) for s in stage_quant_steps]
            if len(steps) != self.progressive_stages:
                raise ValueError("stage_quant_steps length must match progressive_stages")
        self.stage_quant_steps = tuple(steps)

        self.encoder = ResidualEncoder(9, self.residual_dim, base_ch, num_down)
        self.predictor = nn.Sequential(
            nn.Conv2d(self.semantic_dim, self.residual_dim, 3, padding=1),
            ResBlock(self.residual_dim),
            ResBlock(self.residual_dim),
            nn.Conv2d(self.residual_dim, self.residual_dim, 3, padding=1),
        )
        self.scale_net = nn.Sequential(
            nn.Conv2d(self.semantic_dim, self.residual_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.residual_dim, self.residual_dim, 3, padding=1),
        )
        self.extra_scale_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.semantic_dim + self.residual_dim, self.residual_dim, 3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.residual_dim, self.residual_dim, 3, padding=1),
                )
                for _ in range(self.progressive_stages - 1)
            ]
        )
        for net in self.extra_scale_nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.constant_(net[-1].bias, -3.0)
        self.extra_gate_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.semantic_dim + self.residual_dim, self.residual_dim, 3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.residual_dim, self.residual_dim, 3, padding=1),
                )
                for _ in range(self.progressive_stages - 1)
            ]
        )
        for net in self.extra_gate_nets:
            nn.init.constant_(net[-1].bias, self.gate_init_bias)
        self.extra_decoders = nn.ModuleList(
            [ResidualDecoder(self.semantic_dim, self.residual_dim, base_ch, num_down) for _ in range(self.progressive_stages - 1)]
        ) if self.stage_correction_decoder else nn.ModuleList()
        self.decoder = ResidualDecoder(self.semantic_dim, self.residual_dim, base_ch, num_down)

    def _quantize(self, residual: torch.Tensor, step: float) -> torch.Tensor:
        symbols = residual / float(step)
        if self.training:
            return symbols + torch.empty_like(symbols).uniform_(-0.5, 0.5)
        return symbols.round()

    def forward(self, x: torch.Tensor, base: torch.Tensor, z_s: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_in = torch.cat([x, base, x - base], dim=1)
        y = self.encoder(enc_in)
        mu = self.predictor(z_s)

        y_hat = mu
        stage_bpps: list[torch.Tensor] = []
        stage_abs: list[torch.Tensor] = []
        stage_scales: list[torch.Tensor] = []
        stage_gates: list[torch.Tensor] = []
        stage_gate_maps: list[torch.Tensor | None] = []
        stage_gate_probs: list[torch.Tensor | None] = []
        stage_gate_logits: list[torch.Tensor | None] = []
        q_symbols_all: list[torch.Tensor] = []
        stage_deltas: list[torch.Tensor] = []
        stage_y_hats: list[torch.Tensor] = []
        bits_all: list[torch.Tensor] = []

        for stage_idx, step in enumerate(self.stage_quant_steps):
            residual = y - y_hat
            raw_q_symbols = self._quantize(residual, step)
            gate = None
            if stage_idx == 0:
                scales = F.softplus(self.scale_net(z_s)) + self.scale_floor
            else:
                scale_in = torch.cat([z_s, y_hat.detach()], dim=1)
                scales = F.softplus(self.extra_scale_nets[stage_idx - 1](scale_in)) + self.scale_floor
                if self.gate_extra_stages:
                    gate_logit = self.extra_gate_nets[stage_idx - 1](scale_in)
                    gate_prob = torch.sigmoid(gate_logit)
                    if self.gate_topk_frac > 0:
                        flat = gate_prob.flatten(1)
                        k = max(1, int(flat.shape[1] * self.gate_topk_frac))
                        kth = torch.topk(flat, k, dim=1).values[:, -1].view(-1, 1, 1, 1)
                        gate_hard = (gate_prob >= kth).to(gate_prob.dtype)
                        if self.training:
                            gate = gate_hard.detach() - gate_prob.detach() + gate_prob
                        else:
                            gate = gate_hard
                    elif self.training and self.gate_soft_train:
                        gate = gate_prob
                    elif self.training:
                        gate_hard = (gate_prob > self.gate_threshold).to(gate_prob.dtype)
                        gate = gate_hard.detach() - gate_prob.detach() + gate_prob
                    else:
                        gate = (gate_prob > self.gate_threshold).to(gate_prob.dtype)
                    stage_gates.append(gate.detach().mean())
                    stage_gate_maps.append(gate)
                    stage_gate_probs.append(gate_prob)
                    stage_gate_logits.append(gate_logit)
            if gate is None:
                gated_q_symbols = raw_q_symbols
                bits = gaussian_bits(raw_q_symbols, scales)
                stage_gates.append(raw_q_symbols.new_tensor(1.0))
                stage_gate_maps.append(None)
                stage_gate_probs.append(None)
                stage_gate_logits.append(None)
            else:
                gated_q_symbols = raw_q_symbols * gate
                bits = gaussian_bits(raw_q_symbols, scales) * gate
            num_pixels = x.shape[0] * x.shape[-2] * x.shape[-1]
            stage_bpps.append(bits.sum() / max(1, num_pixels))
            stage_abs.append(gated_q_symbols.detach().abs().mean())
            stage_scales.append(scales.mean())
            delta = gated_q_symbols * float(step)
            q_symbols_all.append(gated_q_symbols)
            stage_deltas.append(delta)
            bits_all.append(bits)
            y_hat = y_hat + delta
            stage_y_hats.append(y_hat)

        base_safe = base.clamp(1e-4, 1.0 - 1e-4)
        base_logit = torch.logit(base_safe)
        stage0_correction = self.decoder(stage_y_hats[0], z_s)
        stage0_x_hat = torch.sigmoid(base_logit + stage0_correction)
        if self.stage_correction_decoder and len(stage_y_hats) > 1:
            correction = stage0_correction
            for extra_idx, delta in enumerate(stage_deltas[1:]):
                correction = correction + self.extra_decoders[extra_idx](delta, z_s)
        else:
            correction = self.decoder(y_hat, z_s)
        x_hat = torch.sigmoid(base_logit + correction)
        residual_bpp = torch.stack(stage_bpps).sum()
        pred_loss = F.smooth_l1_loss(mu, y.detach())
        residual_symbols = q_symbols_all[0]
        total_bits = torch.stack([b.sum() for b in bits_all]).sum()
        out: dict[str, torch.Tensor] = {
            "x_hat": x_hat,
            "stage0_x_hat": stage0_x_hat,
            "y": y,
            "mu": mu,
            "residual_symbols": residual_symbols,
            "q_symbols": residual_symbols,
            "y_hat": y_hat,
            "scales": stage_scales[0],
            "bits": total_bits,
            "residual_bpp": residual_bpp,
            "pred_loss": pred_loss,
            "residual_abs_mean": torch.stack(stage_abs).mean(),
            "residual_std": residual_symbols.detach().std(),
            "scale_mean": torch.stack(stage_scales).mean(),
        }
        for i, value in enumerate(stage_bpps):
            out[f"stage{i}_bpp"] = value
            out[f"stage{i}_residual_abs_mean"] = stage_abs[i]
            out[f"stage{i}_scale_mean"] = stage_scales[i]
            out[f"stage{i}_gate_mean"] = stage_gates[i]
            if stage_gate_maps[i] is not None:
                out[f"stage{i}_gate_map"] = stage_gate_maps[i]
            if stage_gate_probs[i] is not None:
                out[f"stage{i}_gate_prob"] = stage_gate_probs[i]
            if stage_gate_logits[i] is not None:
                out[f"stage{i}_gate_logit"] = stage_gate_logits[i]
        return out
