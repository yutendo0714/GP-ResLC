"""Minimal Stage-A VQ autoencoder for scratch GP-ResLC research.

Stage A learns a compact semantic/generative code s before residual coding is
introduced. The default 16x16 latent grid with 1024 codes has a fixed semantic
index cost of roughly 0.039 bpp for 256x256 crops. Setting num_down=5 gives an
8x8 semantic grid, roughly 0.0098 bpp before entropy coding, which is closer to
the final cheap semantic code plus unpredictable residual design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SelfAttention2d(nn.Module):
    """Small spatial self-attention block for low-resolution latent maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        heads = min(int(num_heads), int(channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.channels = int(channels)
        self.num_heads = int(heads)
        self.head_dim = self.channels // self.num_heads
        self.norm = nn.GroupNorm(_norm_groups(self.channels), self.channels)
        self.qkv = nn.Conv2d(self.channels, self.channels * 3, 1)
        self.proj = nn.Conv2d(self.channels, self.channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).view(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(-2, -1)
        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        y = torch.matmul(attn, v.transpose(-2, -1))
        y = y.transpose(-2, -1).contiguous().view(b, c, h, w)
        return x + self.proj(y)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class VQOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    perplexity: torch.Tensor
    avg_probs: torch.Tensor


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 1024, dim: int = 256, beta: float = 0.25, entropy_tau: float = 1.0):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.beta = float(beta)
        self.entropy_tau = float(entropy_tau)
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.register_buffer("usage_ema", torch.zeros(self.codebook_size))

    def forward(self, z: torch.Tensor) -> VQOutput:
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, c)
        emb = self.embedding.weight
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ emb.t() + emb.pow(2).sum(1)
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        embed_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z_q.detach(), z)
        loss = embed_loss + self.beta * commit_loss
        z_q_st = z + (z_q - z).detach()

        one_hot = F.one_hot(indices, self.codebook_size).float()
        avg_probs = one_hot.mean(0)
        if self.training:
            with torch.no_grad():
                self.usage_ema.mul_(0.99).add_(avg_probs, alpha=0.01)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        tau = max(self.entropy_tau, 1e-6)
        soft_assign = F.softmax(-dist / tau, dim=1)
        soft_avg_probs = soft_assign.mean(0)
        # Store soft probabilities in avg_probs only for the hard VQOutput? No: the
        # caller needs both. Attach as an attribute to keep the dataclass small.
        out = VQOutput(z_q_st, indices.view(b, h, w), loss, perplexity, avg_probs)
        out.soft_avg_probs = soft_avg_probs
        return out


    @torch.no_grad()
    def restart_dead_codes(
        self,
        z: torch.Tensor,
        threshold: float = 1e-5,
        max_fraction: float = 0.10,
        noise_std: float = 0.01,
    ) -> int:
        """Reinitialize a limited number of low-usage embeddings from encoder outputs."""
        if z.numel() == 0:
            return 0
        flat = z.detach().permute(0, 2, 3, 1).contiguous().view(-1, z.shape[1])
        dead = torch.nonzero(self.usage_ema < float(threshold), as_tuple=False).flatten()
        if dead.numel() == 0:
            return 0
        max_restart = max(1, int(self.codebook_size * float(max_fraction)))
        if dead.numel() > max_restart:
            perm = torch.randperm(dead.numel(), device=dead.device)[:max_restart]
            dead = dead[perm]
        src = flat[torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)]
        if noise_std > 0:
            src = src + torch.randn_like(src) * float(noise_std)
        self.embedding.weight.data[dead] = src.to(self.embedding.weight.dtype)
        alive = self.usage_ema[self.usage_ema >= float(threshold)]
        fill = alive.mean() if alive.numel() > 0 else self.usage_ema.new_tensor(float(threshold))
        self.usage_ema[dead] = fill.clamp_min(float(threshold))
        return int(dead.numel())


def _channel_schedule(base_ch: int, latent_dim: int, num_down: int) -> list[int]:
    if num_down < 3 or num_down > 5:
        raise ValueError("num_down must be between 3 and 5 for the scratch VQ-AE")
    mults = [1, 2, 3, 4, 4][:num_down]
    return [base_ch * m for m in mults[:-1]] + [latent_dim]


class ScratchVQAutoencoder(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        latent_dim: int = 256,
        base_ch: int = 96,
        vq_beta: float = 0.25,
        vq_entropy_tau: float = 1.0,
        num_down: int = 4,
        decoder_attention: bool = False,
        extra_decoder_blocks: int = 0,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.vq_beta = float(vq_beta)
        self.vq_entropy_tau = float(vq_entropy_tau)
        self.num_down = int(num_down)
        self.decoder_attention = bool(decoder_attention)
        self.extra_decoder_blocks = int(extra_decoder_blocks)

        enc_layers: list[nn.Module] = [
            nn.Conv2d(3, base_ch, 3, padding=1),
            ResBlock(base_ch),
        ]
        in_ch = base_ch
        channels = _channel_schedule(base_ch, latent_dim, self.num_down)
        for out_ch in channels:
            enc_layers.append(Down(in_ch, out_ch))
            in_ch = out_ch
        enc_layers.extend([
            ResBlock(latent_dim),
            nn.GroupNorm(_norm_groups(latent_dim), latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, 1),
        ])
        self.encoder = nn.Sequential(*enc_layers)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, beta=vq_beta, entropy_tau=vq_entropy_tau)

        refine_layers: list[nn.Module] = []
        for _ in range(self.extra_decoder_blocks):
            block = ResBlock(latent_dim)
            nn.init.zeros_(block.net[-1].weight)
            nn.init.zeros_(block.net[-1].bias)
            refine_layers.append(block)
        if self.decoder_attention:
            refine_layers.append(SelfAttention2d(latent_dim))
        self.latent_refine = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()

        dec_layers: list[nn.Module] = [
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            ResBlock(latent_dim),
        ]
        rev_channels = list(reversed(channels[:-1])) + [base_ch]
        in_ch = latent_dim
        for out_ch in rev_channels:
            dec_layers.append(Up(in_ch, out_ch))
            in_ch = out_ch
        dec_layers.extend([
            nn.GroupNorm(_norm_groups(base_ch), base_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 3, 3, padding=1),
        ])
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> VQOutput:
        return self.quantizer(self.encoder(x))

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z_q = self.latent_refine(z_q)
        return torch.sigmoid(self.decoder(z_q))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        vq = self.quantizer(z_e)
        x_hat = self.decode(vq.quantized)
        bpp_index = self.semantic_index_bpp(x.shape[-2], x.shape[-1], vq.indices.shape[-2], vq.indices.shape[-1])
        entropy = -(vq.avg_probs * (vq.avg_probs + 1e-10).log()).sum()
        entropy_norm = entropy / math.log(self.codebook_size)
        usage_frac = (vq.avg_probs > 0).float().mean()
        soft_entropy = -(vq.soft_avg_probs * (vq.soft_avg_probs + 1e-10).log()).sum()
        soft_entropy_norm = soft_entropy / math.log(self.codebook_size)
        soft_perplexity = torch.exp(soft_entropy)
        soft_usage_frac = (vq.soft_avg_probs > (1.0 / self.codebook_size) * 0.1).float().mean()
        return {
            "x_hat": x_hat,
            "z_e": z_e,
            "z_q": vq.quantized,
            "indices": vq.indices,
            "vq_loss": vq.loss,
            "perplexity": vq.perplexity,
            "codebook_entropy": entropy,
            "codebook_entropy_norm": entropy_norm,
            "codebook_usage_frac": usage_frac,
            "soft_codebook_entropy_norm": soft_entropy_norm,
            "soft_perplexity": soft_perplexity,
            "soft_codebook_usage_frac": soft_usage_frac,
            "semantic_bpp_fixed": x.new_tensor(bpp_index),
        }

    def semantic_index_bpp(self, image_h: int, image_w: int, latent_h: int, latent_w: int) -> float:
        bits_per_index = math.ceil(math.log2(self.codebook_size))
        return float(latent_h * latent_w * bits_per_index / max(1, image_h * image_w))
