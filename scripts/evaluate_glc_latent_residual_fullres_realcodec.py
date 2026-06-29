#!/usr/bin/env python3
"""Full-resolution real-codec evaluation for the GLC-latent scratch branch.

This script upgrades the scratch/full-design branch from 256x256 center crops
to the GLC image protocol:

- load original-resolution images;
- pad to a multiple of 64 with replicate padding;
- count transmitted bytes over the original, unpadded pixel count;
- serialize semantic VQ indices with fixed-width packing;
- decode semantic indices back through the Stage-A codebook;
- arithmetic-code residual symbols with torchac;
- decode from transmitted semantic + residual streams and crop to original size.

It is still a research evaluator, but both transmitted streams used by the
decoder are byte-backed rather than reusing source-side tensors.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import lpips as lpips_lib
from DISTS_pytorch import DISTS

from gp_reslc.real_codec import (
    _decode_gaussian_symbols,
    _encode_gaussian_symbols,
    _pack_fixed_width,
    _unpack_fixed_width,
    STREAM_STRUCT,
)
from gp_reslc.scratch import ScratchVQAutoencoder, GLCLatentResidualBottleneck
from gp_reslc.scratch.glc_latent_residual import gaussian_bits, gaussian_bits_stable
from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict


SCRATCH_MAGIC = b"GPSR1"
SCRATCH_VERSION = 1
SCRATCH_HEADER_STRUCT = struct.Struct("<5sBIIIIHHHHHHIhhI")


class FullResFolder:
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.JPEG")

    def __init__(self, root: str, limit: int = 0):
        paths: list[str] = []
        for ext in self.EXTS:
            paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        self.paths = sorted(paths)
        if limit > 0:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise RuntimeError(f"no images found in {root}")
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        for path in self.paths:
            img = Image.open(path).convert("RGB")
            yield self.to_tensor(img), os.path.basename(path), path


def load_stage_a(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    args = dict(ckpt.get("args", {}))
    if "codebook_size" not in args:
        args = dict(ckpt.get("stage_a_args", {}))
    model = ScratchVQAutoencoder(
        args["codebook_size"],
        args["latent_dim"],
        args["base_ch"],
        args.get("vq_beta", 0.25),
        args.get("vq_entropy_tau", 1.0),
        args.get("num_down", 4),
        decoder_attention=args.get("decoder_attention", False),
        extra_decoder_blocks=args.get("extra_decoder_blocks", 0),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[load_model] non-strict model load missing={missing} unexpected={unexpected}", flush=True)
    model.eval()
    return model, args


def load_vqgan(weights: str, device: str):
    glc = GLC_Image(inplace=False).to(device)
    glc.load_state_dict(get_state_dict(weights), strict=True)
    glc.eval()
    for p in glc.parameters():
        p.requires_grad_(False)
    return glc.vqgan


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = dict(ckpt["args"])
    stage_a, stage_a_args = load_stage_a(args["stage_a_ckpt"], device)
    if "stage_a_model" in ckpt:
        stage_a.load_state_dict(ckpt["stage_a_model"])
    model = GLCLatentResidualBottleneck(
        semantic_dim=stage_a_args["latent_dim"],
        target_dim=256,
        residual_dim=args.get("residual_dim", 24),
        hidden_dim=args.get("hidden_dim", 256),
        quant_step=args.get("quant_step", 0.5),
        scale_floor=args.get("scale_floor", 0.11),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[load_model] non-strict model load missing={missing} unexpected={unexpected}", flush=True)
    model.eval()
    return stage_a, model, args


def to_glc_range(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def from_glc_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0, 1)


def cuda_sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def pad_to_glc_multiple(x: torch.Tensor, multiple: int = 64):
    _, _, h, w = x.shape
    pl, pr, pt, pb = GLC_Image.get_padding_size(h, w, multiple)
    return F.pad(x, (pl, pr, pt, pb), mode="replicate"), (pl, pr, pt, pb)


def crop_to_original(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x[..., :h, :w]


@torch.no_grad()
def target_latent(vqgan, x01_padded: torch.Tensor) -> torch.Tensor:
    return vqgan.encoder(to_glc_range(x01_padded)).detach()


@torch.no_grad()
def encode_decode_semantic(stage_a: ScratchVQAutoencoder, x01_padded: torch.Tensor):
    vq = stage_a.encode(x01_padded)
    bit_width = int(math.ceil(math.log2(stage_a.codebook_size)))
    sem_bytes = _pack_fixed_width(vq.indices, bit_width)
    decoded = _unpack_fixed_width(
        sem_bytes,
        int(vq.indices.numel()),
        bit_width,
        device=x01_padded.device,
    )
    b, h, w = vq.indices.shape
    z_dec = stage_a.quantizer.embedding(decoded).view(b, h, w, stage_a.latent_dim)
    z_dec = z_dec.permute(0, 3, 1, 2).contiguous()
    semantic_decode_max_abs = float((z_dec - vq.quantized).abs().max().item())
    return z_dec.detach(), vq.indices.detach(), sem_bytes, bit_width, semantic_decode_max_abs


def build_payload(
    *,
    orig_h: int,
    orig_w: int,
    pad_h: int,
    pad_w: int,
    sem_h: int,
    sem_w: int,
    sem_bit_width: int,
    sem_bytes: bytes,
    residual_shape: tuple[int, int, int],
    residual_stream,
) -> bytes:
    res_c, res_h, res_w = residual_shape
    header = SCRATCH_HEADER_STRUCT.pack(
        SCRATCH_MAGIC,
        SCRATCH_VERSION,
        int(orig_h),
        int(orig_w),
        int(pad_h),
        int(pad_w),
        int(sem_h),
        int(sem_w),
        int(sem_bit_width),
        int(res_c),
        int(res_h),
        int(res_w),
        len(sem_bytes),
        int(residual_stream.lo),
        int(residual_stream.hi),
        len(residual_stream.data),
    )
    stream_header = STREAM_STRUCT.pack(
        int(residual_stream.lo),
        int(residual_stream.hi),
        len(residual_stream.data),
    )
    return b"".join([header, sem_bytes, stream_header, residual_stream.data])


def decode_latent_from_symbols(
    model,
    z_s,
    mu,
    q_symbols,
    delta_gate_mode: str,
    delta_scale: float = 1.0,
    adaptive_delta_scale: bool = False,
    delta_scale_min: float = 0.0,
    delta_scale_max: float = 1.0,
    delta_lowpass_kernel: int = 0,
    delta_split_kernel: int = 0,
    delta_low_scale: float = 1.0,
    delta_high_scale: float = 1.0,
    progressive_residual: bool = False,
    stage1_channels: int = 0,
    stage1_delta_scale: float = 1.0,
    stage2_delta_scale: float = 1.0,
):
    target_size = mu.shape[-2:]
    z_up = F.interpolate(z_s, size=target_size, mode="bilinear", align_corners=False)
    q_residual = q_symbols * model.quant_step
    if progressive_residual:
        c1 = int(stage1_channels) if stage1_channels > 0 else max(1, model.residual_dim // 2)
        c1 = max(1, min(model.residual_dim - 1, c1))
        q_stage1 = torch.zeros_like(q_residual)
        q_stage2 = torch.zeros_like(q_residual)
        q_stage1[:, :c1] = q_residual[:, :c1]
        q_stage2[:, c1:] = q_residual[:, c1:]
        zero_residual = torch.zeros_like(q_residual)
        delta_stage1 = model.residual_decoder_stage1(q_stage1, mu, z_up)
        delta_stage2 = model.residual_decoder_stage2(q_stage2, mu, z_up)
        if delta_gate_mode == "zero_center":
            delta_stage1 = delta_stage1 - model.residual_decoder_stage1(zero_residual, mu, z_up)
            delta_stage2 = delta_stage2 - model.residual_decoder_stage2(zero_residual, mu, z_up)
        elif delta_gate_mode in {"payload_hard", "payload_ste"}:
            hard_activity = (q_symbols.detach().abs().sum(dim=1, keepdim=True) > 0).to(delta_stage1.dtype)
            delta_stage1 = delta_stage1 * hard_activity
            delta_stage2 = delta_stage2 * hard_activity
        elif delta_gate_mode != "none":
            raise ValueError(f"unsupported delta_gate_mode: {delta_gate_mode}")
        residual_delta = delta_stage1 * float(stage1_delta_scale) + delta_stage2 * float(stage2_delta_scale)
    else:
        residual_delta = model.residual_decoder(q_residual, mu, z_up)
        if delta_gate_mode == "zero_center":
            residual_delta = residual_delta - model.residual_decoder(torch.zeros_like(q_residual), mu, z_up)
        elif delta_gate_mode in {"payload_hard", "payload_ste"}:
            hard_activity = (q_symbols.detach().abs().sum(dim=1, keepdim=True) > 0).to(residual_delta.dtype)
            residual_delta = residual_delta * hard_activity
        elif delta_gate_mode != "none":
            raise ValueError(f"unsupported delta_gate_mode: {delta_gate_mode}")
    if adaptive_delta_scale:
        adaptive_scale = model.delta_scale_net(
            q_residual,
            mu,
            z_up,
            scale_min=delta_scale_min,
            scale_max=delta_scale_max,
        )
        residual_delta = residual_delta * adaptive_scale
    else:
        adaptive_scale = torch.ones_like(residual_delta[:, :1])
    if delta_split_kernel and delta_split_kernel > 1:
        k = int(delta_split_kernel)
        if k % 2 == 0:
            raise ValueError("delta_split_kernel must be odd")
        low_delta = F.avg_pool2d(residual_delta, kernel_size=k, stride=1, padding=k // 2)
        high_delta = residual_delta - low_delta
        residual_delta = float(delta_low_scale) * low_delta + float(delta_high_scale) * high_delta
    if delta_lowpass_kernel and delta_lowpass_kernel > 1:
        k = int(delta_lowpass_kernel)
        if k % 2 == 0:
            raise ValueError("delta_lowpass_kernel must be odd")
        residual_delta = F.avg_pool2d(residual_delta, kernel_size=k, stride=1, padding=k // 2)
    residual_delta = residual_delta * float(delta_scale)
    return mu + residual_delta, adaptive_scale


def _apply_residual_delta_for_selector(
    model,
    q_symbols: torch.Tensor,
    mu: torch.Tensor,
    z_up: torch.Tensor,
    delta_gate_mode: str,
    delta_scale: float,
) -> torch.Tensor:
    q_residual = q_symbols * model.quant_step
    residual_delta = model.residual_decoder(q_residual, mu, z_up)
    if delta_gate_mode == "zero_center":
        residual_delta = residual_delta - model.residual_decoder(torch.zeros_like(q_residual), mu, z_up)
    elif delta_gate_mode in {"payload_hard", "payload_ste"}:
        hard_activity = (q_symbols.detach().abs().sum(dim=1, keepdim=True) > 0).to(residual_delta.dtype)
        residual_delta = residual_delta * hard_activity
    elif delta_gate_mode != "none":
        raise ValueError(f"unsupported delta_gate_mode for encoder selector: {delta_gate_mode}")
    return residual_delta * float(delta_scale)


def image_guided_selector_forward(
    *,
    model,
    vqgan,
    z_s: torch.Tensor,
    target_latent: torch.Tensor,
    x01_padded: torch.Tensor,
    delta_gate_mode: str,
    force_topk_frac: float,
    hard_topk: bool,
    entropy_mode: str,
    max_symbol_abs: float,
    delta_scale: float,
    selector_loss: str,
    selector_metric_max_side: int,
    selector_latent_max_side: int,
    selector_l1_weight: float,
    selector_lpips_weight: float,
    selector_dists_weight: float,
    selector_latent_weight: float,
    lpips_fn,
    dists_fn,
) -> dict[str, torch.Tensor]:
    """Encoder-side residual selection using first-order image/perceptual payoff."""
    if force_topk_frac <= 0:
        raise ValueError("image_guided_selector_forward requires force_topk_frac > 0")
    target_size = target_latent.shape[-2:]
    z_up = F.interpolate(z_s, size=target_size, mode="bilinear", align_corners=False).detach()
    with torch.no_grad():
        mu = model.predictor(z_s, target_size).detach()
        scales = model.scale_net(z_up, mu, model.scale_floor).detach()
        residual_latent = model.residual_encoder(target_latent, mu, z_up)
        symbols = residual_latent / model.quant_step
        rounded_symbols = symbols.round()
        forced_sign = torch.where(symbols >= 0, torch.ones_like(symbols), -torch.ones_like(symbols))
        candidate_symbols = torch.where(rounded_symbols == 0, forced_sign, rounded_symbols).detach()
    with torch.enable_grad():
        probe_symbols = torch.zeros_like(candidate_symbols, requires_grad=True)
        residual_delta = _apply_residual_delta_for_selector(
            model, probe_symbols, mu, z_up, delta_gate_mode, delta_scale
        )
        latent_probe = mu + residual_delta
        loss = x01_padded.new_tensor(0.0)
        if selector_latent_weight > 0 or "latent" in selector_loss:
            loss = loss + float(selector_latent_weight) * F.smooth_l1_loss(latent_probe, target_latent.detach())
        if any(w > 0 for w in (selector_l1_weight, selector_lpips_weight, selector_dists_weight)):
            latent_for_image = latent_probe
            x_for_image = x01_padded
            if selector_latent_max_side and selector_latent_max_side > 0:
                lh, lw = latent_probe.shape[-2:]
                side = max(lh, lw)
                if side > selector_latent_max_side:
                    scale = float(selector_latent_max_side) / float(side)
                    new_lh = max(1, int(round(lh * scale)))
                    new_lw = max(1, int(round(lw * scale)))
                    latent_for_image = F.interpolate(latent_probe, size=(new_lh, new_lw), mode="bilinear", align_corners=False)
                    x_for_image = F.interpolate(x01_padded, size=(new_lh * 16, new_lw * 16), mode="area")
            x_probe = ((vqgan.generator(latent_for_image) + 1.0) * 0.5).clamp(0, 1)
            x_ref, x_probe_ref = metric_inputs(x_for_image, x_probe, selector_metric_max_side)
            if selector_l1_weight > 0 or "l1" in selector_loss:
                loss = loss + float(selector_l1_weight) * F.smooth_l1_loss(x_probe_ref, x_ref)
            if selector_lpips_weight > 0 or "lpips" in selector_loss:
                loss = loss + float(selector_lpips_weight) * lpips_fn(x_probe_ref * 2 - 1, x_ref * 2 - 1).mean()
            if selector_dists_weight > 0 or "dists" in selector_loss:
                loss = loss + float(selector_dists_weight) * dists_fn(x_probe_ref, x_ref).mean()
        grad = torch.autograd.grad(loss, probe_symbols, retain_graph=False, create_graph=False)[0]
    first_order_loss_delta = grad.detach() * candidate_symbols
    topk_scores = (-first_order_loss_delta).clamp_min(0.0)
    if float(topk_scores.max().item()) <= 0.0:
        topk_scores = first_order_loss_delta.abs()
    flat_scores = topk_scores.flatten(1)
    k = max(1, min(flat_scores.shape[1], int(flat_scores.shape[1] * float(force_topk_frac))))
    topk_idx = flat_scores.topk(k, dim=1).indices
    topk_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    topk_mask.scatter_(1, topk_idx, True)
    topk_mask = topk_mask.view_as(candidate_symbols)
    if hard_topk:
        q_symbols = torch.where(topk_mask, candidate_symbols, torch.zeros_like(candidate_symbols))
    else:
        q_symbols = candidate_symbols
    if max_symbol_abs > 0:
        q_symbols = q_symbols.clamp(-float(max_symbol_abs), float(max_symbol_abs))
    if entropy_mode == "stable":
        bits = gaussian_bits_stable(q_symbols, scales)
    elif entropy_mode == "clamped":
        bits = gaussian_bits(q_symbols, scales)
    else:
        raise ValueError(f"unknown entropy_mode: {entropy_mode}")
    num_pixels = target_latent.shape[0] * (target_latent.shape[-2] * 16) * (target_latent.shape[-1] * 16)
    residual_delta = _apply_residual_delta_for_selector(
        model, q_symbols, mu, z_up, delta_gate_mode, delta_scale=1.0
    )
    latent_hat = mu + residual_delta
    zeros = target_latent.new_zeros(())
    return {
        "latent_hat": latent_hat,
        "mu": mu,
        "q_symbols": q_symbols.detach(),
        "scales": scales.detach(),
        "residual_bpp": bits.sum() / max(1, num_pixels),
        "rounded_abs_mean": q_symbols.detach().abs().mean(),
        "rounded_nonzero_frac": (q_symbols.detach().abs() > 0).float().mean(),
        "rounded_max_abs": q_symbols.detach().abs().max(),
        "scale_mean": scales.detach().mean(),
        "stage1_rounded_nonzero_frac": zeros,
        "stage2_rounded_nonzero_frac": zeros,
        "stage1_delta_abs_mean": residual_delta.detach().abs().mean(),
        "stage2_delta_abs_mean": zeros,
        "selector_loss_value": loss.detach(),
    }


def metric_inputs(x: torch.Tensor, x_hat: torch.Tensor, max_side: int):
    if max_side <= 0:
        return x, x_hat
    h, w = x.shape[-2:]
    side = max(h, w)
    if side <= max_side:
        return x, x_hat
    scale = max_side / float(side)
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    return (
        F.interpolate(x, size=size, mode="area"),
        F.interpolate(x_hat, size=size, mode="area"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--padding", type=int, default=64)
    ap.add_argument("--metric_max_side", type=int, default=0, help="Optional dev downsample for LPIPS/DISTS. 0 keeps original resolution.")
    ap.add_argument("--save_recon", action="store_true")
    ap.add_argument("--save_limit", type=int, default=8)
    ap.add_argument("--print_every", type=int, default=1)
    ap.add_argument("--delta_scale", type=float, default=1.0, help="Global decoder-side scale for residual delta; fixed per model, no side bits.")
    ap.add_argument("--adaptive_delta_scale", action="store_true", help="Use learned decoder-side residual gamma map. No side bits.")
    ap.add_argument("--delta_scale_min", type=float, default=0.0)
    ap.add_argument("--delta_scale_max", type=float, default=1.0)
    ap.add_argument("--delta_lowpass_kernel", type=int, default=0, help="Optional fixed odd avg-pool kernel on latent residual delta. No side bits.")
    ap.add_argument("--delta_split_kernel", type=int, default=0, help="Optional odd avg-pool kernel for low/high latent residual split. No side bits.")
    ap.add_argument("--delta_low_scale", type=float, default=1.0)
    ap.add_argument("--delta_high_scale", type=float, default=1.0)
    ap.add_argument("--disable_residual", action="store_true", help="Decode the semantic/base latent only and skip residual analysis/coding.")
    ap.add_argument("--override_topk_score_mode", default="",
                    choices=["", "abs", "latent_error", "latent_error_sq", "latent_grad", "latent_grad_improve", "learned_selector"],
                    help="Override checkpoint top-k score mode for analysis-only evaluation.")
    ap.add_argument("--encoder_selector_loss", default="none",
                    choices=["none", "l1", "l1_latent", "lpips", "dists", "mix"],
                    help="Use an encoder-side image/perceptual-gradient selector instead of the checkpoint selector.")
    ap.add_argument("--selector_metric_max_side", type=int, default=512,
                    help="Downsample side length for image-gradient selector losses; 0 keeps full resolution.")
    ap.add_argument("--selector_latent_max_side", type=int, default=0,
                    help="Downsample latent before generator only for selector scoring; 0 keeps full latent resolution.")
    ap.add_argument("--selector_l1_weight", type=float, default=1.0)
    ap.add_argument("--selector_lpips_weight", type=float, default=0.0)
    ap.add_argument("--selector_dists_weight", type=float, default=0.0)
    ap.add_argument("--selector_latent_weight", type=float, default=0.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU is not visible; stop evaluation.")
    torch.set_grad_enabled(False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = out_dir / "recon"
    if args.save_recon:
        recon_dir.mkdir(parents=True, exist_ok=True)

    stage_a, model, cfg = load_model(args.ckpt, device)
    for module in (stage_a, model):
        for p in module.parameters():
            p.requires_grad_(False)
    vqgan = load_vqgan(args.glc_weights, device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    dists_fn = DISTS().to(device).eval()
    dataset = FullResFolder(args.data, args.limit)

    rows = []
    sums: dict[str, float] = {}
    delta_gate_mode = cfg.get("delta_gate_mode", "none")
    force_topk_frac = float(cfg.get("force_topk_frac", 0.0))
    hard_topk = bool(cfg.get("hard_topk", False))
    entropy_mode = cfg.get("entropy_mode", "clamped")
    max_symbol_abs = float(cfg.get("max_symbol_abs", 0.0))
    progressive_residual = bool(cfg.get("progressive_residual", False))
    stage1_channels = int(cfg.get("stage1_channels", 0))
    stage1_delta_scale = float(cfg.get("stage1_delta_scale", 1.0))
    stage2_delta_scale = float(cfg.get("stage2_delta_scale", 1.0))
    progressive_stage_topk = bool(cfg.get("progressive_stage_topk", False))
    stage1_topk_frac = float(cfg.get("stage1_topk_frac", 0.0))
    stage2_topk_frac = float(cfg.get("stage2_topk_frac", 0.0))
    topk_score_mode = str(cfg.get("topk_score_mode", "abs"))
    if args.override_topk_score_mode:
        topk_score_mode = args.override_topk_score_mode

    for idx, (x_cpu, name, path) in enumerate(dataset):
        x = x_cpu.unsqueeze(0).to(device, non_blocking=True)
        orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
        pixels = orig_h * orig_w
        x_pad, pads = pad_to_glc_multiple(x, args.padding)
        pad_h, pad_w = int(x_pad.shape[-2]), int(x_pad.shape[-1])
        cuda_sync(device)
        start = time.perf_counter()
        z_s, sem_indices, sem_bytes, sem_bit_width, semantic_decode_max_abs = encode_decode_semantic(stage_a, x_pad)
        y = target_latent(vqgan, x_pad)
        if args.encoder_selector_loss != "none" and not args.disable_residual:
            if progressive_residual:
                raise RuntimeError("encoder_selector_loss is currently implemented for the single residual decoder only")
            selector_l1_weight = args.selector_l1_weight
            selector_lpips_weight = args.selector_lpips_weight
            selector_dists_weight = args.selector_dists_weight
            selector_latent_weight = args.selector_latent_weight
            if args.encoder_selector_loss == "l1_latent" and selector_latent_weight <= 0:
                selector_latent_weight = 1.0
            elif args.encoder_selector_loss == "lpips" and selector_lpips_weight <= 0:
                selector_lpips_weight = 1.0
            elif args.encoder_selector_loss == "dists" and selector_dists_weight <= 0:
                selector_dists_weight = 1.0
            elif args.encoder_selector_loss == "mix":
                if selector_lpips_weight <= 0:
                    selector_lpips_weight = 0.5
                if selector_dists_weight <= 0:
                    selector_dists_weight = 1.0
                if selector_latent_weight <= 0:
                    selector_latent_weight = 0.25
            out = image_guided_selector_forward(
                model=model,
                vqgan=vqgan,
                z_s=z_s,
                target_latent=y,
                x01_padded=x_pad,
                delta_gate_mode=delta_gate_mode,
                force_topk_frac=force_topk_frac,
                hard_topk=hard_topk,
                entropy_mode=entropy_mode,
                max_symbol_abs=max_symbol_abs,
                delta_scale=args.delta_scale,
                selector_loss=args.encoder_selector_loss,
                selector_metric_max_side=args.selector_metric_max_side,
                selector_latent_max_side=args.selector_latent_max_side,
                selector_l1_weight=selector_l1_weight,
                selector_lpips_weight=selector_lpips_weight,
                selector_dists_weight=selector_dists_weight,
                selector_latent_weight=selector_latent_weight,
                lpips_fn=lpips_fn,
                dists_fn=dists_fn,
            )
        else:
            out = model(
                z_s,
                y,
                use_residual=not args.disable_residual,
                delta_gate_mode=delta_gate_mode,
                force_topk_frac=force_topk_frac,
                hard_topk=hard_topk,
                entropy_mode=entropy_mode,
                max_symbol_abs=max_symbol_abs,
                progressive_residual=progressive_residual,
                stage1_channels=stage1_channels,
                stage1_delta_scale=stage1_delta_scale,
                stage2_delta_scale=stage2_delta_scale,
                progressive_stage_topk=progressive_stage_topk,
                stage1_topk_frac=stage1_topk_frac,
                stage2_topk_frac=stage2_topk_frac,
                topk_score_mode=topk_score_mode,
            )
        out_base = model(z_s, y, use_residual=False)
        x_base_pad = from_glc_range(vqgan.generator(out_base["latent_hat"])).clamp(0, 1)
        cuda_sync(device)
        analysis_time = time.perf_counter() - start

        q_symbols = out["q_symbols"].detach()
        scales = out["scales"].detach()
        if args.disable_residual:
            encode_time = 0.0
            decode_time = 0.0
            q_dec = q_symbols
            latent_dec = out["latent_hat"]
            adaptive_scale = torch.ones_like(out["latent_hat"][:, :1])
            payload = sem_bytes
            residual_ac_len = 0
            residual_stream_len = 0
            x_hat_pad = x_base_pad
        else:
            cuda_sync(device)
            enc_start = time.perf_counter()
            residual_stream = _encode_gaussian_symbols(q_symbols, scales)
            payload = build_payload(
                orig_h=orig_h,
                orig_w=orig_w,
                pad_h=pad_h,
                pad_w=pad_w,
                sem_h=int(sem_indices.shape[-2]),
                sem_w=int(sem_indices.shape[-1]),
                sem_bit_width=sem_bit_width,
                sem_bytes=sem_bytes,
                residual_shape=(int(q_symbols.shape[1]), int(q_symbols.shape[2]), int(q_symbols.shape[3])),
                residual_stream=residual_stream,
            )
            residual_ac_len = len(residual_stream.data)
            residual_stream_len = len(residual_stream.data) + STREAM_STRUCT.size
            cuda_sync(device)
            encode_time = time.perf_counter() - enc_start

            cuda_sync(device)
            dec_start = time.perf_counter()
            q_dec = _decode_gaussian_symbols(residual_stream, scales).reshape_as(q_symbols).to(device)
            latent_dec, adaptive_scale = decode_latent_from_symbols(
                model,
                z_s,
                out["mu"],
                q_dec,
                delta_gate_mode,
                args.delta_scale,
                adaptive_delta_scale=args.adaptive_delta_scale,
                delta_scale_min=args.delta_scale_min,
                delta_scale_max=args.delta_scale_max,
                delta_lowpass_kernel=args.delta_lowpass_kernel,
                delta_split_kernel=args.delta_split_kernel,
                delta_low_scale=args.delta_low_scale,
                delta_high_scale=args.delta_high_scale,
                progressive_residual=progressive_residual,
                stage1_channels=stage1_channels,
                stage1_delta_scale=stage1_delta_scale,
                stage2_delta_scale=stage2_delta_scale,
            )
            x_hat_pad = from_glc_range(vqgan.generator(latent_dec)).clamp(0, 1)
            cuda_sync(device)
            decode_time = time.perf_counter() - dec_start

        x_hat = crop_to_original(x_hat_pad, orig_h, orig_w)
        x_base = crop_to_original(x_base_pad, orig_h, orig_w)
        x_metric, x_hat_metric = metric_inputs(x, x_hat, args.metric_max_side)
        _, x_base_metric = metric_inputs(x, x_base, args.metric_max_side)
        lp = float(lpips_fn(x_hat_metric * 2 - 1, x_metric * 2 - 1).mean().item())
        ds = float(dists_fn(x_hat_metric, x_metric).mean().item())
        base_lp = float(lpips_fn(x_base_metric * 2 - 1, x_metric * 2 - 1).mean().item())
        base_ds = float(dists_fn(x_base_metric, x_metric).mean().item())
        l1 = float((x_hat - x).abs().mean().item())
        mse = float(F.mse_loss(x_hat, x).item())
        base_l1 = float((x_base - x).abs().mean().item())
        residual_proxy_bpp_orig = float(out["residual_bpp"].item()) * (pad_h * pad_w) / float(pixels)
        semantic_bits_bpp = float(sem_indices.numel() * sem_bit_width / pixels)
        semantic_stream_bpp = float(len(sem_bytes) * 8.0 / pixels)
        residual_ac_bpp = float(residual_ac_len * 8.0 / pixels)
        residual_stream_bpp = float(residual_stream_len * 8.0 / pixels)
        total_payload_bpp = float(len(payload) * 8.0 / pixels)
        decode_symbol_max_abs = float((q_dec - q_symbols).abs().max().item())
        row = {
            "name": name,
            "path": path,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "semantic_bits_bpp": semantic_bits_bpp,
            "semantic_stream_bpp": semantic_stream_bpp,
            "semantic_bit_width": float(sem_bit_width),
            "semantic_decode_max_abs": semantic_decode_max_abs,
            "residual_proxy_bpp": residual_proxy_bpp_orig,
            "residual_ac_bpp": residual_ac_bpp,
            "residual_stream_bpp": residual_stream_bpp,
            "total_payload_bpp": total_payload_bpp,
            "base_l1": base_l1,
            "base_lpips": base_lp,
            "base_dists": base_ds,
            "l1": l1,
            "mse": mse,
            "lpips": lp,
            "dists": ds,
            "decode_symbol_max_abs": decode_symbol_max_abs,
            "forward_decode_max_abs": float("nan"),
            "rounded_abs_mean": float(out["rounded_abs_mean"].item()),
            "rounded_nonzero_frac": float(out["rounded_nonzero_frac"].item()),
            "rounded_max_abs": float(out["rounded_max_abs"].item()),
            "scale_mean": float(out["scale_mean"].item()),
            "delta_scale": float(args.delta_scale),
            "adaptive_delta_scale_enabled": float(args.adaptive_delta_scale),
            "adaptive_delta_scale_mean": float(adaptive_scale.mean().item()),
            "adaptive_delta_scale_min": float(adaptive_scale.amin().item()),
            "adaptive_delta_scale_max": float(adaptive_scale.amax().item()),
            "delta_lowpass_kernel": float(args.delta_lowpass_kernel),
            "delta_split_kernel": float(args.delta_split_kernel),
            "delta_low_scale": float(args.delta_low_scale),
            "delta_high_scale": float(args.delta_high_scale),
            "disable_residual": float(args.disable_residual),
            "progressive_residual": float(progressive_residual),
            "progressive_stage_topk": float(progressive_stage_topk),
            "stage1_channels": float(stage1_channels),
            "stage1_topk_frac": float(stage1_topk_frac),
            "stage2_topk_frac": float(stage2_topk_frac),
            "topk_score_mode_latent_error": float(topk_score_mode == "latent_error"),
            "topk_score_mode_latent_error_sq": float(topk_score_mode == "latent_error_sq"),
            "topk_score_mode_latent_grad": float(topk_score_mode == "latent_grad"),
            "topk_score_mode_latent_grad_improve": float(topk_score_mode == "latent_grad_improve"),
            "topk_score_mode_learned_selector": float(topk_score_mode == "learned_selector"),
            "encoder_selector_enabled": float(args.encoder_selector_loss != "none"),
            "encoder_selector_loss_value": float(out.get("selector_loss_value", torch.zeros((), device=device)).item()),
            "stage1_delta_scale": float(stage1_delta_scale),
            "stage2_delta_scale": float(stage2_delta_scale),
            "stage1_rounded_nonzero_frac": float(out["stage1_rounded_nonzero_frac"].item()),
            "stage2_rounded_nonzero_frac": float(out["stage2_rounded_nonzero_frac"].item()),
            "stage1_delta_abs_mean": float(out["stage1_delta_abs_mean"].item()),
            "stage2_delta_abs_mean": float(out["stage2_delta_abs_mean"].item()),
            "analysis_time_sec": analysis_time,
            "encode_time_sec": encode_time,
            "decode_time_sec": decode_time,
            "payload_bytes": float(len(payload)),
            "semantic_bytes": float(len(sem_bytes)),
            "residual_ac_bytes": float(residual_ac_len),
            "padding_l": float(pads[0]),
            "padding_r": float(pads[1]),
            "padding_t": float(pads[2]),
            "padding_b": float(pads[3]),
        }
        rows.append(row)
        for k, v in row.items():
            if k not in {"name", "path"}:
                sums[k] = sums.get(k, 0.0) + float(v)
        if args.save_recon and idx < args.save_limit:
            save_image(x_hat.detach().cpu(), recon_dir / Path(name).with_suffix(".png").name)
        if args.print_every > 0 and (idx == 0 or (idx + 1) % args.print_every == 0 or (idx + 1) == len(dataset)):
            print(
                f"[{idx + 1}/{len(dataset)}] {name} payload_bpp={total_payload_bpp:.5f} "
                f"sem={semantic_stream_bpp:.5f} res_ac={residual_ac_bpp:.5f} "
                f"LPIPS {base_lp:.4f}->{lp:.4f} DISTS {base_ds:.4f}->{ds:.4f} "
                f"enc={encode_time:.3f}s dec={decode_time:.3f}s",
                flush=True,
            )
        del x, x_pad, z_s, y, out, out_base, x_hat_pad, x_base_pad, x_hat, x_base, q_dec, latent_dec, adaptive_scale
        torch.cuda.empty_cache()

    metrics = {k: v / max(1, len(rows)) for k, v in sums.items()}
    metrics.update({
        "num_images": len(rows),
        "ckpt": args.ckpt,
        "data": args.data,
        "metric_max_side": args.metric_max_side,
        "padding": args.padding,
        "delta_gate_mode": delta_gate_mode,
        "force_topk_frac": force_topk_frac,
        "hard_topk": hard_topk,
        "entropy_mode": entropy_mode,
        "max_symbol_abs": max_symbol_abs,
        "delta_scale": args.delta_scale,
        "adaptive_delta_scale": args.adaptive_delta_scale,
        "delta_scale_min": args.delta_scale_min,
        "delta_scale_max": args.delta_scale_max,
        "delta_lowpass_kernel": args.delta_lowpass_kernel,
        "delta_split_kernel": args.delta_split_kernel,
        "delta_low_scale": args.delta_low_scale,
        "delta_high_scale": args.delta_high_scale,
        "disable_residual": args.disable_residual,
        "progressive_residual": progressive_residual,
        "progressive_stage_topk": progressive_stage_topk,
        "stage1_channels": stage1_channels,
        "stage1_topk_frac": stage1_topk_frac,
        "stage2_topk_frac": stage2_topk_frac,
        "topk_score_mode": topk_score_mode,
        "encoder_selector_loss": args.encoder_selector_loss,
        "selector_metric_max_side": args.selector_metric_max_side,
        "selector_latent_max_side": args.selector_latent_max_side,
        "selector_l1_weight": args.selector_l1_weight,
        "selector_lpips_weight": args.selector_lpips_weight,
        "selector_dists_weight": args.selector_dists_weight,
        "selector_latent_weight": args.selector_latent_weight,
        "stage1_delta_scale": stage1_delta_scale,
        "stage2_delta_scale": stage2_delta_scale,
    })
    fields = sorted({k for row in rows for k in row.keys()})
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
