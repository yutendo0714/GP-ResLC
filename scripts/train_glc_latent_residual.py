#!/usr/bin/env python3
"""Train the GLC-latent residual scratch branch.

This is the top-conference/high-upside branch: a cheap Stage-A semantic VQ code
predicts the frozen GLC/VQGAN synthesis latent, and only a low-dimensional
unpredictable residual is entropy-modeled.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

import lpips as lpips_lib
from DISTS_pytorch import DISTS

from gp_reslc.scratch import ScratchVQAutoencoder, GLCLatentResidualBottleneck
from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


class RecursiveCropFolder(Dataset):
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.JPEG")

    def __init__(self, root: str, size: int = 256):
        paths: list[str] = []
        for ext in self.EXTS:
            paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        self.paths = sorted(paths)
        if not self.paths:
            raise RuntimeError(f"no images found in {root}")
        self.size = int(size)
        self.t = transforms.Compose([
            transforms.RandomCrop(self.size, pad_if_needed=True, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if min(img.size) < self.size:
            s = self.size / min(img.size)
            img = img.resize((max(self.size, int(img.size[0] * s) + 1),
                              max(self.size, int(img.size[1] * s) + 1)))
        return self.t(img)


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
    model.load_state_dict(ckpt["model"])
    return model, args


def load_glc_vqgan(weights: str, device: str):
    glc = GLC_Image(inplace=False).to(device)
    glc.load_state_dict(get_state_dict(weights), strict=True)
    glc.eval()
    for p in glc.parameters():
        p.requires_grad_(False)
    return glc.vqgan


def to_glc_range(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def from_glc_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0, 1)


@torch.no_grad()
def semantic_forward(stage_a: ScratchVQAutoencoder, x: torch.Tensor):
    vq = stage_a.encode(x)
    bpp = x.new_tensor(stage_a.semantic_index_bpp(x.shape[-2], x.shape[-1], vq.indices.shape[-2], vq.indices.shape[-1]))
    return vq.quantized.detach(), bpp


def trainable_semantic_forward(stage_a: ScratchVQAutoencoder, x: torch.Tensor):
    vq = stage_a.encode(x)
    bpp = x.new_tensor(stage_a.semantic_index_bpp(x.shape[-2], x.shape[-1], vq.indices.shape[-2], vq.indices.shape[-1]))
    return vq.quantized, bpp, vq.loss


@torch.no_grad()
def target_latent(vqgan, x: torch.Tensor) -> torch.Tensor:
    return vqgan.encoder(to_glc_range(x)).detach()


def make_panel(x: torch.Tensor, mu_img: torch.Tensor, x_hat: torch.Tensor, n: int = 4) -> torch.Tensor:
    rows = []
    for i in range(min(n, x.shape[0])):
        rows.extend([x[i].detach().cpu(), mu_img[i].detach().cpu(), x_hat[i].detach().cpu()])
    return make_grid(torch.stack(rows), nrow=3).clamp(0, 1)


def quick_val(stage_a, vqgan, model, loader, device, lpips_fn, dists_fn, train_stage_a: bool, use_residual: bool, quant_mode: str, delta_gate_mode: str, force_topk_frac: float, hard_topk: bool, entropy_mode: str, max_symbol_abs: float, delta_scale: float, adaptive_delta_scale: bool, delta_scale_min: float, delta_scale_max: float, progressive_residual: bool, stage1_channels: int, stage1_delta_scale: float, stage2_delta_scale: float, progressive_stage_topk: bool, stage1_topk_frac: float, stage2_topk_frac: float, topk_score_mode: str, compute_stage1_metrics: bool = False):
    stage_a.eval()
    model.eval()
    x = next(iter(loader)).to(device, non_blocking=True)
    with torch.set_grad_enabled(train_stage_a):
        if train_stage_a:
            z_s, semantic_bpp, _ = trainable_semantic_forward(stage_a, x)
        else:
            z_s, semantic_bpp = semantic_forward(stage_a, x)
    y = target_latent(vqgan, x)
    out = model(z_s, y, use_residual=use_residual, quant_mode=quant_mode, delta_gate_mode=delta_gate_mode, force_topk_frac=force_topk_frac, hard_topk=hard_topk, entropy_mode=entropy_mode, max_symbol_abs=max_symbol_abs, delta_scale=delta_scale, adaptive_delta_scale=adaptive_delta_scale, delta_scale_min=delta_scale_min, delta_scale_max=delta_scale_max, progressive_residual=progressive_residual, stage1_channels=stage1_channels, stage1_delta_scale=stage1_delta_scale, stage2_delta_scale=stage2_delta_scale, progressive_stage_topk=progressive_stage_topk, stage1_topk_frac=stage1_topk_frac, stage2_topk_frac=stage2_topk_frac, topk_score_mode=topk_score_mode)
    out_base = model(z_s, y, use_residual=False)
    x_hat = from_glc_range(vqgan.generator(out["latent_hat"])).clamp(0, 1)
    x_stage1 = from_glc_range(vqgan.generator(out["latent_stage1_hat"])).clamp(0, 1) if compute_stage1_metrics else None
    x_base = from_glc_range(vqgan.generator(out_base["latent_hat"])).clamp(0, 1)
    metrics = {
        "semantic_bpp": float(semantic_bpp.item()),
        "residual_bpp": float(out["residual_bpp"].item()),
        "total_bpp": float((semantic_bpp + out["residual_bpp"]).item()),
        "base_l1": F.l1_loss(x_base, x).item(),
        "base_lpips": lpips_fn(x_base * 2 - 1, x * 2 - 1).mean().item(),
        "base_dists": dists_fn(x_base, x).mean().item(),
        "l1": F.l1_loss(x_hat, x).item(),
        "mse": F.mse_loss(x_hat, x).item(),
        "lpips": lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean().item(),
        "dists": dists_fn(x_hat, x).mean().item(),
        "stage1_l1": F.l1_loss(x_stage1, x).item() if x_stage1 is not None else float("nan"),
        "stage1_lpips": lpips_fn(x_stage1 * 2 - 1, x * 2 - 1).mean().item() if x_stage1 is not None else float("nan"),
        "stage1_dists": dists_fn(x_stage1, x).mean().item() if x_stage1 is not None else float("nan"),
        "pred_loss": out["pred_loss"].item(),
        "latent_loss": out["latent_loss"].item(),
        "residual_abs_mean": out["residual_abs_mean"].item(),
        "residual_std": out["residual_std"].item(),
        "rounded_abs_mean": out["rounded_abs_mean"].item(),
        "rounded_nonzero_frac": out["rounded_nonzero_frac"].item(),
        "stage1_rounded_nonzero_frac": out["stage1_rounded_nonzero_frac"].item(),
        "stage2_rounded_nonzero_frac": out["stage2_rounded_nonzero_frac"].item(),
        "delta_active_frac": out["delta_active_frac"].item(),
        "scale_mean": out["scale_mean"].item(),
        "mu_abs_mean": out["mu_abs_mean"].item(),
        "delta_abs_mean": out["delta_abs_mean"].item(),
        "stage1_delta_abs_mean": out["stage1_delta_abs_mean"].item(),
        "stage2_delta_abs_mean": out["stage2_delta_abs_mean"].item(),
        "adaptive_delta_scale_mean": out["adaptive_delta_scale_mean"].item(),
        "adaptive_delta_scale_min": out["adaptive_delta_scale_min"].item(),
        "adaptive_delta_scale_max": out["adaptive_delta_scale_max"].item(),
        "panel": make_panel(x, x_base, x_hat),
    }
    model.train()
    if train_stage_a:
        stage_a.train()
    return metrics


def _metric_inputs(x_ref: torch.Tensor, x_hat: torch.Tensor, max_side: int):
    if max_side and max_side > 0:
        h, w = x_ref.shape[-2:]
        side = max(h, w)
        if side > max_side:
            scale = float(max_side) / float(side)
            size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
            return (
                F.interpolate(x_ref, size=size, mode="area"),
                F.interpolate(x_hat, size=size, mode="area"),
            )
    return x_ref, x_hat


def _topk_mask_from_scores(scores: torch.Tensor, frac: float) -> torch.Tensor:
    flat_scores = scores.flatten(1)
    k = max(1, min(flat_scores.shape[1], int(flat_scores.shape[1] * float(frac))))
    topk_idx = flat_scores.topk(k, dim=1).indices
    mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask.view_as(scores)


def _selector_delta(model, q_symbols: torch.Tensor, mu: torch.Tensor, z_up: torch.Tensor, delta_gate_mode: str, delta_scale: float):
    q_residual = q_symbols * model.quant_step
    residual_delta = model.residual_decoder(q_residual, mu, z_up)
    if delta_gate_mode == "zero_center":
        residual_delta = residual_delta - model.residual_decoder(torch.zeros_like(q_residual), mu, z_up)
    elif delta_gate_mode in {"payload_hard", "payload_ste"}:
        hard_activity = (q_symbols.detach().abs().sum(dim=1, keepdim=True) > 0).to(residual_delta.dtype)
        residual_delta = residual_delta * hard_activity
    elif delta_gate_mode != "none":
        raise ValueError(f"unsupported delta_gate_mode for selector teacher: {delta_gate_mode}")
    return residual_delta * float(delta_scale)


def mixed_perceptual_teacher_mask(
    *,
    model,
    vqgan,
    z_s: torch.Tensor,
    target_latent: torch.Tensor,
    x01: torch.Tensor,
    force_topk_frac: float,
    delta_gate_mode: str,
    delta_scale: float,
    max_symbol_abs: float,
    metric_max_side: int,
    latent_max_side: int,
    l1_weight: float,
    lpips_weight: float,
    dists_weight: float,
    latent_weight: float,
    lpips_fn,
    dists_fn,
):
    target_size = target_latent.shape[-2:]
    z_up = F.interpolate(z_s, size=target_size, mode="bilinear", align_corners=False).detach()
    with torch.no_grad():
        mu = model.predictor(z_s, target_size).detach()
        residual_latent = model.residual_encoder(target_latent, mu, z_up)
        symbols = residual_latent / model.quant_step
        rounded_symbols = symbols.round()
        forced_sign = torch.where(symbols >= 0, torch.ones_like(symbols), -torch.ones_like(symbols))
        candidate_symbols = torch.where(rounded_symbols == 0, forced_sign, rounded_symbols).detach()
        if max_symbol_abs > 0:
            candidate_symbols = candidate_symbols.clamp(-float(max_symbol_abs), float(max_symbol_abs))
    with torch.enable_grad():
        probe_symbols = torch.zeros_like(candidate_symbols, requires_grad=True)
        residual_delta = _selector_delta(model, probe_symbols, mu, z_up, delta_gate_mode, delta_scale)
        latent_probe = mu + residual_delta
        teacher_loss = x01.new_tensor(0.0)
        if latent_weight > 0:
            teacher_loss = teacher_loss + float(latent_weight) * F.smooth_l1_loss(latent_probe, target_latent.detach())
        if any(w > 0 for w in (l1_weight, lpips_weight, dists_weight)):
            latent_for_image = latent_probe
            x_for_image = x01
            if latent_max_side and latent_max_side > 0:
                lh, lw = latent_probe.shape[-2:]
                side = max(lh, lw)
                if side > latent_max_side:
                    scale = float(latent_max_side) / float(side)
                    new_lh = max(1, int(round(lh * scale)))
                    new_lw = max(1, int(round(lw * scale)))
                    latent_for_image = F.interpolate(latent_probe, size=(new_lh, new_lw), mode="bilinear", align_corners=False)
                    x_for_image = F.interpolate(x01, size=(new_lh * 16, new_lw * 16), mode="area")
            x_probe = from_glc_range(vqgan.generator(latent_for_image))
            x_ref, x_probe_ref = _metric_inputs(x_for_image, x_probe, metric_max_side)
            if l1_weight > 0:
                teacher_loss = teacher_loss + float(l1_weight) * F.smooth_l1_loss(x_probe_ref, x_ref)
            if lpips_weight > 0:
                teacher_loss = teacher_loss + float(lpips_weight) * lpips_fn(x_probe_ref * 2 - 1, x_ref * 2 - 1).mean()
            if dists_weight > 0:
                teacher_loss = teacher_loss + float(dists_weight) * dists_fn(x_probe_ref, x_ref).mean()
        grad = torch.autograd.grad(teacher_loss, probe_symbols, retain_graph=False, create_graph=False)[0]
    first_order_loss_delta = grad.detach() * candidate_symbols
    teacher_scores = (-first_order_loss_delta).clamp_min(0.0)
    if float(teacher_scores.max().item()) <= 0.0:
        teacher_scores = first_order_loss_delta.abs()
    return _topk_mask_from_scores(teacher_scores, force_topk_frac).detach().float(), teacher_scores.detach(), teacher_loss.detach()


def selector_bce_loss(selector_scores: torch.Tensor, teacher_mask: torch.Tensor, pos_weight_cap: float):
    positives = teacher_mask.sum().clamp_min(1.0)
    negatives = (teacher_mask.numel() - teacher_mask.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(max=float(pos_weight_cap))
    loss = F.binary_cross_entropy_with_logits(
        selector_scores,
        teacher_mask,
        pos_weight=selector_scores.new_tensor(float(pos_weight.item())),
    )
    pred_mask = _topk_mask_from_scores(selector_scores.detach(), float(teacher_mask.sum().item()) / float(teacher_mask.numel()))
    teacher_bool = teacher_mask > 0.5
    intersection = (pred_mask & teacher_bool).float().sum()
    precision = intersection / pred_mask.float().sum().clamp_min(1.0)
    recall = intersection / teacher_bool.float().sum().clamp_min(1.0)
    return loss, precision.detach(), recall.detach(), pos_weight.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--stage_a_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="experiments/glc_latent_residual")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--residual_dim", type=int, default=24)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--quant_step", type=float, default=0.5)
    ap.add_argument("--quant_mode", choices=["noise", "ste"], default="noise")
    ap.add_argument("--delta_gate_mode", choices=["none", "payload_ste", "payload_hard", "zero_center"], default="none")
    ap.add_argument("--force_topk_frac", type=float, default=0.0)
    ap.add_argument("--hard_topk", action="store_true")
    ap.add_argument("--entropy_mode", choices=["clamped", "stable"], default="clamped")
    ap.add_argument("--max_symbol_abs", type=float, default=0.0)
    ap.add_argument("--scale_floor", type=float, default=0.11)
    ap.add_argument("--delta_scale", type=float, default=1.0, help="Fixed decoder-side residual delta scale used during training/eval. No side bits.")
    ap.add_argument("--adaptive_delta_scale", action="store_true", help="Use a decoder-side learned gamma map from transmitted residual symbols and semantic prediction. No side bits.")
    ap.add_argument("--delta_scale_min", type=float, default=0.0)
    ap.add_argument("--delta_scale_max", type=float, default=1.0)
    ap.add_argument("--train_delta_gate_only", action="store_true", help="Freeze all bottleneck modules except the adaptive delta-scale head.")
    ap.add_argument("--train_selector_only", action="store_true", help="Freeze all bottleneck modules except the learned residual selector head.")
    ap.add_argument("--selector_teacher_mode", choices=["none", "latent_grad", "latent_grad_improve", "mixed"], default="none", help="Teacher mask used to distill learned_selector top-k selection.")
    ap.add_argument("--lambda_selector_distill", type=float, default=0.0)
    ap.add_argument("--selector_pos_weight_cap", type=float, default=2048.0)
    ap.add_argument("--selector_teacher_metric_max_side", type=int, default=256)
    ap.add_argument("--selector_teacher_latent_max_side", type=int, default=32)
    ap.add_argument("--selector_teacher_l1_weight", type=float, default=0.5)
    ap.add_argument("--selector_teacher_lpips_weight", type=float, default=0.5)
    ap.add_argument("--selector_teacher_dists_weight", type=float, default=1.0)
    ap.add_argument("--selector_teacher_latent_weight", type=float, default=0.25)
    ap.add_argument("--progressive_residual", action="store_true", help="Split residual channels into stage1/stage2 decoders for progressive residual learning.")
    ap.add_argument("--stage1_channels", type=int, default=0)
    ap.add_argument("--stage1_delta_scale", type=float, default=1.0)
    ap.add_argument("--stage2_delta_scale", type=float, default=1.0)
    ap.add_argument("--progressive_stage_topk", action="store_true", help="When progressive residual is enabled, allocate top-k symbols separately to stage1/stage2 channels.")
    ap.add_argument("--stage1_topk_frac", type=float, default=0.0, help="Stage1 top-k fraction. Defaults to --force_topk_frac when <=0.")
    ap.add_argument("--stage2_topk_frac", type=float, default=0.0, help="Stage2 top-k fraction. Defaults to --force_topk_frac when <=0.")
    ap.add_argument("--topk_score_mode", choices=["abs", "latent_error", "latent_error_sq", "latent_grad", "latent_grad_improve", "learned_selector"], default="abs", help="Encoder-side score used to choose sparse residual symbols.")
    ap.add_argument("--init_progressive_decoders_from_single", choices=["none", "stage1", "both"], default="none", help="Initialize progressive residual decoder(s) from the pretrained single residual decoder after resume loading.")
    ap.add_argument("--predictor_only_iters", type=int, default=0)
    ap.add_argument("--train_stage_a", action="store_true")
    ap.add_argument("--freeze_predictor", action="store_true")
    ap.add_argument("--lambda_R", type=float, default=1.0)
    ap.add_argument("--lambda_l1", type=float, default=0.3)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_dists", type=float, default=1.0)
    ap.add_argument("--lambda_pred", type=float, default=1.0)
    ap.add_argument("--lambda_latent", type=float, default=1.0)
    ap.add_argument("--lambda_vq", type=float, default=0.0)
    ap.add_argument("--lambda_active_l1_improve", type=float, default=0.0)
    ap.add_argument("--active_l1_margin", type=float, default=0.0)
    ap.add_argument("--lambda_active_latent_improve", type=float, default=0.0)
    ap.add_argument("--active_latent_margin", type=float, default=0.0)
    ap.add_argument("--lambda_base_l1_improve", type=float, default=0.0)
    ap.add_argument("--base_l1_margin", type=float, default=0.0)
    ap.add_argument("--lambda_base_lpips_improve", type=float, default=0.0)
    ap.add_argument("--base_lpips_margin", type=float, default=0.0)
    ap.add_argument("--lambda_base_dists_improve", type=float, default=0.0)
    ap.add_argument("--base_dists_margin", type=float, default=0.0)
    ap.add_argument("--lambda_rounded_abs", type=float, default=0.0)
    ap.add_argument("--lambda_delta_abs", type=float, default=0.0)
    ap.add_argument("--lambda_adaptive_delta_scale_mean", type=float, default=0.0)
    ap.add_argument("--lambda_stage1_l1", type=float, default=0.0)
    ap.add_argument("--lambda_stage1_lpips", type=float, default=0.0)
    ap.add_argument("--lambda_stage1_dists", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume_weights_only", action="store_true")
    ap.add_argument("--reset_best_on_resume", action="store_true")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="gp-reslc-research")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU is not visible. Stop here and restart the container before training.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_a, stage_a_args = load_stage_a(args.stage_a_ckpt, device)
    if args.train_stage_a:
        stage_a.train()
    else:
        stage_a.eval()
        for p in stage_a.parameters():
            p.requires_grad_(False)
    vqgan = load_glc_vqgan(args.glc_weights, device)
    model = GLCLatentResidualBottleneck(
        semantic_dim=stage_a_args["latent_dim"],
        target_dim=256,
        residual_dim=args.residual_dim,
        hidden_dim=args.hidden_dim,
        quant_step=args.quant_step,
        scale_floor=args.scale_floor,
    ).to(device).train()
    if args.freeze_predictor:
        model.predictor.eval()
        for p in model.predictor.parameters():
            p.requires_grad_(False)
    if args.train_delta_gate_only:
        if not args.adaptive_delta_scale:
            raise RuntimeError("--train_delta_gate_only requires --adaptive_delta_scale")
        for p in model.parameters():
            p.requires_grad_(False)
        model.delta_scale_net.train()
        for p in model.delta_scale_net.parameters():
            p.requires_grad_(True)
    if args.train_selector_only:
        if args.topk_score_mode != "learned_selector":
            raise RuntimeError("--train_selector_only requires --topk_score_mode learned_selector")
        if args.selector_teacher_mode == "none" or args.lambda_selector_distill <= 0:
            raise RuntimeError("--train_selector_only requires --selector_teacher_mode and --lambda_selector_distill > 0")
        for p in model.parameters():
            p.requires_grad_(False)
        model.selector_net.train()
        for p in model.selector_net.parameters():
            p.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.train_stage_a:
        params += [p for p in stage_a.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    it = 0
    best_val_dists = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[resume] non-strict model load missing={missing} unexpected={unexpected}", flush=True)
        if args.train_stage_a and "stage_a_model" in ckpt:
            stage_a.load_state_dict(ckpt["stage_a_model"])
        if "optimizer" in ckpt and not args.resume_weights_only:
            opt.load_state_dict(ckpt["optimizer"])
            for group in opt.param_groups:
                group["lr"] = args.lr
        it = int(ckpt.get("it", 0))
        best_val_dists = float(ckpt.get("best_val_dists", best_val_dists))
        if args.init_progressive_decoders_from_single != "none":
            model.residual_decoder_stage1.load_state_dict(model.residual_decoder.state_dict())
            copied = ["stage1"]
            if args.init_progressive_decoders_from_single == "both":
                model.residual_decoder_stage2.load_state_dict(model.residual_decoder.state_dict())
                copied.append("stage2")
            print(f"[resume] initialized progressive decoders from single residual decoder: {','.join(copied)}", flush=True)
        if args.reset_best_on_resume:
            best_val_dists = float("inf")
        print(f"[resume] {args.resume} it={it} best_val_dists={best_val_dists:.6f}", flush=True)

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    dists_fn = DISTS().to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    for p in dists_fn.parameters():
        p.requires_grad_(False)

    use_wandb = (not args.no_wandb) and args.wandb_mode != "disabled" and _WANDB
    if use_wandb:
        cfg = vars(args).copy()
        cfg.update({f"stage_a/{k}": v for k, v in stage_a_args.items()})
        wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=cfg, dir=str(ROOT))

    train_loader = DataLoader(
        RecursiveCropFolder(args.data, args.size),
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        RecursiveCropFolder(args.val or args.data, args.size),
        batch_size=min(args.bs, 4),
        shuffle=True,
        num_workers=max(0, min(args.num_workers, 2)),
        drop_last=True,
        pin_memory=True,
    )

    while it < args.iters:
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            if args.train_stage_a:
                z_s, semantic_bpp, vq_loss = trainable_semantic_forward(stage_a, x)
            else:
                z_s, semantic_bpp = semantic_forward(stage_a, x)
                vq_loss = x.new_tensor(0.0)
            y = target_latent(vqgan, x)
            use_residual = it >= args.predictor_only_iters
            out = model(z_s, y, use_residual=use_residual, quant_mode=args.quant_mode, delta_gate_mode=args.delta_gate_mode, force_topk_frac=args.force_topk_frac, hard_topk=args.hard_topk, entropy_mode=args.entropy_mode, max_symbol_abs=args.max_symbol_abs, delta_scale=args.delta_scale, adaptive_delta_scale=args.adaptive_delta_scale, delta_scale_min=args.delta_scale_min, delta_scale_max=args.delta_scale_max, progressive_residual=args.progressive_residual, stage1_channels=args.stage1_channels, stage1_delta_scale=args.stage1_delta_scale, stage2_delta_scale=args.stage2_delta_scale, progressive_stage_topk=args.progressive_stage_topk, stage1_topk_frac=args.stage1_topk_frac, stage2_topk_frac=args.stage2_topk_frac, topk_score_mode=args.topk_score_mode)
            out_base = model(z_s, y, use_residual=False)
            selector_distill = x.new_tensor(0.0)
            selector_precision = x.new_tensor(0.0)
            selector_recall = x.new_tensor(0.0)
            selector_pos_weight = x.new_tensor(0.0)
            selector_teacher_loss = x.new_tensor(0.0)
            if args.selector_teacher_mode != "none" and use_residual:
                if args.selector_teacher_mode == "mixed":
                    teacher_mask, _, selector_teacher_loss = mixed_perceptual_teacher_mask(
                        model=model,
                        vqgan=vqgan,
                        z_s=z_s.detach(),
                        target_latent=y.detach(),
                        x01=x.detach(),
                        force_topk_frac=args.force_topk_frac,
                        delta_gate_mode=args.delta_gate_mode,
                        delta_scale=args.delta_scale,
                        max_symbol_abs=args.max_symbol_abs,
                        metric_max_side=args.selector_teacher_metric_max_side,
                        latent_max_side=args.selector_teacher_latent_max_side,
                        l1_weight=args.selector_teacher_l1_weight,
                        lpips_weight=args.selector_teacher_lpips_weight,
                        dists_weight=args.selector_teacher_dists_weight,
                        latent_weight=args.selector_teacher_latent_weight,
                        lpips_fn=lpips_fn,
                        dists_fn=dists_fn,
                    )
                else:
                    teacher_out = model(
                        z_s.detach(), y.detach(), use_residual=True, quant_mode=args.quant_mode,
                        delta_gate_mode=args.delta_gate_mode, force_topk_frac=args.force_topk_frac,
                        hard_topk=True, entropy_mode=args.entropy_mode, max_symbol_abs=args.max_symbol_abs,
                        delta_scale=args.delta_scale, adaptive_delta_scale=args.adaptive_delta_scale,
                        delta_scale_min=args.delta_scale_min, delta_scale_max=args.delta_scale_max,
                        progressive_residual=False, topk_score_mode=args.selector_teacher_mode,
                    )
                    teacher_mask = teacher_out["topk_mask"].detach()
                selector_distill, selector_precision, selector_recall, selector_pos_weight = selector_bce_loss(
                    out["selector_scores"], teacher_mask, args.selector_pos_weight_cap
                )
            need_stage1_loss = args.progressive_residual and (args.lambda_stage1_l1 > 0 or args.lambda_stage1_lpips > 0 or args.lambda_stage1_dists > 0)
            x_hat = from_glc_range(vqgan.generator(out["latent_hat"]))
            x_stage1 = from_glc_range(vqgan.generator(out["latent_stage1_hat"])) if need_stage1_loss else None
            x_base = from_glc_range(vqgan.generator(out_base["latent_hat"]))

            l1 = F.l1_loss(x_hat, x)
            lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean()
            ds = dists_fn(x_hat, x).mean()
            total_bpp = semantic_bpp + out["residual_bpp"]
            if need_stage1_loss:
                stage1_l1 = F.l1_loss(x_stage1, x)
                stage1_lp = lpips_fn(x_stage1 * 2 - 1, x * 2 - 1).mean()
                stage1_ds = dists_fn(x_stage1, x).mean()
            else:
                stage1_l1 = x.new_tensor(0.0)
                stage1_lp = x.new_tensor(0.0)
                stage1_ds = x.new_tensor(0.0)
            if args.lambda_active_l1_improve > 0 and use_residual:
                active = F.interpolate(out["delta_activity"], size=x.shape[-2:], mode="nearest")
                err_hat = (x_hat - x).abs().mean(dim=1, keepdim=True)
                err_base = (x_base.detach() - x).abs().mean(dim=1, keepdim=True)
                active_l1_improve = (active * F.relu(err_hat - err_base + args.active_l1_margin)).sum() / active.sum().clamp_min(1.0)
            else:
                active_l1_improve = x.new_tensor(0.0)
            if args.lambda_active_latent_improve > 0 and use_residual:
                latent_active = out["delta_activity"]
                latent_err_hat = (out["latent_hat"] - y.detach()).abs().mean(dim=1, keepdim=True)
                latent_err_base = (out["mu"].detach() - y.detach()).abs().mean(dim=1, keepdim=True)
                active_latent_improve = (
                    latent_active * F.relu(latent_err_hat - latent_err_base + args.active_latent_margin)
                ).sum() / latent_active.sum().clamp_min(1.0)
            else:
                active_latent_improve = x.new_tensor(0.0)
            if args.lambda_base_l1_improve > 0 and use_residual:
                with torch.no_grad():
                    base_l1_ref = F.l1_loss(x_base.detach(), x)
                base_l1_improve = F.relu(l1 - base_l1_ref + args.base_l1_margin)
            else:
                base_l1_improve = x.new_tensor(0.0)
            if args.lambda_base_lpips_improve > 0 and use_residual:
                with torch.no_grad():
                    base_lpips_ref = lpips_fn(x_base.detach() * 2 - 1, x * 2 - 1).mean()
                base_lpips_improve = F.relu(lp - base_lpips_ref + args.base_lpips_margin)
            else:
                base_lpips_improve = x.new_tensor(0.0)
            if args.lambda_base_dists_improve > 0 and use_residual:
                with torch.no_grad():
                    base_dists_ref = dists_fn(x_base.detach(), x).mean()
                base_dists_improve = F.relu(ds - base_dists_ref + args.base_dists_margin)
            else:
                base_dists_improve = x.new_tensor(0.0)
            loss = (
                args.lambda_R * total_bpp
                + args.lambda_l1 * l1
                + args.lambda_lpips * lp
                + args.lambda_dists * ds
                + args.lambda_pred * out["pred_loss"]
                + args.lambda_latent * out["latent_loss"]
                + args.lambda_vq * vq_loss
                + args.lambda_active_l1_improve * active_l1_improve
                + args.lambda_active_latent_improve * active_latent_improve
                + args.lambda_base_l1_improve * base_l1_improve
                + args.lambda_base_lpips_improve * base_lpips_improve
                + args.lambda_base_dists_improve * base_dists_improve
                + args.lambda_rounded_abs * out["rounded_abs_mean"]
                + args.lambda_delta_abs * out["delta_abs_mean"]
                + args.lambda_adaptive_delta_scale_mean * out["adaptive_delta_scale_mean"]
                + args.lambda_selector_distill * selector_distill
                + args.lambda_stage1_l1 * stage1_l1
                + args.lambda_stage1_lpips * stage1_lp
                + args.lambda_stage1_dists * stage1_ds
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if it % args.log_every == 0:
                base_lp = lpips_fn(x_base * 2 - 1, x * 2 - 1).mean().item()
                base_ds = dists_fn(x_base, x).mean().item()
                print(
                    f"[it {it}] loss={loss.item():.4f} total_bpp={total_bpp.item():.5f} "
                    f"sem={semantic_bpp.item():.5f} res={out['residual_bpp'].item():.5f} "
                    f"l1={l1.item():.4f} lpips={lp.item():.4f} dists={ds.item():.4f} "
                    f"base_lpips={base_lp:.4f} base_dists={base_ds:.4f} "
                    f"pred={out['pred_loss'].item():.4f} latent={out['latent_loss'].item():.4f} "
                    f"act_l1={active_l1_improve.item():.4f} "
                    f"act_lat={active_latent_improve.item():.4f} "
                    f"base_l1_reg={base_l1_improve.item():.4f} "
                    f"base_lp_reg={base_lpips_improve.item():.4f} "
                    f"base_ds_reg={base_dists_improve.item():.4f} "
                    f"sel_loss={selector_distill.item():.4f} sel_p={selector_precision.item():.3f} sel_r={selector_recall.item():.3f} "
                    f"round_reg={out['rounded_abs_mean'].item():.4f} "
                    f"delta_reg={out['delta_abs_mean'].item():.4f} "
                    f"adscale={out['adaptive_delta_scale_mean'].item():.3f} "
                    f"s1_l1={stage1_l1.item():.4f} s1_lp={stage1_lp.item():.4f} s1_ds={stage1_ds.item():.4f} "
                    f"r_abs={out['residual_abs_mean'].item():.3f} "
                    f"round_abs={out['rounded_abs_mean'].item():.3f} "
                    f"round_nz={out['rounded_nonzero_frac'].item():.3f} "
                    f"s1_nz={out['stage1_rounded_nonzero_frac'].item():.3f} "
                    f"s2_nz={out['stage2_rounded_nonzero_frac'].item():.3f} "
                    f"delta_act={out['delta_active_frac'].item():.3f} "
                    f"scale={out['scale_mean'].item():.3f} "
                    f"qmode={args.quant_mode} dgate={args.delta_gate_mode} topk={args.force_topk_frac:.4f} hard_topk={int(args.hard_topk)} "
                    f"prog_topk={int(args.progressive_stage_topk)} s1_topk={args.stage1_topk_frac:.4f} s2_topk={args.stage2_topk_frac:.4f} "
                    f"score={args.topk_score_mode} entropy={args.entropy_mode} maxsym={args.max_symbol_abs:.2f} dscale={args.delta_scale:.2f} "
                    f"use_residual={int(use_residual)}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/total_bpp": total_bpp.item(),
                        "train/semantic_bpp": semantic_bpp.item(),
                        "train/residual_bpp": out["residual_bpp"].item(),
                        "train/l1": l1.item(),
                        "train/lpips": lp.item(),
                        "train/dists": ds.item(),
                        "train/base_lpips": base_lp,
                        "train/base_dists": base_ds,
                        "train/pred_loss": out["pred_loss"].item(),
                        "train/latent_loss": out["latent_loss"].item(),
                        "train/active_l1_improve": active_l1_improve.item(),
                        "train/active_latent_improve": active_latent_improve.item(),
                        "train/base_l1_improve": base_l1_improve.item(),
                        "train/base_lpips_improve": base_lpips_improve.item(),
                        "train/base_dists_improve": base_dists_improve.item(),
                        "train/selector_distill": selector_distill.item(),
                        "train/selector_precision": selector_precision.item(),
                        "train/selector_recall": selector_recall.item(),
                        "train/selector_pos_weight": selector_pos_weight.item(),
                        "train/selector_teacher_loss": selector_teacher_loss.item(),
                        "train/selector_scores_mean": out["selector_scores_mean"].item(),
                        "train/selector_scores_std": out["selector_scores_std"].item(),
                        "train/rounded_abs_regularizer": out["rounded_abs_mean"].item(),
                        "train/delta_abs_regularizer": out["delta_abs_mean"].item(),
                        "train/residual_abs_mean": out["residual_abs_mean"].item(),
                        "train/residual_std": out["residual_std"].item(),
                        "train/rounded_abs_mean": out["rounded_abs_mean"].item(),
                        "train/rounded_nonzero_frac": out["rounded_nonzero_frac"].item(),
                        "train/stage1_rounded_nonzero_frac": out["stage1_rounded_nonzero_frac"].item(),
                        "train/stage2_rounded_nonzero_frac": out["stage2_rounded_nonzero_frac"].item(),
                        "train/delta_active_frac": out["delta_active_frac"].item(),
                        "train/scale_mean": out["scale_mean"].item(),
                        "train/mu_abs_mean": out["mu_abs_mean"].item(),
                        "train/delta_abs_mean": out["delta_abs_mean"].item(),
                        "train/stage1_delta_abs_mean": out["stage1_delta_abs_mean"].item(),
                        "train/stage2_delta_abs_mean": out["stage2_delta_abs_mean"].item(),
                        "train/stage1_l1": stage1_l1.item(),
                        "train/stage1_lpips": stage1_lp.item(),
                        "train/stage1_dists": stage1_ds.item(),
                        "train/adaptive_delta_scale_mean": out["adaptive_delta_scale_mean"].item(),
                        "train/adaptive_delta_scale_min": out["adaptive_delta_scale_min"].item(),
                        "train/adaptive_delta_scale_max": out["adaptive_delta_scale_max"].item(),
                        "train/use_residual": float(use_residual),
                        "train/force_topk_frac": args.force_topk_frac,
                        "train/progressive_stage_topk": float(args.progressive_stage_topk),
                        "train/stage1_topk_frac": args.stage1_topk_frac,
                        "train/stage2_topk_frac": args.stage2_topk_frac,
                        "train/topk_score_mode_latent_error": float(args.topk_score_mode == "latent_error"),
                        "train/topk_score_mode_latent_error_sq": float(args.topk_score_mode == "latent_error_sq"),
                        "train/topk_score_mode_latent_grad": float(args.topk_score_mode == "latent_grad"),
                        "train/topk_score_mode_latent_grad_improve": float(args.topk_score_mode == "latent_grad_improve"),
                        "train/topk_score_mode_learned_selector": float(args.topk_score_mode == "learned_selector"),
                        "train/hard_topk": float(args.hard_topk),
                        "train/entropy_mode_stable": float(args.entropy_mode == "stable"),
                        "train/max_symbol_abs": args.max_symbol_abs,
                        "train/delta_scale": args.delta_scale,
                    }, step=it)

            if it % args.eval_every == 0:
                val = quick_val(stage_a, vqgan, model, val_loader, device, lpips_fn, dists_fn, args.train_stage_a, use_residual, args.quant_mode, args.delta_gate_mode, args.force_topk_frac, args.hard_topk, args.entropy_mode, args.max_symbol_abs, args.delta_scale, args.adaptive_delta_scale, args.delta_scale_min, args.delta_scale_max, args.progressive_residual, args.stage1_channels, args.stage1_delta_scale, args.stage2_delta_scale, args.progressive_stage_topk, args.stage1_topk_frac, args.stage2_topk_frac, args.topk_score_mode, compute_stage1_metrics=need_stage1_loss)
                panel_path = out_dir / f"val_panel_{it:07d}.png"
                save_image(val["panel"], panel_path)
                print(
                    f"  [val {it}] total_bpp={val['total_bpp']:.5f} res={val['residual_bpp']:.5f} "
                    f"base_lpips={val['base_lpips']:.4f} lpips={val['lpips']:.4f} "
                    f"base_dists={val['base_dists']:.4f} dists={val['dists']:.4f} "
                    f"pred={val['pred_loss']:.4f} latent={val['latent_loss']:.4f} "
                    f"round_nz={val['rounded_nonzero_frac']:.3f} s1_nz={val['stage1_rounded_nonzero_frac']:.3f} "
                    f"s2_nz={val['stage2_rounded_nonzero_frac']:.3f} delta_act={val['delta_active_frac']:.3f}",
                    flush=True,
                )
                if val["dists"] < best_val_dists:
                    best_val_dists = val["dists"]
                    state = {
                        "it": it,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "args": vars(args),
                        "stage_a_args": stage_a_args,
                        "best_val_dists": best_val_dists,
                    }
                    if args.train_stage_a:
                        state["stage_a_model"] = stage_a.state_dict()
                    torch.save(state, out_dir / "glc_latent_residual_best.pt")
                if use_wandb:
                    log = {f"val/{k}": v for k, v in val.items() if k != "panel"}
                    log["val/best_dists"] = best_val_dists
                    log["val/panel"] = wandb.Image(str(panel_path))
                    wandb.log(log, step=it)
                model.train()
                if args.train_stage_a:
                    stage_a.train()

            if it > 0 and it % args.save_every == 0:
                state = {
                    "it": it,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": vars(args),
                    "stage_a_args": stage_a_args,
                    "best_val_dists": best_val_dists,
                }
                if args.train_stage_a:
                    state["stage_a_model"] = stage_a.state_dict()
                torch.save(state, out_dir / f"glc_latent_residual_{it:07d}.pt")
            it += 1
            if it >= args.iters:
                break

    state = {
        "it": it,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "args": vars(args),
        "stage_a_args": stage_a_args,
        "best_val_dists": best_val_dists,
    }
    if args.train_stage_a:
        state["stage_a_model"] = stage_a.state_dict()
    torch.save(state, out_dir / "glc_latent_residual_final.pt")
    if use_wandb:
        wandb.save(str(out_dir / "glc_latent_residual_final.pt"))
        wandb.finish()


if __name__ == "__main__":
    main()
