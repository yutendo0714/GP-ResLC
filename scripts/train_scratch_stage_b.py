#!/usr/bin/env python3
"""Train Scratch GP-ResLC Stage B: semantic-conditioned residual coding.

Stage A provides a cheap semantic/generative code s. Stage B freezes that code,
learns mu_theta(s), and optimizes an entropy proxy for only the unpredictable
residual r = y - mu_theta(s).
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

from gp_reslc.scratch import ScratchVQAutoencoder, ScratchResidualBottleneck, ScratchProgressiveResidualBottleneck

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


class CropFolder(Dataset):
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

    def __init__(self, root: str, size: int = 256):
        self.paths = sorted(sum([glob.glob(os.path.join(root, e)) for e in self.EXTS], []))
        if not self.paths:
            raise RuntimeError(f"no images found in {root}")
        self.size = int(size)
        self.t = transforms.Compose([
            transforms.RandomCrop(self.size, pad_if_needed=True, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.paths[i]).convert("RGB")
        if min(img.size) < self.size:
            s = self.size / min(img.size)
            img = img.resize((max(self.size, int(img.size[0] * s) + 1),
                              max(self.size, int(img.size[1] * s) + 1)))
        return self.t(img)


def load_stage_a(path: str, device: str) -> tuple[ScratchVQAutoencoder, dict]:
    ckpt = torch.load(path, map_location=device)
    a = dict(ckpt.get("args", {}))
    if "codebook_size" not in a:
        a = dict(ckpt.get("stage_a_args", {}))
    required = ["codebook_size", "latent_dim", "base_ch"]
    missing = [k for k in required if k not in a]
    if missing:
        raise RuntimeError(f"Stage-A checkpoint is missing args: {missing}")
    model = ScratchVQAutoencoder(
        a["codebook_size"],
        a["latent_dim"],
        a["base_ch"],
        a.get("vq_beta", 0.25),
        a.get("vq_entropy_tau", 1.0),
        a.get("num_down", 4),
        decoder_attention=a.get("decoder_attention", False),
        extra_decoder_blocks=a.get("extra_decoder_blocks", 0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, a


@torch.no_grad()
def semantic_forward(stage_a: ScratchVQAutoencoder, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vq = stage_a.encode(x)
    base = stage_a.decode(vq.quantized).clamp(0, 1)
    bpp = x.new_tensor(stage_a.semantic_index_bpp(x.shape[-2], x.shape[-1], vq.indices.shape[-2], vq.indices.shape[-1]))
    return vq.quantized.detach(), base.detach(), bpp


def parse_stage_quant_steps(raw: str, quant_step: float, stages: int) -> tuple[float, ...]:
    if raw:
        steps = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
        if len(steps) != int(stages):
            raise ValueError("--stage_quant_steps length must match --progressive_stages")
        return steps
    return tuple(float(quant_step) * (0.5 ** i) for i in range(int(stages)))


def build_stage_b_model(args, stage_a_args, device):
    codec = getattr(args, "residual_codec", "single")
    common = dict(
        semantic_dim=stage_a_args["latent_dim"],
        residual_dim=args.residual_dim,
        base_ch=args.base_ch,
        num_down=stage_a_args.get("num_down", 4),
        quant_step=args.quant_step,
    )
    if codec == "single":
        return ScratchResidualBottleneck(**common).to(device)
    if codec == "progressive":
        return ScratchProgressiveResidualBottleneck(
            **common,
            progressive_stages=args.progressive_stages,
            stage_quant_steps=parse_stage_quant_steps(args.stage_quant_steps, args.quant_step, args.progressive_stages),
            gate_extra_stages=args.progressive_gate,
            gate_threshold=args.gate_threshold,
            gate_init_bias=args.gate_init_bias,
            gate_soft_train=args.gate_soft_train,
            stage_correction_decoder=args.stage_correction_decoder,
            gate_topk_frac=args.gate_topk_frac,
        ).to(device)
    raise ValueError(f"unknown residual codec: {codec}")


def load_compatible_state(model, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    model_state = model.state_dict()
    kept = {k: v for k, v in state.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
    missing = sorted(set(model_state) - set(kept))
    skipped = sorted(set(state) - set(kept))
    model_state.update(kept)
    model.load_state_dict(model_state)
    print(f"[init_from] loaded {len(kept)} tensors from {ckpt_path}; missing={len(missing)} skipped={len(skipped)}", flush=True)
    if missing[:12]:
        print("[init_from] first missing: " + ", ".join(missing[:12]), flush=True)


def make_panel(x: torch.Tensor, base: torch.Tensor, x_hat: torch.Tensor, n: int = 4) -> torch.Tensor:
    n = min(n, x.shape[0])
    rows = []
    for i in range(n):
        rows.extend([x[i].detach().cpu(), base[i].detach().cpu(), x_hat[i].detach().cpu()])
    return make_grid(torch.stack(rows), nrow=3).clamp(0, 1)


def selected_region_improvement_loss(out: dict[str, torch.Tensor], x: torch.Tensor, x_hat: torch.Tensor, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Encourage hard-gated fine residual positions to improve stage-0 local error.

    The gate is decoder-computable, so this does not introduce side information.
    We detach the mask and stage-0 reconstruction: the loss trains the selected
    residual/correction path to be useful instead of moving the selector itself
    through a soft shortcut.
    """
    if "stage1_gate_map" not in out or "stage0_x_hat" not in out:
        zero = x.new_tensor(0.0)
        return zero, zero
    gate = out["stage1_gate_map"].detach()
    mask = gate.mean(dim=1, keepdim=True)
    mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
    denom = mask.sum().clamp_min(1.0)
    stage0_err = (out["stage0_x_hat"].detach().clamp(0, 1) - x).abs().mean(dim=1, keepdim=True)
    final_err = (x_hat - x).abs().mean(dim=1, keepdim=True)
    loss = (F.relu(final_err - stage0_err + float(margin)) * mask).sum() / denom
    return loss, mask.mean()


def selected_region_lpips_improvement_loss(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lpips_spatial_fn,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perceptual selected-region improvement using LPIPS spatial maps."""
    if "stage1_gate_map" not in out or "stage0_x_hat" not in out:
        zero = x.new_tensor(0.0)
        return zero, zero
    gate = out["stage1_gate_map"].detach()
    mask = gate.mean(dim=1, keepdim=True)
    stage0 = out["stage0_x_hat"].detach().clamp(0, 1)
    stage0_map = lpips_spatial_fn(stage0 * 2 - 1, x * 2 - 1).detach()
    final_map = lpips_spatial_fn(x_hat * 2 - 1, x * 2 - 1)
    mask = F.interpolate(mask, size=final_map.shape[-2:], mode="nearest")
    denom = mask.sum().clamp_min(1.0)
    loss = (F.relu(final_map - stage0_map + float(margin)) * mask).sum() / denom
    return loss, mask.mean()


def selected_region_vgg_feature_improvement_loss(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    x_hat: torch.Tensor,
    dists_fn,
    layers: tuple[int, ...],
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DISTS-adjacent selected-region improvement via local VGG feature errors."""
    if "stage1_gate_map" not in out or "stage0_x_hat" not in out:
        zero = x.new_tensor(0.0)
        return zero, zero
    gate = out["stage1_gate_map"].detach()
    mask = gate.mean(dim=1, keepdim=True)
    stage0 = out["stage0_x_hat"].detach().clamp(0, 1)
    with torch.no_grad():
        target_feats = dists_fn.forward_once(x)
        stage0_feats = dists_fn.forward_once(stage0)
    final_feats = dists_fn.forward_once(x_hat)
    total = x.new_tensor(0.0)
    used = 0
    for idx in layers:
        if idx < 0 or idx >= len(final_feats):
            continue
        target = target_feats[idx].detach()
        stage0_err = (stage0_feats[idx].detach() - target).abs().mean(dim=1, keepdim=True)
        final_err = (final_feats[idx] - target).abs().mean(dim=1, keepdim=True)
        layer_mask = F.interpolate(mask, size=final_err.shape[-2:], mode="nearest")
        denom = layer_mask.sum().clamp_min(1.0)
        total = total + (F.relu(final_err - stage0_err + float(margin)) * layer_mask).sum() / denom
        used += 1
    if used == 0:
        zero = x.new_tensor(0.0)
        return zero, mask.mean()
    return total / used, mask.mean()


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def gate_base_error_target_loss(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    base: torch.Tensor,
    topk_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teach decoder-side fine-stage gates to predict Stage-A failure regions.

    The target uses x only during training. At test time the gate is still
    produced from decoder-available z_s/y_hat, so no side information is added.
    """
    if "stage1_gate_logit" not in out:
        zero = x.new_tensor(0.0)
        return zero, zero
    logits = out["stage1_gate_logit"]
    frac = float(topk_frac) if float(topk_frac) > 0 else 0.10
    frac = max(1.0 / max(1, logits.shape[-2] * logits.shape[-1]), min(frac, 1.0))
    with torch.no_grad():
        err = (base.detach().clamp(0, 1) - x).abs().mean(dim=1, keepdim=True)
        err = F.adaptive_avg_pool2d(err, logits.shape[-2:])
        flat = err.flatten(1)
        k = max(1, int(flat.shape[1] * frac))
        kth = torch.topk(flat, k, dim=1).values[:, -1].view(-1, 1, 1, 1)
        target_spatial = (err >= kth).to(logits.dtype)
        target = target_spatial.expand_as(logits)
        pos_frac = target.mean().clamp(1e-4, 0.999)
        pos_weight = ((1.0 - pos_frac) / pos_frac).detach()
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return loss, target.mean()


def quick_val(stage_a, model, loader, device, lpips_fn, dists_fn):
    model.eval()
    x = next(iter(loader)).to(device)
    z_s, base, semantic_bpp = semantic_forward(stage_a, x)
    out = model(x, base, z_s)
    x_hat = out["x_hat"].clamp(0, 1)
    base_lpips = lpips_fn(base * 2 - 1, x * 2 - 1).mean().item()
    ours_lpips = lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean().item()
    base_dists = dists_fn(base, x).mean().item()
    ours_dists = dists_fn(x_hat, x).mean().item()
    panel = make_panel(x, base, x_hat)
    model.train()
    metrics = {
        "base_l1": F.l1_loss(base, x).item(),
        "base_lpips": base_lpips,
        "base_dists": base_dists,
        "l1": F.l1_loss(x_hat, x).item(),
        "mse": F.mse_loss(x_hat, x).item(),
        "lpips": ours_lpips,
        "dists": ours_dists,
        "semantic_bpp": float(semantic_bpp.item()),
        "residual_bpp": float(out["residual_bpp"].item()),
        "total_bpp": float((semantic_bpp + out["residual_bpp"]).item()),
        "pred_loss": float(out["pred_loss"].item()),
        "residual_abs_mean": float(out["residual_abs_mean"].item()),
        "residual_std": float(out["residual_std"].item()),
        "scale_mean": float(out["scale_mean"].item()),
        "panel": panel,
    }
    for k, v in out.items():
        if k.startswith("stage") and k.endswith(("bpp", "residual_abs_mean", "scale_mean", "gate_mean")):
            metrics[k] = float(v.item())
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="experiments/scratch_stage_b")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base_ch", type=int, default=96)
    ap.add_argument("--residual_dim", type=int, default=32)
    ap.add_argument("--quant_step", type=float, default=0.5)
    ap.add_argument("--residual_codec", choices=["single", "progressive"], default="single")
    ap.add_argument("--progressive_stages", type=int, default=2)
    ap.add_argument("--stage_quant_steps", default="", help="comma-separated quant steps, e.g. 1.0,0.5")
    ap.add_argument("--progressive_gate", action="store_true", help="decoder-side hard gate for stages after stage 0")
    ap.add_argument("--gate_threshold", type=float, default=0.2)
    ap.add_argument("--gate_init_bias", type=float, default=-2.0)
    ap.add_argument("--gate_soft_train", action="store_true", help="use soft gates during training and hard gates during eval")
    ap.add_argument("--stage_correction_decoder", action="store_true", help="use extra correction decoders for stages after stage 0")
    ap.add_argument("--gate_topk_frac", type=float, default=0.0, help="decoder-side top-k fraction for extra-stage gates; 0 disables")
    ap.add_argument("--lambda_gate_error_target", type=float, default=0.0, help="teach fine-stage gate to predict high Stage-A base-error regions")
    ap.add_argument("--gate_error_target_topk_frac", type=float, default=0.0, help="top-k spatial fraction for gate target; defaults to gate_topk_frac or 0.10")
    ap.add_argument("--train_only_extra_stages", action="store_true", help="freeze base residual path and train only extra stage modules")
    ap.add_argument("--lambda_R", type=float, default=1.0)
    ap.add_argument("--lambda_l1", type=float, default=0.5)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_dists", type=float, default=1.0)
    ap.add_argument("--lambda_pred", type=float, default=0.1)
    ap.add_argument("--lambda_res_abs", type=float, default=0.0)
    ap.add_argument("--lambda_stage_improve", type=float, default=0.0)
    ap.add_argument("--stage_improve_margin", type=float, default=0.0)
    ap.add_argument("--lambda_stage_lpips_improve", type=float, default=0.0, help="penalize final LPIPS worse than detached stage0 reconstruction")
    ap.add_argument("--stage_lpips_improve_margin", type=float, default=0.0)
    ap.add_argument("--lambda_selected_improve", type=float, default=0.0, help="local L1 improvement loss on decoder-selected fine-stage regions")
    ap.add_argument("--selected_improve_margin", type=float, default=0.0)
    ap.add_argument("--lambda_selected_lpips_improve", type=float, default=0.0, help="LPIPS-spatial improvement loss on decoder-selected fine-stage regions")
    ap.add_argument("--selected_lpips_improve_margin", type=float, default=0.0)
    ap.add_argument("--lambda_selected_vgg_improve", type=float, default=0.0, help="DISTS-adjacent VGG feature improvement loss on decoder-selected fine-stage regions")
    ap.add_argument("--selected_vgg_improve_margin", type=float, default=0.0)
    ap.add_argument("--selected_vgg_layers", default="2,3,4", help="comma-separated DISTS/VGG feature indices used by selected VGG improvement")
    ap.add_argument("--lambda_stage0_bpp_guard", type=float, default=0.0, help="penalize stage0 bpp above target so gains cannot leak back to coarse stage")
    ap.add_argument("--stage0_bpp_target", type=float, default=0.0)
    ap.add_argument("--lambda_stage1_scale_guard", type=float, default=0.0, help="penalize fine-stage entropy scale inflation above target")
    ap.add_argument("--stage1_scale_target", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--resume", default=None, help="optional exact Stage-B checkpoint to resume from")
    ap.add_argument("--init_from", default=None, help="optional compatible Stage-B checkpoint for partial initialization")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="gp-reslc-vcip")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="offline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Scratch Stage B training expects CUDA; stop and restart the container if GPU disappeared.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_a, stage_a_args = load_stage_a(args.stage_a_ckpt, device)
    model = build_stage_b_model(args, stage_a_args, device).train()
    if args.init_from:
        load_compatible_state(model, args.init_from, device)
    if args.train_only_extra_stages:
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("extra_"))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"[train_only_extra_stages] trainable={trainable} frozen={frozen}", flush=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    lpips_spatial_fn = None
    if args.lambda_selected_lpips_improve > 0:
        lpips_spatial_fn = lpips_lib.LPIPS(net="alex", spatial=True).to(device).eval()
    dists_fn = DISTS().to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    if lpips_spatial_fn is not None:
        for p in lpips_spatial_fn.parameters():
            p.requires_grad_(False)
    for p in dists_fn.parameters():
        p.requires_grad_(False)

    use_wandb = (not args.no_wandb) and args.wandb_mode != "disabled" and _WANDB
    if use_wandb:
        cfg = vars(args).copy()
        cfg.update({f"stage_a/{k}": v for k, v in stage_a_args.items()})
        wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=cfg, dir=str(ROOT))

    train_loader = DataLoader(CropFolder(args.data, args.size), batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(CropFolder(args.val or args.data, args.size), batch_size=min(args.bs, 8), shuffle=True,
                            num_workers=max(0, min(args.num_workers, 2)), drop_last=True, pin_memory=True)

    it = 0
    best_val_dists = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            for group in opt.param_groups:
                group["lr"] = args.lr
        it = int(ckpt.get("it", 0))
        best_val_dists = float(ckpt.get("best_val_dists", best_val_dists))
        print(f"[resume] loaded {args.resume} at it={it} best_val_dists={best_val_dists}", flush=True)

    while it < args.iters:
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            z_s, base, semantic_bpp = semantic_forward(stage_a, x)
            out = model(x, base, z_s)
            x_hat = out["x_hat"].clamp(0, 1)
            l1 = F.l1_loss(x_hat, x)
            lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean()
            ds = dists_fn(x_hat, x).mean()
            total_bpp = semantic_bpp + out["residual_bpp"]
            stage_improve_loss = x.new_tensor(0.0)
            selected_improve_loss = x.new_tensor(0.0)
            selected_lpips_improve_loss = x.new_tensor(0.0)
            selected_vgg_improve_loss = x.new_tensor(0.0)
            gate_error_target_loss = x.new_tensor(0.0)
            gate_error_target_mean = x.new_tensor(0.0)
            selected_mask_mean = x.new_tensor(0.0)
            selected_lpips_mask_mean = x.new_tensor(0.0)
            selected_vgg_mask_mean = x.new_tensor(0.0)
            stage0_bpp_guard = x.new_tensor(0.0)
            stage1_scale_guard = x.new_tensor(0.0)
            stage_lpips_improve_loss = x.new_tensor(0.0)
            stage0_ds_value = None
            stage0_lpips_value = None
            if args.lambda_stage_improve > 0 and "stage0_x_hat" in out:
                stage0_ds = dists_fn(out["stage0_x_hat"].detach().clamp(0, 1), x).mean()
                stage0_ds_value = float(stage0_ds.item())
                stage_improve_loss = F.relu(ds - stage0_ds + args.stage_improve_margin)
            if args.lambda_stage_lpips_improve > 0 and "stage0_x_hat" in out:
                stage0_lpips = lpips_fn(out["stage0_x_hat"].detach().clamp(0, 1) * 2 - 1, x * 2 - 1).mean()
                stage0_lpips_value = float(stage0_lpips.item())
                stage_lpips_improve_loss = F.relu(lp - stage0_lpips + args.stage_lpips_improve_margin)
            if args.lambda_selected_improve > 0:
                selected_improve_loss, selected_mask_mean = selected_region_improvement_loss(
                    out, x, x_hat, args.selected_improve_margin
                )
            if args.lambda_selected_lpips_improve > 0:
                if lpips_spatial_fn is None:
                    raise RuntimeError("LPIPS spatial loss was not initialized")
                selected_lpips_improve_loss, selected_lpips_mask_mean = selected_region_lpips_improvement_loss(
                    out, x, x_hat, lpips_spatial_fn, args.selected_lpips_improve_margin
                )
            if args.lambda_selected_vgg_improve > 0:
                selected_vgg_improve_loss, selected_vgg_mask_mean = selected_region_vgg_feature_improvement_loss(
                    out, x, x_hat, dists_fn, parse_int_list(args.selected_vgg_layers), args.selected_vgg_improve_margin
                )
            if args.lambda_gate_error_target > 0:
                gate_target_frac = args.gate_error_target_topk_frac or args.gate_topk_frac or 0.10
                gate_error_target_loss, gate_error_target_mean = gate_base_error_target_loss(
                    out, x, base, gate_target_frac
                )
            if args.lambda_stage0_bpp_guard > 0 and args.stage0_bpp_target > 0 and "stage0_bpp" in out:
                stage0_bpp_guard = F.relu(out["stage0_bpp"] - x.new_tensor(args.stage0_bpp_target))
            if args.lambda_stage1_scale_guard > 0 and args.stage1_scale_target > 0 and "stage1_scale_mean" in out:
                stage1_scale_guard = F.relu(out["stage1_scale_mean"] - x.new_tensor(args.stage1_scale_target))
            loss = (args.lambda_R * total_bpp + args.lambda_l1 * l1 + args.lambda_lpips * lp
                    + args.lambda_dists * ds + args.lambda_pred * out["pred_loss"]
                    + args.lambda_res_abs * out["residual_symbols"].abs().mean()
                    + args.lambda_stage_improve * stage_improve_loss
                    + args.lambda_stage_lpips_improve * stage_lpips_improve_loss
                    + args.lambda_selected_improve * selected_improve_loss
                    + args.lambda_selected_lpips_improve * selected_lpips_improve_loss
                    + args.lambda_selected_vgg_improve * selected_vgg_improve_loss
                    + args.lambda_gate_error_target * gate_error_target_loss
                    + args.lambda_stage0_bpp_guard * stage0_bpp_guard
                    + args.lambda_stage1_scale_guard * stage1_scale_guard)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if it % args.log_every == 0:
                stage_msg = ""
                if "stage0_bpp" in out:
                    stage_parts = []
                    idx = 0
                    while f"stage{idx}_bpp" in out:
                        part = f"s{idx}={out[f'stage{idx}_bpp'].item():.4f}"
                        if f"stage{idx}_gate_mean" in out:
                            part += f"/g{idx}={out[f'stage{idx}_gate_mean'].item():.3f}"
                        stage_parts.append(part)
                        idx += 1
                    stage_msg = " " + " ".join(stage_parts)
                print(f"[it {it}] loss={loss.item():.4f} total_bpp={total_bpp.item():.4f} "
                      f"res_bpp={out['residual_bpp'].item():.4f} sem_bpp={semantic_bpp.item():.5f}{stage_msg} "
                      f"l1={l1.item():.4f} lpips={lp.item():.4f} dists={ds.item():.4f} "
                      f"pred={out['pred_loss'].item():.4f} r_abs={out['residual_abs_mean'].item():.3f} "
                      f"scale={out['scale_mean'].item():.3f} simpr={stage_improve_loss.item():.4f} "
                      f"slpimpr={stage_lpips_improve_loss.item():.4f} selimpr={selected_improve_loss.item():.4f} "
                      f"sellpimpr={selected_lpips_improve_loss.item():.4f} selvggimpr={selected_vgg_improve_loss.item():.4f} "
                      f"getarg={gate_error_target_loss.item():.4f} s0guard={stage0_bpp_guard.item():.5f} "
                      f"s1scaleguard={stage1_scale_guard.item():.5f}", flush=True)
                if use_wandb:
                    train_log = {"train/loss": loss.item(), "train/total_bpp": total_bpp.item(),
                                 "train/semantic_bpp": semantic_bpp.item(), "train/residual_bpp": out["residual_bpp"].item(),
                                 "train/l1": l1.item(), "train/lpips": lp.item(), "train/dists": ds.item(),
                                 "train/pred_loss": out["pred_loss"].item(),
                                 "train/residual_abs_mean": out["residual_abs_mean"].item(),
                                 "train/residual_std": out["residual_std"].item(),
                                 "train/scale_mean": out["scale_mean"].item(),
                                 "train/stage_improve_loss": stage_improve_loss.item(),
                                 "train/stage_lpips_improve_loss": stage_lpips_improve_loss.item(),
                                 "train/selected_improve_loss": selected_improve_loss.item(),
                                 "train/selected_lpips_improve_loss": selected_lpips_improve_loss.item(),
                                 "train/selected_vgg_improve_loss": selected_vgg_improve_loss.item(),
                                 "train/gate_error_target_loss": gate_error_target_loss.item(),
                                 "train/gate_error_target_mean": gate_error_target_mean.item(),
                                 "train/selected_mask_mean": selected_mask_mean.item(),
                                 "train/selected_lpips_mask_mean": selected_lpips_mask_mean.item(),
                                 "train/selected_vgg_mask_mean": selected_vgg_mask_mean.item(),
                                 "train/stage0_bpp_guard": stage0_bpp_guard.item(),
                                 "train/stage1_scale_guard": stage1_scale_guard.item()}
                    if stage0_ds_value is not None:
                        train_log["train/stage0_dists_detached"] = stage0_ds_value
                    if stage0_lpips_value is not None:
                        train_log["train/stage0_lpips_detached"] = stage0_lpips_value
                    for k, v in out.items():
                        if k.startswith("stage") and k.endswith(("bpp", "residual_abs_mean", "scale_mean", "gate_mean")):
                            train_log[f"train/{k}"] = float(v.item())
                    wandb.log(train_log, step=it)

            if it % args.eval_every == 0:
                val = quick_val(stage_a, model, val_loader, device, lpips_fn, dists_fn)
                panel_path = out_dir / f"val_panel_{it:07d}.png"
                save_image(val["panel"], panel_path)
                val_stage_msg = ""
                if "stage0_bpp" in val:
                    val_parts = []
                    idx = 0
                    while f"stage{idx}_bpp" in val:
                        part = f"s{idx}={val[f'stage{idx}_bpp']:.4f}"
                        if f"stage{idx}_gate_mean" in val:
                            part += f"/g{idx}={val[f'stage{idx}_gate_mean']:.3f}"
                        val_parts.append(part)
                        idx += 1
                    val_stage_msg = " " + " ".join(val_parts)
                print(f"  [val {it}] total_bpp={val['total_bpp']:.4f} res_bpp={val['residual_bpp']:.4f}{val_stage_msg} "
                      f"base_lpips={val['base_lpips']:.4f} lpips={val['lpips']:.4f} "
                      f"base_dists={val['base_dists']:.4f} dists={val['dists']:.4f} "
                      f"r_abs={val['residual_abs_mean']:.3f} scale={val['scale_mean']:.3f}", flush=True)
                if val["dists"] < best_val_dists:
                    best_val_dists = val["dists"]
                    torch.save({"it": it, "model": model.state_dict(), "optimizer": opt.state_dict(),
                                "args": vars(args), "stage_a_args": stage_a_args,
                                "best_val_dists": best_val_dists},
                               out_dir / "stage_b_best.pt")
                if use_wandb:
                    val_log = {"val/total_bpp": val["total_bpp"], "val/semantic_bpp": val["semantic_bpp"],
                               "val/residual_bpp": val["residual_bpp"], "val/l1": val["l1"], "val/mse": val["mse"],
                               "val/lpips": val["lpips"], "val/dists": val["dists"], "val/best_dists": best_val_dists,
                               "val/base_l1": val["base_l1"], "val/base_lpips": val["base_lpips"],
                               "val/base_dists": val["base_dists"], "val/pred_loss": val["pred_loss"],
                               "val/residual_abs_mean": val["residual_abs_mean"],
                               "val/residual_std": val["residual_std"], "val/scale_mean": val["scale_mean"],
                               "val/panel": wandb.Image(str(panel_path))}
                    for k, v in val.items():
                        if k.startswith("stage"):
                            val_log[f"val/{k}"] = float(v)
                    wandb.log(val_log, step=it)

            if it > 0 and it % args.save_every == 0:
                torch.save({"it": it, "model": model.state_dict(), "optimizer": opt.state_dict(),
                            "args": vars(args), "stage_a_args": stage_a_args}, out_dir / f"stage_b_{it:07d}.pt")
            it += 1
            if it >= args.iters:
                break

    torch.save({"it": it, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "args": vars(args), "stage_a_args": stage_a_args}, out_dir / "stage_b_final.pt")
    if use_wandb:
        wandb.save(str(out_dir / "stage_b_final.pt"))
        wandb.finish()


if __name__ == "__main__":
    main()
