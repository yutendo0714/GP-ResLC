#!/usr/bin/env python3
"""Train a decoder-side continuous synthesizer for omitted stage-3 residuals.

This branch tests the high-upside GP-ResLC hypothesis:

  omit selected arithmetic-coded residual symbols
  -> reconstruct them with a decoder-computable generator-side module
  -> improve image quality at exactly the same serialized bpp as hard omission

The module sees only GLC decoder-available signals (`common_params`,
`y_hat_so_far`, and `q` through the loaded checkpoint modules).  No side map or
random seed is transmitted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from DISTS_pytorch import DISTS
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from scripts.evaluate_real_codec import build_gp_reslc
from src.models.image_model import GLC_Image
from src.models.loss import LPIPSLoss, get_lpips_model
from src.utils.test_utils import from_0_1_to_minus1_1, from_minus1_1_to_0_1, init_func
from gp_reslc.prior_predictor import (
    StageResidualSelectiveValueSynthesizer,
    StageResidualValueSynthesizer,
)
from gp_reslc.real_codec import (
    _apply_gp_reslc_params,
    _apply_stage_q_condition,
    _stage_delta_and_scales,
    _stage_q_shift,
)

try:
    import wandb
    _WANDB = True
except Exception:
    wandb = None
    _WANDB = False


class CropFolder(Dataset):
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

    def __init__(self, root: str, size: int = 256, max_images: int = 0):
        self.paths = sorted(sum([
            glob.glob(os.path.join(root, "**", ext), recursive=True) for ext in self.EXTS
        ], []))
        if max_images and max_images > 0:
            self.paths = self.paths[:max_images]
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        if min(img.size) < self.size:
            scale = self.size / min(img.size)
            img = img.resize((
                max(self.size, int(img.size[0] * scale) + 1),
                max(self.size, int(img.size[1] * scale) + 1),
            ))
        return from_0_1_to_minus1_1(self.t(img))


def freeze_all(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def disable_inplace(module: torch.nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def stage3_value_forward(
    net: GLC_Image,
    x: torch.Tensor,
    q: int,
    predictor_param_mode: str,
    predictor_delta_bound: float,
    rho_threshold: float,
    stage3_send_frac: float,
    stage3_send_score_mode: str,
    train_all_stage3: bool,
    additive: bool,
    return_base: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    curr_q_enc = net.q_enc[q:q + 1]
    curr_q_dec = net.q_dec[q:q + 1]

    with torch.no_grad():
        y_ori = net.vqgan.encoder(x)
        y = net.enc(y_ori, curr_q_enc)
        z = net.hyper_enc(y)
        z_indices = net.z_vq.get_indices(z).reshape(-1)
        z_hat = net.z_vq.get_quan_feat(
            z_indices.reshape(-1, 1),
            (z.shape[0], z.shape[2], z.shape[3], z.shape[1]),
        )
        params = net.y_prior_fusion(net.hyper_dec(z_hat))
        params, _ = _apply_gp_reslc_params(
            net, z_hat, params, q, predictor_param_mode, predictor_delta_bound)
        q_shift = _stage_q_shift(net, q, params.device, params.dtype)

    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift).detach()
    b, c, h, w = y.shape
    masks = net.get_mask_four_parts(b, c, h, w, y.dtype, y.device)

    y_hat_so_far = None
    y_hat_so_far_base = None
    q_dec_map = torch.zeros_like(quant_step)
    selected = None
    synth_values = None
    target_values = None
    gate_values = None
    rho_stage3 = None

    for stage_idx, mask in enumerate(masks):
        if stage_idx == 0:
            cur_scales, cur_means = scales, means
            rho, _ = net.stage_quant_gate.forward_stage(0, common_params)
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, 0, common_params, None, cur_scales,
                predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))
        else:
            prior_in = torch.cat((y_hat_so_far.detach(), common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if stage_idx == 1
                else net.y_spatial_prior_adaptor_2 if stage_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            rho, _ = net.stage_quant_gate.forward_stage(stage_idx, common_params, y_hat_so_far.detach())
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, stage_idx, common_params, y_hat_so_far.detach(),
                cur_scales, predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))

        q_enc = 1.0 / (quant_step * rho)
        q_dec_map = q_dec_map + quant_step * rho * mask
        means_hat = (cur_means + delta) * mask
        y_res = (y * q_enc - means_hat) * mask
        y_q_true = torch.round(y_res)
        y_q_use = y_q_true

        if stage_idx == 3:
            active = mask > 0.5
            rho_full = rho.expand_as(mask)
            if 0.0 <= float(stage3_send_frac) <= 1.0:
                if stage3_send_score_mode != "latent_mse":
                    raise ValueError(
                        "training currently supports only stage3_send_score_mode=latent_mse")
                # Match the real-codec counted send-control branch: transmit the
                # top-k z-cells by latent MSE benefit and train synthesis only on
                # the omitted cells.  This avoids training on a rho-threshold
                # distribution while evaluating on a transmitted send mask.
                rec_send = (y_q_true + means_hat) * (quant_step * rho) * mask
                rec_drop = means_hat * (quant_step * rho) * mask
                benefit = ((rec_drop - y).square() - (rec_send - y).square()).clamp_min(0.0) * mask
                benefit = benefit.mean(dim=1, keepdim=True)
                pooled = F.adaptive_avg_pool2d(benefit, (z.shape[2], z.shape[3]))
                send_map = torch.zeros_like(pooled)
                frac = max(0.0, min(float(stage3_send_frac), 1.0))
                flat = pooled.flatten(1)
                k = int(round(frac * flat.shape[1]))
                if k > 0:
                    k = min(k, flat.shape[1])
                    idx = torch.topk(flat, k=k, dim=1).indices
                    send_flat = torch.zeros_like(flat)
                    send_flat.scatter_(1, idx, 1.0)
                    send_map = send_flat.reshape_as(pooled)
                send_y = F.interpolate(send_map, size=(h, w), mode="nearest") > 0.5
                selected = active & (~send_y)
            else:
                selected = active if train_all_stage3 else (active & (rho_full >= float(rho_threshold)))
            synth_out = net.stage_residual_value_synthesizer.forward_stage(
                3, common_params, y_hat_so_far.detach())
            if isinstance(synth_out, tuple):
                synth_raw, synth_gate = synth_out
                synth = synth_raw * synth_gate
                gate_values = synth_gate[selected]
            else:
                synth = synth_out
                gate_values = x.new_ones((int(selected.sum().item()),), dtype=x.dtype, device=x.device)
            if additive:
                y_q_use = y_q_true + torch.where(
                    selected, synth.to(dtype=y_q_true.dtype), torch.zeros_like(y_q_true))
            else:
                y_q_use = torch.where(selected, synth.to(dtype=y_q_true.dtype), y_q_true)
            synth_values = synth[selected]
            target_values = y_q_true[selected].detach()
            rho_stage3 = rho_full[active].detach()

        y_hat_part = y_q_use + means_hat
        y_hat_part_base = y_q_true + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part
        y_hat_so_far_base = (
            y_hat_part_base if y_hat_so_far_base is None else y_hat_so_far_base + y_hat_part_base
        )

    y_hat = y_hat_so_far * q_dec_map
    y_hat = net.dec(y_hat, curr_q_dec)
    x_hat = net.vqgan.generator(y_hat)

    x_hat_base = None
    if return_base:
        with torch.no_grad():
            y_hat_base = (y_hat_so_far_base * q_dec_map).detach()
            y_hat_base = net.dec(y_hat_base, curr_q_dec)
            x_hat_base = net.vqgan.generator(y_hat_base)

    if selected is None or synth_values is None or target_values is None or rho_stage3 is None:
        raise RuntimeError("failed to build stage-3 value synthesis forward pass")
    stats = {
        "selected_frac": float(selected.float().mean().item()),
        "selected_count": float(selected.sum().item()),
        "rho_stage3_mean": float(rho_stage3.mean().item()),
        "target_abs_mean": float(target_values.abs().mean().item()) if target_values.numel() else 0.0,
        "synth_abs_mean": synth_values.abs().mean() if synth_values.numel() else x_hat.new_tensor(0.0),
        "gate_mean": gate_values.mean() if gate_values is not None and gate_values.numel() else x_hat.new_tensor(1.0),
        "synth_mse_to_true": F.mse_loss(synth_values, target_values) if synth_values.numel() else x_hat.new_tensor(0.0),
        "nonzero_frac_selected": float((target_values.abs() > 0.5).float().mean().item()) if target_values.numel() else 0.0,
    }
    if return_base:
        return x_hat, x_hat_base, stats
    return x_hat, stats


def save_checkpoint(path: Path, base_ckpt: dict, net: GLC_Image, args: argparse.Namespace, it: int) -> None:
    state = dict(base_ckpt)
    state["stage_residual_value_synthesizer"] = net.stage_residual_value_synthesizer.state_dict()
    state["stage_value_bound"] = float(args.value_bound)
    state["stage_value_depth"] = int(args.depth)
    state["stage_value_selective"] = bool(args.selective)
    state["stage_value_gate_init_prob"] = float(args.gate_init_prob)
    state["stage_value_train_iter"] = int(it)
    state["stage_value_rho_threshold"] = float(args.rho_threshold)
    state["stage_value_stage3_send_frac"] = float(args.stage3_send_frac)
    state["stage_value_stage3_send_score_mode"] = str(args.stage3_send_score_mode)
    state["stage_value_train_all_stage3"] = bool(args.train_all_stage3)
    state["stage_value_additive"] = bool(args.additive)
    state["predictor_param_mode"] = args.predictor_param_mode
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    init_func()
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--data", default="/dpl/open-images-v6/train/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--q_choices", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--predictor_param_mode", default="stage_residual_entropy_quant_gate")
    ap.add_argument("--predictor_delta_bound", type=float, default=0.08)
    ap.add_argument("--rho_threshold", type=float, default=1.20)
    ap.add_argument("--stage3_send_frac", type=float, default=-1.0,
                    help="If in [0,1], train on omitted positions from the real-codec stage-3 send mask.")
    ap.add_argument("--stage3_send_score_mode", default="latent_mse",
                    choices=["latent_mse"])
    ap.add_argument("--train_all_stage3", action="store_true")
    ap.add_argument("--additive", action="store_true",
                    help="Add the synthesized residual to transmitted stage-3 symbols instead of replacing omitted symbols.")
    ap.add_argument("--value_bound", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--selective", action="store_true",
                    help="Use a decoder-computable confidence gate for no-side selective synthesis.")
    ap.add_argument("--gate_init_prob", type=float, default=0.25)
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=2.0)
    ap.add_argument("--lambda_dists", type=float, default=2.0)
    ap.add_argument("--lambda_value", type=float, default=0.01)
    ap.add_argument("--lambda_gate", type=float, default=0.0,
                    help="Penalize average synthesis gate probability on selected coefficients.")
    ap.add_argument("--lambda_safe_lpips", type=float, default=0.0,
                    help="Hinge penalty if LPIPS is worse than the frozen baseline by more than safe_lpips_margin.")
    ap.add_argument("--lambda_safe_dists", type=float, default=0.0,
                    help="Hinge penalty if DISTS is worse than the frozen baseline by more than safe_dists_margin.")
    ap.add_argument("--safe_lpips_margin", type=float, default=0.0)
    ap.add_argument("--safe_dists_margin", type=float, default=0.0)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--wandb_project", default="gp-reslc-mainline")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run but is not available")
    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = torch.load(args.base_ckpt, map_location="cpu")
    net = build_gp_reslc(args.glc_weights, args.base_ckpt, interpolate=False, device=device)
    disable_inplace(net)
    freeze_all(net)
    synth_cls = StageResidualSelectiveValueSynthesizer if args.selective else StageResidualValueSynthesizer
    if args.selective:
        net.stage_residual_value_synthesizer = synth_cls(
            net.N,
            value_bound=args.value_bound,
            depth=args.depth,
            gate_init_prob=args.gate_init_prob,
        ).to(device)
    else:
        net.stage_residual_value_synthesizer = synth_cls(
            net.N, value_bound=args.value_bound, depth=args.depth).to(device)
    for p in net.stage_residual_value_synthesizer.parameters():
        p.requires_grad_(True)
    net.stage_residual_value_synthesizer.train()

    lpips_loss = LPIPSLoss(get_lpips_model()).to(device).eval()
    dists_loss = DISTS().to(device).eval()
    for p in lpips_loss.parameters():
        p.requires_grad_(False)
    for p in dists_loss.parameters():
        p.requires_grad_(False)

    loader = DataLoader(
        CropFolder(args.data, args.crop, args.max_images),
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(net.stage_residual_value_synthesizer.parameters(), lr=args.lr)

    use_wandb = _WANDB and not args.no_wandb and args.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   mode=args.wandb_mode, config=vars(args))

    manifest = {
        "base_ckpt": args.base_ckpt,
        "data": args.data,
        "rho_threshold": args.rho_threshold,
        "stage3_send_frac": args.stage3_send_frac,
        "stage3_send_score_mode": args.stage3_send_score_mode,
        "value_bound": args.value_bound,
        "depth": args.depth,
        "selective": args.selective,
        "gate_init_prob": args.gate_init_prob,
        "train_all_stage3": args.train_all_stage3,
        "additive": args.additive,
        "iters": args.iters,
        "q_choices": args.q_choices,
        "lr": args.lr,
        "batch_size": args.bs,
        "crop": args.crop,
        "lambda_l1": args.lambda_l1,
        "lambda_lpips": args.lambda_lpips,
        "lambda_dists": args.lambda_dists,
        "lambda_value": args.lambda_value,
        "lambda_gate": args.lambda_gate,
        "lambda_safe_lpips": args.lambda_safe_lpips,
        "lambda_safe_dists": args.lambda_safe_dists,
        "safe_lpips_margin": args.safe_lpips_margin,
        "safe_dists_margin": args.safe_dists_margin,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    it = 0
    while it < args.iters:
        for x in loader:
            if it >= args.iters:
                break
            x = x.to(device, non_blocking=True)
            q = random.choice(args.q_choices)
            need_base = args.lambda_safe_lpips > 0.0 or args.lambda_safe_dists > 0.0
            out = stage3_value_forward(
                net, x, q, args.predictor_param_mode, args.predictor_delta_bound,
                args.rho_threshold, args.stage3_send_frac, args.stage3_send_score_mode,
                args.train_all_stage3, args.additive,
                return_base=need_base)
            if need_base:
                x_hat, x_hat_base, stats = out
            else:
                x_hat, stats = out
                x_hat_base = None
            d_l1 = F.l1_loss(x_hat, x)
            d_lpips = lpips_loss(x_hat.clamp(-1, 1), x).mean()
            d_dists = dists_loss(
                from_minus1_1_to_0_1(x_hat.clamp(-1, 1)),
                from_minus1_1_to_0_1(x),
            ).mean()
            safe_lpips = x_hat.new_tensor(0.0)
            safe_dists = x_hat.new_tensor(0.0)
            base_lpips = x_hat.new_tensor(0.0)
            base_dists = x_hat.new_tensor(0.0)
            if need_base and x_hat_base is not None:
                with torch.no_grad():
                    base_lpips = lpips_loss(x_hat_base.clamp(-1, 1), x).mean()
                    base_dists = dists_loss(
                        from_minus1_1_to_0_1(x_hat_base.clamp(-1, 1)),
                        from_minus1_1_to_0_1(x),
                    ).mean()
                safe_lpips = F.relu(d_lpips - base_lpips - float(args.safe_lpips_margin))
                safe_dists = F.relu(d_dists - base_dists - float(args.safe_dists_margin))
            synth_abs = stats["synth_abs_mean"]
            gate_mean = stats["gate_mean"]
            loss = (
                args.lambda_l1 * d_l1
                + args.lambda_lpips * d_lpips
                + args.lambda_dists * d_dists
                + args.lambda_value * synth_abs
                + args.lambda_gate * gate_mean
                + args.lambda_safe_lpips * safe_lpips
                + args.lambda_safe_dists * safe_dists
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.stage_residual_value_synthesizer.parameters(), 1.0)
            opt.step()

            if it % args.log_every == 0:
                row = {
                    "it": it,
                    "q": q,
                    "loss": float(loss.item()),
                    "l1": float(d_l1.item()),
                    "lpips": float(d_lpips.item()),
                    "dists": float(d_dists.item()),
                    "base_lpips": float(base_lpips.item()),
                    "base_dists": float(base_dists.item()),
                    "safe_lpips": float(safe_lpips.item()),
                    "safe_dists": float(safe_dists.item()),
                    "synth_abs_mean": float(synth_abs.item()),
                    "gate_mean": float(gate_mean.item()),
                    "synth_mse_to_true": float(stats["synth_mse_to_true"].item()),
                    "selected_frac": stats["selected_frac"],
                    "selected_count": stats["selected_count"],
                    "rho_stage3_mean": stats["rho_stage3_mean"],
                    "target_abs_mean": stats["target_abs_mean"],
                    "nonzero_frac_selected": stats["nonzero_frac_selected"],
                }
                print(json.dumps(row), flush=True)
                if use_wandb:
                    wandb.log(row, step=it)

            if args.save_every > 0 and it > 0 and it % args.save_every == 0:
                save_checkpoint(out_dir / f"value_synth_{it:06d}.pt", base_ckpt, net, args, it)
            it += 1

    save_checkpoint(out_dir / "value_synth_final.pt", base_ckpt, net, args, it)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
