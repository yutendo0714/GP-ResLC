#!/usr/bin/env python3
"""Train a decoder-side symbol synthesizer for omitted stage-3 residuals.

This is a mainline GP-ResLC diagnostic/implementation step:

  transmit fewer arithmetic-coded y_q residual symbols
  -> reconstruct omitted symbols from decoder-available GLC context

The target is the quantized residual symbol that the real codec would have
sent.  The synthesizer is trained only from `z_hat/q/context/y_hat_so_far`,
so it can be used at decode time without side information.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from scripts.evaluate_real_codec import build_gp_reslc
from src.models.image_model import GLC_Image
from src.utils.test_utils import from_0_1_to_minus1_1, init_func
from gp_reslc.prior_predictor import StageResidualSymbolSynthesizer
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


def set_frozen(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def stage_symbol_batch(
    net: GLC_Image,
    x: torch.Tensor,
    q: int,
    predictor_param_mode: str,
    predictor_delta_bound: float,
    rho_threshold: float,
    radius: int,
    train_all_stage3: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return logits and targets for selected stage-3 residual symbols."""
    with torch.no_grad():
        curr_q_enc = net.q_enc[q:q + 1]
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
        common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
        b, c, h, w = y.shape
        masks = net.get_mask_four_parts(b, c, h, w, y.dtype, y.device)
        y_hat_so_far = None
        selected = None
        target = None
        synth_context = None
        rho_mean = 1.0

        for stage_idx, mask in enumerate(masks):
            if stage_idx == 0:
                cur_scales, cur_means = scales, means
                rho, _ = net.stage_quant_gate.forward_stage(0, common_params)
                delta, cur_scales = _stage_delta_and_scales(
                    net.stage_residual_predictor, 0, common_params, None, cur_scales,
                    predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                    getattr(net, "stage_residual_refiner", None))
            else:
                prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
                adaptor = (
                    net.y_spatial_prior_adaptor_1 if stage_idx == 1
                    else net.y_spatial_prior_adaptor_2 if stage_idx == 2
                    else net.y_spatial_prior_adaptor_3
                )
                cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
                rho, _ = net.stage_quant_gate.forward_stage(stage_idx, common_params, y_hat_so_far)
                delta, cur_scales = _stage_delta_and_scales(
                    net.stage_residual_predictor, stage_idx, common_params, y_hat_so_far,
                    cur_scales, predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                    getattr(net, "stage_residual_refiner", None))

            q_enc = 1.0 / (quant_step * rho)
            means_hat = (cur_means + delta) * mask
            y_res = (y * q_enc - means_hat) * mask
            y_q = torch.round(y_res)

            if stage_idx == 3:
                active = mask > 0.5
                rho_full = rho.expand_as(mask)
                selected = active if train_all_stage3 else (active & (rho_full >= float(rho_threshold)))
                clipped = y_q.clamp(-int(radius), int(radius)).to(torch.long) + int(radius)
                target = clipped
                synth_context = y_hat_so_far
                rho_mean = float(rho_full[active].mean().item())

            y_hat_part = y_q + means_hat
            y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

        if selected is None or target is None:
            raise RuntimeError("failed to build stage-3 symbol batch")
        common_detached = common_params.detach()
        if synth_context is None:
            raise RuntimeError("failed to capture decoder context before stage 3")
        yhat_detached = synth_context.detach()

    logits = net.stage_residual_symbol_synthesizer.forward_stage(3, common_detached, yhat_detached)
    stats = {
        "selected_frac": float(selected.float().mean().item()),
        "selected_count": float(selected.sum().item()),
        "rho_stage3_mean": rho_mean,
        "target_abs_mean": float((target.float() - float(radius)).abs()[selected].mean().item())
        if bool(selected.any()) else 0.0,
        "target_clip_frac": float(((target == 0) | (target == 2 * int(radius))).float()[selected].mean().item())
        if bool(selected.any()) else 0.0,
    }
    return logits, target.detach(), selected.detach(), stats


def save_checkpoint(path: Path, base_ckpt: dict, net: GLC_Image, args: argparse.Namespace, it: int) -> None:
    state = dict(base_ckpt)
    state["stage_residual_symbol_synthesizer"] = net.stage_residual_symbol_synthesizer.state_dict()
    state["stage_symbol_radius"] = int(args.radius)
    state["stage_symbol_depth"] = int(args.depth)
    state["stage_symbol_train_iter"] = int(it)
    state["stage_symbol_rho_threshold"] = float(args.rho_threshold)
    state["stage_symbol_train_all_stage3"] = bool(args.train_all_stage3)
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
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--q_choices", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--predictor_param_mode", default="stage_residual_entropy_quant_gate")
    ap.add_argument("--predictor_delta_bound", type=float, default=0.08)
    ap.add_argument("--rho_threshold", type=float, default=1.25)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--train_all_stage3", action="store_true")
    ap.add_argument("--zero_class_weight", type=float, default=0.1,
                    help="CE weight for the zero residual symbol; lower values focus learning on rare nonzero omitted symbols.")
    ap.add_argument("--nonzero_class_weight", type=float, default=1.0)
    ap.add_argument("--train_nonzero_only", action="store_true",
                    help="Train only on selected positions whose target omitted residual symbol is nonzero.")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
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
    net.stage_residual_symbol_synthesizer = StageResidualSymbolSynthesizer(
        net.N, radius=args.radius, depth=args.depth).to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.stage_residual_symbol_synthesizer.parameters():
        p.requires_grad_(True)
    set_frozen(net.vqgan)
    net.stage_residual_symbol_synthesizer.train()

    loader = DataLoader(
        CropFolder(args.data, args.crop, args.max_images),
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(net.stage_residual_symbol_synthesizer.parameters(), lr=args.lr)

    use_wandb = _WANDB and not args.no_wandb and args.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   mode=args.wandb_mode, config=vars(args))

    manifest = {
        "base_ckpt": args.base_ckpt,
        "data": args.data,
        "rho_threshold": args.rho_threshold,
        "radius": args.radius,
        "depth": args.depth,
        "train_all_stage3": args.train_all_stage3,
        "iters": args.iters,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    it = 0
    while it < args.iters:
        for x in loader:
            if it >= args.iters:
                break
            x = x.to(device, non_blocking=True)
            q = random.choice(args.q_choices)
            logits, target, selected, stats = stage_symbol_batch(
                net, x, q, args.predictor_param_mode, args.predictor_delta_bound,
                args.rho_threshold, args.radius, args.train_all_stage3)
            if not bool(selected.any()):
                continue
            k = logits.shape[2]
            flat_logits = logits.permute(0, 1, 3, 4, 2).reshape(-1, k)
            flat_target = target.reshape(-1)
            flat_selected = selected.reshape(-1)
            if args.train_nonzero_only:
                flat_selected = flat_selected & (flat_target != int(args.radius))
            if not bool(flat_selected.any()):
                continue
            class_weight = torch.full((k,), float(args.nonzero_class_weight), device=device)
            class_weight[int(args.radius)] = float(args.zero_class_weight)
            loss = F.cross_entropy(
                flat_logits[flat_selected],
                flat_target[flat_selected],
                weight=class_weight,
            )
            pred = logits.argmax(dim=2)
            acc = (pred[selected] == target[selected]).float().mean()
            nonzero = selected & (target != int(args.radius))
            nonzero_acc = (pred[nonzero] == target[nonzero]).float().mean() if bool(nonzero.any()) else loss.new_tensor(0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.stage_residual_symbol_synthesizer.parameters(), 1.0)
            opt.step()

            if it % args.log_every == 0:
                row = {
                    "it": it,
                    "q": q,
                    "loss": float(loss.item()),
                    "acc": float(acc.item()),
                    "nonzero_acc": float(nonzero_acc.item()),
                    "nonzero_frac_selected": float(nonzero.float().sum().item() / max(float(selected.float().sum().item()), 1.0)),
                    **stats,
                }
                print(json.dumps(row), flush=True)
                if use_wandb:
                    wandb.log(row, step=it)
            if args.save_every > 0 and it > 0 and it % args.save_every == 0:
                save_checkpoint(out_dir / f"symbol_synth_{it:06d}.pt", base_ckpt, net, args, it)
            it += 1

    save_checkpoint(out_dir / "symbol_synth_final.pt", base_ckpt, net, args, it)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
