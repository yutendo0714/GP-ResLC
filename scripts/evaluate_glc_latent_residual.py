#!/usr/bin/env python3
"""Deterministic center-crop evaluation for the GLC-latent residual branch."""

from __future__ import annotations

import argparse
import csv
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


class CenterCropFolder(Dataset):
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.JPEG")

    def __init__(self, root: str, size: int = 256, limit: int = 0):
        paths: list[str] = []
        for ext in self.EXTS:
            paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        paths = sorted(paths)
        if limit > 0:
            paths = paths[:limit]
        if not paths:
            raise RuntimeError(f"no images found in {root}")
        self.paths = paths
        self.size = int(size)
        self.t = transforms.Compose([transforms.CenterCrop(self.size), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if min(img.size) < self.size:
            s = self.size / min(img.size)
            img = img.resize((max(self.size, int(img.size[0] * s) + 1),
                              max(self.size, int(img.size[1] * s) + 1)))
        return self.t(img), os.path.basename(self.paths[i])


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
    model.eval()
    return model, args


def load_vqgan(weights: str, device: str):
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
def semantic_forward(stage_a, x):
    vq = stage_a.encode(x)
    bpp = x.new_tensor(stage_a.semantic_index_bpp(x.shape[-2], x.shape[-1], vq.indices.shape[-2], vq.indices.shape[-1]))
    return vq.quantized.detach(), bpp


@torch.no_grad()
def target_latent(vqgan, x):
    return vqgan.encoder(to_glc_range(x)).detach()


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = dict(ckpt["args"])
    stage_a_ckpt = args["stage_a_ckpt"]
    stage_a, stage_a_args = load_stage_a(stage_a_ckpt, device)
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
    model.load_state_dict(ckpt["model"])
    model.eval()
    return stage_a, model, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--save_panels", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU is not visible; stop evaluation.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_a, model, cfg = load_model(args.ckpt, device)
    vqgan = load_vqgan(args.glc_weights, device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    dists_fn = DISTS().to(device).eval()
    loader = DataLoader(CenterCropFolder(args.data, args.size, args.limit), batch_size=args.bs,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    sums: dict[str, float] = {}
    panels = []
    n_img = 0
    with torch.no_grad():
        for x, names in loader:
            x = x.to(device, non_blocking=True)
            z_s, sem_bpp = semantic_forward(stage_a, x)
            y = target_latent(vqgan, x)
            delta_gate_mode = cfg.get("delta_gate_mode", "none")
            force_topk_frac = float(cfg.get("force_topk_frac", 0.0))
            hard_topk = bool(cfg.get("hard_topk", False))
            entropy_mode = cfg.get("entropy_mode", "clamped")
            max_symbol_abs = float(cfg.get("max_symbol_abs", 0.0))
            out = model(
                z_s,
                y,
                use_residual=not args.no_residual,
                delta_gate_mode=delta_gate_mode,
                force_topk_frac=force_topk_frac,
                hard_topk=hard_topk,
                entropy_mode=entropy_mode,
                max_symbol_abs=max_symbol_abs,
            )
            out_base = model(z_s, y, use_residual=False)
            x_hat = from_glc_range(vqgan.generator(out["latent_hat"])).clamp(0, 1)
            x_base = from_glc_range(vqgan.generator(out_base["latent_hat"])).clamp(0, 1)
            lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).flatten()
            ds = dists_fn(x_hat, x).flatten()
            base_lp = lpips_fn(x_base * 2 - 1, x * 2 - 1).flatten()
            base_ds = dists_fn(x_base, x).flatten()
            l1 = (x_hat - x).abs().flatten(1).mean(1)
            mse = F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(1)
            base_l1 = (x_base - x).abs().flatten(1).mean(1)
            total_bpp = sem_bpp + out["residual_bpp"]
            for i, name in enumerate(names):
                row = {
                    "name": name,
                    "semantic_bpp": float(sem_bpp.item()),
                    "residual_bpp_batch": float(out["residual_bpp"].item()),
                    "total_bpp_batch": float(total_bpp.item()),
                    "base_l1": float(base_l1[i].item()),
                    "base_lpips": float(base_lp[i].item()),
                    "base_dists": float(base_ds[i].item()),
                    "l1": float(l1[i].item()),
                    "mse": float(mse[i].item()),
                    "lpips": float(lp[i].item()),
                    "dists": float(ds[i].item()),
                    "pred_loss_batch": float(out["pred_loss"].item()),
                    "latent_loss_batch": float(out["latent_loss"].item()),
                    "residual_abs_mean_batch": float(out["residual_abs_mean"].item()),
                    "rounded_abs_mean_batch": float(out["rounded_abs_mean"].item()),
                    "rounded_nonzero_frac_batch": float(out["rounded_nonzero_frac"].item()),
                    "delta_active_frac_batch": float(out["delta_active_frac"].item()),
                    "scale_mean_batch": float(out["scale_mean"].item()),
                }
                rows.append(row)
                for k, v in row.items():
                    if k != "name":
                        sums[k] = sums.get(k, 0.0) + float(v)
                n_img += 1
            if len(panels) < args.save_panels:
                for i in range(min(args.save_panels - len(panels), x.shape[0])):
                    panels.extend([x[i].detach().cpu(), x_base[i].detach().cpu(), x_hat[i].detach().cpu()])

    metrics = {k: v / max(1, n_img) for k, v in sums.items()}
    metrics.update({"num_images": n_img, "ckpt": args.ckpt, "data": args.data, "no_residual": args.no_residual})
    fields = sorted({k for r in rows for k in r})
    fields.remove("name")
    fields = ["name"] + fields
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    if panels:
        save_image(make_grid(torch.stack(panels), nrow=3).clamp(0, 1), out_dir / "panel.png")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
