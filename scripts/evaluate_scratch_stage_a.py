#!/usr/bin/env python3
"""Deterministic evaluation for Scratch GP-ResLC Stage-A checkpoints."""

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

from gp_reslc.scratch import ScratchVQAutoencoder


class CenterCropFolder(Dataset):
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

    def __init__(self, root: str, size: int = 256, limit: int = 0):
        paths = sorted(sum([glob.glob(os.path.join(root, e)) for e in self.EXTS], []))
        if limit > 0:
            paths = paths[:limit]
        if not paths:
            raise RuntimeError(f"no images found in {root}")
        self.paths = paths
        self.size = int(size)
        self.t = transforms.Compose([
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, str]:
        img = Image.open(self.paths[i]).convert("RGB")
        if min(img.size) < self.size:
            s = self.size / min(img.size)
            img = img.resize((max(self.size, int(img.size[0] * s) + 1),
                              max(self.size, int(img.size[1] * s) + 1)))
        return self.t(img), os.path.basename(self.paths[i])


def load_model(path: str, device: str) -> tuple[ScratchVQAutoencoder, dict]:
    ckpt = torch.load(path, map_location=device)
    a = dict(ckpt.get("args", {}))
    if "codebook_size" not in a:
        a = dict(ckpt.get("stage_a_args", {}))
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
    return model, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_panels", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Stage-A evaluation expects CUDA; stop if GPU disappeared.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, device)
    loader = DataLoader(CenterCropFolder(args.data, args.size, args.limit), batch_size=args.bs,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    dists_fn = DISTS().to(device).eval()
    rows = []
    sums: dict[str, float] = {}
    n_img = 0
    panels = []
    with torch.no_grad():
        for x, names in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            x_hat = out["x_hat"].clamp(0, 1)
            lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).flatten()
            ds = dists_fn(x_hat, x).flatten()
            l1 = (x_hat - x).abs().flatten(1).mean(1)
            mse = F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
            for i, name in enumerate(names):
                row = {
                    "name": name,
                    "l1": float(l1[i].item()),
                    "mse": float(mse[i].item()),
                    "psnr": float(psnr[i].item()),
                    "lpips": float(lp[i].item()),
                    "dists": float(ds[i].item()),
                    "semantic_bpp_fixed": float(out["semantic_bpp_fixed"].item()),
                    "perplexity_batch": float(out["perplexity"].item()),
                    "entropy_norm_batch": float(out["codebook_entropy_norm"].item()),
                    "usage_frac_batch": float(out["codebook_usage_frac"].item()),
                }
                rows.append(row)
                for k, v in row.items():
                    if k != "name":
                        sums[k] = sums.get(k, 0.0) + float(v)
                n_img += 1
            if len(panels) < args.save_panels:
                m = min(args.save_panels - len(panels), x.shape[0])
                for i in range(m):
                    panels.extend([x[i].detach().cpu(), x_hat[i].detach().cpu()])
    metrics = {k: v / max(1, n_img) for k, v in sums.items()}
    metrics.update({"num_images": n_img, "ckpt": args.ckpt, "data": args.data})
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    if panels:
        save_image(make_grid(torch.stack(panels), nrow=2).clamp(0, 1), out_dir / "panel.png")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
