#!/usr/bin/env python3
"""Deterministic evaluation for Scratch GP-ResLC Stage-B checkpoints."""

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

from gp_reslc.scratch import ScratchVQAutoencoder, ScratchResidualBottleneck, ScratchProgressiveResidualBottleneck


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
    a = dict(ckpt.get("args", {}))
    if "codebook_size" not in a:
        a = dict(ckpt.get("stage_a_args", {}))
    model = ScratchVQAutoencoder(a["codebook_size"], a["latent_dim"], a["base_ch"],
                                 a.get("vq_beta", 0.25), a.get("vq_entropy_tau", 1.0),
                                 a.get("num_down", 4),
                                 decoder_attention=a.get("decoder_attention", False),
                                 extra_decoder_blocks=a.get("extra_decoder_blocks", 0)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, a


def parse_stage_quant_steps(raw: str, quant_step: float, stages: int) -> tuple[float, ...]:
    if raw:
        steps = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
        if len(steps) != int(stages):
            raise ValueError("stage_quant_steps length must match progressive_stages")
        return steps
    return tuple(float(quant_step) * (0.5 ** i) for i in range(int(stages)))


def load_stage_b(path: str, device: str, gate_threshold_override: float | None = None, gate_topk_frac_override: float | None = None):
    ckpt = torch.load(path, map_location=device)
    args = dict(ckpt["args"])
    if gate_threshold_override is not None:
        args["gate_threshold"] = float(gate_threshold_override)
    if gate_topk_frac_override is not None:
        args["gate_topk_frac"] = float(gate_topk_frac_override)
    stage_a, stage_a_args = load_stage_a(args["stage_a_ckpt"], device)
    common = dict(
        semantic_dim=stage_a_args["latent_dim"],
        residual_dim=args.get("residual_dim", 32),
        base_ch=args.get("base_ch", 96),
        num_down=stage_a_args.get("num_down", 4),
        quant_step=args.get("quant_step", 1.0),
    )
    if args.get("residual_codec", "single") == "progressive":
        model = ScratchProgressiveResidualBottleneck(
            **common,
            progressive_stages=args.get("progressive_stages", 2),
            stage_quant_steps=parse_stage_quant_steps(args.get("stage_quant_steps", ""), args.get("quant_step", 1.0), args.get("progressive_stages", 2)),
            gate_extra_stages=args.get("progressive_gate", False),
            gate_threshold=args.get("gate_threshold", 0.2),
            gate_init_bias=args.get("gate_init_bias", -2.0),
            gate_soft_train=args.get("gate_soft_train", False),
            stage_correction_decoder=args.get("stage_correction_decoder", False),
            gate_topk_frac=args.get("gate_topk_frac", 0.0),
        ).to(device)
    else:
        model = ScratchResidualBottleneck(**common).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return stage_a, model, args


@torch.no_grad()
def semantic_forward(stage_a, x):
    vq = stage_a.encode(x)
    base = stage_a.decode(vq.quantized).clamp(0, 1)
    bpp = x.new_tensor(stage_a.semantic_index_bpp(x.shape[-2], x.shape[-1], vq.indices.shape[-2], vq.indices.shape[-1]))
    return vq.quantized.detach(), base.detach(), bpp


def make_panel(x, base, x_hat, n=4):
    n = min(n, x.shape[0])
    rows = []
    for i in range(n):
        rows.extend([x[i].detach().cpu(), base[i].detach().cpu(), x_hat[i].detach().cpu()])
    return make_grid(torch.stack(rows), nrow=3).clamp(0, 1)


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
    ap.add_argument("--gate_threshold_override", type=float, default=None, help="override checkpoint gate threshold for deterministic decoder-side rate knob")
    ap.add_argument("--gate_topk_frac_override", type=float, default=None, help="override checkpoint top-k gate fraction")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Stage-B evaluation expects CUDA; stop if GPU disappeared.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_a, model, cfg = load_stage_b(args.ckpt, device, args.gate_threshold_override, args.gate_topk_frac_override)
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
            z_s, base, sem_bpp = semantic_forward(stage_a, x)
            out = model(x, base, z_s)
            x_hat = out["x_hat"].clamp(0, 1)
            base_lp = lpips_fn(base * 2 - 1, x * 2 - 1).flatten()
            ours_lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).flatten()
            base_ds = dists_fn(base, x).flatten()
            ours_ds = dists_fn(x_hat, x).flatten()
            base_l1 = (base - x).abs().flatten(1).mean(1)
            ours_l1 = (x_hat - x).abs().flatten(1).mean(1)
            base_mse = F.mse_loss(base, x, reduction="none").flatten(1).mean(1)
            ours_mse = F.mse_loss(x_hat, x, reduction="none").flatten(1).mean(1)
            for i, name in enumerate(names):
                row = {
                    "name": name,
                    "semantic_bpp": float(sem_bpp.item()),
                    "residual_bpp_batch": float(out["residual_bpp"].item()),
                    "total_bpp_batch": float((sem_bpp + out["residual_bpp"]).item()),
                    "base_l1": float(base_l1[i].item()),
                    "l1": float(ours_l1[i].item()),
                    "base_mse": float(base_mse[i].item()),
                    "mse": float(ours_mse[i].item()),
                    "base_lpips": float(base_lp[i].item()),
                    "lpips": float(ours_lp[i].item()),
                    "base_dists": float(base_ds[i].item()),
                    "dists": float(ours_ds[i].item()),
                    "residual_abs_mean_batch": float(out["residual_abs_mean"].item()),
                    "scale_mean_batch": float(out["scale_mean"].item()),
                }
                for k, v in out.items():
                    if k.startswith("stage") and k.endswith(("bpp", "residual_abs_mean", "scale_mean", "gate_mean")):
                        row[f"{k}_batch"] = float(v.item())
                rows.append(row)
                for k, v in row.items():
                    if k != "name":
                        sums[k] = sums.get(k, 0.0) + float(v)
                n_img += 1
            if len(panels) < args.save_panels:
                m = min(args.save_panels - len(panels), x.shape[0])
                for i in range(m):
                    panels.extend([x[i].detach().cpu(), base[i].detach().cpu(), x_hat[i].detach().cpu()])
    metrics = {k: v / max(1, n_img) for k, v in sums.items()}
    metrics.update({"num_images": n_img, "ckpt": args.ckpt, "data": args.data})
    if args.gate_threshold_override is not None:
        metrics["gate_threshold_override"] = args.gate_threshold_override
    if args.gate_topk_frac_override is not None:
        metrics["gate_topk_frac_override"] = args.gate_topk_frac_override
    fieldnames = sorted({k for row in rows for k in row.keys()})
    if "name" in fieldnames:
        fieldnames.remove("name")
        fieldnames = ["name"] + fieldnames
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
