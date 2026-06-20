#!/usr/bin/env python3
"""Analyze spatial correlation between GP-ResLC rho maps and reconstruction/error cues."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_keys(root: Path):
    return [p.stem for p in sorted(root.iterdir()) if p.suffix.lower() in IMG_EXTS]


def find_image(root: Path, key: str):
    for ext in IMG_EXTS:
        p = root / f"{key}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(root / key)


def load_rgb(path: Path):
    return ToTensor()(Image.open(path).convert("RGB"))


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    mask = torch.isfinite(a) & torch.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    if float(den) <= 1e-12:
        return float("nan")
    return float((a * b).sum() / den)


def texture_maps(x: torch.Tensor):
    gray = x.mean(0, keepdim=True).unsqueeze(0)
    local = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
    var = F.avg_pool2d((gray - local).pow(2), kernel_size=7, stride=1, padding=3)[0, 0]
    gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))[0, 0]
    gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))[0, 0]
    grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
    return var, grad


def align_maps(*maps):
    h = min(m.shape[-2] for m in maps)
    w = min(m.shape[-1] for m in maps)
    return [m[..., :h, :w] for m in maps]


def summarize(rows):
    keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    out = {"count": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], float) and math.isfinite(r[k])]
        if vals:
            t = torch.tensor(vals)
            out[f"{k}_mean"] = float(t.mean())
            out[f"{k}_std"] = float(t.std(unbiased=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--anchor", required=True, help="GLC q directory")
    ap.add_argument("--test", required=True, help="GP-ResLC q directory")
    ap.add_argument("--rho_dir", required=True, help="directory containing *_rho_q*.pt")
    ap.add_argument("--q_index", type=int, default=3)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    orig = Path(args.orig)
    anchor = Path(args.anchor)
    test = Path(args.test)
    rho_dir = Path(args.rho_dir)

    keys = []
    for key in image_keys(orig):
        rho_pt = rho_dir / f"{key}_rho_q{args.q_index}.pt"
        if rho_pt.exists() and any((anchor / f"{key}{e}").exists() for e in IMG_EXTS) and any((test / f"{key}{e}").exists() for e in IMG_EXTS):
            keys.append(key)
    if not keys:
        raise RuntimeError("no paired images/rho maps")

    rows = []
    for key in keys:
        x = load_rgb(find_image(orig, key))
        a = load_rgb(find_image(anchor, key))
        y = load_rgb(find_image(test, key))
        rho_path = rho_dir / f"{key}_rho_q{args.q_index}.pt"
        try:
            rho = torch.load(rho_path, map_location="cpu", weights_only=True).float()
        except TypeError:
            rho = torch.load(rho_path, map_location="cpu").float()
        if rho.ndim == 3:
            rho = rho[0]
        x, a, y, rho = align_maps(x, a, y, rho)
        base_err = (x - a).abs().mean(0)
        ours_err = (x - y).abs().mean(0)
        err_improve = base_err - ours_err
        tex_var, grad = texture_maps(x)
        tex_var, grad, rho = align_maps(tex_var, grad, rho)
        base_err, ours_err, err_improve = align_maps(base_err, ours_err, err_improve)

        thresh = torch.quantile(rho.flatten(), 0.75)
        high = rho >= thresh
        low = ~high
        def masked_mean(m, mask):
            return float(m[mask].mean()) if mask.any() else float("nan")

        rows.append({
            "image": key,
            "rho_mean": float(rho.mean()),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "rho_std": float(rho.std(unbiased=False)),
            "corr_rho_base_err": pearson(rho, base_err),
            "corr_rho_ours_err": pearson(rho, ours_err),
            "corr_rho_err_improve": pearson(rho, err_improve),
            "corr_rho_texture_var": pearson(rho, tex_var),
            "corr_rho_grad": pearson(rho, grad),
            "highrho_base_err": masked_mean(base_err, high),
            "lowrho_base_err": masked_mean(base_err, low),
            "highrho_ours_err": masked_mean(ours_err, high),
            "lowrho_ours_err": masked_mean(ours_err, low),
            "highrho_err_improve": masked_mean(err_improve, high),
            "lowrho_err_improve": masked_mean(err_improve, low),
            "highrho_texture_var": masked_mean(tex_var, high),
            "lowrho_texture_var": masked_mean(tex_var, low),
            "highrho_grad": masked_mean(grad, high),
            "lowrho_grad": masked_mean(grad, low),
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    Path(args.out_json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
