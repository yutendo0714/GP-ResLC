#!/usr/bin/env python3
"""Analyze stage-quant gate placement against local error and LPIPS-spatial sensitivity."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict, from_0_1_to_minus1_1
from src.utils.lpips.lpips import LPIPS as RawLPIPS
from gp_reslc.prior_predictor import StageQuantGate, train_forward

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PAD = 64


class LPIPSSpatialLoss(torch.nn.Module):
    def __init__(self, lpips_model, use_input_norm=True, range_norm=True):
        super().__init__()
        self.perceptual = lpips_model
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        if self.use_input_norm:
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        return self.perceptual(target.contiguous(), pred.contiguous())


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


def image_paths(root: Path, max_images: int | None):
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"no images in {root}")
    return paths


def texture_maps(x01: torch.Tensor):
    gray = x01.mean(1, keepdim=True)
    local = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
    var = F.avg_pool2d((gray - local).pow(2), kernel_size=7, stride=1, padding=3)
    gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
    return var, grad


def crop(x, pads):
    pl, pr, pt, pb = pads
    return F.pad(x, (-pl, -pr, -pt, -pb))


def summarize(rows):
    out = {"count": len(rows)}
    keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], float) and math.isfinite(r[k])]
        if vals:
            t = torch.tensor(vals)
            out[f"{k}_mean"] = float(t.mean())
            out[f"{k}_std"] = float(t.std(unbiased=False))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--q_index", type=int, default=1)
    ap.add_argument("--stage_rho_max", type=float, default=None)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(args.glc_weights), strict=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    rho_max = args.stage_rho_max if args.stage_rho_max is not None else ck.get("stage_rho_max", 1.5)
    net.stage_quant_gate = StageQuantGate(net.N, rho_max=rho_max)
    if "stage_quant_gate" in ck:
        net.stage_quant_gate.load_state_dict(ck["stage_quant_gate"])
    else:
        net.stage_quant_gate.load_state_dict(ck)
    net.to(device).eval()

    lpips_spatial = LPIPSSpatialLoss(RawLPIPS(net="alex", spatial=True, verbose=False)).to(device).eval()
    for p in lpips_spatial.parameters():
        p.requires_grad_(False)

    rows = []
    for path in image_paths(Path(args.input), args.max_images):
        img = Image.open(path).convert("RGB")
        x = from_0_1_to_minus1_1(ToTensor()(img)).unsqueeze(0).to(device)
        _, _, h, w = x.shape
        pads = GLC_Image.get_padding_size(h, w, PAD)
        xp = F.pad(x, pads, mode="replicate")
        base = train_forward(net, xp, args.q_index, use_predictor=False, predictor_param_mode="stage_quant_gate")
        ours = train_forward(net, xp, args.q_index, use_predictor=True, predictor_param_mode="stage_quant_gate")
        x0 = crop(xp, pads)
        base_x = crop(base["x_hat"].clamp(-1, 1), pads)
        ours_x = crop(ours["x_hat"].clamp(-1, 1), pads)
        rho = ours["gate_rho"].mean(1, keepdim=True)
        ptex = ours["gate_p_tex"].mean(1, keepdim=True)
        rho = crop(F.interpolate(rho, size=xp.shape[-2:], mode="nearest"), pads)
        ptex = crop(F.interpolate(ptex, size=xp.shape[-2:], mode="nearest"), pads)

        base_err = (x0 - base_x).abs().mean(1, keepdim=True)
        ours_err = (x0 - ours_x).abs().mean(1, keepdim=True)
        err_improve = base_err - ours_err
        base_lp = lpips_spatial(base_x, x0).clamp_min(0)
        ours_lp = lpips_spatial(ours_x, x0).clamp_min(0)
        lp_delta = ours_lp - base_lp
        tex, grad = texture_maps((x0 + 1) / 2)

        high = rho >= torch.quantile(rho.flatten(), 0.75)
        low = rho <= torch.quantile(rho.flatten(), 0.25)

        def mm(m, mask):
            return float(m[mask].mean()) if mask.any() else float("nan")

        bit_y_base = float(base["bit_y"].item() / (h * w))
        bit_y_ours = float(ours["bit_y"].item() / (h * w))
        rows.append({
            "image": path.name,
            "q": args.q_index,
            "bpp_y_base": bit_y_base,
            "bpp_y_ours": bit_y_ours,
            "delta_bpp_y": bit_y_ours - bit_y_base,
            "rho_mean": float(rho.mean()),
            "rho_std": float(rho.std(unbiased=False)),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "ptex_mean": float(ptex.mean()),
            "corr_rho_base_err": pearson(rho, base_err),
            "corr_rho_ours_err": pearson(rho, ours_err),
            "corr_rho_err_improve": pearson(rho, err_improve),
            "corr_rho_lpips_delta": pearson(rho, lp_delta),
            "corr_rho_texture": pearson(rho, tex),
            "corr_rho_grad": pearson(rho, grad),
            "highrho_base_err": mm(base_err, high),
            "lowrho_base_err": mm(base_err, low),
            "highrho_lpips_delta": mm(lp_delta, high),
            "lowrho_lpips_delta": mm(lp_delta, low),
            "highrho_grad": mm(grad, high),
            "lowrho_grad": mm(grad, low),
        })
        print(path.name, f"delta_bpp_y={rows[-1]['delta_bpp_y']:.5f}", f"corr_lpips_delta={rows[-1]['corr_rho_lpips_delta']:.3f}")

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
