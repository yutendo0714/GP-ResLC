#!/usr/bin/env python3
"""Rank qualitative examples by per-image rate/perceptual changes."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

import lpips as lpips_lib
from DISTS_pytorch import DISTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_keys(root: Path):
    return [p.stem for p in sorted(root.iterdir()) if p.suffix.lower() in IMG_EXTS]


def load_img(path: Path):
    return ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)


def find_image(root: Path, key: str):
    for ext in IMG_EXTS:
        p = root / f"{key}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(root / key)


def load_bpp_map(q_dir: Path, keys):
    bpp_json = q_dir / "bpp.json"
    if bpp_json.exists():
        data = json.loads(bpp_json.read_text())
        images = data.get("images", {})
        return {k: float(images[k]["bpp"]) for k in keys if k in images}

    items = q_dir / "items.txt"
    if items.exists():
        vals = []
        for line in items.read_text().splitlines():
            m = re.search(r"\bbpp\s*=\s*([0-9.eE+-]+)", line)
            if m:
                vals.append(float(m.group(1)))
        if len(vals) >= len(keys):
            return dict(zip(keys, vals[:len(keys)]))

    res = q_dir / "res.txt"
    if res.exists():
        m = re.search(r"\bbpp\s*=\s*([0-9.eE+-]+)", res.read_text())
        if m:
            return {k: float(m.group(1)) for k in keys}
    return {}


@torch.no_grad()
def metrics(x, y, lpips_fn, dists_fn):
    h = min(x.shape[2], y.shape[2])
    w = min(x.shape[3], y.shape[3])
    x = x[..., :h, :w]
    y = y[..., :h, :w]
    mse = torch.mean((x - y) ** 2).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
    lp = lpips_fn(x * 2 - 1, y * 2 - 1).item()
    ds = dists_fn(x, y).item()
    return psnr, lp, ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--anchor", required=True, help="baseline recon q directory")
    ap.add_argument("--test", required=True, help="proposed recon q directory")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    orig = Path(args.orig)
    anchor = Path(args.anchor)
    test = Path(args.test)
    keys = [k for k in image_keys(orig) if (anchor / f"{k}.png").exists() or any((anchor / f"{k}{e}").exists() for e in IMG_EXTS)]
    keys = [k for k in keys if any((test / f"{k}{e}").exists() for e in IMG_EXTS)]
    if not keys:
        raise RuntimeError("no paired images")

    anchor_bpp = load_bpp_map(anchor, keys)
    test_bpp = load_bpp_map(test, keys)

    lpips_fn = lpips_lib.LPIPS(net="alex").to(DEVICE).eval()
    dists_fn = DISTS().to(DEVICE).eval()

    rows = []
    for key in keys:
        x = load_img(find_image(orig, key))
        ya = load_img(find_image(anchor, key))
        yt = load_img(find_image(test, key))
        a_psnr, a_lpips, a_dists = metrics(x, ya, lpips_fn, dists_fn)
        t_psnr, t_lpips, t_dists = metrics(x, yt, lpips_fn, dists_fn)
        ab = anchor_bpp.get(key, float("nan"))
        tb = test_bpp.get(key, float("nan"))
        rows.append({
            "image": key,
            "anchor_bpp": ab,
            "test_bpp": tb,
            "bpp_delta_pct": (tb / ab - 1.0) * 100.0 if ab == ab and tb == tb and ab > 0 else float("nan"),
            "anchor_psnr": a_psnr,
            "test_psnr": t_psnr,
            "psnr_delta": t_psnr - a_psnr,
            "anchor_lpips": a_lpips,
            "test_lpips": t_lpips,
            "lpips_delta": t_lpips - a_lpips,
            "anchor_dists": a_dists,
            "test_dists": t_dists,
            "dists_delta": t_dists - a_dists,
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    candidates = [r for r in rows if r["bpp_delta_pct"] < 0 and r["dists_delta"] <= 0]
    candidates.sort(key=lambda r: (r["dists_delta"], r["bpp_delta_pct"]))
    print(f"paired={len(rows)} candidates={len(candidates)} out={out}")
    for r in candidates[:args.topk]:
        print(
            f"{r['image']:10s} bpp={r['test_bpp']:.4f} ({r['bpp_delta_pct']:+.1f}%) "
            f"DISTS {r['anchor_dists']:.4f}->{r['test_dists']:.4f} ({r['dists_delta']:+.4f}) "
            f"LPIPS {r['anchor_lpips']:.4f}->{r['test_lpips']:.4f} ({r['lpips_delta']:+.4f})"
        )


if __name__ == "__main__":
    main()
