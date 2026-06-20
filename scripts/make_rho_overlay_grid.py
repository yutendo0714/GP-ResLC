#!/usr/bin/env python3
"""Make paper-oriented grids with original/GLC/GP-ResLC/rho overlay columns."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def find_image(root: Path, key: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{key}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(root / key)


def load_ranked_images(path: Path, max_images: int) -> list[str]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                bpp_delta = float(row["bpp_delta_pct"])
                dists_delta = float(row["dists_delta"])
            except (KeyError, ValueError):
                continue
            if bpp_delta < 0.0 and dists_delta <= 0.0:
                rows.append(row)
    rows.sort(key=lambda r: (float(r["dists_delta"]), float(r["bpp_delta_pct"])))
    return [r["image"] for r in rows[:max_images]]


def load_metrics(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = row
    return out


def resize_keep(img: Image.Image, width: int) -> Image.Image:
    scale = width / img.width
    height = max(1, int(round(img.height * scale)))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def rho_heatmap(rho: torch.Tensor, rho_min: float, rho_max: float) -> Image.Image:
    if rho.ndim == 3:
        rho = rho[0]
    arr = rho.detach().float().cpu().numpy()
    arr = np.clip((arr - rho_min) / max(rho_max - rho_min, 1e-6), 0.0, 1.0)

    # Blue -> green -> yellow -> red. High rho means stronger decoder-side
    # residual suppression, so warm areas are the most aggressively coarsened.
    r = np.clip(1.5 * arr - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(arr - 0.55) * 2.4, 0.0, 1.0)
    b = np.clip(1.25 - 1.8 * arr, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def make_overlay(base: Image.Image, rho_path: Path, width: int, rho_min: float, rho_max: float, alpha: float) -> Image.Image:
    try:
        rho = torch.load(rho_path, map_location="cpu", weights_only=True)
    except TypeError:
        rho = torch.load(rho_path, map_location="cpu")
    heat = rho_heatmap(rho, rho_min=rho_min, rho_max=rho_max).resize(base.size, Image.Resampling.NEAREST)
    overlay = Image.blend(base, heat, alpha=alpha)
    return resize_keep(overlay, width)


def metric_label(key: str, row: dict[str, str]) -> str:
    if not row:
        return key
    return (
        f"{key} | bpp {float(row['anchor_bpp']):.4f}->{float(row['test_bpp']):.4f} "
        f"({float(row['bpp_delta_pct']):+.1f}%) | "
        f"DISTS {float(row['anchor_dists']):.4f}->{float(row['test_dists']):.4f} "
        f"({float(row['dists_delta']):+.4f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--anchor", required=True, help="GLC q directory")
    ap.add_argument("--test", required=True, help="GP-ResLC q directory")
    ap.add_argument("--rho_dir", required=True)
    ap.add_argument("--q_index", type=int, default=3)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--rank_csv", default=None)
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--metrics_csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--rho_min", type=float, default=1.0)
    ap.add_argument("--rho_max", type=float, default=1.4)
    ap.add_argument("--alpha", type=float, default=0.55)
    args = ap.parse_args()

    if args.images:
        images = args.images[:args.max_images]
    elif args.rank_csv:
        images = load_ranked_images(Path(args.rank_csv), args.max_images)
    else:
        raise ValueError("provide --images or --rank_csv")
    if not images:
        raise RuntimeError("no images selected")

    orig = Path(args.orig)
    anchor = Path(args.anchor)
    test = Path(args.test)
    rho_dir = Path(args.rho_dir)
    metrics = load_metrics(Path(args.metrics_csv) if args.metrics_csv else None)

    margin = 16
    label_h = 48
    col_gap = 10
    row_gap = 20
    cols = ["Original", "GLC q3", "GP-ResLC q3", "rho overlay"]

    prepared = []
    max_h = 0
    for key in images:
        base = Image.open(find_image(orig, key)).convert("RGB")
        ims = [
            resize_keep(base, args.width),
            resize_keep(Image.open(find_image(anchor, key)).convert("RGB"), args.width),
            resize_keep(Image.open(find_image(test, key)).convert("RGB"), args.width),
            make_overlay(
                base,
                rho_dir / f"{key}_rho_q{args.q_index}.pt",
                width=args.width,
                rho_min=args.rho_min,
                rho_max=args.rho_max,
                alpha=args.alpha,
            ),
        ]
        max_h = max(max_h, *(im.height for im in ims))
        prepared.append((key, ims))

    canvas_w = margin * 2 + len(cols) * args.width + (len(cols) - 1) * col_gap
    row_h = label_h + max_h
    canvas_h = margin * 2 + len(prepared) * row_h + (len(prepared) - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = margin
    for key, ims in prepared:
        draw.text((margin, y), metric_label(key, metrics.get(key, {})), fill=(0, 0, 0))
        x = margin
        for title, im in zip(cols, ims):
            draw.text((x, y + 20), title, fill=(55, 55, 55))
            canvas.paste(im, (x, y + label_h))
            x += args.width + col_gap
        y += row_h + row_gap

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
