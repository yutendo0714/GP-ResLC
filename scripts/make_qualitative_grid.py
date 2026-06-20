#!/usr/bin/env python3
"""Make a qualitative comparison grid for original / anchor / test reconstructions."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def find_image(root: Path, key: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{key}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(root / key)


def load_metrics(path: str | None):
    if not path:
        return {}
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["image"]] = row
    return rows


def resize_keep(img: Image.Image, width: int) -> Image.Image:
    scale = width / img.width
    h = max(1, int(round(img.height * scale)))
    return img.resize((width, h), Image.Resampling.LANCZOS)


def label(draw: ImageDraw.ImageDraw, xy, text: str):
    draw.text(xy, text, fill=(10, 10, 10))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--metrics_csv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=280)
    args = ap.parse_args()

    orig = Path(args.orig)
    anchor = Path(args.anchor)
    test = Path(args.test)
    metrics = load_metrics(args.metrics_csv)
    margin = 14
    label_h = 44
    col_gap = 10
    row_gap = 18
    cols = ["Original", "GLC q3", "GP-ResLC q3"]

    prepared = []
    max_h = 0
    for key in args.images:
        ims = [
            resize_keep(Image.open(find_image(orig, key)).convert("RGB"), args.width),
            resize_keep(Image.open(find_image(anchor, key)).convert("RGB"), args.width),
            resize_keep(Image.open(find_image(test, key)).convert("RGB"), args.width),
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
        row = metrics.get(key, {})
        sub = key
        if row:
            sub = (
                f"{key} | bpp {float(row['test_bpp']):.4f} "
                f"({float(row['bpp_delta_pct']):+.1f}%) | "
                f"DISTS {float(row['anchor_dists']):.4f}->{float(row['test_dists']):.4f}"
            )
        draw.text((margin, y), sub, fill=(0, 0, 0))
        x = margin
        for title, im in zip(cols, ims):
            draw.text((x, y + 18), title, fill=(55, 55, 55))
            canvas.paste(im, (x, y + label_h))
            x += args.width + col_gap
        y += row_h + row_gap

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
