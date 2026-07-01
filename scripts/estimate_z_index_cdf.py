#!/usr/bin/env python3
"""Estimate q-specific static entropy tables for GLC z VQ indices."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from scripts.evaluate_real_codec import PAD, build_glc
from src.models.image_model import GLC_Image
from src.utils.test_utils import from_0_1_to_minus1_1, init_func


EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def image_paths(root: str) -> list[str]:
    paths: list[str] = []
    for ext in EXTS:
        paths.extend(glob.glob(str(Path(root) / "**" / ext), recursive=True))
    return sorted(paths)


def crop_image(img: Image.Image, crop: int, rng: random.Random, random_crop: bool) -> Image.Image:
    if crop <= 0:
        return img
    w, h = img.size
    if w < crop or h < crop:
        scale = float(crop) / float(min(w, h))
        img = img.resize((max(crop, int(round(w * scale))), max(crop, int(round(h * scale)))))
        w, h = img.size
    if random_crop:
        left = rng.randint(0, w - crop)
        top = rng.randint(0, h - crop)
    else:
        left = (w - crop) // 2
        top = (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--max_images", type=int, default=2000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--random_crop", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing count per codeword.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return ap.parse_args()


def main() -> None:
    init_func()
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    rng = random.Random(args.seed)
    paths = image_paths(args.data)
    rng.shuffle(paths)
    if args.max_images and args.max_images > 0:
        paths = paths[:args.max_images]
    if not paths:
        raise RuntimeError(f"no images found under {args.data}")

    net = build_glc(args.glc_weights, args.device)
    codebook_size = int(net.codebook_size)
    counts = torch.full((max(args.q_indexes) + 1, codebook_size), float(args.alpha), dtype=torch.float64)
    used_images = 0
    skipped = 0

    for idx, path in enumerate(paths, 1):
        try:
            img = Image.open(path).convert("RGB")
            img = crop_image(img, args.crop, rng, args.random_crop)
            w, h = img.size
            x = from_0_1_to_minus1_1(ToTensor()(img)).unsqueeze(0).to(args.device)
            pl, pr, pt, pb = GLC_Image.get_padding_size(h, w, PAD)
            x = F.pad(x, (pl, pr, pt, pb), mode="replicate")
            with torch.no_grad():
                y_ori = net.vqgan.encoder(x)
                for q in args.q_indexes:
                    y = net.enc(y_ori, net.q_enc[q:q + 1])
                    z = net.hyper_enc(y)
                    z_indices = net.z_vq.get_indices(z).reshape(-1).to(device="cpu", dtype=torch.long)
                    counts[q] += torch.bincount(z_indices, minlength=codebook_size).to(dtype=torch.float64)
            used_images += 1
        except Exception as exc:
            skipped += 1
            print(f"[skip] {path}: {exc}", flush=True)
        if idx == 1 or idx % 100 == 0:
            print(f"[{idx}/{len(paths)}] used={used_images} skipped={skipped}", flush=True)

    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)
    payload = {
        "probs": probs.to(dtype=torch.float32),
        "counts": counts,
        "entropy_bits": entropy.to(dtype=torch.float32),
        "codebook_size": codebook_size,
        "q_indexes": list(args.q_indexes),
        "source_data": str(args.data),
        "used_images": used_images,
        "skipped_images": skipped,
        "crop": int(args.crop),
        "random_crop": bool(args.random_crop),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    sidecar = out.with_suffix(out.suffix + ".json")
    sidecar.write_text(json.dumps({
        "out": str(out),
        "source_data": str(args.data),
        "used_images": used_images,
        "skipped_images": skipped,
        "crop": int(args.crop),
        "random_crop": bool(args.random_crop),
        "alpha": float(args.alpha),
        "entropy_bits": [float(entropy[q].item()) for q in args.q_indexes],
    }, indent=2) + "\n")
    print(f"saved {out}", flush=True)
    for q in args.q_indexes:
        print(f"q{q}: entropy={float(entropy[q].item()):.4f} bits/index", flush=True)


if __name__ == "__main__":
    main()
