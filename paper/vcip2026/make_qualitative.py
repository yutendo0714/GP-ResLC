#!/usr/bin/env python3
"""Compact single-column qualitative comparison: Original | GLC | SAQ."""
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from DISTS_pytorch import DISTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path("/tmp/claude-0/-workspace-GP-ResLC/5de43b82-93ee-41d6-90dd-3fa9537bbfc0/scratchpad")
ORIG = SCRATCH / "kodak"
GLC = SCRATCH / "qual_glc" / "q3"
SAQ = SCRATCH / "qual_saq" / "q3"
OUT = Path(__file__).resolve().parent / "figures"
NAME = "kodim23"            # parrots (landscape, rich texture)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dists = DISTS().to(DEVICE).eval()
glc_bpp = json.load(open(GLC / "bpp.json"))["images"][NAME]["bpp"]
saq_bpp = json.load(open(SAQ / "bpp.json"))["images"][NAME]["bpp"]


@torch.no_grad()
def dval(a, b):
    ta = ToTensor()(a).unsqueeze(0).to(DEVICE)
    tb = ToTensor()(b).unsqueeze(0).to(DEVICE)
    return float(dists(ta, tb).item())


orig = Image.open(ORIG / f"{NAME}.png").convert("RGB")
g = Image.open(GLC / f"{NAME}.png").convert("RGB")
s = Image.open(SAQ / f"{NAME}.png").convert("RGB")
gd, sd = dval(orig, g), dval(orig, s)
saved = 100.0 * (saq_bpp - glc_bpp) / glc_bpp

fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85))
titles = ["Original", f"GLC\n{glc_bpp:.4f} bpp", f"SAQ (ours)\n{saq_bpp:.4f} bpp"]
for ax, img, t in zip(axes, [orig, g, s], titles):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(t, fontsize=10)
fig.tight_layout(w_pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"qualitative.{ext}", dpi=220, bbox_inches="tight")
print(f"wrote qualitative.pdf | GLC {glc_bpp:.4f} bpp DISTS {gd:.3f} | "
      f"SAQ {saq_bpp:.4f} bpp ({saved:+.1f}%) DISTS {sd:.3f}")
