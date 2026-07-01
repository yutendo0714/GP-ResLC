#!/usr/bin/env python3
"""Gate (gamma) mechanism analysis + overlay figure for the VCIP paper.

Loads the frozen GLC codec plus the trained decoder-side controller, computes
the per-location quantization-gate map gamma>=1 from decoder-available context,
and correlates it with local statistics of the image / base reconstruction.
Also renders a gamma-overlay figure for a few images.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_real_codec import build_gp_reslc  # noqa: E402
from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import from_0_1_to_minus1_1  # noqa: E402
from gp_reslc.prior_predictor import (  # noqa: E402
    forward_four_part_prior_with_stage_residual_quant_gate,
)

GLC_W = str(ROOT / "pretrained" / "GLC_image.pth.tar")
CKPT = str(ROOT / "experiments" / "stage_safe_rdo_gate_from_sb03_2000" / "v2_final.pt")
KODAK = Path("/tmp/claude-0/-workspace-GP-ResLC/5de43b82-93ee-41d6-90dd-3fa9537bbfc0/scratchpad/kodak")
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
Q = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pearson(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a * b).sum() / den) if float(den) > 1e-12 else float("nan")


def texture_grad(x):
    gray = x.mean(0, keepdim=True).unsqueeze(0)
    local = F.avg_pool2d(gray, 7, 1, 3)
    var = F.avg_pool2d((gray - local) ** 2, 7, 1, 3)[0, 0]
    gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))[0, 0]
    gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))[0, 0]
    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    return var, grad


@torch.no_grad()
def gamma_map(net, x_padded, q):
    """Return gamma at latent resolution and the base GLC reconstruction."""
    curr_q_enc = net.q_enc[q:q + 1]
    y_ori = net.vqgan.encoder(x_padded)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)
    index = net.z_vq.get_indices(z)
    z_hat = net.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    q_shift = net.q_embed[q:q + 1].to(params.device, params.dtype)
    out = forward_four_part_prior_with_stage_residual_quant_gate(
        net, y, params, net.stage_residual_predictor, net.stage_quant_gate,
        predictor_delta_bound=0.0, q_shift=q_shift)
    rho_map = out[4]                       # (1, C, h, w)
    gamma = rho_map.mean(dim=1)[0]         # (h, w) per-location average gate
    base = net.test(x_padded, q)["x_hat"].clamp(-1, 1)
    return gamma, base


def main():
    net = build_gp_reslc(GLC_W, CKPT, interpolate=False, device=DEVICE)
    imgs = sorted(KODAK.glob("*.png"))
    rows = []
    overlays = []
    for p in imgs:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        x = from_0_1_to_minus1_1(ToTensor()(img)).unsqueeze(0).to(DEVICE)
        pl, pr, pt, pb = GLC_Image.get_padding_size(h, w, 64)
        xp = F.pad(x, (pl, pr, pt, pb), mode="replicate")
        gamma, base = gamma_map(net, xp, Q)
        hl, wl = gamma.shape
        x01 = (xp[0].clamp(-1, 1) + 1) / 2
        base01 = (base[0] + 1) / 2
        err = (x01 - base01).abs().mean(0, keepdim=True).unsqueeze(0)   # (1,1,H,W)
        var, grad = texture_grad(x01.cpu())
        # pool image-space maps to latent resolution, drop a 1-cell border
        def pool(m):
            return F.adaptive_avg_pool2d(m, (hl, wl))[0, 0]
        err_l = pool(err.cpu())
        var_l = pool(var.unsqueeze(0).unsqueeze(0))
        grad_l = pool(grad.unsqueeze(0).unsqueeze(0))
        g = gamma.cpu()
        sl = (slice(1, -1), slice(1, -1))
        gc_, ec, vc, grc = g[sl], err_l[sl], var_l[sl], grad_l[sl]
        rows.append({
            "image": p.stem,
            "gamma_mean": float(g.mean()), "gamma_std": float(g.std()),
            "gamma_min": float(g.min()), "gamma_max": float(g.max()),
            "corr_gamma_err": pearson(gc_, ec),
            "corr_gamma_texture": pearson(gc_, vc),
            "corr_gamma_grad": pearson(gc_, grc),
        })
        # high vs low gamma region stats (top/bottom quartile)
        thr_hi = torch.quantile(gc_.flatten(), 0.75)
        thr_lo = torch.quantile(gc_.flatten(), 0.25)
        hi = gc_ >= thr_hi; lo = gc_ <= thr_lo
        rows[-1]["hi_err"] = float(ec[hi].mean()); rows[-1]["lo_err"] = float(ec[lo].mean())
        rows[-1]["hi_grad"] = float(grc[hi].mean()); rows[-1]["lo_grad"] = float(grc[lo].mean())
        overlays.append((p.stem, x01.cpu().numpy(), gamma.cpu().numpy(),
                         (pl, pt, h, w)))

    keys = [k for k in rows[0] if k != "image"]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "std": float(np.std([r[k] for r in rows]))} for k in keys}
    (OUT / "gate_analysis.json").write_text(json.dumps({"q": Q, "summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- overlay figure: pick 3 representative images ----
    pick = [o for o in overlays if o[0] in {"kodim05", "kodim08", "kodim23", "kodim19", "kodim01"}][:3]
    if len(pick) < 3:
        pick = overlays[:3]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0))
    for ci, (name, x01, gam, (pl, pt, h, w)) in enumerate(pick):
        rgb_pad = np.transpose(x01, (1, 2, 0))
        Hp, Wp = rgb_pad.shape[:2]
        # upsample gamma to padded image size, then crop both to the valid region
        gam_up = np.array(Image.fromarray(gam).resize((Wp, Hp), Image.BILINEAR))
        rgb = rgb_pad[pt:pt + h, pl:pl + w]
        gam_c = gam_up[pt:pt + h, pl:pl + w]
        vmin, vmax = np.percentile(gam_c, 3), np.percentile(gam_c, 97)
        axes[0, ci].imshow(rgb)
        axes[0, ci].set_title(name, fontsize=11)
        axes[0, ci].axis("off")
        im = axes[1, ci].imshow(gam_c, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[1, ci].axis("off")
        cbar = fig.colorbar(im, ax=axes[1, ci], fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    axes[0, 0].text(-0.08, 0.5, "Input", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    axes[1, 0].text(-0.08, 0.5, r"Gate $\gamma$", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"gate_map.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "gate_map.pdf")


if __name__ == "__main__":
    main()
