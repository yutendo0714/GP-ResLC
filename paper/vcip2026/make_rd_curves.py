#!/usr/bin/env python3
"""Rate-quality curves: 2 datasets x 4 metrics (FID, KID, DISTS, LPIPS).

Linear axes with per-point operating-point labels (q0..q3). At a fixed quality
(y), the proposed curve sits to the left of GLC, i.e. fewer bits for the same
quality; the q-labels make the per-operating-point correspondence explicit.
Everything is read from the stored evaluation CSVs.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
RC = ROOT / "experiments" / "real_codec"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CSVS = {
    "CLIC2020 test": RC / "clic2020_test_stage_safe_rdo_gate_from_sb03_2000_compare_metrics.csv",
    "DIV2K validation": RC / "div2k_stage_safe_rdo_gate_from_sb03_2000_compare_metrics.csv",
}
BASE = {"glc", "GLC"}
OURS = {"SafeRDO"}
METRICS = [
    ("FID", "FID $\\downarrow$"),
    ("KID", "KID ($\\times10^{3}$) $\\downarrow$"),
    ("DISTS", "DISTS $\\downarrow$"),
    ("LPIPS", "LPIPS $\\downarrow$"),
]
KID_SCALE = 1e3
GLC_C, OURS_C = "#1f77b4", "#e8590c"   # blue (GLC) / orange (ours)

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
    "legend.fontsize": 15, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "axes.linewidth": 1.0, "lines.linewidth": 2.4, "lines.markersize": 8.5,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def _isnum(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def load(path):
    base, ours = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {k: (float(v) if _isnum(v) else v) for k, v in r.items()}
            (base if row["run"] in BASE else ours if row["run"] in OURS else []).append(row)
    base.sort(key=lambda d: d["q"]); ours.sort(key=lambda d: d["q"])
    return base, ours


fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.6))
for ri, (dset, path) in enumerate(CSVS.items()):
    base, ours = load(path)
    for ci, (mkey, mlabel) in enumerate(METRICS):
        ax = axes[ri, ci]
        sc = KID_SCALE if mkey == "KID" else 1.0
        bx = [d["bpp"] for d in base]; by = [d[mkey] * sc for d in base]
        ox = [d["bpp"] for d in ours]; oy = [d[mkey] * sc for d in ours]
        ax.plot(bx, by, color=GLC_C, linestyle="-", marker="o", label="GLC")
        ax.plot(ox, oy, color=OURS_C, linestyle="-", marker="s", label="SAQ (ours)")
        ax.grid(True, color="0.85", linewidth=0.6); ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.margins(x=0.12, y=0.12)
        if ri == 1:
            ax.set_xlabel("Rate [bpp]")
        ax.set_ylabel(mlabel, labelpad=2)
        if ri == 0 and ci == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.95)

fig.tight_layout(rect=(0.035, 0.0, 1.0, 1.0), w_pad=1.2, h_pad=1.4)
for ri, dset in enumerate(CSVS):
    pos = axes[ri, 0].get_position()
    fig.text(0.008, (pos.y0 + pos.y1) / 2.0, dset, rotation=90,
             va="center", ha="center", fontsize=16, fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"rd_curves.{ext}", dpi=200, bbox_inches="tight")
print("wrote", OUT / "rd_curves.pdf")
