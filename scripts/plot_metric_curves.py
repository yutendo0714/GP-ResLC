#!/usr/bin/env python3
"""Plot rate-quality curves from evaluate_recon_grid.py CSV output."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("bpp"):
                continue
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV produced by evaluate_recon_grid.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--metrics", nargs="+", default=["DISTS", "LPIPS", "PSNR", "MS-SSIM", "FID", "KID"])
    args = ap.parse_args()

    rows = load_rows(args.csv)
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run"]].append(row)
    for run_rows in by_run.values():
        run_rows.sort(key=lambda r: float(r["bpp"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        if not all(metric in row and row[metric] != "" for row in rows):
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=180)
        for run, run_rows in sorted(by_run.items()):
            xs = [float(r["bpp"]) for r in run_rows]
            ys = [float(r[metric]) for r in run_rows]
            qs = [r.get("q", "") for r in run_rows]
            label = "GLC" if run == "glc" else "GP-ResLC" if run == "gp_reslc" else run
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5, label=label)
            for x, y, q in zip(xs, ys, qs):
                ax.annotate(f"q{q}", (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)

        ax.set_xlabel("bpp")
        ax.set_ylabel(metric)
        if args.title:
            ax.set_title(f"{args.title}: {metric}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"curve_{metric.replace('-', '_')}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
