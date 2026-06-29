#!/usr/bin/env python3
"""Create a MIRU-friendly 2x4 perceptual metric curve panel.

Rows: CLIC2020 test, DIV2K validation.
Columns: FID, KID, DISTS, LPIPS.
The output is intentionally larger and less tick-dense than the first draft
figure so it remains readable when placed across the paper width.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper/miru2026/figures"
DATASETS = [
    ("CLIC2020 test", ROOT / "experiments/real_codec/clic2020_test_real_metrics.csv"),
    ("DIV2K validation", ROOT / "experiments/real_codec/div2k_real_metrics.csv"),
]
METRICS = ["FID", "KID", "DISTS", "LPIPS"]
RUN_LABELS = {
    "glc_real": "GLC",
    "gp_rho116_real": "GP-ResLC",
}
RUN_STYLES = {
    "glc_real": dict(color="#1f77b4", marker="o", linestyle="-", linewidth=2.2, markersize=5.5),
    "gp_rho116_real": dict(color="#ff7f0e", marker="s", linestyle="-", linewidth=2.2, markersize=5.5),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        run = row["run"]
        if run not in RUN_LABELS:
            continue
        out.setdefault(run, []).append(row)
    for pts in out.values():
        pts.sort(key=lambda r: float(r["bpp"]))
    return out


def metric_values(points: list[dict[str, str]], metric: str) -> list[float]:
    vals = [float(p[metric]) for p in points]
    if metric == "KID":
        vals = [v * 1000.0 for v in vals]
    return vals


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.solid_capstyle": "round",
    })
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), dpi=260, constrained_layout=True)

    for row_idx, (dataset_name, csv_path) in enumerate(DATASETS):
        runs = grouped(read_rows(csv_path))
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            all_x: list[float] = []
            all_y: list[float] = []
            for run_name in ("glc_real", "gp_rho116_real"):
                pts = runs[run_name]
                xs = [float(p["bpp"]) for p in pts]
                ys = metric_values(pts, metric)
                all_x.extend(xs)
                all_y.extend(ys)
                ax.plot(xs, ys, label=RUN_LABELS[run_name], **RUN_STYLES[run_name])
                for p, x, y in zip(pts, xs, ys):
                    ax.annotate(f"q{p['q']}", (x, y), textcoords="offset points",
                                xytext=(4, 4), fontsize=8, color=RUN_STYLES[run_name]["color"])

            ax.set_title(metric if metric != "KID" else "KID (x1e-3)")
            if col_idx == 0:
                ax.set_ylabel(dataset_name)
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("bpp")
            ax.grid(True, color="#d1d5db", alpha=0.55, linewidth=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            x_pad = (max(all_x) - min(all_x)) * 0.12
            y_pad = (max(all_y) - min(all_y)) * 0.18 if max(all_y) > min(all_y) else 0.01
            ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
            ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.045))
    out = FIG_DIR / "result_curves_clic_div2k_perceptual_2x4_readable.png"
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
