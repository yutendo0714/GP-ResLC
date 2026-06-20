#!/usr/bin/env python3
"""Summarize bpp savings at matched perceptual metric values."""

import argparse
import csv
from pathlib import Path

import numpy as np


LOWER_IS_BETTER = {"DISTS", "LPIPS", "FID", "KID"}
HIGHER_IS_BETTER = {"PSNR", "MS-SSIM"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def interp_bpp_at_metric(rows: list[dict[str, str]], metric: str, target: float) -> float | None:
    xs = np.array([float(r[metric]) for r in rows], dtype=np.float64)
    ys = np.array([float(r["bpp"]) for r in rows], dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if target < xs[0] or target > xs[-1]:
        return None
    return float(np.interp(target, xs, ys))


def summarize(path: Path, anchor_name: str, metrics: list[str]) -> list[dict[str, str]]:
    rows = load_rows(path)
    runs = sorted({r["run"] for r in rows})
    if anchor_name not in runs:
        raise RuntimeError(f"{path} does not contain anchor run {anchor_name!r}")
    glc = sorted([r for r in rows if r["run"] == anchor_name], key=lambda r: int(r["q"]))

    out = []
    for run in runs:
        if run == anchor_name:
            continue
        test = sorted([r for r in rows if r["run"] == run], key=lambda r: int(r["q"]))
        for metric in metrics:
            deltas = []
            points = []
            for anchor in glc:
                target = float(anchor[metric])
                bpp_test = interp_bpp_at_metric(test, metric, target)
                if bpp_test is None:
                    continue
                bpp_glc = float(anchor["bpp"])
                delta = (bpp_test / bpp_glc - 1.0) * 100.0
                deltas.append(delta)
                points.append(f"q{anchor['q']}:{delta:+.2f}%")
            if not deltas:
                continue
            arr = np.array(deltas, dtype=np.float64)
            out.append({
                "source": str(path),
                "run": run,
                "metric": metric,
                "matched_points": str(len(deltas)),
                "mean_bpp_delta_pct": f"{float(arr.mean()):+.2f}",
                "median_bpp_delta_pct": f"{float(np.median(arr)):+.2f}",
                "min_bpp_delta_pct": f"{float(arr.min()):+.2f}",
                "max_bpp_delta_pct": f"{float(arr.max()):+.2f}",
                "per_q": "; ".join(points),
            })
    return out


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Matched-metric bpp summary",
        "",
        "Negative bpp delta means GP-ResLC needs fewer bits than GLC at the same metric value.",
        "Targets are the GLC q points that fall inside the GP-ResLC metric range.",
        "",
        "| source | run | metric | points | mean | median | range | per-q deltas |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        source = Path(r["source"]).stem
        lines.append(
            f"| {source} | {r['run']} | {r['metric']} | {r['matched_points']} | "
            f"{r['mean_bpp_delta_pct']}% | {r['median_bpp_delta_pct']}% | "
            f"{r['min_bpp_delta_pct']}..{r['max_bpp_delta_pct']}% | {r['per_q']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="merged metrics CSV")
    ap.add_argument("--anchor", default="GLC")
    ap.add_argument("--metrics", nargs="+", default=["DISTS", "FID", "LPIPS"])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    all_rows = []
    for csv_path in args.csv:
        all_rows.extend(summarize(Path(csv_path), args.anchor, args.metrics))
    if not all_rows:
        raise RuntimeError("no matched metric points found")
    write_csv(all_rows, Path(args.out_csv))
    write_md(all_rows, Path(args.out_md))
    for row in all_rows:
        print(
            f"{Path(row['source']).stem} {row['run']} {row['metric']}: "
            f"mean {row['mean_bpp_delta_pct']}% over {row['matched_points']} points"
        )


if __name__ == "__main__":
    main()
