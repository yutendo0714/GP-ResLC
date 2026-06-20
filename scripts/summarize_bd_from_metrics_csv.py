#!/usr/bin/env python3
"""Compute BD-rate summaries from evaluate_recon_grid.py CSV files."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from eval_metrics import bd_rate


METRICS = ["DISTS", "LPIPS", "PSNR", "MS-SSIM", "FID", "KID"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(path: Path, anchor: str, metrics: list[str]) -> list[dict[str, str]]:
    rows = load_rows(path)
    by_run = defaultdict(list)
    for row in rows:
        if row.get("bpp"):
            by_run[row["run"]].append(row)
    if anchor not in by_run:
        raise RuntimeError(f"anchor {anchor!r} not found in {path}")
    for run_rows in by_run.values():
        run_rows.sort(key=lambda r: float(r["bpp"]))

    anchor_rows = by_run[anchor]
    rate_a = [float(r["bpp"]) for r in anchor_rows]
    out = []
    for run, run_rows in sorted(by_run.items()):
        if run == anchor:
            continue
        row = {"source": str(path), "run": run}
        rate_t = [float(r["bpp"]) for r in run_rows]
        for metric in metrics:
            try:
                metric_a = [float(r[metric]) for r in anchor_rows]
                metric_t = [float(r[metric]) for r in run_rows]
                row[metric] = f"{bd_rate(rate_a, metric_a, rate_t, metric_t, metric):+.2f}"
            except Exception:
                row[metric] = "nan"
        out.append(row)
    return out


def write_csv(rows: list[dict[str, str]], path: Path, metrics: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "run", *metrics])
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path, metrics: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BD-rate summary",
        "",
        "Negative means lower bitrate than the anchor at matched metric.",
        "",
        "| source | run | " + " | ".join(metrics) + " |",
        "|---|---|" + "|".join(["---:"] * len(metrics)) + "|",
    ]
    for row in rows:
        source = Path(row["source"]).stem
        vals = " | ".join(f"{row[m]}%" for m in metrics)
        lines.append(f"| {source} | {row['run']} | {vals} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--anchor", default="GLC")
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    all_rows = []
    for path in args.csv:
        all_rows.extend(summarize(Path(path), args.anchor, args.metrics))
    if not all_rows:
        raise RuntimeError("no non-anchor runs found")
    write_csv(all_rows, Path(args.out_csv), args.metrics)
    write_md(all_rows, Path(args.out_md), args.metrics)
    for row in all_rows:
        print(Path(row["source"]).stem, row["run"], {m: row[m] for m in args.metrics})


if __name__ == "__main__":
    main()
