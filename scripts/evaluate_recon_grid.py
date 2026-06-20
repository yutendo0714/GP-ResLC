#!/usr/bin/env python3
"""Evaluate reconstruction grids produced by test_v2.py.

Example:
  python scripts/evaluate_recon_grid.py \
    --orig /dpl/kodak \
    --run final=experiments/eval/final \
    --run ckpt_3000=experiments/eval/ckpt_3000 \
    --q_indexes 0 1 2 3 \
    --patch 64 \
    --out_json experiments/eval/metrics.json
"""

import argparse
import csv
import json
import re
from pathlib import Path

from eval_metrics import distribution_metrics, full_reference
from src.utils.test_utils import init_func


def load_bpp(q_dir: Path) -> dict:
    bpp_path = q_dir / "bpp.json"
    if bpp_path.exists():
        data = json.loads(bpp_path.read_text())
        row = {"bpp": float(data.get("avg_bpp", 0.0))}
        if "avg_bpp_y" in data:
            row["bpp_y"] = float(data["avg_bpp_y"])
        if "avg_bpp_z" in data:
            row["bpp_z"] = float(data["avg_bpp_z"])
        return row

    res_path = q_dir / "res.txt"
    if res_path.exists():
        match = re.search(r"\bbpp\s*=\s*([0-9.eE+-]+)", res_path.read_text())
        if match:
            return {"bpp": float(match.group(1))}
    return {}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    init_func()
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--run", action="append", required=True,
                    help="name=path where path contains q0/q1/... folders")
    ap.add_argument("--q_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--split_patch_num", type=int, default=2)
    ap.add_argument("--kid_subset_size", type=int, default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    results = []
    for run_arg in args.run:
        name, root = parse_run(run_arg)
        for q in args.q_indexes:
            q_dir = root / f"q{q}"
            if not q_dir.exists():
                raise FileNotFoundError(q_dir)
            row = {"run": name, "q": q}
            row.update(load_bpp(q_dir))
            row.update(full_reference(args.orig, str(q_dir)))
            row.update(distribution_metrics(
                args.orig,
                str(q_dir),
                patch=args.patch,
                split_patch_num=args.split_patch_num,
                kid_subset_size=args.kid_subset_size,
                return_patch_count=True,
            ))
            results.append(row)
            print(
                f"{name:12s} q{q}: "
                f"bpp={row.get('bpp', 0.0):.4f} "
                f"PSNR={row['PSNR']:.4f} LPIPS={row['LPIPS']:.4f} "
                f"DISTS={row['DISTS']:.4f} FID={row['FID']:.4f} KID={row['KID']:.4f}",
                flush=True,
            )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    out_csv = Path(args.out_csv) if args.out_csv else out_json.with_suffix(".csv")
    if results:
        preferred = [
            "run", "q", "bpp", "bpp_y", "bpp_z",
            "PSNR", "MS-SSIM", "LPIPS", "DISTS", "FID", "KID",
            "FID_PATCHES", "KID_PATCHES", "FID_PATCH_SIZE", "FID_SPLIT_PATCH_NUM",
        ]
        keys = set().union(*(row.keys() for row in results))
        fieldnames = [key for key in preferred if key in keys]
        fieldnames += sorted(keys.difference(fieldnames))
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
