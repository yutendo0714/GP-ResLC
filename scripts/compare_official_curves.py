#!/usr/bin/env python3
"""Compare GP-ResLC real-codec results with graph-extracted official GLC curves."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_metrics import bd_rate


METRICS = ["FID", "KID", "DISTS", "LPIPS", "PSNR", "MS-SSIM"]
LOWER_BETTER = {"FID", "KID", "DISTS", "LPIPS"}
DATASET_TO_REAL_CSV = {
    "CLIC 2020": REPO_ROOT / "experiments/real_codec/clic2020_test_real_metrics.csv",
    "DIV2K": REPO_ROOT / "experiments/real_codec/div2k_real_metrics.csv",
    "Kodak": REPO_ROOT / "experiments/real_codec/kodak_real_metrics.csv",
}
DATASET_SLUG = {"CLIC 2020": "clic2020", "DIV2K": "div2k", "Kodak": "kodak"}
OFFICIAL_NOTE = (
    "Graph-extracted from official GLC paper plots as supplied by the project owner; "
    "FID/KID plot ordinates were read from log2-scale paper figures."
)


OFFICIAL_ROWS = [
    # CLIC 2020
    ("CLIC 2020", "FCC", "FID", 0, 0.022632, 26.872159),
    ("CLIC 2020", "HiFiC", "FID", 0, 0.021926, 19.577890),
    ("CLIC 2020", "HiFiC", "FID", 1, 0.025470, 17.104507),
    ("CLIC 2020", "HiFiC", "FID", 2, 0.033113, 14.701990),
    ("CLIC 2020", "HiFiC", "FID", 3, 0.043695, 11.756862),
    ("CLIC 2020", "MS-ILLM", "FID", 0, 0.021792, 11.977931),
    ("CLIC 2020", "MS-ILLM", "FID", 1, 0.027973, 8.329505),
    ("CLIC 2020", "MS-ILLM", "FID", 2, 0.034675, 6.255030),
    ("CLIC 2020", "GLC", "FID", 0, 0.020985, 6.255030),
    ("CLIC 2020", "GLC", "FID", 1, 0.024630, 5.277212),
    ("CLIC 2020", "GLC", "FID", 2, 0.029132, 4.653658),
    ("CLIC 2020", "GLC", "FID", 3, 0.033231, 4.441897),
    ("CLIC 2020", "FCC", "KID", 0, 0.022601, 0.009599),
    ("CLIC 2020", "HiFiC", "KID", 0, 0.021889, 0.005937),
    ("CLIC 2020", "HiFiC", "KID", 1, 0.025453, 0.004882),
    ("CLIC 2020", "HiFiC", "KID", 2, 0.033085, 0.004015),
    ("CLIC 2020", "HiFiC", "KID", 3, 0.043670, 0.003041),
    ("CLIC 2020", "MS-ILLM", "KID", 0, 0.021773, 0.002781),
    ("CLIC 2020", "MS-ILLM", "KID", 1, 0.027976, 0.001744),
    ("CLIC 2020", "MS-ILLM", "KID", 2, 0.034650, 0.001060),
    ("CLIC 2020", "GLC", "KID", 0, 0.020958, 0.001358),
    ("CLIC 2020", "GLC", "KID", 1, 0.024610, 0.001014),
    ("CLIC 2020", "GLC", "KID", 2, 0.029117, 0.000820),
    ("CLIC 2020", "GLC", "KID", 3, 0.033222, 0.000737),
    ("CLIC 2020", "FCC", "DISTS", 0, 0.022637, 0.150629),
    ("CLIC 2020", "HiFiC", "DISTS", 0, 0.021916, 0.154088),
    ("CLIC 2020", "HiFiC", "DISTS", 1, 0.025469, 0.144969),
    ("CLIC 2020", "HiFiC", "DISTS", 2, 0.033095, 0.130660),
    ("CLIC 2020", "HiFiC", "DISTS", 3, 0.043687, 0.119182),
    ("CLIC 2020", "MS-ILLM", "DISTS", 0, 0.021799, 0.116824),
    ("CLIC 2020", "MS-ILLM", "DISTS", 1, 0.027983, 0.098585),
    ("CLIC 2020", "MS-ILLM", "DISTS", 2, 0.034670, 0.085849),
    ("CLIC 2020", "GLC", "DISTS", 0, 0.020978, 0.082075),
    ("CLIC 2020", "GLC", "DISTS", 1, 0.024631, 0.074214),
    ("CLIC 2020", "GLC", "DISTS", 2, 0.029140, 0.069025),
    ("CLIC 2020", "GLC", "DISTS", 3, 0.033229, 0.066667),
    ("CLIC 2020", "FCC", "LPIPS", 0, 0.022634, 0.350922),
    ("CLIC 2020", "HiFiC", "LPIPS", 0, 0.021921, 0.150461),
    ("CLIC 2020", "HiFiC", "LPIPS", 1, 0.025470, 0.135945),
    ("CLIC 2020", "HiFiC", "LPIPS", 2, 0.033099, 0.115668),
    ("CLIC 2020", "HiFiC", "LPIPS", 3, 0.043686, 0.098848),
    ("CLIC 2020", "MS-ILLM", "LPIPS", 0, 0.021785, 0.165207),
    ("CLIC 2020", "MS-ILLM", "LPIPS", 1, 0.027973, 0.141244),
    ("CLIC 2020", "MS-ILLM", "LPIPS", 2, 0.034661, 0.114055),
    ("CLIC 2020", "GLC", "LPIPS", 0, 0.020981, 0.154378),
    ("CLIC 2020", "GLC", "LPIPS", 1, 0.024636, 0.140553),
    ("CLIC 2020", "GLC", "LPIPS", 2, 0.029125, 0.130645),
    ("CLIC 2020", "GLC", "LPIPS", 3, 0.033220, 0.125346),
    ("CLIC 2020", "FCC", "PSNR", 0, 0.022617, 25.153242),
    ("CLIC 2020", "HiFiC", "PSNR", 0, 0.021891, 27.440079),
    ("CLIC 2020", "HiFiC", "PSNR", 1, 0.025466, 27.805501),
    ("CLIC 2020", "HiFiC", "PSNR", 2, 0.033109, 28.459725),
    ("CLIC 2020", "HiFiC", "PSNR", 3, 0.043705, 29.096267),
    ("CLIC 2020", "MS-ILLM", "PSNR", 0, 0.021788, 26.597250),
    ("CLIC 2020", "MS-ILLM", "PSNR", 1, 0.027979, 27.375246),
    ("CLIC 2020", "MS-ILLM", "PSNR", 2, 0.034663, 27.870334),
    ("CLIC 2020", "GLC", "PSNR", 0, 0.020984, 24.074656),
    ("CLIC 2020", "GLC", "PSNR", 1, 0.024663, 24.528487),
    ("CLIC 2020", "GLC", "PSNR", 2, 0.029145, 24.923379),
    ("CLIC 2020", "GLC", "PSNR", 3, 0.033238, 25.165029),
    ("CLIC 2020", "FCC", "MS-SSIM", 0, 0.022655, 0.836273),
    ("CLIC 2020", "HiFiC", "MS-SSIM", 0, 0.021933, 0.883000),
    ("CLIC 2020", "HiFiC", "MS-SSIM", 1, 0.025464, 0.892182),
    ("CLIC 2020", "HiFiC", "MS-SSIM", 2, 0.033093, 0.907909),
    ("CLIC 2020", "MS-ILLM", "MS-SSIM", 0, 0.021804, 0.865182),
    ("CLIC 2020", "MS-ILLM", "MS-SSIM", 1, 0.027990, 0.884455),
    ("CLIC 2020", "MS-ILLM", "MS-SSIM", 2, 0.034665, 0.898000),
    ("CLIC 2020", "GLC", "MS-SSIM", 0, 0.021005, 0.836091),
    ("CLIC 2020", "GLC", "MS-SSIM", 1, 0.024665, 0.848909),
    ("CLIC 2020", "GLC", "MS-SSIM", 2, 0.029149, 0.859182),
    ("CLIC 2020", "GLC", "MS-SSIM", 3, 0.033222, 0.865091),
    # DIV2K
    ("DIV2K", "FCC", "FID", 0, 0.033083, 45.408646),
    ("DIV2K", "HiFiC", "FID", 0, 0.029008, 36.262829),
    ("DIV2K", "HiFiC", "FID", 1, 0.033574, 33.297516),
    ("DIV2K", "HiFiC", "FID", 2, 0.043145, 29.184549),
    ("DIV2K", "MS-ILLM", "FID", 0, 0.029312, 26.385590),
    ("DIV2K", "MS-ILLM", "FID", 1, 0.037768, 20.908447),
    ("DIV2K", "MS-ILLM", "FID", 2, 0.044515, 18.540232),
    ("DIV2K", "GLC", "FID", 0, 0.023478, 14.325955),
    ("DIV2K", "GLC", "FID", 1, 0.027300, 13.103571),
    ("DIV2K", "GLC", "FID", 2, 0.031832, 12.125733),
    ("DIV2K", "GLC", "FID", 3, 0.036026, 11.869858),
    ("DIV2K", "FCC", "KID", 0, 0.033081, 0.011681),
    ("DIV2K", "HiFiC", "KID", 0, 0.029007, 0.008231),
    ("DIV2K", "HiFiC", "KID", 1, 0.033569, 0.007019),
    ("DIV2K", "HiFiC", "KID", 2, 0.043148, 0.005886),
    ("DIV2K", "MS-ILLM", "KID", 0, 0.029293, 0.004142),
    ("DIV2K", "MS-ILLM", "KID", 1, 0.037778, 0.002932),
    ("DIV2K", "MS-ILLM", "KID", 2, 0.044512, 0.002470),
    ("DIV2K", "GLC", "KID", 0, 0.023451, 0.001135),
    ("DIV2K", "GLC", "KID", 1, 0.027273, 0.000882),
    ("DIV2K", "GLC", "KID", 2, 0.031818, 0.000738),
    ("DIV2K", "GLC", "KID", 3, 0.036027, 0.000743),
    ("DIV2K", "FCC", "DISTS", 0, 0.033071, 0.155840),
    ("DIV2K", "HiFiC", "DISTS", 0, 0.029004, 0.164000),
    ("DIV2K", "HiFiC", "DISTS", 1, 0.033571, 0.155200),
    ("DIV2K", "HiFiC", "DISTS", 2, 0.043128, 0.141120),
    ("DIV2K", "MS-ILLM", "DISTS", 0, 0.029301, 0.125920),
    ("DIV2K", "MS-ILLM", "DISTS", 1, 0.037763, 0.108160),
    ("DIV2K", "MS-ILLM", "DISTS", 2, 0.044505, 0.102400),
    ("DIV2K", "GLC", "DISTS", 0, 0.023467, 0.090400),
    ("DIV2K", "GLC", "DISTS", 1, 0.027284, 0.083520),
    ("DIV2K", "GLC", "DISTS", 2, 0.031835, 0.077920),
    ("DIV2K", "GLC", "DISTS", 3, 0.036027, 0.075680),
    ("DIV2K", "FCC", "LPIPS", 0, 0.033087, 0.339362),
    ("DIV2K", "HiFiC", "LPIPS", 0, 0.028989, 0.176596),
    ("DIV2K", "HiFiC", "LPIPS", 1, 0.033571, 0.161968),
    ("DIV2K", "HiFiC", "LPIPS", 2, 0.043144, 0.139096),
    ("DIV2K", "MS-ILLM", "LPIPS", 0, 0.029286, 0.187234),
    ("DIV2K", "MS-ILLM", "LPIPS", 1, 0.037779, 0.161702),
    ("DIV2K", "MS-ILLM", "LPIPS", 2, 0.044489, 0.137766),
    ("DIV2K", "GLC", "LPIPS", 0, 0.023467, 0.184043),
    ("DIV2K", "GLC", "LPIPS", 1, 0.027284, 0.170213),
    ("DIV2K", "GLC", "LPIPS", 2, 0.031820, 0.159309),
    ("DIV2K", "GLC", "LPIPS", 3, 0.036011, 0.152926),
    ("DIV2K", "FCC", "PSNR", 0, 0.033079, 23.280000),
    ("DIV2K", "HiFiC", "PSNR", 0, 0.028992, 25.021538),
    ("DIV2K", "HiFiC", "PSNR", 1, 0.033575, 25.366154),
    ("DIV2K", "HiFiC", "PSNR", 2, 0.043118, 25.975385),
    ("DIV2K", "MS-ILLM", "PSNR", 0, 0.029299, 24.375385),
    ("DIV2K", "MS-ILLM", "PSNR", 1, 0.037780, 25.046154),
    ("DIV2K", "MS-ILLM", "PSNR", 2, 0.044488, 25.489231),
    ("DIV2K", "GLC", "PSNR", 0, 0.023465, 21.513846),
    ("DIV2K", "GLC", "PSNR", 1, 0.027291, 21.907692),
    ("DIV2K", "GLC", "PSNR", 2, 0.031827, 22.289231),
    ("DIV2K", "GLC", "PSNR", 3, 0.036008, 22.590769),
    ("DIV2K", "FCC", "MS-SSIM", 0, 0.033070, 0.819289),
    ("DIV2K", "HiFiC", "MS-SSIM", 0, 0.029011, 0.857372),
    ("DIV2K", "HiFiC", "MS-SSIM", 1, 0.033592, 0.868995),
    ("DIV2K", "HiFiC", "MS-SSIM", 2, 0.043133, 0.888903),
    ("DIV2K", "MS-ILLM", "MS-SSIM", 0, 0.029296, 0.840927),
    ("DIV2K", "MS-ILLM", "MS-SSIM", 1, 0.037769, 0.865533),
    ("DIV2K", "MS-ILLM", "MS-SSIM", 2, 0.044509, 0.880247),
    ("DIV2K", "GLC", "MS-SSIM", 0, 0.023457, 0.783555),
    ("DIV2K", "GLC", "MS-SSIM", 1, 0.027278, 0.800000),
    ("DIV2K", "GLC", "MS-SSIM", 2, 0.031835, 0.813354),
    ("DIV2K", "GLC", "MS-SSIM", 3, 0.036013, 0.821020),
    # Kodak
    ("Kodak", "FCC", "DISTS", 0, 0.030013, 0.183130),
    ("Kodak", "HiFiC", "DISTS", 0, 0.025391, 0.208000),
    ("Kodak", "HiFiC", "DISTS", 1, 0.028957, 0.199652),
    ("Kodak", "HiFiC", "DISTS", 2, 0.033294, 0.189913),
    ("Kodak", "HiFiC", "DISTS", 3, 0.043165, 0.172348),
    ("Kodak", "MS-ILLM", "DISTS", 0, 0.028743, 0.161391),
    ("Kodak", "MS-ILLM", "DISTS", 1, 0.037131, 0.135826),
    ("Kodak", "GLC", "DISTS", 0, 0.024692, 0.112696),
    ("Kodak", "GLC", "DISTS", 1, 0.028615, 0.103826),
    ("Kodak", "GLC", "DISTS", 2, 0.033165, 0.097913),
    ("Kodak", "GLC", "DISTS", 3, 0.037374, 0.095130),
    ("Kodak", "FCC", "LPIPS", 0, 0.030000, 0.381544),
    ("Kodak", "HiFiC", "LPIPS", 0, 0.025384, 0.216107),
    ("Kodak", "HiFiC", "LPIPS", 1, 0.028959, 0.201678),
    ("Kodak", "HiFiC", "LPIPS", 2, 0.033301, 0.186242),
    ("Kodak", "HiFiC", "LPIPS", 3, 0.043151, 0.159396),
    ("Kodak", "MS-ILLM", "LPIPS", 0, 0.028740, 0.218456),
    ("Kodak", "MS-ILLM", "LPIPS", 1, 0.037123, 0.187248),
    ("Kodak", "GLC", "LPIPS", 0, 0.024699, 0.196644),
    ("Kodak", "GLC", "LPIPS", 1, 0.028616, 0.180201),
    ("Kodak", "GLC", "LPIPS", 2, 0.033164, 0.167114),
    ("Kodak", "GLC", "LPIPS", 3, 0.037356, 0.161074),
    ("Kodak", "FCC", "PSNR", 0, 0.030002, 22.879046),
    ("Kodak", "HiFiC", "PSNR", 0, 0.025380, 24.105622),
    ("Kodak", "HiFiC", "PSNR", 1, 0.028980, 24.442930),
    ("Kodak", "HiFiC", "PSNR", 2, 0.033315, 24.734242),
    ("Kodak", "HiFiC", "PSNR", 3, 0.043151, 25.316865),
    ("Kodak", "MS-ILLM", "PSNR", 0, 0.028734, 23.824532),
    ("Kodak", "MS-ILLM", "PSNR", 1, 0.037139, 24.356048),
    ("Kodak", "GLC", "PSNR", 0, 0.024706, 21.320273),
    ("Kodak", "GLC", "PSNR", 1, 0.028611, 21.729131),
    ("Kodak", "GLC", "PSNR", 2, 0.033172, 22.076661),
    ("Kodak", "GLC", "PSNR", 3, 0.037344, 22.275980),
    ("Kodak", "FCC", "MS-SSIM", 0, 0.030004, 0.769217),
    ("Kodak", "HiFiC", "MS-SSIM", 0, 0.025373, 0.810630),
    ("Kodak", "HiFiC", "MS-SSIM", 1, 0.028959, 0.825253),
    ("Kodak", "HiFiC", "MS-SSIM", 2, 0.033303, 0.837788),
    ("Kodak", "MS-ILLM", "MS-SSIM", 0, 0.028754, 0.801905),
    ("Kodak", "MS-ILLM", "MS-SSIM", 1, 0.037135, 0.828080),
    ("Kodak", "GLC", "MS-SSIM", 0, 0.024676, 0.748694),
    ("Kodak", "GLC", "MS-SSIM", 1, 0.028611, 0.767988),
    ("Kodak", "GLC", "MS-SSIM", 2, 0.033160, 0.781874),
    ("Kodak", "GLC", "MS-SSIM", 3, 0.037361, 0.791336),
]


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def official_dict_rows() -> list[dict[str, str]]:
    rows = []
    for dataset, model, metric, q, bpp, value in OFFICIAL_ROWS:
        rows.append({
            "dataset": dataset,
            "source": "official_glc_paper_graph_extract",
            "model": model,
            "metric": metric,
            "q": str(q),
            "bpp": f"{bpp:.6f}",
            "value": f"{value:.6f}",
            "note": OFFICIAL_NOTE,
        })
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def local_real_long_rows() -> list[dict[str, str]]:
    rows = []
    for dataset, csv_path in DATASET_TO_REAL_CSV.items():
        for row in read_csv(csv_path):
            if row["run"] not in {"glc_real", "gp_rho116_real"}:
                continue
            model = "GLC-local-real" if row["run"] == "glc_real" else "GP-ResLC-real"
            for metric in METRICS:
                if metric not in row or row[metric] == "":
                    continue
                rows.append({
                    "dataset": dataset,
                    "source": "local_real_codec",
                    "model": model,
                    "metric": metric,
                    "q": row["q"],
                    "bpp": f"{float(row['bpp']):.12f}",
                    "value": f"{float(row[metric]):.12f}",
                    "note": str(csv_path.relative_to(REPO_ROOT)),
                })
    return rows


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["metric"])].append(row)
    for vals in grouped.values():
        vals.sort(key=lambda r: (int(r["q"]), float(r["bpp"])))
    return grouped


def curve_arrays(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    ordered = sorted(rows, key=lambda r: float(r["bpp"]))
    return [float(r["bpp"]) for r in ordered], [float(r["value"]) for r in ordered]


def sanity_rows(official_rows: list[dict[str, str]], local_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    local_map = {
        (r["dataset"], r["metric"], r["q"]): r
        for r in local_rows
        if r["model"] == "GLC-local-real"
    }
    out = []
    for off in official_rows:
        if off["model"] != "GLC":
            continue
        local = local_map.get((off["dataset"], off["metric"], off["q"]))
        if local is None:
            continue
        metric = off["metric"]
        off_bpp, local_bpp = float(off["bpp"]), float(local["bpp"])
        off_val, local_val = float(off["value"]), float(local["value"])
        if metric in LOWER_BETTER:
            quality_delta = (local_val / off_val - 1.0) * 100.0
            quality_label = "positive=worse-local"
        else:
            quality_delta = (local_val / off_val - 1.0) * 100.0
            quality_label = "positive=better-local"
        out.append({
            "dataset": off["dataset"],
            "metric": metric,
            "q": off["q"],
            "official_bpp": fmt_float(off_bpp, 6),
            "local_bpp": fmt_float(local_bpp, 6),
            "bpp_delta_pct": fmt_float((local_bpp / off_bpp - 1.0) * 100.0, 2),
            "official_value": fmt_float(off_val, 6),
            "local_value": fmt_float(local_val, 6),
            "value_delta": fmt_float(local_val - off_val, 6),
            "quality_delta_pct": fmt_float(quality_delta, 2),
            "quality_delta_interpretation": quality_label,
        })
    return out


def sanity_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["metric"])].append(row)
    out = []
    for (dataset, metric), vals in sorted(grouped.items()):
        bpp_delta = np.array([float(r["bpp_delta_pct"]) for r in vals], dtype=np.float64)
        quality_delta = np.array([float(r["quality_delta_pct"]) for r in vals], dtype=np.float64)
        value_delta = np.array([float(r["value_delta"]) for r in vals], dtype=np.float64)
        out.append({
            "dataset": dataset,
            "metric": metric,
            "points": str(len(vals)),
            "mean_bpp_delta_pct": fmt_float(float(bpp_delta.mean()), 2),
            "mean_abs_bpp_delta_pct": fmt_float(float(np.abs(bpp_delta).mean()), 2),
            "mean_quality_delta_pct": fmt_float(float(quality_delta.mean()), 2),
            "mean_abs_quality_delta_pct": fmt_float(float(np.abs(quality_delta).mean()), 2),
            "mean_value_delta": fmt_float(float(value_delta.mean()), 6),
            "max_abs_quality_delta_pct": fmt_float(float(np.abs(quality_delta).max()), 2),
        })
    return out


def bd_rows(official_rows: list[dict[str, str]], local_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = group_rows(official_rows + local_rows)
    out = []
    for dataset in DATASET_TO_REAL_CSV:
        for metric in METRICS:
            anchor = grouped.get((dataset, "GLC", metric), [])
            test = grouped.get((dataset, "GP-ResLC-real", metric), [])
            if len(anchor) < 4 or len(test) < 4:
                continue
            rate_a, metric_a = curve_arrays(anchor)
            rate_t, metric_t = curve_arrays(test)
            try:
                bd = bd_rate(rate_a, metric_a, rate_t, metric_t, metric)
            except Exception:
                bd = float("nan")
            out.append({
                "dataset": dataset,
                "anchor": "official-graph-GLC",
                "test": "GP-ResLC-real",
                "metric": metric,
                "bd_rate_pct": fmt_float(bd, 2),
                "anchor_points": str(len(anchor)),
                "test_points": str(len(test)),
                "note": "Cross-source comparison; use local real-codec GLC as primary paper anchor.",
            })
    return out


def interp_bpp_at_metric(rows: list[dict[str, str]], target: float) -> float | None:
    xs = np.array([float(r["value"]) for r in rows], dtype=np.float64)
    ys = np.array([float(r["bpp"]) for r in rows], dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    uniq_x, uniq_idx = np.unique(xs, return_index=True)
    xs = uniq_x
    ys = ys[uniq_idx]
    if len(xs) < 2 or target < xs[0] or target > xs[-1]:
        return None
    return float(np.interp(target, xs, ys))


def matched_rows(official_rows: list[dict[str, str]], local_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = group_rows(official_rows + local_rows)
    out = []
    for dataset in DATASET_TO_REAL_CSV:
        for metric in METRICS:
            anchor = grouped.get((dataset, "GLC", metric), [])
            test = grouped.get((dataset, "GP-ResLC-real", metric), [])
            if not anchor or not test:
                continue
            deltas = []
            points = []
            for a in sorted(anchor, key=lambda r: int(r["q"])):
                target = float(a["value"])
                test_bpp = interp_bpp_at_metric(test, target)
                if test_bpp is None:
                    continue
                anchor_bpp = float(a["bpp"])
                delta = (test_bpp / anchor_bpp - 1.0) * 100.0
                deltas.append(delta)
                points.append(f"q{a['q']}:{delta:+.2f}%")
            if not deltas:
                continue
            arr = np.asarray(deltas, dtype=np.float64)
            out.append({
                "dataset": dataset,
                "anchor": "official-graph-GLC",
                "test": "GP-ResLC-real",
                "metric": metric,
                "matched_points": str(len(deltas)),
                "mean_bpp_delta_pct": fmt_float(float(arr.mean()), 2),
                "median_bpp_delta_pct": fmt_float(float(np.median(arr)), 2),
                "min_bpp_delta_pct": fmt_float(float(arr.min()), 2),
                "max_bpp_delta_pct": fmt_float(float(arr.max()), 2),
                "per_q": "; ".join(points),
                "note": "Targets are official graph-extracted GLC q points inside GP-ResLC real metric range.",
            })
    return out


def write_md_table(rows: list[dict[str, str]], path: Path, title: str, intro: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.extend(intro)
    if intro:
        lines.append("")
    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        lines.append("| " + " | ".join(fieldnames) + " |")
        lines.append("|" + "|".join(["---"] * len(fieldnames)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_manifest(out_dir: Path) -> None:
    lines = [
        "# Official Curve Comparison",
        "",
        "This directory compares GP-ResLC real-codec results with graph-extracted points from the official GLC paper plots.",
        "",
        "Important: this is a cross-source positioning aid. The VCIP main rate claim should use the paired local real-codec comparison against local GLC, because it keeps model implementation, codec accounting, patch extraction, and metric code fixed.",
        "",
        "Files:",
        "",
        "- `official_extracted_metrics_long.csv`: graph-extracted official GLC/FCC/HiFiC/MS-ILLM points.",
        "- `official_plus_gp_reslc_real_long.csv`: official curves plus local GP-ResLC and local GLC real-codec points in long format.",
        "- `official_vs_local_glc_sanity.csv/md`: official GLC graph points versus local real-codec GLC points at the same dataset/metric/q.",
        "- `gp_reslc_real_vs_official_glc_bd.csv/md`: GP-ResLC real-codec BD-rate against official graph-extracted GLC.",
        "- `gp_reslc_real_vs_official_glc_matched.csv/md`: GP-ResLC real-codec bpp deltas at official GLC matched metric targets.",
        "- `curves/`: paper-style curves with official competitors and GP-ResLC real-codec points. FID/KID are plotted with log2 y-scale.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def make_plots(combined_rows: list[dict[str, str]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = group_rows(combined_rows)
    styles = {
        "GLC": dict(marker="o", linewidth=2.0),
        "HiFiC": dict(marker="s", linewidth=1.5),
        "MS-ILLM": dict(marker="^", linewidth=1.5),
        "FCC": dict(marker="x", linewidth=0.0),
        "GP-ResLC-real": dict(marker="*", linewidth=2.0, color="black", markersize=11),
    }
    for dataset in DATASET_TO_REAL_CSV:
        for metric in METRICS:
            has_official = any(
                r["dataset"] == dataset
                and r["metric"] == metric
                and r["source"] == "official_glc_paper_graph_extract"
                for r in combined_rows
            )
            if not has_official:
                continue
            fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=180)
            plotted = False
            for model in ["FCC", "HiFiC", "MS-ILLM", "GLC", "GP-ResLC-real"]:
                rows = grouped.get((dataset, model, metric), [])
                if not rows:
                    continue
                ordered = sorted(rows, key=lambda r: float(r["bpp"]))
                x = [float(r["bpp"]) for r in ordered]
                y = [float(r["value"]) for r in ordered]
                style = dict(styles.get(model, {}))
                ax.plot(x, y, label=model, **style)
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            if metric in {"FID", "KID"}:
                try:
                    ax.set_yscale("log", base=2)
                except TypeError:
                    ax.set_yscale("log", basey=2)
            ax.set_xlabel("bpp")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            ax.set_title(f"{dataset} / {metric}")
            fig.tight_layout()
            metric_slug = metric.lower().replace("-", "_")
            curve_dir = out_dir / "curves" / DATASET_SLUG[dataset]
            curve_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(curve_dir / f"curve_{metric_slug}.png")
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "experiments/paper_assets/official_curve_comparison"),
    )
    ap.add_argument("--skip_plots", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    official_rows = official_dict_rows()
    local_rows = local_real_long_rows()
    combined = official_rows + local_rows

    official_fields = ["dataset", "source", "model", "metric", "q", "bpp", "value", "note"]
    write_csv(official_rows, out_dir / "official_extracted_metrics_long.csv", official_fields)
    write_csv(combined, out_dir / "official_plus_gp_reslc_real_long.csv", official_fields)

    sanity = sanity_rows(official_rows, local_rows)
    write_csv(sanity, out_dir / "official_vs_local_glc_sanity.csv")
    write_md_table(
        sanity,
        out_dir / "official_vs_local_glc_sanity.md",
        "Official GLC vs Local Real-Codec GLC Sanity",
        [
            "This table quantifies how far the local real-codec GLC reproduction is from the graph-extracted official GLC curve.",
            "Use it to decide whether official-curve comparison is safe to quote. Large CLIC FID/KID gaps should be treated as protocol/source mismatch, not as a model conclusion.",
        ],
    )
    sanity_summary = sanity_summary_rows(sanity)
    write_csv(sanity_summary, out_dir / "official_vs_local_glc_sanity_summary.csv")
    write_md_table(
        sanity_summary,
        out_dir / "official_vs_local_glc_sanity_summary.md",
        "Official GLC vs Local Real-Codec GLC Sanity Summary",
        [
            "Mean deltas summarize whether the local GLC reproduction is numerically close to the graph-extracted official GLC curve.",
            "For lower-is-better metrics, positive quality delta means local is worse; for PSNR/MS-SSIM, positive means local is better.",
        ],
    )

    bd = bd_rows(official_rows, local_rows)
    write_csv(bd, out_dir / "gp_reslc_real_vs_official_glc_bd.csv")
    write_md_table(
        bd,
        out_dir / "gp_reslc_real_vs_official_glc_bd.md",
        "GP-ResLC Real vs Official GLC BD-Rate",
        [
            "Negative BD-rate means GP-ResLC needs fewer real-codec bits than the graph-extracted official GLC curve at matched metric quality.",
            "This is a cross-source comparison and should be secondary to paired local real-codec BD-rate.",
        ],
    )

    matched = matched_rows(official_rows, local_rows)
    write_csv(matched, out_dir / "gp_reslc_real_vs_official_glc_matched.csv")
    write_md_table(
        matched,
        out_dir / "gp_reslc_real_vs_official_glc_matched.md",
        "GP-ResLC Real vs Official GLC Matched-Metric Bpp",
        [
            "Negative bpp delta means GP-ResLC uses fewer real-codec bits at an official graph-extracted GLC metric target.",
            "Targets outside the GP-ResLC metric range are skipped.",
        ],
    )

    write_manifest(out_dir)
    if not args.skip_plots:
        make_plots(combined, out_dir)

    print(f"wrote {out_dir}")
    for row in bd:
        print(row["dataset"], row["metric"], row["bd_rate_pct"])


if __name__ == "__main__":
    main()
