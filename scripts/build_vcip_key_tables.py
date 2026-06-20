#!/usr/bin/env python3
"""Build key VCIP paper tables from GP-ResLC metric artifacts.

The default package is paper-facing real-codec evaluation: bitrate is measured from
serialized payload bytes produced by scripts/evaluate_real_codec.py. Estimated-bpp
validation artifacts can still be passed explicitly with --bd_csv/--matched_csv,
but they are no longer the default VCIP package.
"""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_REAL_BD = [
    "experiments/real_codec/clic2020_test_real_bd_rate_summary.csv",
    "experiments/real_codec/div2k_real_bd_rate_summary.csv",
    "experiments/real_codec/kodak_real_bd_rate_summary.csv",
]
DEFAULT_REAL_MATCHED = [
    "experiments/real_codec/clic2020_test_real_matched_metric_summary.csv",
    "experiments/real_codec/div2k_real_matched_metric_summary.csv",
    "experiments/real_codec/kodak_real_matched_metric_summary.csv",
]
DEFAULT_REAL_METRICS = [
    "experiments/real_codec/clic2020_test_real_metrics.csv",
    "experiments/real_codec/div2k_real_metrics.csv",
    "experiments/real_codec/kodak_real_metrics.csv",
]

DEFAULT_OFFICIAL_COMPARISON_DIR = Path("experiments/paper_assets/official_curve_comparison")

DATASET_NAMES = {
    "clic2020_test_real_metrics": "CLIC2020 test",
    "clic_prof_test_real_metrics": "CLIC professional test",
    "div2k_real_metrics": "DIV2K validation",
    "kodak_real_metrics": "Kodak",
    "clic_prof_test_metrics_officialpatch": "CLIC professional test local",
    "div2k_metrics_officialpatch": "DIV2K valid local",
    "kodak_glc_gp112_gp116_metrics": "Kodak",
    "clic_prof_valid_glc_gp112_gp116_metrics": "CLIC professional valid",
    "clic_mobile_valid_metrics": "CLIC mobile valid",
}

DATASET_ORDER = [
    "CLIC2020 test",
    "CLIC professional test",
    "DIV2K validation",
    "Kodak",
    "CLIC professional test local",
    "CLIC professional valid",
    "CLIC mobile valid",
]

GATE_JSONS = [
    (
        "Kodak q3",
        Path("experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_kodak_q3.json"),
    ),
    (
        "CLIC professional valid q3",
        Path("experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_clic_q3.json"),
    ),
    (
        "CLIC mobile valid q3",
        Path("experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/gate_corr_clic_mobile_q3.json"),
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_csvs(root: Path, paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel in paths:
        rows.extend(read_csv(root / rel))
    return rows


def source_name(source: str) -> str:
    stem = Path(source).stem
    return DATASET_NAMES.get(stem, stem)


def dataset_sort_key(name: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(name), name)
    except ValueError:
        return (len(DATASET_ORDER), name)


def pct(value: str | float) -> str:
    val = float(value)
    return f"{val:+.2f}%"


def normalize_run(run: str) -> str:
    if run == "gp_rho116_real":
        return "rho1.16 real"
    if run == "GP-ResLC-rho1.16":
        return "rho1.16 est"
    if run.startswith("GP-ResLC-"):
        return run.replace("GP-ResLC-", "")
    return run


def build_bd_table(rows: list[dict[str, str]]) -> list[str]:
    rows = [r for r in rows if r.get("run") in {"gp_rho116_real", "GP-ResLC-rho1.16"}]
    rows.sort(key=lambda r: dataset_sort_key(source_name(r["source"])))
    lines = [
        "## Main Real-Codec BD-Rate Table",
        "",
        "Negative values mean lower serialized bitrate than GLC real codec at matched metric.",
        "",
        "| dataset | model | DISTS | FID | LPIPS | PSNR | MS-SSIM | KID |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {run} | {dists} | {fid} | {lpips} | {psnr} | {msssim} | {kid} |".format(
                dataset=source_name(row["source"]),
                run=normalize_run(row["run"]),
                dists=pct(row["DISTS"]),
                fid=pct(row["FID"]),
                lpips=pct(row["LPIPS"]),
                psnr=pct(row["PSNR"]),
                msssim=pct(row["MS-SSIM"]),
                kid=pct(row["KID"]),
            )
        )
    return lines


def build_matched_table(rows: list[dict[str, str]]) -> list[str]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    datasets: set[str] = set()
    for row in rows:
        if row.get("run") not in {"gp_rho116_real", "GP-ResLC-rho1.16"}:
            continue
        if row.get("metric") not in {"DISTS", "FID", "LPIPS"}:
            continue
        dataset = source_name(row["source"])
        datasets.add(dataset)
        selected[(dataset, row["metric"])] = row

    lines = [
        "## Matched-Metric Real-Bpp Table",
        "",
        "Negative values mean GP-ResLC needs fewer serialized bits than GLC at the same metric value.",
        "",
        "| dataset | DISTS mean | DISTS points | FID mean | FID points | LPIPS mean | LPIPS points |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(datasets, key=dataset_sort_key):
        dists = selected.get((dataset, "DISTS"))
        fid = selected.get((dataset, "FID"))
        lpips = selected.get((dataset, "LPIPS"))
        lines.append(
            "| {dataset} | {dists_mean} | {dists_n} | {fid_mean} | {fid_n} | {lpips_mean} | {lpips_n} |".format(
                dataset=dataset,
                dists_mean=pct(dists["mean_bpp_delta_pct"]) if dists else "n/a",
                dists_n=dists["matched_points"] if dists else "0",
                fid_mean=pct(fid["mean_bpp_delta_pct"]) if fid else "n/a",
                fid_n=fid["matched_points"] if fid else "0",
                lpips_mean=pct(lpips["mean_bpp_delta_pct"]) if lpips else "n/a",
                lpips_n=lpips["matched_points"] if lpips else "0",
            )
        )
    return lines


def build_real_bpp_table(rows: list[dict[str, str]]) -> list[str]:
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row.get("_source", ""), []).append(row)

    lines = [
        "## Per-Q Real-Bpp Savings",
        "",
        "These deltas are measured from actual serialized payload bytes. The fixed-width z stream and compact header are unchanged, so savings come from the arithmetic-coded y stream.",
        "",
        "| dataset | q0 | q1 | q2 | q3 | y-stream range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, source_rows in sorted(by_source.items(), key=lambda kv: dataset_sort_key(source_name(kv[0]))):
        glc = {int(r["q"]): r for r in source_rows if r["run"] == "glc_real"}
        gp = {int(r["q"]): r for r in source_rows if r["run"] == "gp_rho116_real"}
        if not glc or not gp:
            continue
        deltas = []
        y_deltas = []
        for q in range(4):
            d = (float(gp[q]["bpp"]) / float(glc[q]["bpp"]) - 1.0) * 100.0
            dy = (float(gp[q]["bpp_y"]) / float(glc[q]["bpp_y"]) - 1.0) * 100.0
            deltas.append(d)
            y_deltas.append(dy)
        lines.append(
            "| {dataset} | {q0} | {q1} | {q2} | {q3} | {yrange} |".format(
                dataset=source_name(source),
                q0=pct(deltas[0]),
                q1=pct(deltas[1]),
                q2=pct(deltas[2]),
                q3=pct(deltas[3]),
                yrange=f"{min(y_deltas):+.2f}..{max(y_deltas):+.2f}%",
            )
        )
    return lines


def load_gate_summary(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return data["summary"]


def build_gate_table(root: Path) -> list[str]:
    lines = [
        "## Mechanism Table",
        "",
        "q3 gate-correlation analysis for the lead rho1.16 checkpoint. This analysis uses the existing validation reconstructions and is mechanism evidence rather than a bitrate source.",
        "",
        "| dataset | mean rho | rho std | corr base err | corr texture | corr gradient | high/low base err | high/low gradient |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, rel_path in GATE_JSONS:
        summary = load_gate_summary(root / rel_path)
        err_ratio = summary["highrho_base_err_mean"] / summary["lowrho_base_err_mean"]
        grad_ratio = summary["highrho_grad_mean"] / summary["lowrho_grad_mean"]
        lines.append(
            "| {label} | {rho:.4f} | {rho_std:.4f} | {cerr:.3f} | {ctex:.3f} | {cgrad:.3f} | {er:.2f}x | {gr:.2f}x |".format(
                label=label,
                rho=summary["rho_mean_mean"],
                rho_std=summary["rho_std_mean"],
                cerr=summary["corr_rho_base_err_mean"],
                ctex=summary["corr_rho_texture_var_mean"],
                cgrad=summary["corr_rho_grad_mean"],
                er=err_ratio,
                gr=grad_ratio,
            )
        )
    return lines


def build_official_comparison_table(root: Path) -> list[str]:
    comp_dir = root / DEFAULT_OFFICIAL_COMPARISON_DIR
    bd_path = comp_dir / "gp_reslc_real_vs_official_glc_bd.csv"
    matched_path = comp_dir / "gp_reslc_real_vs_official_glc_matched.csv"
    if not bd_path.exists() or not matched_path.exists():
        return []

    bd_rows = read_csv(bd_path)
    matched_rows = read_csv(matched_path)
    datasets = ["CLIC 2020", "DIV2K", "Kodak"]
    metrics = ["DISTS", "FID", "LPIPS"]
    bd = {(r["dataset"], r["metric"]): r for r in bd_rows}
    matched = {(r["dataset"], r["metric"]): r for r in matched_rows}
    notes = {
        "CLIC 2020": "Full 428-image protocol; local real GLC matches official curve closely, and GP-ResLC improves DISTS/FID.",
        "DIV2K": "Best cross-source sanity; official GLC and local real GLC nearly coincide.",
        "Kodak": "Official plot has no FID/KID; real-codec bpp is about 5% above graph bpp while quality matches.",
    }

    lines = [
        "## Official GLC Curve Comparison",
        "",
        "Secondary comparison against graph-extracted official GLC paper curves. Negative values mean GP-ResLC real codec needs fewer bits than the official graph-extracted GLC anchor. This is not the primary paper anchor; the main tables above use paired local real-codec GLC.",
        "",
        "| dataset | DISTS BD | FID BD | LPIPS BD | matched DISTS | matched FID | matched LPIPS | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in datasets:
        vals = []
        for metric in metrics:
            row = bd.get((dataset, metric))
            vals.append(pct(row["bd_rate_pct"]) if row else "n/a")
        for metric in metrics:
            row = matched.get((dataset, metric))
            vals.append(pct(row["mean_bpp_delta_pct"]) if row else "n/a")
        lines.append(f"| {dataset} | " + " | ".join(vals) + f" | {notes[dataset]} |")
    lines.extend([
        "",
        "Artifacts: `experiments/paper_assets/official_curve_comparison/`. FID/KID official figures are plotted with log2 y-scale; CSV files store the decoded numeric values.",
    ])
    return lines


def build_asset_manifest() -> list[str]:
    return [
        "## Figure and Artifact Manifest",
        "",
        "- VCIP real-codec package manifest: `experiments/paper_assets/vcip_real_codec_package.md`",
        "- Real codec protocol: `docs/real_codec_protocol.md`",
        "- CLIC2020 test real metrics: `experiments/real_codec/clic2020_test_real_metrics.csv`",
        "- CLIC2020 test real BD summary: `experiments/real_codec/clic2020_test_real_bd_rate_summary.md`",
        "- CLIC2020 test real matched summary: `experiments/real_codec/clic2020_test_real_matched_metric_summary.md`",
        "- DIV2K real metrics: `experiments/real_codec/div2k_real_metrics.csv`",
        "- DIV2K real BD summary: `experiments/real_codec/div2k_real_bd_rate_summary.md`",
        "- DIV2K real matched summary: `experiments/real_codec/div2k_real_matched_metric_summary.md`",
        "- Kodak real metrics: `experiments/real_codec/kodak_real_metrics.csv`",
        "- Kodak real BD summary: `experiments/real_codec/kodak_real_bd_rate_summary.md`",
        "- Kodak real matched summary: `experiments/real_codec/kodak_real_matched_metric_summary.md`",
        "- CLIC2020 test real curves: `experiments/paper_assets/clic2020_test_real_curves/`",
        "- DIV2K real curves: `experiments/paper_assets/div2k_real_curves/`",
        "- Kodak real curves: `experiments/paper_assets/kodak_real_curves/`",
        "- Official paper-curve comparison: `experiments/paper_assets/official_curve_comparison/`",
        "- CLIC2020 real payload/recon: `experiments/real_codec/clic2020_test_glc/`, `experiments/real_codec/clic2020_test_gp_reslc_rho116/`",
        "- DIV2K real payload/recon: `experiments/real_codec/div2k_glc/`, `experiments/real_codec/div2k_gp_reslc_rho116/`",
        "- Kodak real payload/recon: `experiments/real_codec/kodak_glc/`, `experiments/real_codec/kodak_gp_reslc_rho116/`",
        "- Rho overlay figures: `experiments/paper_assets/clic_q3_rho_overlay_top4.png`, `experiments/paper_assets/clic_mobile_q3_rho_overlay_top4.png`, `experiments/paper_assets/kodak_q3_rho_overlay_top4.png`",
        "- Lead checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`",
        "- Lead W&B run: `a2w5fjt4`",
    ]


def write_merged_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    preferred = list(rows[0].keys())
    extra = sorted(set().union(*(row.keys() for row in rows)).difference(preferred))
    fieldnames = preferred + extra
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--bd_csv", action="append", default=None, help="May be passed multiple times. Defaults to real-codec BD summaries.")
    ap.add_argument("--matched_csv", action="append", default=None, help="May be passed multiple times. Defaults to real-codec matched summaries.")
    ap.add_argument("--metrics_csv", action="append", default=None, help="May be passed multiple times. Defaults to real-codec metric CSVs.")
    ap.add_argument("--out", default="experiments/paper_assets/vcip_key_tables.md")
    ap.add_argument("--merged_bd_csv", default="experiments/paper_assets/real_codec_bd_rate_summary_all.csv")
    ap.add_argument("--merged_matched_csv", default="experiments/paper_assets/real_codec_matched_metric_bpp_summary_all.csv")
    ap.add_argument("--merged_metrics_csv", default="experiments/paper_assets/real_codec_metrics_all.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    bd_paths = args.bd_csv or DEFAULT_REAL_BD
    matched_paths = args.matched_csv or DEFAULT_REAL_MATCHED
    metrics_paths = args.metrics_csv or DEFAULT_REAL_METRICS

    bd_rows = read_csvs(root, bd_paths)
    matched_rows = read_csvs(root, matched_paths)
    metric_rows: list[dict[str, str]] = []
    for rel in metrics_paths:
        for row in read_csv(root / rel):
            row = dict(row)
            row["_source"] = rel
            metric_rows.append(row)

    lines = [
        "# VCIP Key Tables for GP-ResLC",
        "",
        "Generated from real-codec metric artifacts by `scripts/build_vcip_key_tables.py`.",
        "Paper-facing bitrate is serialized payload bpp, not estimated likelihood bpp.",
        "",
    ]
    for block in [
        build_bd_table(bd_rows),
        build_matched_table(matched_rows),
        build_real_bpp_table(metric_rows),
        build_official_comparison_table(root),
        build_gate_table(root),
        build_asset_manifest(),
    ]:
        lines.extend(block)
        lines.append("")

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")

    write_merged_csv(root / args.merged_bd_csv, bd_rows)
    write_merged_csv(root / args.merged_matched_csv, matched_rows)
    write_merged_csv(root / args.merged_metrics_csv, metric_rows)
    print(out_path)


if __name__ == "__main__":
    main()
