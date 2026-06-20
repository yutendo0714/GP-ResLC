# make_comparison.py
# ============================================================================
#  (d) 比較オーケストレータ — データセット × 手法 × 指標 の比較を自動生成
#  各手法は「root/q*/（再構成 PNG）＋ root/q*/bpp.json（平均 bpp）」を持つ前提。
#  baseline(GLC) / ours(V1, test_v1.py) / ours_v2(test_v2.py) / 外部手法 を束ねる。
#
#  出力（out/ 以下, データセットごと）:
#    - results.json                : 全曲線点（bpp と全指標）
#    - curve_<dataset>_<metric>.png: 全手法重ね描き（DISTS/LPIPS/FID/PSNR）
#    - bd_rate_<dataset>.md        : markdown BD-rate 表（論文貼り付け用）
#    - summary.md                  : 全データセットの BD-rate を 1 枚に集約
#
#  config の例（compare_config.example.json 参照）:
#  {
#    "out": "/out/comparison",
#    "anchor": "baseline",
#    "datasets": { "Kodak": {"orig": "/data/kodak", "patch": 64},
#                  "CLIC2020": {"orig": "/data/clic2020", "patch": 256} },
#    "methods": {
#      "baseline":  {"Kodak": "/out/baseline_kodak",  "CLIC2020": "/out/baseline_clic"},
#      "ours":      {"Kodak": "/out/ours_kodak",      "CLIC2020": "/out/ours_clic"},
#      "ours_v2":   {"Kodak": "/out/ours_v2_kodak",   "CLIC2020": "/out/ours_v2_clic"},
#      "MS-ILLM":   {"Kodak": "/out/msillm_kodak"}
#    }
#  }
#  実行: python make_comparison.py --config compare_config.example.json
# ============================================================================

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_metrics import full_reference, distribution_metrics, bd_rate

FR_METRICS = ["PSNR", "MS-SSIM", "LPIPS", "DISTS"]
DIST_METRICS = ["FID", "KID"]
ALL = FR_METRICS + DIST_METRICS
PLOT = ["DISTS", "LPIPS", "FID", "PSNR"]


def eval_method_on_dataset(orig, method_root, patch):
    """method_root/q*/ を走査し (bpp, {metrics}) のリスト（bpp 昇順）を返す。"""
    pts = []
    for qdir in sorted(glob.glob(os.path.join(method_root, "q*"))):
        man_path = os.path.join(qdir, "bpp.json")
        if not os.path.exists(man_path):
            continue
        man = json.load(open(man_path))
        m = {"bpp": man["avg_bpp"], "bpp_y": man.get("avg_bpp_y", man["avg_bpp"])}
        m.update(full_reference(orig, qdir))
        m.update(distribution_metrics(orig, qdir, patch=patch))
        pts.append(m)
    pts.sort(key=lambda d: d["bpp"])
    return pts


def compare(cfg):
    out = cfg["out"]
    anchor = cfg.get("anchor", "baseline")
    os.makedirs(out, exist_ok=True)
    results = {}        # results[dataset][method] = [points]
    summary_rows = []   # (dataset, method, {metric: bd})

    for ds, dmeta in cfg["datasets"].items():
        orig = dmeta["orig"]
        patch = dmeta.get("patch", 256)
        results[ds] = {}
        print(f"\n########## dataset: {ds} ##########")
        for name, roots in cfg["methods"].items():
            if ds not in roots:
                continue
            print(f"=== {name} @ {ds} ===")
            results[ds][name] = eval_method_on_dataset(orig, roots[ds], patch)
            for p in results[ds][name]:
                print(f"  bpp={p['bpp']:.4f} " + " ".join(f"{k}={p[k]:.4f}" for k in ALL))

        # 曲線（手法重ね描き）
        for metric in PLOT:
            plt.figure(figsize=(5, 4))
            for name, pts in results[ds].items():
                if not pts:
                    continue
                plt.plot([p["bpp"] for p in pts], [p[metric] for p in pts],
                         marker="o", label=name)
            plt.xlabel("bpp"); plt.ylabel(metric)
            plt.title(f"{ds}: {metric} vs bitrate")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out, f"curve_{ds}_{metric}.png"), dpi=200)
            plt.close()

        # BD-rate 表（markdown）
        if anchor in results[ds] and results[ds][anchor]:
            ap = results[ds][anchor]
            ra = [p["bpp"] for p in ap]
            md = [f"### BD-rate (%) — {ds}  (anchor={anchor}, 負=ビット削減=良)\n",
                  "| method | " + " | ".join(ALL) + " |",
                  "|" + "---|" * (len(ALL) + 1)]
            for name, pts in results[ds].items():
                if name == anchor or not pts:
                    continue
                rt = [p["bpp"] for p in pts]
                cells, row = [], {}
                for metric in ALL:
                    try:
                        bd = bd_rate(ra, [p[metric] for p in ap], rt, [p[metric] for p in pts], metric)
                        cells.append(f"{bd:+.2f}"); row[metric] = bd
                    except Exception:
                        cells.append("n/a"); row[metric] = None
                md.append(f"| {name} | " + " | ".join(cells) + " |")
                summary_rows.append((ds, name, row))
            open(os.path.join(out, f"bd_rate_{ds}.md"), "w").write("\n".join(md) + "\n")
            print("\n".join(md))

    # 全データセット集約
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)
    sm = ["# BD-rate summary (anchor=%s)\n" % anchor,
          "| dataset | method | " + " | ".join(ALL) + " |",
          "|" + "---|" * (len(ALL) + 2)]
    for ds, name, row in summary_rows:
        cells = [f"{row[m]:+.2f}" if row.get(m) is not None else "n/a" for m in ALL]
        sm.append(f"| {ds} | {name} | " + " | ".join(cells) + " |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(sm) + "\n")
    print(f"\n→ {out}/ に results.json / curve_*.png / bd_rate_*.md / summary.md を出力。")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    compare(json.load(open(args.config)))


if __name__ == "__main__":
    main()
