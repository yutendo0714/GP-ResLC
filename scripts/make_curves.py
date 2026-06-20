# make_curves.py
# ============================================================================
#  VCIP — R-P / R-D 曲線・BD-rate 表・図の自動生成（論文用）
#  test_v1.py が書き出した recon フォルダ群（各 q に bpp.json）を読み、
#  eval_metrics.py で全指標を計算 → results.json / 曲線 PNG / BD-rate 表を出力。
#  外部ベースライン（MS-ILLM 等）も「recon フォルダ + bpp.json」を用意すれば追加可能。
#
#  config.json の例:
#  {
#    "orig": "/data/kodak",
#    "out":  "/out/curves_kodak",
#    "anchor": "baseline",
#    "patch": 64,
#    "methods": {
#      "baseline": "/out/baseline_kodak",
#      "ours":     "/out/ours_kodak",
#      "MS-ILLM":  "/out/msillm_kodak"     # 任意（各 q サブフォルダ + bpp.json）
#    }
#  }
#  実行: python make_curves.py --config config.json
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
PLOT = ["DISTS", "LPIPS", "FID", "PSNR"]  # 論文の主要 R-P / R-D 図


def eval_method(orig, method_root, patch):
    """method_root/q*/ を走査し、各 q の (avg_bpp, {metrics}) を返す（bpp 昇順）。"""
    points = []
    for qdir in sorted(glob.glob(os.path.join(method_root, "q*"))):
        man_path = os.path.join(qdir, "bpp.json")
        if not os.path.exists(man_path):
            print(f"[skip] bpp.json なし: {qdir}")
            continue
        man = json.load(open(man_path))
        m = {}
        m.update(full_reference(orig, qdir))
        m.update(distribution_metrics(orig, qdir, patch=patch))
        m["bpp"] = man["avg_bpp"]
        m["bpp_y"] = man.get("avg_bpp_y", man["avg_bpp"])
        points.append(m)
        print(f"  {os.path.basename(qdir)}: bpp={m['bpp']:.4f} "
              + " ".join(f"{k}={m[k]:.4f}" for k in ALL))
    points.sort(key=lambda d: d["bpp"])
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    orig = cfg["orig"]
    out = cfg["out"]
    anchor = cfg.get("anchor", "baseline")
    patch = cfg.get("patch", 256)
    os.makedirs(out, exist_ok=True)

    # 1) 全手法の曲線点を計算
    results = {}
    for name, root in cfg["methods"].items():
        print(f"=== {name} ===")
        results[name] = eval_method(orig, root, patch)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)

    # 2) 曲線 PNG（bpp vs 各指標）
    for metric in PLOT:
        plt.figure(figsize=(5, 4))
        for name, pts in results.items():
            xs = [p["bpp"] for p in pts]
            ys = [p[metric] for p in pts]
            plt.plot(xs, ys, marker="o", label=name)
        plt.xlabel("bpp")
        plt.ylabel(metric)
        plt.title(f"{metric} vs bitrate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"curve_{metric}.png"), dpi=200)
        plt.close()

    # 3) BD-rate 表（anchor 比, 各指標）
    lines = [f"BD-rate (%) vs anchor={anchor}  (負 = ビット削減=良)"]
    header = "method".ljust(16) + "".join(m.rjust(10) for m in ALL)
    lines.append(header)
    ap_pts = results[anchor]
    ra = [p["bpp"] for p in ap_pts]
    for name, pts in results.items():
        if name == anchor:
            continue
        rt = [p["bpp"] for p in pts]
        row = name.ljust(16)
        for metric in ALL:
            ma = [p[metric] for p in ap_pts]
            mt = [p[metric] for p in pts]
            try:
                bd = bd_rate(ra, ma, rt, mt, metric)
                row += f"{bd:+10.2f}"
            except Exception:
                row += "       n/a"
        lines.append(row)
    table = "\n".join(lines)
    print("\n" + table)
    open(os.path.join(out, "bd_rate.txt"), "w").write(table + "\n")
    print(f"\n→ {out}/ に results.json / curve_*.png / bd_rate.txt を出力。")


if __name__ == "__main__":
    main()
