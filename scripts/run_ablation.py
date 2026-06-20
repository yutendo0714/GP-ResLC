# run_ablation.py
# ============================================================================
#  (e) 統合アブレーション — V2 の効果（q 条件化・知覚ゲート）を一括検証
#  既存チェックポイントから各条件の推論（test_v1.py / test_v2.py を subprocess）→
#  make_comparison.compare() で ablation 表（markdown）と曲線を自動生成する。
#
#  検証する条件（kind）:
#    baseline : GLC そのもの            （test_v1.py --method baseline）
#    v1       : P_θ（主張①）            （test_v1.py --method ours --ckpt）
#    v2       : q 条件化(±gate, ckpt 依存) （test_v2.py --ckpt [--interpolate]）
#  → V1 vs V2(q-cond) で「1モデル化の代償の有無」、
#    V2(no_gate) vs V2(gate) で「知覚ゲート②の効果（主張②）」を切り分ける。
#
#  事前に学習しておくチェックポイント（学習は重いので別途・並行で）:
#    python train_v1.py ... --out ./ckpt_v1                       # v1
#    python train_v2.py ... --no_gate --out ./ckpt_v2_nogate      # v2 q-cond only
#    python train_v2.py ... --out ./ckpt_v2_gate                  # v2 q-cond + gate
#
#  実行: python run_ablation.py --config ablation_config.example.json
#        既に推論済みなら: --skip_inference（比較のみ再生成）
# ============================================================================

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from make_comparison import compare

SCRIPT_DIR = Path(__file__).resolve().parent


def _slug(s):
    return re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_")


def run_inference(cfg):
    glc = cfg["glc_weights"]
    orig = cfg["dataset"]["orig"]
    qs = [str(q) for q in cfg["q_indexes"]]
    out_root = cfg["out"]
    recon_roots = {}  # variant_name -> recon root

    for v in cfg["variants"]:
        name = v["name"]
        rec = os.path.join(out_root, _slug(name))
        recon_roots[name] = rec
        kind = v["kind"]

        if kind == "baseline":
            cmd = [sys.executable, str(SCRIPT_DIR / "test_v1.py"), "--glc_weights", glc,
                   "--input", orig, "--out", rec, "--method", "baseline",
                   "--q_indexes", *qs]
        elif kind == "v1":
            cmd = [sys.executable, str(SCRIPT_DIR / "test_v1.py"), "--glc_weights", glc,
                   "--ckpt", v["ckpt"], "--input", orig, "--out", rec,
                   "--method", "ours", "--q_indexes", *qs]
        elif kind == "v2":
            cmd = [sys.executable, str(SCRIPT_DIR / "test_v2.py"), "--glc_weights", glc,
                   "--ckpt", v["ckpt"], "--input", orig, "--out", rec,
                   "--q_indexes", *qs]
            if v.get("interpolate"):
                cmd.append("--interpolate")
        else:
            raise ValueError(f"未知の kind: {kind}")

        print(f"\n>>> [{name}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    return recon_roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_inference", action="store_true",
                    help="推論をスキップし、既存 recon で比較のみ再生成")
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    out_root = cfg["out"]
    if args.skip_inference:
        recon_roots = {v["name"]: os.path.join(out_root, _slug(v["name"]))
                       for v in cfg["variants"]}
    else:
        recon_roots = run_inference(cfg)

    # make_comparison 用の config を組み立てて比較
    ds_name = cfg["dataset"]["name"]
    comp_cfg = {
        "out": os.path.join(out_root, "comparison"),
        "anchor": cfg.get("anchor", "baseline"),
        "datasets": {ds_name: {"orig": cfg["dataset"]["orig"],
                               "patch": cfg["dataset"].get("patch", 256)}},
        "methods": {name: {ds_name: root} for name, root in recon_roots.items()},
    }
    json.dump(comp_cfg, open(os.path.join(out_root, "comparison_config.json"), "w"), indent=2)
    compare(comp_cfg)
    print(f"\n=== ablation 完了 → {out_root}/comparison/ に bd_rate_{ds_name}.md / curve_*.png ===")
    print("読み方: V1 vs V2(q-cond) = 1モデル化の代償, V2(no_gate) vs V2(gate) = 知覚ゲート②の効果。")


if __name__ == "__main__":
    main()
