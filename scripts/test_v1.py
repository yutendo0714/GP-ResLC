# test_v1.py
# ============================================================================
#  VCIP Phase V1 — フル解像度推論（再構成 PNG + bpp.json を書き出し）
#  GLC の test_image.py と同じ集計（64 パディング, bpp = bit / 原画素）で、
#  prior_predictor を注入した推論を行う。baseline(GLC) / ours(P_θ) を切替。
#  出力フォルダは make_curves.py / eval_metrics.py の入力になる。
#
#  実行例（ours, 全 q を Kodak で）:
#    python test_v1.py --glc_weights /weights/GLC_image \
#        --ckpt ./ckpt_v1/prior_predictor_final.pt \
#        --input /data/kodak --out /out/ours_kodak --method ours --q_indexes 0 1 2 3
#  baseline（GLC そのもの）:
#    python test_v1.py --glc_weights /weights/GLC_image \
#        --input /data/kodak --out /out/baseline_kodak --method baseline --q_indexes 0 1 2 3
# ============================================================================

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict, from_0_1_to_minus1_1, write_image
from gp_reslc.prior_predictor import PriorPredictor, train_forward

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
PAD = 64  # GLC と同じ


def build_net(weights, ckpt, device):
    net = GLC_Image(inplace=True)
    net.load_state_dict(get_state_dict(weights), strict=True)
    net.prior_predictor = PriorPredictor(net.N)
    if ckpt:  # ours のとき学習済み P_θ をロード
        net.prior_predictor.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return net.to(device).eval()


@torch.no_grad()
def run_one(net, x, q, use_predictor, predictor_param_mode, predictor_delta_bound):
    _, _, H, W = x.shape
    pl, pr, pt, pb = GLC_Image.get_padding_size(H, W, PAD)
    xp = F.pad(x, (pl, pr, pt, pb), mode="replicate")
    out = train_forward(net, xp, q, use_predictor=use_predictor,
                        predictor_param_mode=predictor_param_mode,
                        predictor_delta_bound=predictor_delta_bound)
    x_hat = F.pad(out["x_hat"], (-pl, -pr, -pt, -pb))          # 元サイズに戻す
    bpp = (out["bit_y"].item() + out["bit_z"]) / (H * W)        # GLC と同じ集計
    bpp_y = out["bit_y"].item() / (H * W)
    bpp_z = out["bit_z"] / (H * W)
    return x_hat, bpp, bpp_y, bpp_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--ckpt", default=None, help="ours のとき prior_predictor の学習済み重み")
    ap.add_argument("--input", required=True, help="評価画像フォルダ（原画）")
    ap.add_argument("--out", required=True, help="出力ルート（method/q ごとにサブフォルダ）")
    ap.add_argument("--method", choices=["baseline", "ours"], default="ours")
    ap.add_argument("--q_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--predictor_param_mode", choices=["mean", "scale_mean", "all", "latent_residual"], default="scale_mean")
    ap.add_argument("--predictor_delta_bound", type=float, default=0.0,
                    help="Bound predictor delta by bound*tanh(delta/bound); 0 disables")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_predictor = (args.method == "ours")
    if use_predictor and not args.ckpt:
        print("[warn] --method ours だが --ckpt 未指定 → 未学習 P_θ（zero-init=GLC 同等）で推論。")

    net = build_net(args.glc_weights, args.ckpt if use_predictor else None, device)
    paths = sorted(sum([glob.glob(os.path.join(args.input, e)) for e in EXTS], []))
    assert paths, f"画像が見つかりません: {args.input}"

    for q in args.q_indexes:
        save_dir = os.path.join(args.out, f"q{q}")
        os.makedirs(save_dir, exist_ok=True)
        manifest = {"method": args.method, "q": q, "images": {}}
        bpps = []
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            x = from_0_1_to_minus1_1(ToTensor()(Image.open(p).convert("RGB"))).unsqueeze(0).to(device)
            x_hat, bpp, bpp_y, bpp_z = run_one(net, x, q, use_predictor, args.predictor_param_mode, args.predictor_delta_bound)
            write_image(os.path.join(save_dir, f"{name}.png"), x_hat)
            manifest["images"][name] = {"bpp": bpp, "bpp_y": bpp_y, "bpp_z": bpp_z}
            bpps.append(bpp)
            print(f"[{args.method} q={q}] {name}: bpp={bpp:.4f} (y={bpp_y:.4f}, z={bpp_z:.4f})")
        manifest["avg_bpp"] = sum(bpps) / len(bpps)
        manifest["avg_bpp_y"] = sum(m["bpp_y"] for m in manifest["images"].values()) / len(bpps)
        json.dump(manifest, open(os.path.join(save_dir, "bpp.json"), "w"), indent=2)
        print(f"== {args.method} q={q}: avg_bpp={manifest['avg_bpp']:.4f} "
              f"(y={manifest['avg_bpp_y']:.4f}) → {save_dir}")


if __name__ == "__main__":
    main()
