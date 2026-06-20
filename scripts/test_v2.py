# test_v2.py
# ============================================================================
#  VCIP Phase V2 — q 条件化（1モデル全レート）＋知覚ゲート の推論
#  V1 と同じ集計（64 パディング, bpp = bit/原画素）。
#  --interpolate で GLC の interpolate_q() を呼び、q_embed も 4→64 に補間して
#  連続レート（64 点）の R-P 曲線用 recon を書き出す。
#
#  実行例（離散 4 点）:
#    python test_v2.py --glc_weights /weights/GLC_image --ckpt ./ckpt_v2/v2_final.pt \
#      --input /data/kodak --out /out/ours_v2_kodak --q_indexes 0 1 2 3
#  実行例（連続 64 点）:
#    python test_v2.py --glc_weights /weights/GLC_image --ckpt ./ckpt_v2/v2_final.pt \
#      --input /data/kodak --out /out/ours_v2_kodak_dense --interpolate \
#      --q_indexes 0 8 16 24 32 40 48 56 63
# ============================================================================

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict, from_0_1_to_minus1_1, write_image
from gp_reslc.prior_predictor import PriorPredictor, train_forward
from gp_reslc.perceptual_gate import PerceptualGate

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
PAD = 64


def expand_qembed(qe):
    """[4,N,1,1] → [64,N,1,1]（GLC interpolate_q と同じ bilinear 補間）。"""
    N = qe.shape[1]
    w = F.interpolate(qe.view(1, 1, 4, N), size=(64, N), mode="bilinear", align_corners=True)
    return w.view(64, N, 1, 1)


def build_net(weights, ckpt_path, interpolate, device):
    net = GLC_Image(inplace=True)
    net.load_state_dict(get_state_dict(weights), strict=True)
    net.prior_predictor = PriorPredictor(net.N)

    ck = torch.load(ckpt_path, map_location="cpu")
    net.prior_predictor.load_state_dict(ck["prior_predictor"])
    q_embed = ck["q_embed"]                                   # [4,N,1,1]
    use_gate = ck.get("use_gate", False)
    if use_gate:
        net.perceptual_gate = PerceptualGate(net.N, rho_max=ck.get("rho_max", 2.0), rho_min=ck.get("rho_min", 0.5), rho_mode=ck.get("rho_mode", "hard"), softplus_shift=ck.get("gate_softplus_shift", 2.0), softplus_tau=ck.get("gate_softplus_tau", 1.0))
        net.perceptual_gate.load_state_dict(ck["perceptual_gate"])
    else:
        net.perceptual_gate = None

    if interpolate:
        net.interpolate_q()                                  # q_enc/q_dec 4→64
        q_embed = expand_qembed(q_embed)                     # q_embed も 4→64

    net.q_embed = q_embed.to(device)
    return net.to(device).eval(), use_gate


@torch.no_grad()
def run_one(net, x, q, use_gate, predictor_param_mode, predictor_delta_bound):
    _, _, H, W = x.shape
    pl, pr, pt, pb = GLC_Image.get_padding_size(H, W, PAD)
    xp = F.pad(x, (pl, pr, pt, pb), mode="replicate")
    out = train_forward(net, xp, q, use_predictor=True,
                        gate=net.perceptual_gate if use_gate else None,
                        q_shift=net.q_embed[q:q + 1],
                        predictor_param_mode=predictor_param_mode,
                        predictor_delta_bound=predictor_delta_bound)
    x_hat = F.pad(out["x_hat"], (-pl, -pr, -pt, -pb))
    bpp = (out["bit_y"].item() + out["bit_z"]) / (H * W)
    return x_hat, bpp, out["bit_y"].item() / (H * W), out["bit_z"] / (H * W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--ckpt", required=True, help="train_v2.py の v2_*.pt")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interpolate", action="store_true", help="64 点の連続レートで評価")
    ap.add_argument("--q_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--predictor_param_mode", choices=["mean", "scale_mean", "all", "latent_residual"], default="scale_mean")
    ap.add_argument("--predictor_delta_bound", type=float, default=0.0,
                    help="Bound predictor delta by bound*tanh(delta/bound); 0 disables")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, use_gate = build_net(args.glc_weights, args.ckpt, args.interpolate, device)
    qmax = net.q_embed.shape[0]
    paths = sorted(sum([glob.glob(os.path.join(args.input, e)) for e in EXTS], []))
    assert paths, f"画像が見つかりません: {args.input}"

    for q in args.q_indexes:
        assert 0 <= q < qmax, f"q={q} は範囲外（0..{qmax-1}, --interpolate で 0..63）"
        save_dir = os.path.join(args.out, f"q{q}")
        os.makedirs(save_dir, exist_ok=True)
        manifest = {"method": "ours_v2", "q": q, "interpolate": args.interpolate, "images": {}}
        bpps = []
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            x = from_0_1_to_minus1_1(ToTensor()(Image.open(p).convert("RGB"))).unsqueeze(0).to(device)
            x_hat, bpp, bpp_y, bpp_z = run_one(net, x, q, use_gate, args.predictor_param_mode, args.predictor_delta_bound)
            write_image(os.path.join(save_dir, f"{name}.png"), x_hat)
            manifest["images"][name] = {"bpp": bpp, "bpp_y": bpp_y, "bpp_z": bpp_z}
            bpps.append(bpp)
            print(f"[ours_v2 q={q}] {name}: bpp={bpp:.4f} (y={bpp_y:.4f}, z={bpp_z:.4f})")
        manifest["avg_bpp"] = sum(bpps) / len(bpps)
        manifest["avg_bpp_y"] = sum(m["bpp_y"] for m in manifest["images"].values()) / len(bpps)
        json.dump(manifest, open(os.path.join(save_dir, "bpp.json"), "w"), indent=2)
        print(f"== ours_v2 q={q}: avg_bpp={manifest['avg_bpp']:.4f} → {save_dir}")


if __name__ == "__main__":
    main()
