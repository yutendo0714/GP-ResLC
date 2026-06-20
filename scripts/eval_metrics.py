# eval_metrics.py
# 全ベースライン横断で使える統一評価スクリプト雛形。
# GLC は src/utils/metric_image.evaluate_quality を同梱しているが、
# 「他手法も同じ土俵で測る」ために、原画フォルダと再構成フォルダ＋bppリストから
# bpp / PSNR / MS-SSIM / LPIPS / DISTS / FID / KID / BD-rate を一括算出する。
#
# 依存（GLC requirements に含まれる）:
#   pip install lpips DISTS_pytorch pytorch-msssim torchmetrics torch_fidelity numpy scipy pillow
#
# 使い方（1品質点）:
#   python eval_metrics.py --orig /data/kodak --recon /out/glc/q2 --patch 64
# 使い方（BD-rate, 複数品質点を JSON で）:
#   curves.json = {"GLC":[[bpp,DISTS],...], "Ours":[[bpp,DISTS],...]}  # DISTS は小さいほど良い
#   python eval_metrics.py --bd curves.json --anchor GLC --metric DISTS

import argparse
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import lpips as lpips_lib
from DISTS_pytorch import DISTS
from pytorch_msssim import ms_ssim
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance
from src.utils._update_patch_fid import update_patch_fid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOWER_BETTER = {"LPIPS", "DISTS", "FID", "KID"}  # BD-rate の符号処理に使用


# ----------------------------------------------------------------------
# 画像入出力
# ----------------------------------------------------------------------
def _load(path):
    return ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)


def _paired_paths(orig_dir, recon_dir):
    """basename（拡張子無視）で原画と再構成を対応付ける。"""
    recon = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(recon_dir, "*"))}
    pairs = []
    for op in sorted(glob.glob(os.path.join(orig_dir, "*"))):
        key = os.path.splitext(os.path.basename(op))[0]
        if key in recon:
            pairs.append((op, recon[key]))
    if not pairs:
        raise RuntimeError("対応する画像が見つかりません。フォルダとファイル名を確認してください。")
    return pairs


# ----------------------------------------------------------------------
# フルリファレンス指標（PSNR / MS-SSIM / LPIPS / DISTS）
# ----------------------------------------------------------------------
@torch.no_grad()
def full_reference(orig_dir, recon_dir):
    lpips_fn = lpips_lib.LPIPS(net="alex").to(DEVICE).eval()
    dists_fn = DISTS().to(DEVICE).eval()
    psnr_l, msssim_l, lpips_l, dists_l = [], [], [], []

    for op, rp in _paired_paths(orig_dir, recon_dir):
        x = _load(op).to(DEVICE)
        y = _load(rp).to(DEVICE)
        # サイズ不一致（パディング等）を中央クロップで吸収
        h = min(x.shape[2], y.shape[2]); w = min(x.shape[3], y.shape[3])
        x, y = x[..., :h, :w], y[..., :h, :w]

        mse = torch.mean((x - y) ** 2).item()
        psnr_l.append(10.0 * np.log10(1.0 / max(mse, 1e-10)))
        msssim_l.append(ms_ssim(x, y, data_range=1.0).item())
        lpips_l.append(lpips_fn(x * 2 - 1, y * 2 - 1).item())   # LPIPS は [-1,1] 入力
        dists_l.append(dists_fn(x, y).item())                    # DISTS は [0,1] 入力

    return {
        "PSNR": float(np.mean(psnr_l)),
        "MS-SSIM": float(np.mean(msssim_l)),
        "LPIPS": float(np.mean(lpips_l)),
        "DISTS": float(np.mean(dists_l)),
    }


# ----------------------------------------------------------------------
# 分布指標（FID / KID）— 低解像度データはパッチ化して標本数を確保
# ----------------------------------------------------------------------
@torch.no_grad()
def distribution_metrics(
    orig_dir,
    recon_dir,
    patch=256,
    split_patch_num=2,
    kid_subset_size=None,
    return_patch_count=False,
):
    fid = FrechetInceptionDistance(normalize=False).to(DEVICE)
    if kid_subset_size is None:
        kid = KernelInceptionDistance(normalize=False).to(DEVICE)
    else:
        kid = KernelInceptionDistance(subset_size=kid_subset_size, normalize=False).to(DEVICE)
    patch_count = 0
    for op, rp in _paired_paths(orig_dir, recon_dir):
        xo = _load(op).to(DEVICE)
        xr = _load(rp).to(DEVICE)
        patch_count += int(update_patch_fid(
            xo,
            xr,
            fid_metric=fid,
            kid_metric=kid,
            patch_size=patch,
            split_patch_num=split_patch_num,
        ))
    kid_mean, _ = kid.compute()
    out = {"FID": float(fid.compute().item()), "KID": float(kid_mean.item())}
    if return_patch_count:
        out["FID_PATCHES"] = patch_count
        out["KID_PATCHES"] = patch_count
        out["FID_PATCH_SIZE"] = int(patch)
        out["FID_SPLIT_PATCH_NUM"] = int(split_patch_num)
    return out


# ----------------------------------------------------------------------
# BD-rate（Bjøntegaard）: 同一品質帯での平均ビット削減率[%]
#   metric が「大きいほど良い」(PSNR, MS-SSIM) はそのまま、
#   「小さいほど良い」(LPIPS/DISTS/FID/KID) は符号反転して扱う。
#   返り値が負 = test がアンカーよりビット削減（良い）。
# ----------------------------------------------------------------------
def bd_rate(rate_a, metric_a, rate_t, metric_t, metric_name):
    rate_a, metric_a = np.asarray(rate_a, float), np.asarray(metric_a, float)
    rate_t, metric_t = np.asarray(rate_t, float), np.asarray(metric_t, float)
    if metric_name in LOWER_BETTER:
        metric_a, metric_t = -metric_a, -metric_t
    lr_a, lr_t = np.log(rate_a), np.log(rate_t)

    # log-rate を品質の3次多項式で回帰し、共通品質帯で積分
    pa = np.polyfit(metric_a, lr_a, 3)
    pt = np.polyfit(metric_t, lr_t, 3)
    lo = max(metric_a.min(), metric_t.min())
    hi = min(metric_a.max(), metric_t.max())
    if hi <= lo:
        return float("nan")  # 品質帯が重ならない → BD-rate 定義不可
    Pa = np.polyint(pa); Pt = np.polyint(pt)
    int_a = np.polyval(Pa, hi) - np.polyval(Pa, lo)
    int_t = np.polyval(Pt, hi) - np.polyval(Pt, lo)
    avg_diff = (int_t - int_a) / (hi - lo)
    return float((np.exp(avg_diff) - 1.0) * 100.0)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", type=str, help="原画フォルダ")
    ap.add_argument("--recon", type=str, help="再構成フォルダ（1品質点）")
    ap.add_argument("--patch", type=int, default=256, help="FID/KID パッチサイズ（CLIC/DIV2K は 256）")
    ap.add_argument("--split_patch_num", type=int, default=2, help="2なら公式GLCと同じ half-patch shift を追加")
    ap.add_argument("--kid_subset_size", type=int, default=None, help="未指定ならtorchmetrics既定値。公式GLCも未指定")
    ap.add_argument("--bd", type=str, help="curves.json で BD-rate モード")
    ap.add_argument("--anchor", type=str, default=None)
    ap.add_argument("--metric", type=str, default="DISTS")
    args = ap.parse_args()

    if args.bd:
        import json
        curves = json.load(open(args.bd))  # {name: [[bpp, metric], ...]}
        anchor = args.anchor or list(curves)[0]
        ra = [p[0] for p in curves[anchor]]; ma = [p[1] for p in curves[anchor]]
        print(f"BD-rate（{args.metric}, anchor={anchor}）")
        for name, pts in curves.items():
            if name == anchor:
                continue
            rt = [p[0] for p in pts]; mt = [p[1] for p in pts]
            print(f"  {name:16s}: {bd_rate(ra, ma, rt, mt, args.metric):+7.2f} %")
        return

    res = {}
    res.update(full_reference(args.orig, args.recon))
    res.update(distribution_metrics(
        args.orig,
        args.recon,
        patch=args.patch,
        split_patch_num=args.split_patch_num,
        kid_subset_size=args.kid_subset_size,
    ))
    for k, v in res.items():
        print(f"{k:8s}: {v:.4f}")


if __name__ == "__main__":
    main()
