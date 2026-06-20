# train_v1.py
# ============================================================================
#  VCIP Phase V1 — GLC 凍結 + 生成事前予測器 P_θ のみ学習（wandb 対応）
#  目的: 「事前予測残差エントロピー」で意味コードのビットを増やさずに bit_y を下げる、
#        という主張①を A/B（baseline=GLC / ours=P_θ on）で最短検証する。
#
#  配置: GLC リポジトリ直下に prior_predictor.py と本ファイルを置く。
#  実行例:
#    python train_v1.py --glc_weights /weights/GLC_image \
#        --data /data/openimages_subset --val /data/kodak \
#        --q_index 2 --iters 20000 --bs 8 --out ./ckpt_v1 \
#        --wandb_project gp-reslc-vcip --wandb_name v1_q2
#
#  メモ:
#    - GLC は学習コード未公開のため、prior_predictor.train_forward が test() の学習版。
#    - 256x256 クロップ固定（code_pred_loss の position_emb=latent_size=256=16x16 に整合）。
#    - 損失 = λ_R·bpp_y + λ_d·MSE + λ_lpips·LPIPS + λ_align·CE(code_pred(μ_θ), idx_gt)
#      bit_z は z_vq 凍結で定数 → bit_y の減少がそのまま主張①の利得。
# ============================================================================

import argparse
import glob
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image

from src.models.image_model import GLC_Image
from src.models.loss import (
    get_lpips_model, LPIPSLoss,
    calculate_vqgan_results, cal_ce_Loss, cal_mse_Loss,
)
from src.utils.test_utils import get_state_dict, from_0_1_to_minus1_1, from_minus1_1_to_0_1
from gp_reslc.prior_predictor import PriorPredictor, train_forward

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


# ----------------------------------------------------------------------
class CropFolder(Dataset):
    """フォルダ内画像から 256x256 のランダムクロップを返す（[-1,1]）。"""
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

    def __init__(self, root, size=256):
        self.paths = sorted(sum([glob.glob(os.path.join(root, e)) for e in self.EXTS], []))
        assert self.paths, f"画像が見つかりません: {root}"
        self.size = size
        self.t = transforms.Compose([
            transforms.RandomCrop(size, pad_if_needed=True, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if min(img.size) < self.size:
            s = self.size / min(img.size)
            img = img.resize((max(self.size, int(img.size[0] * s) + 1),
                              max(self.size, int(img.size[1] * s) + 1)))
        return from_0_1_to_minus1_1(self.t(img))


# ----------------------------------------------------------------------
def build_net(weights, device, unfreeze_fusion=False):
    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(weights), strict=True)   # GLC 事前学習済み
    net.prior_predictor = PriorPredictor(net.N)                  # 新規（zero-init ゲート）
    net = net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.prior_predictor.parameters():
        p.requires_grad_(True)
    if unfreeze_fusion:  # 利得が出ないときのピボット用
        for p in net.y_prior_fusion.parameters():
            p.requires_grad_(True)
    return net


@torch.no_grad()
def quick_eval(net, x, q):
    """A/B: baseline(GLC, P_θ off) vs ours(on) の (bpp_y, PSNR) と再構成画像。"""
    net.eval()
    res, recon = {}, {}
    B, _, H, W = x.shape
    for tag, use in (("baseline", False), ("ours", True)):
        out = train_forward(net, x, q, use_predictor=use,
                            predictor_param_mode=getattr(net, "predictor_param_mode", "scale_mean"),
                            predictor_delta_bound=getattr(net, "predictor_delta_bound", 0.0))
        bpp_y = out["bit_y"].item() / (B * H * W)
        mse = torch.mean((x - out["x_hat"].clamp(-1, 1)) ** 2).item()
        psnr = 10 * math.log10(4.0 / max(mse, 1e-10))  # 範囲 [-1,1] → peak²=4
        res[tag] = (bpp_y, psnr)
        recon[tag] = out["x_hat"].clamp(-1, 1)
    net.train()
    return res, recon


def _img_panel(x, recon, n=4):
    """[原画 | baseline | ours] の比較グリッドを作る（wandb 用, [0,1]）。"""
    n = min(n, x.shape[0])
    rows = []
    for i in range(n):
        rows += [from_minus1_1_to_0_1(x[i]),
                 from_minus1_1_to_0_1(recon["baseline"][i]),
                 from_minus1_1_to_0_1(recon["ours"][i])]
    return make_grid(torch.stack(rows), nrow=3).clamp(0, 1)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--data", required=True, help="学習画像フォルダ（OpenImages サブセット等）")
    ap.add_argument("--val", default=None, help="A/B 用の小検証フォルダ（例: Kodak）")
    ap.add_argument("--out", default="./ckpt_v1")
    ap.add_argument("--q_index", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_R", type=float, default=1.0)
    ap.add_argument("--lambda_d", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_mean_pred", type=float, default=0.0,
                    help="Smooth-L1 between corrected common prior mean and y in quantized space")
    ap.add_argument("--lambda_scale_reg", type=float, default=0.0,
                    help="Penalize positive scale deltas to discourage rate reduction by scale inflation")
    ap.add_argument("--lambda_distill", type=float, default=0.0,
                    help="MSE distillation to frozen GLC baseline reconstruction")
    ap.add_argument("--unfreeze_fusion", action="store_true")
    ap.add_argument("--predictor_param_mode", choices=["mean", "scale_mean", "all", "latent_residual"], default="scale_mean")
    ap.add_argument("--predictor_delta_bound", type=float, default=0.0,
                    help="Bound predictor delta by bound*tanh(delta/bound); 0 disables")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=8)
    # wandb
    ap.add_argument("--wandb_project", type=str, default="gp-reslc-vcip")
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    ap.add_argument("--resume", type=str, default=None, help="train_state.pt から学習再開")
    ap.add_argument("--wandb_id", type=str, default=None, help="resume 時に同一 wandb run を継続")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   entity=args.wandb_entity, config=vars(args), mode=args.wandb_mode,
                   id=args.wandb_id, resume="allow" if args.resume else None)
    elif not args.no_wandb:
        print("[warn] wandb 未インストール。`pip install wandb` 推奨。ログ無しで継続。")

    net = build_net(args.glc_weights, device, args.unfreeze_fusion)
    net.predictor_param_mode = args.predictor_param_mode
    net.predictor_delta_bound = args.predictor_delta_bound
    net.train()

    lpips_loss = LPIPSLoss(get_lpips_model()).to(device).eval()
    for p in lpips_loss.parameters():
        p.requires_grad_(False)

    loader = DataLoader(CropFolder(args.data, 256), batch_size=args.bs, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    val = None
    if args.val:
        vds = CropFolder(args.val, 256)
        val = torch.stack([vds[i] for i in range(min(8, len(vds)))]).to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    n_param = sum(p.numel() for p in params) / 1e6
    print(f"学習対象パラメータ: {n_param:.2f} M")
    if use_wandb:
        wandb.summary["trainable_params_M"] = n_param

    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.prior_predictor.load_state_dict(ck["prior_predictor"])
        opt.load_state_dict(ck["optimizer"])
        start_it = ck.get("it", 0)
        print(f"[resume] {args.resume} から再開（it={start_it}）")

    it = start_it
    while it < args.iters:
        for x in loader:
            x = x.to(device)
            out = train_forward(net, x, args.q_index, use_predictor=True,
                                predictor_param_mode=args.predictor_param_mode,
                                predictor_delta_bound=args.predictor_delta_bound)
            B, _, H, W = x.shape

            bpp_y = out["bit_y"] / (B * H * W)
            bpp_z = torch.as_tensor(out["bit_z"], device=x.device, dtype=bpp_y.dtype) / (H * W)
            bpp_total = bpp_y + bpp_z
            d_mse = cal_mse_Loss(x, out["x_hat"]).mean()
            d_lp = lpips_loss(out["x_hat"], x).mean()
            psnr = 10 * math.log10(4.0 / max(d_mse.item(), 1e-10))
            delta_abs = out["delta_params"].detach().abs().mean().item() if out["delta_params"] is not None else 0.0
            mu_mean = out["mu_pred"].detach().mean().item() if out["mu_pred"] is not None else 0.0
            mu_std = out["mu_pred"].detach().std().item() if out["mu_pred"] is not None else 0.0
            zero = bpp_y.new_tensor(0.0)
            if args.lambda_align > 0:
                idx_gt = calculate_vqgan_results(x, net.vqgan)["idx_gt"]
                l_align = cal_ce_Loss(net.code_pred_loss(out["mu_pred"]), idx_gt).mean()
            else:
                l_align = zero

            if args.lambda_mean_pred > 0:
                with torch.no_grad():
                    base_q_enc, _, _, base_mean = net.separate_prior(out["params_base"])
                    target_y = out["y"].detach() * base_q_enc.detach()
                if args.predictor_param_mode == "latent_residual":
                    target_residual_pred = target_y - base_mean.detach()
                    l_mean_pred = F.smooth_l1_loss(out["latent_pred_scaled"], target_residual_pred)
                else:
                    _, _, _, corrected_mean = net.separate_prior(out["params_after"])
                    l_mean_pred = F.smooth_l1_loss(corrected_mean, target_y)
            else:
                l_mean_pred = zero

            if args.lambda_scale_reg > 0 and args.predictor_param_mode != "latent_residual":
                base_scale = out["params_base"][:, net.N:2 * net.N]
                after_scale = out["params_after"][:, net.N:2 * net.N]
                l_scale_reg = F.relu(after_scale - base_scale).mean()
            else:
                l_scale_reg = zero

            if args.lambda_distill > 0:
                with torch.no_grad():
                    base_out = train_forward(net, x, args.q_index, use_predictor=False,
                                             predictor_param_mode=args.predictor_param_mode,
                                predictor_delta_bound=args.predictor_delta_bound)
                    base_x_hat = base_out["x_hat"].detach()
                l_distill = cal_mse_Loss(out["x_hat"], base_x_hat).mean()
            else:
                l_distill = zero

            loss = (args.lambda_R * bpp_y + args.lambda_d * d_mse
                    + args.lambda_lpips * d_lp + args.lambda_align * l_align
                    + args.lambda_mean_pred * l_mean_pred
                    + args.lambda_scale_reg * l_scale_reg
                    + args.lambda_distill * l_distill)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if it % args.log_every == 0:
                print(f"[it {it}] loss={loss.item():.4f} bpp={bpp_total.item():.4f} "
                      f"bpp_y={bpp_y.item():.4f} psnr={psnr:.2f} "
                      f"mse={d_mse.item():.4f} lpips={d_lp.item():.4f} ce={l_align.item():.4f} "
                      f"mean_pred={l_mean_pred.item():.4f} scale_reg={l_scale_reg.item():.4f} "
                      f"distill={l_distill.item():.4f} delta_abs={delta_abs:.5f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/bpp_y": bpp_y.item(),
                               "train/bpp_z": bpp_z.item(), "train/bpp_total": bpp_total.item(),
                               "train/psnr": psnr, "train/mse": d_mse.item(), "train/lpips": d_lp.item(),
                               "train/ce_align": l_align.item(), "train/mean_pred": l_mean_pred.item(),
                               "train/scale_reg": l_scale_reg.item(), "train/distill": l_distill.item(),
                               "pred/delta_abs_mean": delta_abs,
                               "pred/mu_mean": mu_mean, "pred/mu_std": mu_std,
                               "train/lr": args.lr}, step=it)

            if val is not None and it % args.eval_every == 0:
                ab, recon = quick_eval(net, val, args.q_index)
                d_bpp = ab["ours"][0] - ab["baseline"][0]
                print(f"  [A/B it {it}] baseline bpp_y={ab['baseline'][0]:.4f} psnr={ab['baseline'][1]:.2f} "
                      f"| ours bpp_y={ab['ours'][0]:.4f} psnr={ab['ours'][1]:.2f} | Δbpp_y={d_bpp:+.4f}")
                torch.save(net.prior_predictor.state_dict(),
                           os.path.join(args.out, f"prior_predictor_{it}.pt"))
                torch.save({"it": it, "prior_predictor": net.prior_predictor.state_dict(),
                            "optimizer": opt.state_dict()},
                           os.path.join(args.out, "train_state.pt"))   # resume 用（上書き）
                if use_wandb:
                    wandb.log({
                        "ab/baseline_bpp_y": ab["baseline"][0], "ab/baseline_psnr": ab["baseline"][1],
                        "ab/ours_bpp_y": ab["ours"][0], "ab/ours_psnr": ab["ours"][1],
                        "ab/delta_bpp_y": d_bpp,
                        "ab/samples": wandb.Image(_img_panel(val, recon),
                                                  caption="rows: [orig | baseline(GLC) | ours]"),
                    }, step=it)

            it += 1
            if it >= args.iters:
                break

    torch.save(net.prior_predictor.state_dict(),
               os.path.join(args.out, "prior_predictor_final.pt"))
    torch.save({"it": it, "prior_predictor": net.prior_predictor.state_dict(),
                "optimizer": opt.state_dict()},
               os.path.join(args.out, "train_state.pt"))
    print("done. → 同 PSNR/DISTS 帯で baseline vs ours の bpp_y を比較し、主張①を確認。")
    if use_wandb:
        wandb.save(os.path.join(args.out, "prior_predictor_final.pt"))
        wandb.finish()


if __name__ == "__main__":
    main()
