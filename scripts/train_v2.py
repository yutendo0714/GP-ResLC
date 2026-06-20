# train_v2.py
# ============================================================================
#  VCIP Phase V2 — q 条件化（1モデルで全レート）＋ 知覚重要度ゲート② を統合
#  - (b) 可変レート: GLC は元々 gained-VAE で可変レート（q_enc/q_dec 4本＋interpolate_q→64本）。
#        本スクリプトは P_θ / gate を q 条件付き（学習 q_embed を z_hat にシフト加算）にし、
#        毎バッチ q を {0,1,2,3} からサンプルして 1 個の P_θ で全レートをカバーする。
#  - (a) 知覚ゲート: perceptual_gate.PerceptualGate を train_forward に統合（quant_step 変調）。
#
#  GLC を凍結し、{prior_predictor, perceptual_gate, q_embed} のみ学習。
#  実行例:
#    python train_v2.py --glc_weights /weights/GLC_image \
#      --data /data/oi_subset --val /data/kodak --iters 40000 --bs 8 \
#      --out ./ckpt_v2 --wandb_project gp-reslc-vcip --wandb_name v2_qcond_gate
#  知覚ゲートを外して「q 条件化のみ（純粋 (b)）」を検証: --no_gate
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

import random

import torch
from torch import nn
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
from gp_reslc.perceptual_gate import PerceptualGate
from DISTS_pytorch import DISTS

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

try:
    import lpips as lpips_lib
    _LPIPS_LIB = True
except Exception:
    lpips_lib = None
    _LPIPS_LIB = False

NUM_Q = 4  # GLC のレート点数（interpolate_q 前）


class CropFolder(Dataset):
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


def build_net(weights, device, use_gate, rho_max, rho_min=0.5, train_predictor=True,
              rho_mode="hard", softplus_shift=2.0, softplus_tau=1.0, rho_init=1.0):
    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(weights), strict=True)
    net.prior_predictor = PriorPredictor(net.N)
    net.q_embed = nn.Parameter(torch.zeros(NUM_Q, net.N, 1, 1))      # q 条件（zero-init=V1 等価）
    net.perceptual_gate = PerceptualGate(
        net.N, rho_max=rho_max, rho_min=rho_min, rho_mode=rho_mode,
        softplus_shift=softplus_shift, softplus_tau=softplus_tau, rho_init=rho_init,
    ) if use_gate else None
    net = net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.prior_predictor.parameters():
        p.requires_grad_(train_predictor)
    net.q_embed.requires_grad_(True)
    if use_gate:
        for p in net.perceptual_gate.parameters():
            p.requires_grad_(True)
    return net


@torch.no_grad()
def quick_eval(net, x, qs=(0, 1, 2, 3)):
    """各 q で baseline(GLC) vs ours(P_θ+gate+q_embed) の (bpp_y, PSNR)。"""
    net.eval()
    out = {}
    B, _, H, W = x.shape
    for q in qs:
        res = {}
        for tag, use, g, sh in (("baseline", False, None, None),
                                ("ours", True, net.perceptual_gate, net.q_embed[q:q + 1])):
            o = train_forward(net, x, q, use_predictor=use, gate=g, q_shift=sh,
                              predictor_param_mode=getattr(net, "predictor_param_mode", "scale_mean"),
                            predictor_delta_bound=getattr(net, "predictor_delta_bound", 0.0))
            bpp_y = o["bit_y"].item() / (B * H * W)
            mse = torch.mean((x - o["x_hat"].clamp(-1, 1)) ** 2).item()
            res[tag] = (bpp_y, 10 * math.log10(4.0 / max(mse, 1e-10)), o["x_hat"].clamp(-1, 1))
        out[q] = res
    net.train()
    return out


def _panel(x, ab_q, n=4):
    n = min(n, x.shape[0])
    rows = []
    for i in range(n):
        rows += [from_minus1_1_to_0_1(x[i]),
                 from_minus1_1_to_0_1(ab_q["baseline"][2][i]),
                 from_minus1_1_to_0_1(ab_q["ours"][2][i])]
    return make_grid(torch.stack(rows), nrow=3).clamp(0, 1)


def make_gate_sendability_target(x, x_hat, spatial_size, desired_mean, tau=1.0, texture_weight=0.25, edge_weight=0.0):
    """Training-only teacher: high where the current residual is predictable/low.

    The map is re-centered to desired_mean so it shapes the spatial mask without
    fighting the explicit rho_target rate budget.
    """
    with torch.no_grad():
        err = (x.detach().clamp(-1, 1) - x_hat.detach().clamp(-1, 1)).abs().mean(1, keepdim=True)
        err_lr = F.interpolate(err, size=spatial_size, mode="area")
        dims = (2, 3)
        err_mu = err_lr.mean(dims, keepdim=True)
        err_std = err_lr.std(dims, keepdim=True).clamp_min(1e-6)
        low_err = torch.sigmoid((err_mu - err_lr) / (max(float(tau), 1e-6) * err_std))

        gray = x.detach().clamp(-1, 1).mean(1, keepdim=True)
        local = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        var = F.avg_pool2d((gray - local).pow(2), kernel_size=7, stride=1, padding=3)
        tex_lr = F.interpolate(var, size=spatial_size, mode="area")
        tex_mu = tex_lr.mean(dims, keepdim=True)
        tex_std = tex_lr.std(dims, keepdim=True).clamp_min(1e-6)
        texture = torch.sigmoid((tex_lr - tex_mu) / tex_std)

        gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
        gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
        grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
        grad_lr = F.interpolate(grad, size=spatial_size, mode="area")
        grad_mu = grad_lr.mean(dims, keepdim=True)
        grad_std = grad_lr.std(dims, keepdim=True).clamp_min(1e-6)
        edge = torch.sigmoid((grad_lr - grad_mu) / grad_std)

        w = min(max(float(texture_weight), 0.0), 1.0)
        e = max(float(edge_weight), 0.0)
        target = (1.0 - w) * low_err + w * texture - e * edge
        desired = target.new_tensor(float(desired_mean)).clamp(0.05, 0.95)
        target = target - target.mean(dims, keepdim=True) + desired
        return target.clamp(0.05, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="./ckpt_v2")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_R", type=float, default=1.0)
    ap.add_argument("--lambda_d", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--train_lpips_net", choices=["glc_vgg", "alex"], default="glc_vgg",
                    help="LPIPS model used for training loss. eval_metrics.py uses Alex LPIPS.")
    ap.add_argument("--lambda_dists", type=float, default=0.0,
                    help="DISTS perceptual loss weight, computed on [0,1] images")
    ap.add_argument("--lambda_base_l1", type=float, default=0.0,
                    help="Distill ours toward frozen GLC reconstruction with image-space L1")
    ap.add_argument("--lambda_base_lpips", type=float, default=0.0,
                    help="Distill ours toward frozen GLC reconstruction with the training LPIPS backbone")
    ap.add_argument("--base_distill_until", type=int, default=0,
                    help="Apply baseline distillation while it < this value; 0 keeps it active")
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_rho_floor", type=float, default=0.0,
                    help="Penalty for rho < 1 so the gate does not spend extra bits")
    ap.add_argument("--no_gate", action="store_true", help="知覚ゲートを使わない（q 条件化のみ）")
    ap.add_argument("--freeze_predictor", action="store_true", help="P_thetaを凍結し、q_embed/gateのみ学習する")
    ap.add_argument("--rho_max", type=float, default=2.0)
    ap.add_argument("--gate_rho_min", type=float, default=0.5,
                    help="Minimum rho clamp for PerceptualGate; use 1.0 to forbid bit-increasing rho<1")
    ap.add_argument("--gate_rho_mode", type=str, default="hard", choices=["hard", "softplus"],
                    help="Gate rho parameterization. hard preserves old checkpoints; softplus is monotone with positive-side gradient")
    ap.add_argument("--gate_softplus_shift", type=float, default=2.0)
    ap.add_argument("--gate_softplus_tau", type=float, default=1.0)
    ap.add_argument("--gate_rho_init", type=float, default=1.0,
                    help="Initial mean rho. >1 starts from mild bit saving, then learns where to return to rho=1")
    ap.add_argument("--lambda_rho_target", type=float, default=0.0,
                    help="Warmup penalty ReLU(rho_target - mean(rho)); keeps the no-send region alive early")
    ap.add_argument("--rho_target", type=float, default=1.0)
    ap.add_argument("--rho_target_until", type=int, default=0,
                    help="Apply rho_target penalty only while it < this value; 0 keeps it active for all iterations")
    ap.add_argument("--lambda_gate_send", type=float, default=0.0,
                    help="Training-only BCE teacher for spatial sendability; inference still uses z_hat only")
    ap.add_argument("--gate_send_until", type=int, default=0,
                    help="Apply sendability teacher while it < this value; 0 keeps it active")
    ap.add_argument("--gate_send_tau", type=float, default=1.0)
    ap.add_argument("--gate_send_texture_weight", type=float, default=0.25)
    ap.add_argument("--gate_send_edge_weight", type=float, default=0.0,
                    help="Subtract a high-gradient teacher term before recentering, protecting edges from high rho")
    ap.add_argument("--predictor_param_mode", choices=["mean", "scale_mean", "all", "latent_residual"], default="scale_mean")
    ap.add_argument("--predictor_delta_bound", type=float, default=0.0,
                    help="Bound predictor delta by bound*tanh(delta/bound); 0 disables")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=8)
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
    use_gate = not args.no_gate
    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   entity=args.wandb_entity, config=vars(args), mode=args.wandb_mode,
                   id=args.wandb_id, resume="allow" if args.resume else None)

    net = build_net(
        args.glc_weights, device, use_gate, args.rho_max, args.gate_rho_min,
        train_predictor=not args.freeze_predictor, rho_mode=args.gate_rho_mode,
        softplus_shift=args.gate_softplus_shift, softplus_tau=args.gate_softplus_tau,
        rho_init=args.gate_rho_init,
    )
    net.predictor_param_mode = args.predictor_param_mode
    net.predictor_delta_bound = args.predictor_delta_bound
    net.train()
    if args.train_lpips_net == "alex":
        if not _LPIPS_LIB:
            raise RuntimeError("--train_lpips_net alex requires the lpips package")
        lpips_loss = lpips_lib.LPIPS(net="alex").to(device).eval()
    else:
        lpips_loss = LPIPSLoss(get_lpips_model()).to(device).eval()
    for p in lpips_loss.parameters():
        p.requires_grad_(False)
    dists_loss = DISTS().to(device).eval() if args.lambda_dists > 0 else None
    if dists_loss is not None:
        for p in dists_loss.parameters():
            p.requires_grad_(False)

    loader = DataLoader(CropFolder(args.data, 256), batch_size=args.bs, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val = None
    if args.val:
        vds = CropFolder(args.val, 256)
        val = torch.stack([vds[i] for i in range(min(8, len(vds)))]).to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"学習対象: {sum(p.numel() for p in params)/1e6:.2f} M | gate={use_gate} | predictor_train={not args.freeze_predictor}")

    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.prior_predictor.load_state_dict(ck["prior_predictor"])
        with torch.no_grad():
            net.q_embed.copy_(ck["q_embed"].to(device))
        if use_gate and ck.get("perceptual_gate") is not None:
            net.perceptual_gate.load_state_dict(ck["perceptual_gate"])
        if bool(ck.get("use_gate", False)) == use_gate and "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        else:
            print("[warn] use_gate 不一致のため optimizer は初期化のまま継続。")
        start_it = ck.get("it", 0)
        print(f"[resume] {args.resume} から再開（it={start_it}, gate={use_gate}）")

    it = start_it
    while it < args.iters:
        for x in loader:
            x = x.to(device)
            q = random.randint(0, NUM_Q - 1)                          # (b) 毎バッチ q をサンプル
            gate = net.perceptual_gate if use_gate else None
            out = train_forward(net, x, q, use_predictor=True,
                                gate=gate, q_shift=net.q_embed[q:q + 1],
                                predictor_param_mode=args.predictor_param_mode,
                                predictor_delta_bound=args.predictor_delta_bound)
            B, _, H, W = x.shape
            bpp_y = out["bit_y"] / (B * H * W)
            bpp_z = torch.as_tensor(out["bit_z"], device=x.device, dtype=bpp_y.dtype) / (H * W)
            bpp_total = bpp_y + bpp_z
            d_mse = cal_mse_Loss(x, out["x_hat"]).mean()
            d_lp = lpips_loss(out["x_hat"], x).mean()
            base_distill_active = (args.lambda_base_l1 > 0 or args.lambda_base_lpips > 0) and (args.base_distill_until <= 0 or it < args.base_distill_until)
            if base_distill_active:
                with torch.no_grad():
                    base_out = train_forward(net, x, q, use_predictor=False, gate=None, q_shift=None,
                                             predictor_param_mode=args.predictor_param_mode,
                                             predictor_delta_bound=args.predictor_delta_bound)
                    x_hat_base = base_out["x_hat"].detach().clamp(-1, 1)
                x_hat_for_distill = out["x_hat"].clamp(-1, 1)
                l_base_l1 = F.l1_loss(x_hat_for_distill, x_hat_base)
                if args.lambda_base_lpips > 0:
                    l_base_lpips = lpips_loss(x_hat_for_distill, x_hat_base).mean()
                else:
                    l_base_lpips = bpp_y.new_tensor(0.0)
            else:
                l_base_l1 = bpp_y.new_tensor(0.0)
                l_base_lpips = bpp_y.new_tensor(0.0)
            if dists_loss is not None:
                x01 = from_minus1_1_to_0_1(x.clamp(-1, 1))
                xhat01 = from_minus1_1_to_0_1(out["x_hat"].clamp(-1, 1))
                d_dists = dists_loss(x01, xhat01).mean()
            else:
                d_dists = bpp_y.new_tensor(0.0)
            psnr = 10 * math.log10(4.0 / max(d_mse.item(), 1e-10))
            delta_abs = out["delta_params"].detach().abs().mean().item() if out["delta_params"] is not None else 0.0
            mu_mean = out["mu_pred"].detach().mean().item() if out["mu_pred"] is not None else 0.0
            mu_std = out["mu_pred"].detach().std().item() if out["mu_pred"] is not None else 0.0
            rho_mean = out["gate_rho"].detach().mean().item() if out.get("gate_rho") is not None else 1.0
            rho_min = out["gate_rho"].detach().min().item() if out.get("gate_rho") is not None else 1.0
            rho_max = out["gate_rho"].detach().max().item() if out.get("gate_rho") is not None else 1.0
            rho_active = (out["gate_rho"].detach() > 1.0005).float().mean().item() if out.get("gate_rho") is not None else 0.0
            if out.get("gate_p_tex") is not None:
                gate_raw = torch.logit(out["gate_p_tex"].detach().clamp(1e-6, 1 - 1e-6))
                gate_raw_mean = gate_raw.mean().item()
                gate_raw_min = gate_raw.min().item()
                gate_raw_max = gate_raw.max().item()
            else:
                gate_raw_mean = gate_raw_min = gate_raw_max = 0.0
            if args.lambda_align > 0:
                idx_gt = calculate_vqgan_results(x, net.vqgan)["idx_gt"]
                l_align = cal_ce_Loss(net.code_pred_loss(out["mu_pred"]), idx_gt).mean()
            else:
                l_align = bpp_y.new_tensor(0.0)
            if args.lambda_rho_floor > 0 and out.get("gate_rho") is not None:
                l_rho_floor = F.relu(1.0 - out["gate_rho"]).mean()
            else:
                l_rho_floor = bpp_y.new_tensor(0.0)
            target_active = args.rho_target_until <= 0 or it < args.rho_target_until
            if args.lambda_rho_target > 0 and target_active and out.get("gate_rho") is not None:
                l_rho_target = F.relu(out["gate_rho"].new_tensor(args.rho_target) - out["gate_rho"].mean())
            else:
                l_rho_target = bpp_y.new_tensor(0.0)
            send_active = args.gate_send_until <= 0 or it < args.gate_send_until
            if args.lambda_gate_send > 0 and send_active and out.get("gate_p_tex") is not None:
                if args.rho_max > 1.0:
                    desired_p = 0.5 * (1.0 + (args.rho_target - 1.0) / (args.rho_max - 1.0))
                else:
                    desired_p = 0.5
                gate_target = make_gate_sendability_target(
                    x, out["x_hat"], out["gate_p_tex"].shape[-2:], desired_p,
                    tau=args.gate_send_tau, texture_weight=args.gate_send_texture_weight,
                    edge_weight=args.gate_send_edge_weight)
                l_gate_send = F.binary_cross_entropy(out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), gate_target)
                gate_target_mean = gate_target.mean().item()
                gate_target_std = gate_target.std().item()
            else:
                l_gate_send = bpp_y.new_tensor(0.0)
                gate_target_mean = 0.0
                gate_target_std = 0.0
            loss = (args.lambda_R * bpp_y + args.lambda_d * d_mse
                    + args.lambda_lpips * d_lp + args.lambda_dists * d_dists
                    + args.lambda_base_l1 * l_base_l1 + args.lambda_base_lpips * l_base_lpips
                    + args.lambda_align * l_align + args.lambda_rho_floor * l_rho_floor
                    + args.lambda_rho_target * l_rho_target + args.lambda_gate_send * l_gate_send)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if it % args.log_every == 0:
                print(f"[it {it}] q={q} loss={loss.item():.4f} bpp={bpp_total.item():.4f} "
                      f"bpp_y={bpp_y.item():.4f} psnr={psnr:.2f} "
                      f"mse={d_mse.item():.4f} lpips={d_lp.item():.4f} dists={d_dists.item():.4f} ce={l_align.item():.4f} "
                      f"base_l1={l_base_l1.item():.4f} base_lpips={l_base_lpips.item():.4f} "
                      f"rho={rho_mean:.3f}/{rho_min:.3f}/{rho_max:.3f} active={rho_active:.2f} "
                      f"rho_floor={l_rho_floor.item():.4f} rho_target={l_rho_target.item():.4f} "
                      f"gate_send={l_gate_send.item():.4f} delta_abs={delta_abs:.5f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/bpp_y": bpp_y.item(),
                               "train/bpp_z": bpp_z.item(), "train/bpp_total": bpp_total.item(),
                               "train/psnr": psnr, "train/mse": d_mse.item(), "train/lpips": d_lp.item(),
                               "train/dists": d_dists.item(), "train/ce_align": l_align.item(),
                               "train/base_l1": l_base_l1.item(), "train/base_lpips": l_base_lpips.item(),
                               "train/rho_floor": l_rho_floor.item(), "train/rho_target": l_rho_target.item(),
                               "train/gate_send": l_gate_send.item(),
                               "pred/delta_abs_mean": delta_abs,
                               "gate/rho_mean": rho_mean, "gate/rho_min": rho_min, "gate/rho_max": rho_max,
                               "gate/rho_active_frac": rho_active, "gate/raw_mean": gate_raw_mean,
                               "gate/raw_min": gate_raw_min, "gate/raw_max": gate_raw_max,
                               "gate/send_target_mean": gate_target_mean, "gate/send_target_std": gate_target_std,
                               "pred/mu_mean": mu_mean, "pred/mu_std": mu_std,
                               "train/q": q}, step=it)

            if val is not None and it % args.eval_every == 0:
                ab = quick_eval(net, val)
                logd = {}
                for q_ in ab:
                    b, o = ab[q_]["baseline"], ab[q_]["ours"]
                    d = o[0] - b[0]
                    print(f"  [A/B it {it} q={q_}] base bpp_y={b[0]:.4f}/psnr={b[1]:.2f} "
                          f"| ours bpp_y={o[0]:.4f}/psnr={o[1]:.2f} | Δbpp_y={d:+.4f}")
                    logd[f"ab/q{q_}_delta_bpp_y"] = d
                    logd[f"ab/q{q_}_ours_bpp_y"] = o[0]
                    logd[f"ab/q{q_}_ours_psnr"] = o[1]
                if use_wandb:
                    logd["ab/samples_q2"] = wandb.Image(_panel(val, ab[2]),
                                                        caption="q=2 rows: [orig | baseline | ours]")
                    wandb.log(logd, step=it)
                torch.save({"prior_predictor": net.prior_predictor.state_dict(),
                            "q_embed": net.q_embed.detach().cpu(),
                            "perceptual_gate": net.perceptual_gate.state_dict() if use_gate else None,
                            "use_gate": use_gate, "rho_max": args.rho_max, "rho_min": args.gate_rho_min,
                             "rho_mode": args.gate_rho_mode, "gate_softplus_shift": args.gate_softplus_shift,
                             "gate_softplus_tau": args.gate_softplus_tau, "gate_rho_init": args.gate_rho_init},
                           os.path.join(args.out, f"v2_{it}.pt"))
                torch.save({"it": it, "prior_predictor": net.prior_predictor.state_dict(),
                            "q_embed": net.q_embed.detach().cpu(),
                            "perceptual_gate": net.perceptual_gate.state_dict() if use_gate else None,
                            "use_gate": use_gate, "rho_max": args.rho_max, "rho_min": args.gate_rho_min,
                            "rho_mode": args.gate_rho_mode, "gate_softplus_shift": args.gate_softplus_shift,
                            "gate_softplus_tau": args.gate_softplus_tau, "gate_rho_init": args.gate_rho_init,
                            "optimizer": opt.state_dict()},
                           os.path.join(args.out, "train_state.pt"))   # resume 用（上書き）

            it += 1
            if it >= args.iters:
                break

    torch.save({"prior_predictor": net.prior_predictor.state_dict(),
                "q_embed": net.q_embed.detach().cpu(),
                "perceptual_gate": net.perceptual_gate.state_dict() if use_gate else None,
                "use_gate": use_gate, "rho_max": args.rho_max, "rho_min": args.gate_rho_min,
                             "rho_mode": args.gate_rho_mode, "gate_softplus_shift": args.gate_softplus_shift,
                             "gate_softplus_tau": args.gate_softplus_tau, "gate_rho_init": args.gate_rho_init},
               os.path.join(args.out, "v2_final.pt"))
    torch.save({"it": it, "prior_predictor": net.prior_predictor.state_dict(),
                "q_embed": net.q_embed.detach().cpu(),
                "perceptual_gate": net.perceptual_gate.state_dict() if use_gate else None,
                "use_gate": use_gate, "rho_max": args.rho_max, "rho_min": args.gate_rho_min,
                "rho_mode": args.gate_rho_mode, "gate_softplus_shift": args.gate_softplus_shift,
                "gate_softplus_tau": args.gate_softplus_tau, "gate_rho_init": args.gate_rho_init,
                "optimizer": opt.state_dict()},
               os.path.join(args.out, "train_state.pt"))
    print("done. → test_v2.py で全 q（--interpolate で 64 点）の R-P 曲線を生成。")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
