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
from DISTS_pytorch import DISTS

from src.models.image_model import GLC_Image
from src.models.loss import (
    get_lpips_model, LPIPSLoss,
    calculate_vqgan_results, cal_ce_Loss, cal_mse_Loss,
)
from src.utils.test_utils import get_state_dict, from_0_1_to_minus1_1, from_minus1_1_to_0_1
from src.utils.lpips.lpips import LPIPS as RawLPIPS
from gp_reslc.prior_predictor import PriorPredictor, StageResidualPredictor, StageQuantGate, train_forward

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
        self.paths = sorted(sum([
            glob.glob(os.path.join(root, "**", e), recursive=True) for e in self.EXTS
        ], []))
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
def _set_trainable(module, value: bool) -> int:
    if module is None:
        return 0
    count = 0
    for param in module.parameters():
        param.requires_grad_(value)
        count += param.numel()
    return count


def build_net(weights, device, unfreeze_fusion=False, use_stage_residual=False, use_stage_quant_gate=False,
              stage_rho_max=1.5, unfreeze_entropy=False, unfreeze_hyper_dec=False):
    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(weights), strict=True)   # GLC 事前学習済み
    net.prior_predictor = PriorPredictor(net.N)                  # 新規（zero-init ゲート）
    if use_stage_residual:
        net.stage_residual_predictor = StageResidualPredictor(net.N)
    if use_stage_quant_gate:
        net.stage_quant_gate = StageQuantGate(net.N, rho_max=stage_rho_max)
    net = net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    if use_stage_residual:
        for p in net.stage_residual_predictor.parameters():
            p.requires_grad_(True)
    if use_stage_quant_gate:
        for p in net.stage_quant_gate.parameters():
            p.requires_grad_(True)
    if not use_stage_residual and not use_stage_quant_gate:
        for p in net.prior_predictor.parameters():
            p.requires_grad_(True)
    if unfreeze_fusion:  # 利得が出ないときのピボット用
        _set_trainable(net.y_prior_fusion, True)
    if unfreeze_entropy:
        _set_trainable(net.y_prior_fusion, True)
        _set_trainable(getattr(net, "y_spatial_prior_reduction", None), True)
        _set_trainable(net.y_spatial_prior_adaptor_1, True)
        _set_trainable(net.y_spatial_prior_adaptor_2, True)
        _set_trainable(net.y_spatial_prior_adaptor_3, True)
        _set_trainable(net.y_spatial_prior, True)
    if unfreeze_hyper_dec:
        _set_trainable(net.hyper_dec, True)
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




def stage_gate_p_from_rho_target(rho_target: float, rho_max: float, shift: float = 2.0, tau: float = 1.0) -> float:
    """Approximate StageQuantGate p_tex mean that corresponds to a target rho."""
    if rho_max <= 1.0:
        return 0.5
    frac = (float(rho_target) - 1.0) / max(float(rho_max) - 1.0, 1e-6)
    frac = min(max(frac, 1e-5), 1.0 - 1e-5)
    excess = -float(tau) * math.log(1.0 - frac)
    base = F.softplus(torch.tensor(float(shift))).item()
    y = excess + base
    raw = math.log(max(math.expm1(y), 1e-12)) - float(shift)
    return min(max(1.0 / (1.0 + math.exp(-raw)), 0.05), 0.95)


def make_gate_sendability_target(x, x_hat, spatial_size, desired_mean, tau=1.0, texture_weight=0.25, edge_weight=0.0):
    """Training-only teacher for decoder-computable stage gates.

    High target values are assigned to low-error/predictable regions. The map is
    recentered to the p_tex mean implied by rho_target so it shapes spatial use
    without changing the requested average coarsening budget. No teacher signal
    is used at inference time.
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


def make_gate_lpips_sensitivity_target(x, base_x_hat, spatial_size, desired_mean, lpips_spatial_loss,
                                       tau=1.0, edge_weight=0.0):
    """LPIPS-spatial target for stage gates.

    High target values mean a location is perceptually safer to coarsen. The
    target is built from the frozen GLC baseline reconstruction, then recentered
    to the desired mean implied by rho_target so it changes allocation rather
    than the requested average rate budget.
    """
    with torch.no_grad():
        lp = lpips_spatial_loss(base_x_hat.detach().clamp(-1, 1), x.detach().clamp(-1, 1)).clamp_min(0)
        lp_lr = F.interpolate(lp, size=spatial_size, mode="area")
        dims = (2, 3)
        mu = lp_lr.mean(dims, keepdim=True)
        std = lp_lr.std(dims, keepdim=True).clamp_min(1e-6)
        safe = torch.sigmoid((mu - lp_lr) / (max(float(tau), 1e-6) * std))

        if edge_weight > 0:
            gray = x.detach().clamp(-1, 1).mean(1, keepdim=True)
            gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
            gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
            grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
            grad_lr = F.interpolate(grad, size=spatial_size, mode="area")
            gmu = grad_lr.mean(dims, keepdim=True)
            gstd = grad_lr.std(dims, keepdim=True).clamp_min(1e-6)
            edge = torch.sigmoid((grad_lr - gmu) / gstd)
            safe = safe - float(edge_weight) * edge

        desired = safe.new_tensor(float(desired_mean)).clamp(0.05, 0.95)
        safe = safe - safe.mean(dims, keepdim=True) + desired
        return safe.clamp(0.05, 0.95)


def make_gate_measured_sensitivity_target(x, base_x_hat, ours_x_hat, spatial_size, desired_mean, lpips_spatial_loss,
                                          margin=0.0, tau=1.0, edge_weight=0.0):
    """Training-only target from the measured local effect of current coarsening.

    High target values mean the current gated reconstruction does not worsen
    local LPIPS-spatial relative to frozen GLC, so those positions are safer to
    coarsen. The map is recentered to the desired mean implied by rho_target.
    """
    with torch.no_grad():
        base_lp = lpips_spatial_loss(base_x_hat.detach().clamp(-1, 1), x.detach().clamp(-1, 1)).clamp_min(0)
        ours_lp = lpips_spatial_loss(ours_x_hat.detach().clamp(-1, 1), x.detach().clamp(-1, 1)).clamp_min(0)
        lp_delta = F.interpolate(ours_lp - base_lp, size=spatial_size, mode="area")
        dims = (2, 3)
        centered = lp_delta - float(margin)
        scale = centered.std(dims, keepdim=True).clamp_min(1e-6)
        safe = torch.sigmoid(-centered / (max(float(tau), 1e-6) * scale))

        if edge_weight > 0:
            gray = x.detach().clamp(-1, 1).mean(1, keepdim=True)
            gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
            gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
            grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
            grad_lr = F.interpolate(grad, size=spatial_size, mode="area")
            gmu = grad_lr.mean(dims, keepdim=True)
            gstd = grad_lr.std(dims, keepdim=True).clamp_min(1e-6)
            edge = torch.sigmoid((grad_lr - gmu) / gstd)
            safe = safe - float(edge_weight) * edge

        desired = safe.new_tensor(float(desired_mean)).clamp(0.05, 0.95)
        safe = safe - safe.mean(dims, keepdim=True) + desired
        return safe.clamp(0.05, 0.95)


def _standardize_map(v: torch.Tensor, dims=(2, 3)) -> torch.Tensor:
    mu = v.mean(dims, keepdim=True)
    std = v.std(dims, keepdim=True).clamp_min(1e-6)
    return (v - mu) / std


def make_gate_mixed_sensitivity_target(x, base_x_hat, ours_x_hat, spatial_size, desired_mean,
                                       lpips_spatial_loss, l1_weight=1.0, lpips_weight=1.0,
                                       texture_weight=0.0, edge_weight=0.0, margin=0.0, tau=1.0):
    """Mixed local teacher for stage gates.

    High target values mark positions where the current coarsening is locally
    safe: it causes little L1/LPIPS-spatial degradation relative to frozen GLC
    and avoids high edge/texture regions. The map is recentered to the target
    p-map mean, so it reallocates rho spatially without changing the average
    rate budget.
    """
    with torch.no_grad():
        x_ref = x.detach().clamp(-1, 1)
        base = base_x_hat.detach().clamp(-1, 1)
        ours = ours_x_hat.detach().clamp(-1, 1)
        dims = (2, 3)

        base_l1 = (base - x_ref).abs().mean(1, keepdim=True)
        ours_l1 = (ours - x_ref).abs().mean(1, keepdim=True)
        l1_delta = F.interpolate(ours_l1 - base_l1, size=spatial_size, mode="area")

        base_lp = lpips_spatial_loss(base, x_ref).clamp_min(0)
        ours_lp = lpips_spatial_loss(ours, x_ref).clamp_min(0)
        lp_delta = F.interpolate(ours_lp - base_lp, size=spatial_size, mode="area")

        score = l1_delta.new_zeros(l1_delta.shape)
        if l1_weight > 0:
            score = score + float(l1_weight) * _standardize_map(l1_delta, dims)
        if lpips_weight > 0:
            score = score + float(lpips_weight) * _standardize_map(lp_delta, dims)

        gray = x_ref.mean(1, keepdim=True)
        if texture_weight > 0:
            local = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
            var = F.avg_pool2d((gray - local).pow(2), kernel_size=7, stride=1, padding=3)
            tex_lr = F.interpolate(var, size=spatial_size, mode="area")
            score = score + float(texture_weight) * _standardize_map(tex_lr, dims)

        if edge_weight > 0:
            gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
            gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
            grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
            grad_lr = F.interpolate(grad, size=spatial_size, mode="area")
            score = score + float(edge_weight) * _standardize_map(grad_lr, dims)

        safe = torch.sigmoid(-(score - float(margin)) / max(float(tau), 1e-6))
        desired = safe.new_tensor(float(desired_mean)).clamp(0.05, 0.95)
        safe = safe - safe.mean(dims, keepdim=True) + desired
        return safe.clamp(0.05, 0.95)

def _img_panel(x, recon, n=4):
    """[原画 | baseline | ours] の比較グリッドを作る（wandb 用, [0,1]）。"""
    n = min(n, x.shape[0])
    rows = []
    for i in range(n):
        rows += [from_minus1_1_to_0_1(x[i]),
                 from_minus1_1_to_0_1(recon["baseline"][i]),
                 from_minus1_1_to_0_1(recon["ours"][i])]
    return make_grid(torch.stack(rows), nrow=3).clamp(0, 1)


class LPIPSSpatialLoss(torch.nn.Module):
    """LPIPS wrapper that keeps the spatial map for local gate regularization."""

    def __init__(self, lpips_model, use_input_norm=True, range_norm=True):
        super().__init__()
        self.perceptual = lpips_model
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        if self.use_input_norm:
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        return self.perceptual(target.contiguous(), pred.contiguous())


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
    ap.add_argument("--lambda_dists", type=float, default=0.0)
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_mean_pred", type=float, default=0.0,
                    help="Smooth-L1 between corrected common prior mean and y in quantized space")
    ap.add_argument("--lambda_scale_reg", type=float, default=0.0,
                    help="Penalize positive scale deltas to discourage rate reduction by scale inflation")
    ap.add_argument("--lambda_distill", type=float, default=0.0,
                    help="MSE distillation to frozen GLC baseline reconstruction")
    ap.add_argument("--lambda_lpips_distill", type=float, default=0.0,
                    help="LPIPS distillation to frozen GLC baseline reconstruction")
    ap.add_argument("--lambda_dists_distill", type=float, default=0.0,
                    help="DISTS distillation to frozen GLC baseline reconstruction")
    ap.add_argument("--lambda_lpips_hinge", type=float, default=0.0,
                    help="Penalize LPIPS worse than frozen GLC baseline by more than --lpips_hinge_margin")
    ap.add_argument("--lambda_dists_hinge", type=float, default=0.0,
                    help="Penalize DISTS worse than frozen GLC baseline by more than --dists_hinge_margin")
    ap.add_argument("--lpips_hinge_margin", type=float, default=0.0)
    ap.add_argument("--dists_hinge_margin", type=float, default=0.0)
    ap.add_argument("--lambda_R_start", type=float, default=None,
                    help="Initial lambda_R for linear rate-pressure warmup; defaults to --lambda_R")
    ap.add_argument("--rate_warmup_iters", type=int, default=0,
                    help="Linearly ramp lambda_R_start to lambda_R over this many iterations")
    ap.add_argument("--unfreeze_fusion", action="store_true")
    ap.add_argument("--unfreeze_entropy", action="store_true",
                    help="Also train y_prior_fusion, y_spatial_prior_reduction, adaptors, and y_spatial_prior.")
    ap.add_argument("--unfreeze_hyper_dec", action="store_true",
                    help="Also train hyper_dec so z_hat-to-prior features can adapt to residual coding.")
    ap.add_argument("--save_model_state", action="store_true",
                    help="Save net.state_dict() in train_state.pt. Forced on when any GLC module is unfrozen.")
    ap.add_argument("--freeze_aux_module", action="store_true",
                    help="Freeze prior_predictor/stage_residual/stage_quant while training unfrozen GLC modules.")
    ap.add_argument(
        "--predictor_param_mode",
        choices=[
            "mean",
            "scale_mean",
            "all",
            "latent_residual",
            "stage_latent_residual",
            "stage_quant_gate",
            "stage_residual_quant_gate",
        ],
        default="scale_mean",
    )
    ap.add_argument("--predictor_delta_bound", type=float, default=0.0,
                    help="Bound predictor delta by bound*tanh(delta/bound); 0 disables")
    ap.add_argument("--stage_rho_max", type=float, default=1.5,
                    help="Maximum rho for predictor_param_mode=stage_quant_gate")
    ap.add_argument("--lambda_rho_target", type=float, default=0.0,
                    help="Penalty ReLU(rho_target - mean(rho)) for stage_quant_gate.")
    ap.add_argument("--rho_target", type=float, default=1.0)
    ap.add_argument("--lambda_gate_send", type=float, default=0.0,
                    help="Training-only BCE teacher for stage gate sendability; inference sends no map")
    ap.add_argument("--gate_send_tau", type=float, default=1.0)
    ap.add_argument("--gate_send_texture_weight", type=float, default=0.25)
    ap.add_argument("--gate_send_edge_weight", type=float, default=0.0,
                    help="Subtract high-gradient teacher term, protecting edges from high rho")
    ap.add_argument("--gate_send_use_base", action="store_true",
                    help="Build sendability teacher from frozen GLC baseline reconstruction instead of current ours")
    ap.add_argument("--lambda_lpips_spatial_gate_hinge", type=float, default=0.0,
                    help="Penalize coarse gate probability where local spatial LPIPS worsens versus frozen GLC baseline")
    ap.add_argument("--lpips_spatial_gate_margin", type=float, default=0.0)
    ap.add_argument("--lambda_gate_lpips_sens", type=float, default=0.0,
                    help="BCE teacher that allocates high gate probability to low baseline LPIPS-spatial sensitivity regions")
    ap.add_argument("--gate_lpips_sens_tau", type=float, default=1.0)
    ap.add_argument("--gate_lpips_sens_edge_weight", type=float, default=0.0)
    ap.add_argument("--lambda_gate_measured_sens", type=float, default=0.0,
                    help="BCE teacher from measured LPIPS-spatial delta caused by current coarsening")
    ap.add_argument("--gate_measured_sens_tau", type=float, default=1.0)
    ap.add_argument("--gate_measured_sens_margin", type=float, default=0.0)
    ap.add_argument("--gate_measured_sens_edge_weight", type=float, default=0.0)
    ap.add_argument("--lambda_gate_mixed_sens", type=float, default=0.0,
                    help="BCE teacher from mixed local L1/LPIPS sensitivity caused by current coarsening")
    ap.add_argument("--gate_mixed_sens_tau", type=float, default=1.0)
    ap.add_argument("--gate_mixed_sens_margin", type=float, default=0.0)
    ap.add_argument("--gate_mixed_l1_weight", type=float, default=1.0)
    ap.add_argument("--gate_mixed_lpips_weight", type=float, default=1.0)
    ap.add_argument("--gate_mixed_texture_weight", type=float, default=0.0)
    ap.add_argument("--gate_mixed_edge_weight", type=float, default=0.0)
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
    ap.add_argument("--resume_weights_only", action="store_true",
                    help="Load module weights from --resume but start a fresh optimizer and iteration counter.")
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

    use_stage_residual = args.predictor_param_mode in {
        "stage_latent_residual",
        "stage_residual_quant_gate",
    }
    use_stage_quant_gate = args.predictor_param_mode in {
        "stage_quant_gate",
        "stage_residual_quant_gate",
    }
    net = build_net(args.glc_weights, device, args.unfreeze_fusion,
                    use_stage_residual, use_stage_quant_gate, args.stage_rho_max,
                    args.unfreeze_entropy, args.unfreeze_hyper_dec)
    net.predictor_param_mode = args.predictor_param_mode
    net.predictor_delta_bound = args.predictor_delta_bound
    if args.freeze_aux_module:
        if use_stage_residual and use_stage_quant_gate:
            _set_trainable(net.stage_residual_predictor, False)
            _set_trainable(net.stage_quant_gate, False)
        elif use_stage_residual:
            _set_trainable(net.stage_residual_predictor, False)
        elif use_stage_quant_gate:
            _set_trainable(net.stage_quant_gate, False)
        else:
            _set_trainable(net.prior_predictor, False)
    net.train()

    lpips_loss = LPIPSLoss(get_lpips_model()).to(device).eval()
    for p in lpips_loss.parameters():
        p.requires_grad_(False)
    lpips_spatial_loss = None
    if (args.lambda_lpips_spatial_gate_hinge > 0 or args.lambda_gate_lpips_sens > 0
            or args.lambda_gate_measured_sens > 0 or args.lambda_gate_mixed_sens > 0):
        lpips_spatial_loss = LPIPSSpatialLoss(RawLPIPS(net="alex", spatial=True, verbose=False)).to(device).eval()
        for p in lpips_spatial_loss.parameters():
            p.requires_grad_(False)
    need_dists_model = any(v > 0 for v in (
        args.lambda_dists,
        args.lambda_dists_distill,
        args.lambda_dists_hinge,
    ))
    dists_loss = DISTS().to(device).eval() if need_dists_model else None
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
    n_param = sum(p.numel() for p in params) / 1e6
    print(f"学習対象パラメータ: {n_param:.2f} M")
    save_model_state = args.save_model_state or args.unfreeze_fusion or args.unfreeze_entropy or args.unfreeze_hyper_dec
    if use_wandb:
        wandb.summary["trainable_params_M"] = n_param
        wandb.summary["save_model_state"] = bool(save_model_state)

    def add_common_state(state):
        state["unfreeze_fusion"] = bool(args.unfreeze_fusion)
        state["unfreeze_entropy"] = bool(args.unfreeze_entropy)
        state["unfreeze_hyper_dec"] = bool(args.unfreeze_hyper_dec)
        state["freeze_aux_module"] = bool(args.freeze_aux_module)
        if save_model_state:
            state["model_state_dict"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        return state

    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if "model_state_dict" in ck:
            net.load_state_dict(ck["model_state_dict"], strict=False)
        if use_stage_residual:
            net.stage_residual_predictor.load_state_dict(ck["stage_residual_predictor"])
        if use_stage_quant_gate:
            net.stage_quant_gate.load_state_dict(ck["stage_quant_gate"])
        else:
            net.prior_predictor.load_state_dict(ck["prior_predictor"])
        if not args.resume_weights_only:
            opt.load_state_dict(ck["optimizer"])
            start_it = ck.get("it", 0)
            print(f"[resume] {args.resume} から再開（it={start_it}）")
        else:
            print(f"[resume_weights_only] {args.resume} から重みのみロード（fresh optimizer）")

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
            if args.lambda_R_start is not None and args.rate_warmup_iters > 0:
                progress = min(1.0, max(0.0, it / float(args.rate_warmup_iters)))
                lambda_R_eff = args.lambda_R_start + (args.lambda_R - args.lambda_R_start) * progress
            else:
                lambda_R_eff = args.lambda_R
            d_mse = cal_mse_Loss(x, out["x_hat"]).mean()
            d_lp = lpips_loss(out["x_hat"], x).mean()
            if dists_loss is not None:
                d_dists = dists_loss(
                    from_minus1_1_to_0_1(out["x_hat"].clamp(-1, 1)),
                    from_minus1_1_to_0_1(x),
                ).mean()
            else:
                d_dists = bpp_y.new_tensor(0.0)
            psnr = 10 * math.log10(4.0 / max(d_mse.item(), 1e-10))
            delta_abs = out["delta_params"].detach().abs().mean().item() if out["delta_params"] is not None else 0.0
            stage_delta_abs = out["stage_delta_abs"].detach().item() if out.get("stage_delta_abs") is not None else 0.0
            stage_target_abs = out["stage_target_abs"].detach().item() if out.get("stage_target_abs") is not None else 0.0
            mu_mean = out["mu_pred"].detach().mean().item() if out["mu_pred"] is not None else 0.0
            mu_std = out["mu_pred"].detach().std().item() if out["mu_pred"] is not None else 0.0
            rho_mean = out["gate_rho"].detach().mean().item() if out.get("gate_rho") is not None else 1.0
            rho_max = out["gate_rho"].detach().max().item() if out.get("gate_rho") is not None else 1.0
            zero = bpp_y.new_tensor(0.0)
            base_x_hat = None
            if args.lambda_rho_target > 0 and out.get("gate_rho") is not None:
                l_rho_target = F.relu(bpp_y.new_tensor(args.rho_target) - out["gate_rho"].mean())
            else:
                l_rho_target = zero
            if args.lambda_gate_send > 0 and out.get("gate_p_tex") is not None:
                desired_p = stage_gate_p_from_rho_target(args.rho_target, args.stage_rho_max)
                teacher_x_hat = out["x_hat"]
                if args.gate_send_use_base:
                    with torch.no_grad():
                        base_out = train_forward(net, x, args.q_index, use_predictor=False,
                                                 predictor_param_mode=args.predictor_param_mode,
                                                 predictor_delta_bound=args.predictor_delta_bound)
                        base_x_hat = base_out["x_hat"].detach().clamp(-1, 1)
                    teacher_x_hat = base_x_hat
                gate_target = make_gate_sendability_target(
                    x, teacher_x_hat, out["gate_p_tex"].shape[-2:], desired_p,
                    tau=args.gate_send_tau,
                    texture_weight=args.gate_send_texture_weight,
                    edge_weight=args.gate_send_edge_weight,
                )
                if gate_target.shape != out["gate_p_tex"].shape:
                    gate_target = gate_target.expand_as(out["gate_p_tex"])
                l_gate_send = F.binary_cross_entropy(out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), gate_target)
                gate_target_mean = float(gate_target.mean().item())
                gate_target_std = float(gate_target.std().item())
            else:
                l_gate_send = zero
                gate_target_mean = 0.0
                gate_target_std = 0.0

            if args.lambda_gate_lpips_sens > 0 and out.get("gate_p_tex") is not None:
                desired_p = stage_gate_p_from_rho_target(args.rho_target, args.stage_rho_max)
                if base_x_hat is None:
                    with torch.no_grad():
                        base_out = train_forward(net, x, args.q_index, use_predictor=False,
                                                 predictor_param_mode=args.predictor_param_mode,
                                                 predictor_delta_bound=args.predictor_delta_bound)
                        base_x_hat = base_out["x_hat"].detach().clamp(-1, 1)
                lpips_sens_target = make_gate_lpips_sensitivity_target(
                    x, base_x_hat, out["gate_p_tex"].shape[-2:], desired_p, lpips_spatial_loss,
                    tau=args.gate_lpips_sens_tau, edge_weight=args.gate_lpips_sens_edge_weight,
                )
                if lpips_sens_target.shape != out["gate_p_tex"].shape:
                    lpips_sens_target = lpips_sens_target.expand_as(out["gate_p_tex"])
                l_gate_lpips_sens = F.binary_cross_entropy(
                    out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), lpips_sens_target)
                gate_lpips_sens_mean = float(lpips_sens_target.mean().item())
                gate_lpips_sens_std = float(lpips_sens_target.std().item())
            else:
                l_gate_lpips_sens = zero
                gate_lpips_sens_mean = 0.0
                gate_lpips_sens_std = 0.0

            if args.lambda_gate_measured_sens > 0 and out.get("gate_p_tex") is not None:
                desired_p = stage_gate_p_from_rho_target(args.rho_target, args.stage_rho_max)
                if base_x_hat is None:
                    with torch.no_grad():
                        base_out = train_forward(net, x, args.q_index, use_predictor=False,
                                                 predictor_param_mode=args.predictor_param_mode,
                                                 predictor_delta_bound=args.predictor_delta_bound)
                        base_x_hat = base_out["x_hat"].detach().clamp(-1, 1)
                measured_sens_target = make_gate_measured_sensitivity_target(
                    x, base_x_hat, out["x_hat"], out["gate_p_tex"].shape[-2:], desired_p, lpips_spatial_loss,
                    margin=args.gate_measured_sens_margin, tau=args.gate_measured_sens_tau,
                    edge_weight=args.gate_measured_sens_edge_weight,
                )
                if measured_sens_target.shape != out["gate_p_tex"].shape:
                    measured_sens_target = measured_sens_target.expand_as(out["gate_p_tex"])
                l_gate_measured_sens = F.binary_cross_entropy(
                    out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), measured_sens_target)
                gate_measured_sens_mean = float(measured_sens_target.mean().item())
                gate_measured_sens_std = float(measured_sens_target.std().item())
            else:
                l_gate_measured_sens = zero
                gate_measured_sens_mean = 0.0
                gate_measured_sens_std = 0.0

            if args.lambda_gate_mixed_sens > 0 and out.get("gate_p_tex") is not None:
                desired_p = stage_gate_p_from_rho_target(args.rho_target, args.stage_rho_max)
                if base_x_hat is None:
                    with torch.no_grad():
                        base_out = train_forward(net, x, args.q_index, use_predictor=False,
                                                 predictor_param_mode=args.predictor_param_mode,
                                                 predictor_delta_bound=args.predictor_delta_bound)
                        base_x_hat = base_out["x_hat"].detach().clamp(-1, 1)
                mixed_sens_target = make_gate_mixed_sensitivity_target(
                    x, base_x_hat, out["x_hat"], out["gate_p_tex"].shape[-2:], desired_p, lpips_spatial_loss,
                    l1_weight=args.gate_mixed_l1_weight,
                    lpips_weight=args.gate_mixed_lpips_weight,
                    texture_weight=args.gate_mixed_texture_weight,
                    edge_weight=args.gate_mixed_edge_weight,
                    margin=args.gate_mixed_sens_margin,
                    tau=args.gate_mixed_sens_tau,
                )
                if mixed_sens_target.shape != out["gate_p_tex"].shape:
                    mixed_sens_target = mixed_sens_target.expand_as(out["gate_p_tex"])
                l_gate_mixed_sens = F.binary_cross_entropy(
                    out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), mixed_sens_target)
                gate_mixed_sens_mean = float(mixed_sens_target.mean().item())
                gate_mixed_sens_std = float(mixed_sens_target.std().item())
            else:
                l_gate_mixed_sens = zero
                gate_mixed_sens_mean = 0.0
                gate_mixed_sens_std = 0.0
            if args.lambda_align > 0 and out["mu_pred"] is not None:
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
                elif args.predictor_param_mode in {"stage_latent_residual", "stage_residual_quant_gate"}:
                    l_mean_pred = out["stage_mean_pred_loss"]
                elif args.predictor_param_mode == "stage_quant_gate":
                    l_mean_pred = zero
                else:
                    _, _, _, corrected_mean = net.separate_prior(out["params_after"])
                    l_mean_pred = F.smooth_l1_loss(corrected_mean, target_y)
            else:
                l_mean_pred = zero

            if args.lambda_scale_reg > 0 and args.predictor_param_mode not in (
                "latent_residual",
                "stage_latent_residual",
                "stage_quant_gate",
                "stage_residual_quant_gate",
            ):
                base_scale = out["params_base"][:, net.N:2 * net.N]
                after_scale = out["params_after"][:, net.N:2 * net.N]
                l_scale_reg = F.relu(after_scale - base_scale).mean()
            else:
                l_scale_reg = zero

            need_base_loss = any(v > 0 for v in (
                args.lambda_distill,
                args.lambda_lpips_distill,
                args.lambda_dists_distill,
                args.lambda_lpips_hinge,
                args.lambda_dists_hinge,
                args.lambda_lpips_spatial_gate_hinge,
                args.lambda_gate_lpips_sens,
                args.lambda_gate_measured_sens,
                args.lambda_gate_mixed_sens,
            ))
            if need_base_loss and base_x_hat is None:
                with torch.no_grad():
                    base_out = train_forward(net, x, args.q_index, use_predictor=False,
                                             predictor_param_mode=args.predictor_param_mode,
                                             predictor_delta_bound=args.predictor_delta_bound)
                    base_x_hat = base_out["x_hat"].detach().clamp(-1, 1)

            l_distill = cal_mse_Loss(out["x_hat"], base_x_hat).mean() if args.lambda_distill > 0 else zero
            l_lpips_distill = lpips_loss(out["x_hat"].clamp(-1, 1), base_x_hat).mean() if args.lambda_lpips_distill > 0 else zero
            if args.lambda_dists_distill > 0:
                l_dists_distill = dists_loss(
                    from_minus1_1_to_0_1(out["x_hat"].clamp(-1, 1)),
                    from_minus1_1_to_0_1(base_x_hat),
                ).mean()
            else:
                l_dists_distill = zero

            if args.lambda_lpips_hinge > 0:
                with torch.no_grad():
                    base_lp = lpips_loss(base_x_hat, x).mean()
                l_lpips_hinge = F.relu(d_lp - base_lp - args.lpips_hinge_margin)
            else:
                l_lpips_hinge = zero

            if args.lambda_dists_hinge > 0:
                with torch.no_grad():
                    base_dists = dists_loss(
                        from_minus1_1_to_0_1(base_x_hat),
                        from_minus1_1_to_0_1(x),
                    ).mean()
                l_dists_hinge = F.relu(d_dists - base_dists - args.dists_hinge_margin)
            else:
                l_dists_hinge = zero

            if args.lambda_lpips_spatial_gate_hinge > 0 and out.get("gate_p_tex") is not None:
                lp_map = lpips_spatial_loss(out["x_hat"].clamp(-1, 1), x).clamp_min(0)
                with torch.no_grad():
                    base_lp_map = lpips_spatial_loss(base_x_hat, x).clamp_min(0)
                lp_delta = F.interpolate(lp_map - base_lp_map, size=out["gate_p_tex"].shape[-2:], mode="area")
                local_bad = F.relu(lp_delta - args.lpips_spatial_gate_margin)
                gate_weight = out["gate_p_tex"].mean(dim=1, keepdim=True).clamp(0, 1)
                l_lpips_spatial_gate_hinge = (local_bad * gate_weight).mean()
                lpips_spatial_bad_mean = float(local_bad.detach().mean().item())
                lpips_spatial_gate_weight = float(gate_weight.detach().mean().item())
            else:
                l_lpips_spatial_gate_hinge = zero
                lpips_spatial_bad_mean = 0.0
                lpips_spatial_gate_weight = 0.0

            loss = (lambda_R_eff * bpp_y + args.lambda_d * d_mse
                    + args.lambda_lpips * d_lp + args.lambda_dists * d_dists
                    + args.lambda_align * l_align
                    + args.lambda_mean_pred * l_mean_pred
                    + args.lambda_scale_reg * l_scale_reg
                    + args.lambda_distill * l_distill
                    + args.lambda_lpips_distill * l_lpips_distill
                    + args.lambda_dists_distill * l_dists_distill
                    + args.lambda_lpips_hinge * l_lpips_hinge
                    + args.lambda_dists_hinge * l_dists_hinge
                    + args.lambda_lpips_spatial_gate_hinge * l_lpips_spatial_gate_hinge
                    + args.lambda_gate_lpips_sens * l_gate_lpips_sens
                    + args.lambda_gate_measured_sens * l_gate_measured_sens
                    + args.lambda_gate_mixed_sens * l_gate_mixed_sens
                    + args.lambda_rho_target * l_rho_target
                    + args.lambda_gate_send * l_gate_send)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if it % args.log_every == 0:
                print(f"[it {it}] loss={loss.item():.4f} bpp={bpp_total.item():.4f} "
                      f"bpp_y={bpp_y.item():.4f} psnr={psnr:.2f} "
                      f"mse={d_mse.item():.4f} lpips={d_lp.item():.4f} dists={d_dists.item():.4f} ce={l_align.item():.4f} "
                      f"mean_pred={l_mean_pred.item():.4f} scale_reg={l_scale_reg.item():.4f} "
                      f"distill={l_distill.item():.4f}/{l_lpips_distill.item():.4f}/{l_dists_distill.item():.4f} "
                      f"hinge={l_lpips_hinge.item():.4f}/{l_dists_hinge.item():.4f} "
                      f"spgate={l_lpips_spatial_gate_hinge.item():.4f}/{lpips_spatial_bad_mean:.4f}/{lpips_spatial_gate_weight:.3f} "
                      f"rho_t={l_rho_target.item():.4f} "
                      f"gate_send={l_gate_send.item():.4f} gt={gate_target_mean:.3f}/{gate_target_std:.3f} "
                      f"lp_sens={l_gate_lpips_sens.item():.4f} lpt={gate_lpips_sens_mean:.3f}/{gate_lpips_sens_std:.3f} "
                      f"meas_sens={l_gate_measured_sens.item():.4f} mst={gate_measured_sens_mean:.3f}/{gate_measured_sens_std:.3f} "
                      f"mix_sens={l_gate_mixed_sens.item():.4f} mixt={gate_mixed_sens_mean:.3f}/{gate_mixed_sens_std:.3f} R={lambda_R_eff:.3f} "
                      f"delta_abs={delta_abs:.5f} "
                      f"stage_delta_abs={stage_delta_abs:.5f} stage_target_abs={stage_target_abs:.5f} "
                      f"rho={rho_mean:.3f}/{rho_max:.3f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/bpp_y": bpp_y.item(),
                               "train/bpp_z": bpp_z.item(), "train/bpp_total": bpp_total.item(),
                               "train/psnr": psnr, "train/mse": d_mse.item(), "train/lpips": d_lp.item(),
                               "train/dists": d_dists.item(),
                               "train/ce_align": l_align.item(), "train/mean_pred": l_mean_pred.item(),
                               "train/scale_reg": l_scale_reg.item(), "train/distill": l_distill.item(),
                               "train/lpips_distill": l_lpips_distill.item(),
                               "train/dists_distill": l_dists_distill.item(),
                               "train/lpips_hinge": l_lpips_hinge.item(),
                               "train/dists_hinge": l_dists_hinge.item(),
                               "train/lpips_spatial_gate_hinge": l_lpips_spatial_gate_hinge.item(),
                               "train/lpips_spatial_bad_mean": lpips_spatial_bad_mean,
                               "train/lpips_spatial_gate_weight": lpips_spatial_gate_weight,
                               "train/rho_target_loss": l_rho_target.item(),
                               "train/gate_send": l_gate_send.item(),
                               "train/gate_target_mean": gate_target_mean,
                               "train/gate_target_std": gate_target_std,
                               "train/gate_lpips_sens": l_gate_lpips_sens.item(),
                               "train/gate_lpips_sens_mean": gate_lpips_sens_mean,
                               "train/gate_lpips_sens_std": gate_lpips_sens_std,
                               "train/gate_measured_sens": l_gate_measured_sens.item(),
                               "train/gate_measured_sens_mean": gate_measured_sens_mean,
                               "train/gate_measured_sens_std": gate_measured_sens_std,
                               "train/gate_mixed_sens": l_gate_mixed_sens.item(),
                               "train/gate_mixed_sens_mean": gate_mixed_sens_mean,
                               "train/gate_mixed_sens_std": gate_mixed_sens_std,
                               "train/lambda_R_eff": lambda_R_eff,
                               "pred/delta_abs_mean": delta_abs,
                               "pred/stage_delta_abs_mean": stage_delta_abs,
                               "pred/stage_target_abs_mean": stage_target_abs,
                               "pred/mu_mean": mu_mean, "pred/mu_std": mu_std,
                               "gate/rho_mean": rho_mean, "gate/rho_max": rho_max,
                               "train/lr": args.lr}, step=it)

            if val is not None and it % args.eval_every == 0:
                ab, recon = quick_eval(net, val, args.q_index)
                d_bpp = ab["ours"][0] - ab["baseline"][0]
                print(f"  [A/B it {it}] baseline bpp_y={ab['baseline'][0]:.4f} psnr={ab['baseline'][1]:.2f} "
                      f"| ours bpp_y={ab['ours'][0]:.4f} psnr={ab['ours'][1]:.2f} | Δbpp_y={d_bpp:+.4f}")
                state = {"it": it, "optimizer": opt.state_dict(),
                         "predictor_param_mode": args.predictor_param_mode}
                if use_stage_residual:
                    torch.save(net.stage_residual_predictor.state_dict(),
                               os.path.join(args.out, f"stage_residual_predictor_{it}.pt"))
                    state["stage_residual_predictor"] = net.stage_residual_predictor.state_dict()
                if use_stage_quant_gate:
                    torch.save(net.stage_quant_gate.state_dict(),
                               os.path.join(args.out, f"stage_quant_gate_{it}.pt"))
                    state["stage_quant_gate"] = net.stage_quant_gate.state_dict()
                    state["stage_rho_max"] = args.stage_rho_max
                if not use_stage_residual and not use_stage_quant_gate:
                    torch.save(net.prior_predictor.state_dict(),
                               os.path.join(args.out, f"prior_predictor_{it}.pt"))
                    state["prior_predictor"] = net.prior_predictor.state_dict()
                state = add_common_state(state)
                torch.save(state, os.path.join(args.out, "train_state.pt"))   # resume 用（上書き）
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

    state = {"it": it, "optimizer": opt.state_dict(),
             "predictor_param_mode": args.predictor_param_mode}
    if use_stage_residual:
        final_path = os.path.join(args.out, "stage_residual_predictor_final.pt")
        torch.save(net.stage_residual_predictor.state_dict(), final_path)
        state["stage_residual_predictor"] = net.stage_residual_predictor.state_dict()
    if use_stage_quant_gate:
        final_path = os.path.join(args.out, "stage_quant_gate_final.pt")
        torch.save(net.stage_quant_gate.state_dict(), final_path)
        state["stage_quant_gate"] = net.stage_quant_gate.state_dict()
        state["stage_rho_max"] = args.stage_rho_max
    if not use_stage_residual and not use_stage_quant_gate:
        final_path = os.path.join(args.out, "prior_predictor_final.pt")
        torch.save(net.prior_predictor.state_dict(), final_path)
        state["prior_predictor"] = net.prior_predictor.state_dict()
    state = add_common_state(state)
    torch.save(state, os.path.join(args.out, "train_state.pt"))
    print("done. → 同 PSNR/DISTS 帯で baseline vs ours の bpp_y を比較し、主張①を確認。")
    if use_wandb:
        wandb.save(final_path)
        wandb.finish()


if __name__ == "__main__":
    main()
