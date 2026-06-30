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
from src.utils.lpips.lpips import LPIPS as RawLPIPS
from gp_reslc.prior_predictor import (
    PriorPredictor,
    StageLatentControlEncoder,
    StageQuantGate,
    StageResidualEntropyPredictor,
    StageResidualPredictor,
    StageResidualRefiner,
    StageScaleCalibrator,
    StageTinyControlEncoder,
    ternary_expected_nll_bits,
    train_forward,
)
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


def build_net(weights, device, use_gate, rho_max, rho_min=0.5, train_predictor=True,
              rho_mode="hard", softplus_shift=2.0, softplus_tau=1.0, rho_init=1.0,
              use_stage_residual=False, use_stage_quant_gate=False, stage_rho_max=1.5,
              use_stage_entropy=False, stage_scale_log_bound=0.7,
              use_stage_scale_calibrator=False, stage_scale_calib_bound=0.25,
              use_stage_residual_refiner=False, stage_residual_refiner_bound=0.25,
              stage_residual_refiner_depth=3,
              use_tiny_control=False, control_init_prob=0.05, control_prob_one=0.08,
              control_threshold=0.5, control_hard_mode="threshold", control_topk_frac=0.06,
              use_latent_control=False, latent_control_init_prob=0.0025,
              latent_control_prob_nonzero=0.0025, latent_control_topk_frac=0.0025,
              latent_control_delta=0.05, latent_control_groups=16,
              latent_control_hard_mode="topk", latent_control_threshold=0.5):
    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(weights), strict=True)
    net.prior_predictor = PriorPredictor(net.N)
    if use_stage_residual:
        if use_stage_entropy:
            net.stage_residual_predictor = StageResidualEntropyPredictor(
                net.N, scale_log_bound=stage_scale_log_bound)
        else:
            net.stage_residual_predictor = StageResidualPredictor(net.N)
    if use_stage_quant_gate:
        net.stage_quant_gate = StageQuantGate(net.N, rho_max=stage_rho_max)
    if use_stage_scale_calibrator:
        net.stage_scale_calibrator = StageScaleCalibrator(net.N, scale_log_bound=stage_scale_calib_bound)
    if use_stage_residual_refiner:
        net.stage_residual_refiner = StageResidualRefiner(
            net.N,
            scale_log_bound=stage_residual_refiner_bound,
            depth=stage_residual_refiner_depth,
        )
    if use_tiny_control:
        net.tiny_control_encoder = StageTinyControlEncoder(
            net.N, num_q=NUM_Q, init_prob=control_init_prob, threshold=control_threshold,
            hard_mode=control_hard_mode, topk_frac=control_topk_frac)
        net.tiny_control_prob_one = float(control_prob_one)
        net.tiny_control_threshold = float(control_threshold)
        net.tiny_control_hard_mode = str(control_hard_mode)
        net.tiny_control_topk_frac = float(control_topk_frac)
    if use_latent_control:
        net.latent_control_encoder = StageLatentControlEncoder(
            net.N, num_q=NUM_Q, groups=latent_control_groups,
            init_prob=latent_control_init_prob, topk_frac=latent_control_topk_frac,
            hard_mode=latent_control_hard_mode, threshold=latent_control_threshold)
        net.latent_control_prob_nonzero = float(latent_control_prob_nonzero)
        net.latent_control_delta = float(latent_control_delta)
    net.q_embed = nn.Parameter(torch.zeros(NUM_Q, net.N, 1, 1))      # q 条件（zero-init=V1 等価）
    net.perceptual_gate = PerceptualGate(
        net.N, rho_max=rho_max, rho_min=rho_min, rho_mode=rho_mode,
        softplus_shift=softplus_shift, softplus_tau=softplus_tau, rho_init=rho_init,
    ) if use_gate else None
    net = net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    if use_stage_residual:
        for p in net.stage_residual_predictor.parameters():
            p.requires_grad_(train_predictor)
    if use_stage_quant_gate:
        for p in net.stage_quant_gate.parameters():
            p.requires_grad_(train_predictor)
    if use_stage_scale_calibrator:
        for p in net.stage_scale_calibrator.parameters():
            p.requires_grad_(train_predictor)
    if use_stage_residual_refiner:
        for p in net.stage_residual_refiner.parameters():
            p.requires_grad_(train_predictor)
    if use_tiny_control:
        for p in net.tiny_control_encoder.parameters():
            p.requires_grad_(train_predictor)
    if use_latent_control:
        for p in net.latent_control_encoder.parameters():
            p.requires_grad_(train_predictor)
    if not use_stage_residual and not use_stage_quant_gate:
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
            bpp_y = (o["bit_y"] + o.get("bit_control", o["bit_y"].new_tensor(0.0))).item() / (B * H * W)
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


def p_from_rho_target(rho_target: float, rho_max: float) -> float:
    if rho_max <= 1.0:
        return 0.5
    return max(0.05, min(0.95, 0.5 * (1.0 + (float(rho_target) - 1.0) / max(float(rho_max) - 1.0, 1e-6))))


def q_value(default: float, by_q, q: int, name: str) -> float:
    if by_q is None:
        return float(default)
    if len(by_q) == 1:
        return float(by_q[0])
    if len(by_q) != NUM_Q:
        raise ValueError(f"{name} expects 1 or {NUM_Q} values, got {len(by_q)}")
    return float(by_q[int(q)])


def weighted_spatial_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss_map = F.smooth_l1_loss(pred, target.detach(), reduction="none").mean(dim=1, keepdim=True)
    if weight.shape[-2:] != loss_map.shape[-2:]:
        weight = F.interpolate(weight, size=loss_map.shape[-2:], mode="area")
    if weight.shape[1] != 1:
        weight = weight.mean(dim=1, keepdim=True)
    weight = weight.detach().clamp_min(0.0)
    return (loss_map * weight).sum() / weight.sum().clamp_min(1.0)


def any_positive(values) -> bool:
    return values is not None and any(float(v) > 0 for v in values)


def spatial_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="area")
    if a.shape[1] != 1:
        a = a.mean(dim=1, keepdim=True)
    if b.shape[1] != 1:
        b = b.mean(dim=1, keepdim=True)
    af = a.detach().flatten(1)
    bf = b.detach().flatten(1)
    af = af - af.mean(dim=1, keepdim=True)
    bf = bf - bf.mean(dim=1, keepdim=True)
    denom = af.std(dim=1).clamp_min(1e-6) * bf.std(dim=1).clamp_min(1e-6)
    return ((af * bf).mean(dim=1) / denom).mean()


def set_module_trainable(module, enabled: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(enabled)


def load_stage_predictor_compatible(module: nn.Module, state_dict: dict):
    """Load old mean-only stage predictors into mean+scale predictors.

    For StageResidualEntropyPredictor the final conv has 2N output channels:
    the first N are mean deltas and the second N are log-scale deltas. Old
    checkpoints only have N channels, so copy them into the mean half and keep
    the scale half at zero initialization.
    """
    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    current = module.state_dict()
    patched = {k: v.clone() for k, v in current.items()}
    for key, value in state_dict.items():
        if key not in patched:
            continue
        target = patched[key]
        if target.shape == value.shape:
            patched[key] = value
            continue
        if target.ndim >= 1 and target.shape[0] == value.shape[0] * 2 and target.shape[1:] == value.shape[1:]:
            target[:value.shape[0]].copy_(value)
            patched[key] = target
    module.load_state_dict(patched, strict=True)


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


def make_gate_measured_sensitivity_target(x, base_x_hat, ours_x_hat, spatial_size, desired_mean, lpips_spatial_loss,
                                          margin=0.0, tau=1.0, edge_weight=0.0):
    """Measured local safety target for the zero-side-bit rho gate.

    High target values mean current coarsening does not increase local LPIPS
    versus frozen GLC, so those positions are safer to coarsen.
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
                                       texture_weight=0.0, edge_weight=0.0, rate_map=None,
                                       rate_weight=0.0, margin=0.0, tau=1.0):
    """Mixed local teacher for decoder-computable global rho gates.

    High target values mark positions where current coarsening is locally safe:
    it causes little L1/LPIPS-spatial degradation relative to frozen GLC and
    avoids high texture/edge regions. The target is recentered to the requested
    p-map mean, so it reallocates precision spatially without changing the
    average rate budget. The teacher is training-only; inference sends no map.
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

        if rate_map is not None and rate_weight > 0:
            rate_lr = F.interpolate(rate_map.detach(), size=spatial_size, mode="area")
            score = score - float(rate_weight) * _standardize_map(rate_lr, dims)

        safe = torch.sigmoid(-(score - float(margin)) / max(float(tau), 1e-6))
        desired = safe.new_tensor(float(desired_mean)).clamp(0.05, 0.95)
        safe = safe - safe.mean(dims, keepdim=True) + desired
        return safe.clamp(0.05, 0.95)


def make_gate_rdo_sensitivity_target(
    x,
    base_x_hat,
    ours_x_hat,
    spatial_size,
    desired_mean,
    lpips_spatial_loss,
    base_rate_map,
    ours_rate_map,
    l1_weight=1.0,
    lpips_weight=1.0,
    saved_rate_weight=1.0,
    texture_weight=0.0,
    edge_weight=0.0,
    margin=0.0,
    tau=1.0,
):
    """Benefit-cost teacher for safe residual coarsening.

    High target values mean that the current lower-precision residual coding
    saves local estimated y bits while causing little local L1/LPIPS damage
    relative to the frozen GLC reconstruction.  The target is recentered to the
    requested mean so it changes allocation rather than acting as an implicit
    global rho sweep.  It is training-only and sends no side map.
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

        base_rate = F.interpolate(base_rate_map.detach(), size=spatial_size, mode="area")
        ours_rate = F.interpolate(ours_rate_map.detach(), size=spatial_size, mode="area")
        saved_rate = (base_rate - ours_rate).clamp_min(0.0)

        score = l1_delta.new_zeros(l1_delta.shape)
        if l1_weight > 0:
            score = score + float(l1_weight) * _standardize_map(l1_delta, dims)
        if lpips_weight > 0:
            score = score + float(lpips_weight) * _standardize_map(lp_delta, dims)
        if saved_rate_weight > 0:
            score = score - float(saved_rate_weight) * _standardize_map(saved_rate, dims)

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
        return safe.clamp(0.05, 0.95), saved_rate.detach(), score.detach()


def make_control_protect_target(safe_target: torch.Tensor, control_shape, desired_mean: float) -> torch.Tensor:
    """Convert a high=safe coarsening map into a sparse high=protect control map."""
    with torch.no_grad():
        unsafe = 1.0 - safe_target.detach()
        if unsafe.shape[1] != 1:
            unsafe = unsafe.mean(dim=1, keepdim=True)
        if len(control_shape) >= 4:
            target_channels = int(control_shape[1])
            control_spatial_size = tuple(control_shape[-2:])
        else:
            target_channels = 1
            control_spatial_size = tuple(control_shape)
        protect = F.interpolate(unsafe, size=control_spatial_size, mode="area")
        if target_channels > 1:
            protect = protect.expand(-1, target_channels, -1, -1)
        frac = max(0.0, min(float(desired_mean), 0.35))
        target = protect.new_full(protect.shape, 0.005)
        flat = protect.flatten(1)
        k = int(round(frac * flat.shape[1]))
        if k <= 0:
            return target
        k = min(k, flat.shape[1])
        idx = torch.topk(flat, k=k, dim=1).indices
        target_flat = target.flatten(1)
        target_flat.scatter_(1, idx, 0.95)
        return target_flat.reshape_as(target)


class LPIPSSpatialLoss(torch.nn.Module):
    """LPIPS wrapper that keeps the spatial map for local gate teachers."""

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
    ap.add_argument("--lambda_R_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q rate weights. Provide 1 or 4 values for q0..q3.")
    ap.add_argument("--lambda_d", type=float, default=1.0)
    ap.add_argument("--lambda_d_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q MSE weights. Provide 1 or 4 values.")
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_lpips_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q LPIPS weights. Provide 1 or 4 values.")
    ap.add_argument("--train_lpips_net", choices=["glc_vgg", "alex"], default="glc_vgg",
                    help="LPIPS model used for training loss. eval_metrics.py uses Alex LPIPS.")
    ap.add_argument("--lambda_dists", type=float, default=0.0,
                    help="DISTS perceptual loss weight, computed on [0,1] images")
    ap.add_argument("--lambda_dists_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q DISTS weights. Provide 1 or 4 values.")
    ap.add_argument("--lambda_base_l1", type=float, default=0.0,
                    help="Distill ours toward frozen GLC reconstruction with image-space L1")
    ap.add_argument("--lambda_base_lpips", type=float, default=0.0,
                    help="Distill ours toward frozen GLC reconstruction with the training LPIPS backbone")
    ap.add_argument("--lambda_dists_distill", type=float, default=0.0,
                    help="Distill ours toward frozen GLC reconstruction with DISTS")
    ap.add_argument("--lambda_lpips_hinge", type=float, default=0.0,
                    help="Penalize LPIPS worse than frozen GLC baseline by more than --lpips_hinge_margin")
    ap.add_argument("--lambda_lpips_hinge_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q LPIPS hinge weights. Provide 1 or 4 values.")
    ap.add_argument("--lambda_dists_hinge", type=float, default=0.0,
                    help="Penalize DISTS worse than frozen GLC baseline by more than --dists_hinge_margin")
    ap.add_argument("--lambda_dists_hinge_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q DISTS hinge weights. Provide 1 or 4 values.")
    ap.add_argument("--lpips_hinge_margin", type=float, default=0.0)
    ap.add_argument("--dists_hinge_margin", type=float, default=0.0)
    ap.add_argument("--base_distill_until", type=int, default=0,
                    help="Apply baseline distillation while it < this value; 0 keeps it active")
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_mean_pred", type=float, default=0.0,
                    help="Smooth-L1 between corrected prior mean or latent residual prediction and y in quantized space")
    ap.add_argument("--lambda_stage_mean_norm", type=float, default=0.0,
                    help="Normalized stage residual-explanation loss; strengthens gp_mu_stage learning when raw residual targets are small")
    ap.add_argument("--lambda_mean_pred_safe", type=float, default=0.0,
                    help="Mixed-teacher weighted mean prediction loss; learns decoder-computable residual means mainly in safe/predictable regions")
    ap.add_argument("--lambda_mean_pred_safe_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q safe mean prediction weights. Provide 1 or 4 values.")
    ap.add_argument("--lambda_predictor_unsafe_delta", type=float, default=0.0,
                    help="Penalize predictor deltas in mixed-teacher unsafe regions so mean prediction only removes generator-predictable residuals")
    ap.add_argument("--lambda_stage_delta_abs", type=float, default=0.0,
                    help="L1 penalty on stage residual mean corrections to prevent bound-saturated global shifts")
    ap.add_argument("--lambda_rho_floor", type=float, default=0.0,
                    help="Penalty for rho < 1 so the gate does not spend extra bits")
    ap.add_argument("--no_gate", action="store_true", help="知覚ゲートを使わない（q 条件化のみ）")
    ap.add_argument("--freeze_predictor", action="store_true", help="P_thetaを凍結し、q_embed/gateのみ学習する")
    ap.add_argument("--freeze_q_embed", action="store_true", help="q条件embeddingを凍結し、gate/predictorだけを学習する")
    ap.add_argument("--freeze_gate", action="store_true",
                    help="Freeze perceptual gate parameters after loading; useful for predictor-only fine-tunes from a gate checkpoint.")
    ap.add_argument("--freeze_stage_residual_module", action="store_true",
                    help="Freeze StageResidualPredictor while training other modules.")
    ap.add_argument("--freeze_stage_quant_module", action="store_true",
                    help="Freeze StageQuantGate while training other modules.")
    ap.add_argument("--freeze_tiny_control_module", action="store_true",
                    help="Freeze StageTinyControlEncoder while training other modules.")
    ap.add_argument("--predictor_train_start", type=int, default=0,
                    help="Iteration at which predictor/stage-residual parameters become trainable. Use >0 for gate-first staged training.")
    ap.add_argument("--gate_train_start", type=int, default=0,
                    help="Iteration at which perceptual gate parameters become trainable.")
    ap.add_argument("--q_embed_train_start", type=int, default=0,
                    help="Iteration at which q embedding becomes trainable.")
    ap.add_argument("--q_choices", type=int, nargs="+", default=None,
                    help="Restrict training to specific q indexes, e.g. --q_choices 0 1. Default samples all 0..3.")
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
    ap.add_argument("--lambda_rho_target_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q rho-target penalty weights. Provide 1 or 4 values.")
    ap.add_argument("--rho_target", type=float, default=1.0)
    ap.add_argument("--rho_target_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q rho targets. Provide either 1 value or 4 values for q0..q3.")
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
    ap.add_argument("--lambda_gate_measured_sens", type=float, default=0.0,
                    help="BCE teacher from measured local LPIPS delta caused by current coarsening")
    ap.add_argument("--gate_measured_sens_until", type=int, default=0,
                    help="Apply measured sensitivity teacher while it < this value; 0 keeps it active")
    ap.add_argument("--gate_measured_sens_tau", type=float, default=1.0)
    ap.add_argument("--gate_measured_sens_margin", type=float, default=0.0)
    ap.add_argument("--gate_measured_sens_edge_weight", type=float, default=0.0)
    ap.add_argument("--lambda_gate_mixed_sens", type=float, default=0.0,
                    help="BCE teacher from mixed L1/LPIPS/texture/edge local safety caused by current coarsening")
    ap.add_argument("--lambda_gate_mixed_sens_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q mixed-teacher BCE weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_mixed_sens_until", type=int, default=0,
                    help="Apply mixed sensitivity teacher while it < this value; 0 keeps it active")
    ap.add_argument("--gate_mixed_sens_tau", type=float, default=1.0)
    ap.add_argument("--gate_mixed_sens_margin", type=float, default=0.0)
    ap.add_argument("--gate_mixed_l1_weight", type=float, default=1.0)
    ap.add_argument("--gate_mixed_lpips_weight", type=float, default=1.0)
    ap.add_argument("--gate_mixed_texture_weight", type=float, default=0.0)
    ap.add_argument("--gate_mixed_edge_weight", type=float, default=0.0)
    ap.add_argument("--gate_mixed_rate_weight", type=float, default=0.0,
                    help="Weight high estimated-bit regions in the mixed teacher; favors coarsening residuals that actually cost bits.")
    ap.add_argument("--gate_mixed_l1_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q mixed teacher L1 weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_mixed_lpips_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q mixed teacher LPIPS-spatial weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_mixed_texture_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q mixed teacher texture weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_mixed_edge_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q mixed teacher edge-protection weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_mixed_rate_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q mixed teacher rate-potential weights. Provide 1 or 4 values.")
    ap.add_argument("--lambda_gate_rdo_sens", type=float, default=0.0,
                    help="BCE teacher from local saved-rate versus perceptual-damage benefit-cost maps.")
    ap.add_argument("--lambda_gate_rdo_sens_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q RDO-teacher BCE weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_rdo_sens_until", type=int, default=0,
                    help="Apply RDO sensitivity teacher while it < this value; 0 keeps it active")
    ap.add_argument("--gate_rdo_sens_tau", type=float, default=1.0)
    ap.add_argument("--gate_rdo_sens_margin", type=float, default=0.0)
    ap.add_argument("--gate_rdo_l1_weight", type=float, default=1.0)
    ap.add_argument("--gate_rdo_lpips_weight", type=float, default=1.0)
    ap.add_argument("--gate_rdo_saved_rate_weight", type=float, default=1.0)
    ap.add_argument("--gate_rdo_texture_weight", type=float, default=0.0)
    ap.add_argument("--gate_rdo_edge_weight", type=float, default=0.0)
    ap.add_argument("--gate_rdo_l1_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q RDO teacher L1-damage weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_rdo_lpips_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q RDO teacher LPIPS-damage weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_rdo_saved_rate_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q RDO teacher saved-rate weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_rdo_texture_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q RDO teacher texture-protection weights. Provide 1 or 4 values.")
    ap.add_argument("--gate_rdo_edge_weight_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q RDO teacher edge-protection weights. Provide 1 or 4 values.")
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
            "stage_residual_entropy_quant_gate",
            "stage_residual_entropy_quant_gate_scale_calib",
            "stage_residual_entropy_quant_gate_residual_refiner",
            "stage_residual_quant_gate_control",
            "stage_residual_entropy_quant_gate_control",
            "stage_residual_entropy_quant_gate_latent_control",
        ],
        default="scale_mean",
    )
    ap.add_argument("--predictor_delta_bound", type=float, default=0.0,
                    help="Bound predictor delta by bound*tanh(delta/bound); 0 disables")
    ap.add_argument("--stage_rho_max", type=float, default=1.5,
                    help="Maximum rho for predictor_param_mode=stage_quant_gate")
    ap.add_argument("--stage_scale_log_bound", type=float, default=0.7,
                    help="Bound for decoder-computable residual scale multiplier in stage_residual_entropy_quant_gate")
    ap.add_argument("--stage_scale_calib_bound", type=float, default=0.25,
                    help="Bound for extra decoder-computable entropy-scale calibration")
    ap.add_argument("--stage_residual_refiner_bound", type=float, default=0.25,
                    help="Scale-log bound for the additive stage residual refiner")
    ap.add_argument("--stage_residual_refiner_depth", type=int, default=3,
                    help="DepthConvBlock count before the zero-init refiner head")
    ap.add_argument("--control_init_prob", type=float, default=0.05,
                    help="Initial probability for the tiny paid protection-control symbols")
    ap.add_argument("--control_prob_one", type=float, default=0.08,
                    help="Fixed Bernoulli prior P(control=1) used for control entropy cost and real codec")
    ap.add_argument("--control_threshold", type=float, default=0.5,
                    help="Hard threshold for turning control probabilities into transmitted binary symbols")
    ap.add_argument("--control_hard_mode", choices=["threshold", "topk"], default="threshold",
                    help="How control probabilities are converted into transmitted binary symbols")
    ap.add_argument("--control_topk_frac", type=float, default=0.06,
                    help="Fraction of z-resolution control symbols sent as 1 when --control_hard_mode topk")
    ap.add_argument("--lambda_control_protect", type=float, default=0.0,
                    help="BCE teacher for the tiny paid protection-control stream")
    ap.add_argument("--lambda_control_protect_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q control protection weights. Provide 1 or 4 values.")
    ap.add_argument("--control_target_mean", type=float, default=0.06,
                    help="Target mean for sparse protection-control symbols")
    ap.add_argument("--control_target_mean_by_q", type=float, nargs="+", default=None,
                    help="Optional per-q control target means. Provide 1 or 4 values.")
    ap.add_argument("--latent_control_init_prob", type=float, default=0.0025,
                    help="Initial probability for signed latent residual-control nonzero symbols")
    ap.add_argument("--latent_control_prob_nonzero", type=float, default=0.0025,
                    help="Fixed ternary prior P(symbol!=0) used for latent-control entropy cost and real codec")
    ap.add_argument("--latent_control_topk_frac", type=float, default=0.0025,
                    help="Fraction of z-resolution signed latent-control symbols transmitted as nonzero")
    ap.add_argument("--latent_control_hard_mode", type=str, default="topk",
                    choices=["topk", "threshold"],
                    help="Hard symbol selection for signed latent-control stream")
    ap.add_argument("--latent_control_threshold", type=float, default=0.5,
                    help="Nonzero probability threshold when --latent_control_hard_mode=threshold")
    ap.add_argument("--latent_control_delta", type=float, default=0.05,
                    help="Signed latent correction magnitude applied before synthesis transform")
    ap.add_argument("--latent_control_groups", type=int, default=16,
                    help="Channel groups per four-part stage for signed latent-control symbols")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=8)
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
    use_gate = not args.no_gate
    use_wandb = _WANDB and not args.no_wandb
    q_choices = args.q_choices if args.q_choices is not None else list(range(NUM_Q))
    bad_q = [q for q in q_choices if q < 0 or q >= NUM_Q]
    if bad_q:
        raise ValueError(f"--q_choices must be in [0,{NUM_Q - 1}], got {bad_q}")
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   entity=args.wandb_entity, config=vars(args), mode=args.wandb_mode,
                   id=args.wandb_id, resume="allow" if args.resume else None)
        wandb.summary["q_choices"] = q_choices

    use_stage_residual = args.predictor_param_mode in {
        "stage_latent_residual",
        "stage_residual_quant_gate",
        "stage_residual_entropy_quant_gate",
        "stage_residual_entropy_quant_gate_scale_calib",
        "stage_residual_entropy_quant_gate_residual_refiner",
        "stage_residual_quant_gate_control",
        "stage_residual_entropy_quant_gate_control",
        "stage_residual_entropy_quant_gate_latent_control",
    }
    use_stage_quant_gate = args.predictor_param_mode in {
        "stage_quant_gate",
        "stage_residual_quant_gate",
        "stage_residual_entropy_quant_gate",
        "stage_residual_entropy_quant_gate_scale_calib",
        "stage_residual_entropy_quant_gate_residual_refiner",
        "stage_residual_quant_gate_control",
        "stage_residual_entropy_quant_gate_control",
        "stage_residual_entropy_quant_gate_latent_control",
    }
    use_stage_entropy = args.predictor_param_mode in {
        "stage_residual_entropy_quant_gate",
        "stage_residual_entropy_quant_gate_scale_calib",
        "stage_residual_entropy_quant_gate_residual_refiner",
        "stage_residual_entropy_quant_gate_control",
        "stage_residual_entropy_quant_gate_latent_control",
    }
    use_stage_scale_calibrator = args.predictor_param_mode == "stage_residual_entropy_quant_gate_scale_calib"
    use_stage_residual_refiner = args.predictor_param_mode == "stage_residual_entropy_quant_gate_residual_refiner"
    use_tiny_control = args.predictor_param_mode in {
        "stage_residual_quant_gate_control",
        "stage_residual_entropy_quant_gate_control",
    }
    use_latent_control = args.predictor_param_mode == "stage_residual_entropy_quant_gate_latent_control"
    net = build_net(
        args.glc_weights, device, use_gate, args.rho_max, args.gate_rho_min,
        train_predictor=not args.freeze_predictor, rho_mode=args.gate_rho_mode,
        softplus_shift=args.gate_softplus_shift, softplus_tau=args.gate_softplus_tau,
        rho_init=args.gate_rho_init,
        use_stage_residual=use_stage_residual, use_stage_quant_gate=use_stage_quant_gate,
        stage_rho_max=args.stage_rho_max,
        use_stage_entropy=use_stage_entropy,
        stage_scale_log_bound=args.stage_scale_log_bound,
        use_stage_scale_calibrator=use_stage_scale_calibrator,
        stage_scale_calib_bound=args.stage_scale_calib_bound,
        use_stage_residual_refiner=use_stage_residual_refiner,
        stage_residual_refiner_bound=args.stage_residual_refiner_bound,
        stage_residual_refiner_depth=args.stage_residual_refiner_depth,
        use_tiny_control=use_tiny_control,
        control_init_prob=args.control_init_prob,
        control_prob_one=args.control_prob_one,
        control_threshold=args.control_threshold,
        control_hard_mode=args.control_hard_mode,
        control_topk_frac=args.control_topk_frac,
        use_latent_control=use_latent_control,
        latent_control_init_prob=args.latent_control_init_prob,
        latent_control_prob_nonzero=args.latent_control_prob_nonzero,
        latent_control_topk_frac=args.latent_control_topk_frac,
        latent_control_delta=args.latent_control_delta,
        latent_control_groups=args.latent_control_groups,
        latent_control_hard_mode=args.latent_control_hard_mode,
        latent_control_threshold=args.latent_control_threshold,
    )
    net.predictor_param_mode = args.predictor_param_mode
    net.predictor_delta_bound = args.predictor_delta_bound
    if args.freeze_q_embed:
        net.q_embed.requires_grad_(False)
    if use_stage_residual and use_stage_quant_gate and use_tiny_control:
        predictor_module = nn.ModuleList([
            net.stage_residual_predictor,
            net.stage_quant_gate,
            net.tiny_control_encoder,
        ])
    elif use_stage_residual and use_stage_quant_gate and use_latent_control:
        predictor_module = nn.ModuleList([
            net.stage_residual_predictor,
            net.stage_quant_gate,
            net.latent_control_encoder,
        ])
    elif use_stage_residual and use_stage_quant_gate:
        modules = [net.stage_residual_predictor, net.stage_quant_gate]
        if use_stage_scale_calibrator:
            modules.append(net.stage_scale_calibrator)
        if use_stage_residual_refiner:
            modules.append(net.stage_residual_refiner)
        predictor_module = nn.ModuleList(modules)
    elif use_stage_residual:
        predictor_module = net.stage_residual_predictor
    elif use_stage_quant_gate:
        predictor_module = net.stage_quant_gate
    else:
        predictor_module = net.prior_predictor
    if args.freeze_gate and net.perceptual_gate is not None:
        for p in net.perceptual_gate.parameters():
            p.requires_grad_(False)
    if args.freeze_stage_residual_module and use_stage_residual:
        set_module_trainable(net.stage_residual_predictor, False)
    if args.freeze_stage_quant_module and use_stage_quant_gate:
        set_module_trainable(net.stage_quant_gate, False)
    if args.freeze_tiny_control_module and use_tiny_control:
        set_module_trainable(net.tiny_control_encoder, False)
    net.train()
    if args.train_lpips_net == "alex":
        if not _LPIPS_LIB:
            raise RuntimeError("--train_lpips_net alex requires the lpips package")
        lpips_loss = lpips_lib.LPIPS(net="alex").to(device).eval()
    else:
        lpips_loss = LPIPSLoss(get_lpips_model()).to(device).eval()
    for p in lpips_loss.parameters():
        p.requires_grad_(False)
    need_dists_model = any(v > 0 for v in (
        args.lambda_dists,
        args.lambda_dists_distill,
        args.lambda_dists_hinge,
    )) or any_positive(args.lambda_dists_by_q) or any_positive(args.lambda_dists_hinge_by_q)
    dists_loss = DISTS().to(device).eval() if need_dists_model else None
    if dists_loss is not None:
        for p in dists_loss.parameters():
            p.requires_grad_(False)
    lpips_spatial_loss = None
    if (args.lambda_gate_measured_sens > 0 or args.lambda_gate_mixed_sens > 0
            or any_positive(args.lambda_gate_mixed_sens_by_q)
            or args.lambda_gate_rdo_sens > 0
            or any_positive(args.lambda_gate_rdo_sens_by_q)
            or args.lambda_control_protect > 0
            or any_positive(args.lambda_control_protect_by_q)):
        lpips_spatial_loss = LPIPSSpatialLoss(RawLPIPS(net="alex", spatial=True, verbose=False)).to(device).eval()
        for p in lpips_spatial_loss.parameters():
            p.requires_grad_(False)

    loader = DataLoader(CropFolder(args.data, 256), batch_size=args.bs, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val = None
    if args.val:
        vds = CropFolder(args.val, 256)
        val = torch.stack([vds[i] for i in range(min(8, len(vds)))]).to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"学習対象: {sum(p.numel() for p in params)/1e6:.2f} M | gate={use_gate} | predictor_train={not args.freeze_predictor} | q_embed_train={not args.freeze_q_embed} | starts P/G/Q={args.predictor_train_start}/{args.gate_train_start}/{args.q_embed_train_start}")

    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if "prior_predictor" in ck:
            net.prior_predictor.load_state_dict(ck["prior_predictor"])
        if use_stage_residual and "stage_residual_predictor" in ck:
            load_stage_predictor_compatible(net.stage_residual_predictor, ck["stage_residual_predictor"])
        if use_stage_quant_gate and "stage_quant_gate" in ck:
            net.stage_quant_gate.load_state_dict(ck["stage_quant_gate"])
        if use_stage_scale_calibrator and "stage_scale_calibrator" in ck:
            net.stage_scale_calibrator.load_state_dict(ck["stage_scale_calibrator"])
        if use_stage_residual_refiner and "stage_residual_refiner" in ck:
            net.stage_residual_refiner.load_state_dict(ck["stage_residual_refiner"])
        if use_tiny_control and "tiny_control_encoder" in ck:
            net.tiny_control_encoder.load_state_dict(ck["tiny_control_encoder"])
            net.tiny_control_prob_one = float(ck.get("control_prob_one", args.control_prob_one))
            net.tiny_control_encoder.threshold = float(ck.get("control_threshold", args.control_threshold))
            net.tiny_control_encoder.hard_mode = str(ck.get("control_hard_mode", args.control_hard_mode))
            net.tiny_control_encoder.topk_frac = float(ck.get("control_topk_frac", args.control_topk_frac))
            net.tiny_control_threshold = float(ck.get("control_threshold", args.control_threshold))
            net.tiny_control_hard_mode = str(ck.get("control_hard_mode", args.control_hard_mode))
            net.tiny_control_topk_frac = float(ck.get("control_topk_frac", args.control_topk_frac))
        if use_latent_control and "latent_control_encoder" in ck:
            net.latent_control_encoder.load_state_dict(ck["latent_control_encoder"])
            net.latent_control_prob_nonzero = float(
                ck.get("latent_control_prob_nonzero", args.latent_control_prob_nonzero))
            net.latent_control_delta = float(ck.get("latent_control_delta", args.latent_control_delta))
            net.latent_control_encoder.topk_frac = float(
                ck.get("latent_control_topk_frac", args.latent_control_topk_frac))
            net.latent_control_encoder.hard_mode = str(
                ck.get("latent_control_hard_mode", args.latent_control_hard_mode))
            net.latent_control_encoder.threshold = float(
                ck.get("latent_control_threshold", args.latent_control_threshold))
        with torch.no_grad():
            net.q_embed.copy_(ck["q_embed"].to(device))
        if use_gate and ck.get("perceptual_gate") is not None:
            net.perceptual_gate.load_state_dict(ck["perceptual_gate"])
        if not args.resume_weights_only and bool(ck.get("use_gate", False)) == use_gate and "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
            start_it = ck.get("it", 0)
            print(f"[resume] {args.resume} から再開（it={start_it}, gate={use_gate}）")
        else:
            start_it = 0
            print(f"[resume_weights_only] {args.resume} から重みのみロード（fresh optimizer, gate={use_gate}）")

    it = start_it
    while it < args.iters:
        for x in loader:
            x = x.to(device)
            q = random.choice(q_choices)                              # (b) 指定 q 集合からサンプル
            predictor_active = (not args.freeze_predictor) and it >= args.predictor_train_start
            gate_active = use_gate and (not args.freeze_gate) and it >= args.gate_train_start
            q_embed_active = (not args.freeze_q_embed) and it >= args.q_embed_train_start
            set_module_trainable(predictor_module, predictor_active)
            set_module_trainable(net.perceptual_gate if use_gate else None, gate_active)
            if args.freeze_stage_residual_module and use_stage_residual:
                set_module_trainable(net.stage_residual_predictor, False)
            if args.freeze_stage_quant_module and use_stage_quant_gate:
                set_module_trainable(net.stage_quant_gate, False)
            if args.freeze_tiny_control_module and use_tiny_control:
                set_module_trainable(net.tiny_control_encoder, False)
            net.q_embed.requires_grad_(q_embed_active)
            lambda_R_q = q_value(args.lambda_R, args.lambda_R_by_q, q, "--lambda_R_by_q")
            lambda_d_q = q_value(args.lambda_d, args.lambda_d_by_q, q, "--lambda_d_by_q")
            lambda_lpips_q = q_value(args.lambda_lpips, args.lambda_lpips_by_q, q, "--lambda_lpips_by_q")
            lambda_dists_q = q_value(args.lambda_dists, args.lambda_dists_by_q, q, "--lambda_dists_by_q")
            lambda_lpips_hinge_q = q_value(args.lambda_lpips_hinge, args.lambda_lpips_hinge_by_q, q, "--lambda_lpips_hinge_by_q")
            lambda_dists_hinge_q = q_value(args.lambda_dists_hinge, args.lambda_dists_hinge_by_q, q, "--lambda_dists_hinge_by_q")
            lambda_mean_pred_safe_q = q_value(args.lambda_mean_pred_safe, args.lambda_mean_pred_safe_by_q, q, "--lambda_mean_pred_safe_by_q")
            lambda_rho_target_q = q_value(args.lambda_rho_target, args.lambda_rho_target_by_q, q, "--lambda_rho_target_by_q")
            lambda_gate_mixed_sens_q = q_value(args.lambda_gate_mixed_sens, args.lambda_gate_mixed_sens_by_q, q, "--lambda_gate_mixed_sens_by_q")
            lambda_gate_rdo_sens_q = q_value(args.lambda_gate_rdo_sens, args.lambda_gate_rdo_sens_by_q, q, "--lambda_gate_rdo_sens_by_q")
            lambda_control_protect_q = q_value(args.lambda_control_protect, args.lambda_control_protect_by_q, q, "--lambda_control_protect_by_q")
            rho_target_q = q_value(args.rho_target, args.rho_target_by_q, q, "--rho_target_by_q")
            teacher_rho_max = args.stage_rho_max if use_stage_quant_gate else args.rho_max
            control_target_mean_q = q_value(args.control_target_mean, args.control_target_mean_by_q, q, "--control_target_mean_by_q")
            gate_mixed_l1_q = q_value(args.gate_mixed_l1_weight, args.gate_mixed_l1_weight_by_q, q, "--gate_mixed_l1_weight_by_q")
            gate_mixed_lpips_q = q_value(args.gate_mixed_lpips_weight, args.gate_mixed_lpips_weight_by_q, q, "--gate_mixed_lpips_weight_by_q")
            gate_mixed_texture_q = q_value(args.gate_mixed_texture_weight, args.gate_mixed_texture_weight_by_q, q, "--gate_mixed_texture_weight_by_q")
            gate_mixed_edge_q = q_value(args.gate_mixed_edge_weight, args.gate_mixed_edge_weight_by_q, q, "--gate_mixed_edge_weight_by_q")
            gate_mixed_rate_q = q_value(args.gate_mixed_rate_weight, args.gate_mixed_rate_weight_by_q, q, "--gate_mixed_rate_weight_by_q")
            gate_rdo_l1_q = q_value(args.gate_rdo_l1_weight, args.gate_rdo_l1_weight_by_q, q, "--gate_rdo_l1_weight_by_q")
            gate_rdo_lpips_q = q_value(args.gate_rdo_lpips_weight, args.gate_rdo_lpips_weight_by_q, q, "--gate_rdo_lpips_weight_by_q")
            gate_rdo_saved_rate_q = q_value(args.gate_rdo_saved_rate_weight, args.gate_rdo_saved_rate_weight_by_q, q, "--gate_rdo_saved_rate_weight_by_q")
            gate_rdo_texture_q = q_value(args.gate_rdo_texture_weight, args.gate_rdo_texture_weight_by_q, q, "--gate_rdo_texture_weight_by_q")
            gate_rdo_edge_q = q_value(args.gate_rdo_edge_weight, args.gate_rdo_edge_weight_by_q, q, "--gate_rdo_edge_weight_by_q")
            gate = net.perceptual_gate if use_gate else None
            out = train_forward(net, x, q, use_predictor=True,
                                gate=gate, q_shift=net.q_embed[q:q + 1],
                                predictor_param_mode=args.predictor_param_mode,
                                predictor_delta_bound=args.predictor_delta_bound)
            B, _, H, W = x.shape
            bpp_y = out["bit_y"] / (B * H * W)
            bpp_control = out.get("bit_control", bpp_y.new_tensor(0.0)) / (B * H * W)
            bpp_rate = bpp_y + bpp_control
            bpp_z = torch.as_tensor(out["bit_z"], device=x.device, dtype=bpp_y.dtype) / (H * W)
            bpp_total = bpp_rate + bpp_z
            d_mse = cal_mse_Loss(x, out["x_hat"]).mean()
            d_lp = lpips_loss(out["x_hat"], x).mean()
            base_distill_active = any(v > 0 for v in (
                args.lambda_base_l1,
                args.lambda_base_lpips,
                args.lambda_dists_distill,
                lambda_lpips_hinge_q,
                lambda_dists_hinge_q,
            )) and (args.base_distill_until <= 0 or it < args.base_distill_until)
            x_hat_base = None
            base_out_for_teacher = None
            if base_distill_active:
                with torch.no_grad():
                    base_out = train_forward(net, x, q, use_predictor=False, gate=None, q_shift=None,
                                             predictor_param_mode=args.predictor_param_mode,
                                             predictor_delta_bound=args.predictor_delta_bound)
                    base_out_for_teacher = base_out
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
            l_base_dists = bpp_y.new_tensor(0.0)
            l_lpips_hinge = bpp_y.new_tensor(0.0)
            l_dists_hinge = bpp_y.new_tensor(0.0)
            if dists_loss is not None:
                x01 = from_minus1_1_to_0_1(x.clamp(-1, 1))
                xhat01 = from_minus1_1_to_0_1(out["x_hat"].clamp(-1, 1))
                d_dists = dists_loss(x01, xhat01).mean()
            else:
                x01 = None
                xhat01 = None
                d_dists = bpp_y.new_tensor(0.0)
            if x_hat_base is not None:
                if args.lambda_dists_distill > 0:
                    l_base_dists = dists_loss(
                        from_minus1_1_to_0_1(out["x_hat"].clamp(-1, 1)),
                        from_minus1_1_to_0_1(x_hat_base),
                    ).mean()
                if lambda_lpips_hinge_q > 0:
                    with torch.no_grad():
                        base_lp = lpips_loss(x_hat_base, x).mean()
                    l_lpips_hinge = F.relu(d_lp - base_lp - args.lpips_hinge_margin)
                if lambda_dists_hinge_q > 0:
                    with torch.no_grad():
                        base_dists = dists_loss(
                            from_minus1_1_to_0_1(x_hat_base),
                            from_minus1_1_to_0_1(x),
                        ).mean()
                    l_dists_hinge = F.relu(d_dists - base_dists - args.dists_hinge_margin)
            psnr = 10 * math.log10(4.0 / max(d_mse.item(), 1e-10))
            delta_abs = out["delta_params"].detach().abs().mean().item() if out["delta_params"] is not None else 0.0
            mu_mean = out["mu_pred"].detach().mean().item() if out["mu_pred"] is not None else 0.0
            mu_std = out["mu_pred"].detach().std().item() if out["mu_pred"] is not None else 0.0
            latent_pred_abs = out["latent_pred_scaled"].detach().abs().mean().item() if out.get("latent_pred_scaled") is not None else 0.0
            target_residual_abs = 0.0
            stage_delta_abs_tensor = out["stage_delta_abs"] if out.get("stage_delta_abs") is not None else None
            stage_delta_abs = stage_delta_abs_tensor.detach().item() if stage_delta_abs_tensor is not None else 0.0
            stage_target_abs = out["stage_target_abs"].detach().item() if out.get("stage_target_abs") is not None else 0.0
            stage_mean_pred_loss = out["stage_mean_pred_loss"] if out.get("stage_mean_pred_loss") is not None else None
            stage_mean_pred_norm_loss = out["stage_mean_pred_norm_loss"] if out.get("stage_mean_pred_norm_loss") is not None else None
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
            if out.get("control_symbols") is not None:
                control_mean = out["control_symbols"].detach().mean().item()
                control_abs_mean = out["control_symbols"].detach().abs().mean().item()
                control_prob_mean = out["control_prob"].detach().mean().item()
                control_prob_max = out["control_prob"].detach().max().item()
            else:
                control_mean = control_abs_mean = control_prob_mean = control_prob_max = 0.0
            if args.lambda_align > 0 and out.get("mu_pred") is not None:
                idx_gt = calculate_vqgan_results(x, net.vqgan)["idx_gt"]
                l_align = cal_ce_Loss(net.code_pred_loss(out["mu_pred"]), idx_gt).mean()
            else:
                l_align = bpp_y.new_tensor(0.0)
            if args.lambda_mean_pred > 0:
                with torch.no_grad():
                    base_q_enc, _, _, base_mean = net.separate_prior(out["params_base"])
                    target_y = out["y"].detach() * base_q_enc.detach()
                if args.predictor_param_mode in {
                    "stage_latent_residual",
                    "stage_residual_quant_gate",
                    "stage_residual_entropy_quant_gate",
                    "stage_residual_entropy_quant_gate_scale_calib",
                    "stage_residual_entropy_quant_gate_residual_refiner",
                    "stage_residual_quant_gate_control",
                    "stage_residual_entropy_quant_gate_control",
                    "stage_residual_entropy_quant_gate_latent_control",
                }:
                    l_mean_pred = stage_mean_pred_loss if stage_mean_pred_loss is not None else bpp_y.new_tensor(0.0)
                    latent_pred_abs = stage_delta_abs
                    target_residual_abs = stage_target_abs
                elif args.predictor_param_mode == "latent_residual":
                    target_residual_pred = target_y - base_mean.detach()
                    l_mean_pred = F.smooth_l1_loss(out["latent_pred_scaled"], target_residual_pred)
                    target_residual_abs = target_residual_pred.detach().abs().mean().item()
                else:
                    _, _, _, corrected_mean = net.separate_prior(out["params_after"])
                    l_mean_pred = F.smooth_l1_loss(corrected_mean, target_y)
                    target_residual_abs = target_y.detach().abs().mean().item()
            else:
                l_mean_pred = bpp_y.new_tensor(0.0)

            if args.lambda_stage_delta_abs > 0 and stage_delta_abs_tensor is not None:
                l_stage_delta_abs = stage_delta_abs_tensor
            else:
                l_stage_delta_abs = bpp_y.new_tensor(0.0)

            if args.lambda_stage_mean_norm > 0 and stage_mean_pred_norm_loss is not None:
                l_stage_mean_norm = stage_mean_pred_norm_loss
            else:
                l_stage_mean_norm = bpp_y.new_tensor(0.0)

            if args.lambda_rho_floor > 0 and out.get("gate_rho") is not None:
                l_rho_floor = F.relu(1.0 - out["gate_rho"]).mean()
            else:
                l_rho_floor = bpp_y.new_tensor(0.0)
            target_active = args.rho_target_until <= 0 or it < args.rho_target_until
            if lambda_rho_target_q > 0 and target_active and out.get("gate_rho") is not None:
                l_rho_target = F.relu(out["gate_rho"].new_tensor(rho_target_q) - out["gate_rho"].mean())
            else:
                l_rho_target = bpp_y.new_tensor(0.0)
            send_active = args.gate_send_until <= 0 or it < args.gate_send_until
            if args.lambda_gate_send > 0 and send_active and out.get("gate_p_tex") is not None:
                desired_p = p_from_rho_target(rho_target_q, teacher_rho_max)
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
            measured_active = args.gate_measured_sens_until <= 0 or it < args.gate_measured_sens_until
            if args.lambda_gate_measured_sens > 0 and measured_active and out.get("gate_p_tex") is not None:
                if lpips_spatial_loss is None:
                    raise RuntimeError("LPIPS spatial loss was not initialized")
                if x_hat_base is None:
                    with torch.no_grad():
                        base_out = train_forward(net, x, q, use_predictor=False, gate=None, q_shift=None,
                                                 predictor_param_mode=args.predictor_param_mode,
                                                 predictor_delta_bound=args.predictor_delta_bound)
                        x_hat_base = base_out["x_hat"].detach().clamp(-1, 1)
                desired_p = p_from_rho_target(rho_target_q, teacher_rho_max)
                measured_target = make_gate_measured_sensitivity_target(
                    x, x_hat_base, out["x_hat"], out["gate_p_tex"].shape[-2:], desired_p, lpips_spatial_loss,
                    margin=args.gate_measured_sens_margin, tau=args.gate_measured_sens_tau,
                    edge_weight=args.gate_measured_sens_edge_weight)
                if measured_target.shape != out["gate_p_tex"].shape:
                    measured_target = measured_target.expand_as(out["gate_p_tex"])
                l_gate_measured_sens = F.binary_cross_entropy(
                    out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), measured_target)
                gate_measured_sens_mean = measured_target.mean().item()
                gate_measured_sens_std = measured_target.std().item()
            else:
                l_gate_measured_sens = bpp_y.new_tensor(0.0)
                gate_measured_sens_mean = 0.0
                gate_measured_sens_std = 0.0
            gate_rate_corr = 0.0
            mixed_rate_corr = 0.0
            mixed_rate_mean = 0.0
            mixed_rate_std = 0.0
            safe_for_control = None
            mixed_active = args.gate_mixed_sens_until <= 0 or it < args.gate_mixed_sens_until
            if lambda_gate_mixed_sens_q > 0 and mixed_active and out.get("gate_p_tex") is not None:
                if lpips_spatial_loss is None:
                    raise RuntimeError("LPIPS spatial loss was not initialized")
                if x_hat_base is None:
                    with torch.no_grad():
                        base_out = train_forward(net, x, q, use_predictor=False, gate=None, q_shift=None,
                                                 predictor_param_mode=args.predictor_param_mode,
                                                 predictor_delta_bound=args.predictor_delta_bound)
                        x_hat_base = base_out["x_hat"].detach().clamp(-1, 1)
                desired_p = p_from_rho_target(rho_target_q, teacher_rho_max)
                rate_map = net.get_y_gaussian_bits(
                    out["y_q"].detach(), out["scales_hat"].detach()).mean(dim=1, keepdim=True)
                mixed_rate_mean = rate_map.detach().mean().item()
                mixed_rate_std = rate_map.detach().std().item()
                gate_rate_corr = spatial_corr(out["gate_p_tex"], rate_map).item()
                mixed_target = make_gate_mixed_sensitivity_target(
                    x, x_hat_base, out["x_hat"], out["gate_p_tex"].shape[-2:], desired_p, lpips_spatial_loss,
                    l1_weight=gate_mixed_l1_q, lpips_weight=gate_mixed_lpips_q,
                    texture_weight=gate_mixed_texture_q, edge_weight=gate_mixed_edge_q,
                    rate_map=rate_map, rate_weight=gate_mixed_rate_q,
                    margin=args.gate_mixed_sens_margin, tau=args.gate_mixed_sens_tau)
                if mixed_target.shape != out["gate_p_tex"].shape:
                    mixed_target = mixed_target.expand_as(out["gate_p_tex"])
                l_gate_mixed_sens = F.binary_cross_entropy(
                    out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), mixed_target)
                safe_for_control = mixed_target.detach()
                mixed_rate_corr = spatial_corr(mixed_target, rate_map).item()
                gate_mixed_sens_mean = mixed_target.mean().item()
                gate_mixed_sens_std = mixed_target.std().item()
            else:
                l_gate_mixed_sens = bpp_y.new_tensor(0.0)
                gate_mixed_sens_mean = 0.0
                gate_mixed_sens_std = 0.0

            gate_rdo_sens_mean = 0.0
            gate_rdo_sens_std = 0.0
            gate_rdo_saved_mean = 0.0
            gate_rdo_saved_std = 0.0
            gate_rdo_score_std = 0.0
            rdo_active = args.gate_rdo_sens_until <= 0 or it < args.gate_rdo_sens_until
            if lambda_gate_rdo_sens_q > 0 and rdo_active and out.get("gate_p_tex") is not None:
                if lpips_spatial_loss is None:
                    raise RuntimeError("LPIPS spatial loss was not initialized")
                if x_hat_base is None or base_out_for_teacher is None:
                    with torch.no_grad():
                        base_out_for_teacher = train_forward(
                            net, x, q, use_predictor=False, gate=None, q_shift=None,
                            predictor_param_mode=args.predictor_param_mode,
                            predictor_delta_bound=args.predictor_delta_bound)
                        x_hat_base = base_out_for_teacher["x_hat"].detach().clamp(-1, 1)
                desired_p = p_from_rho_target(rho_target_q, teacher_rho_max)
                base_rate_map = net.get_y_gaussian_bits(
                    base_out_for_teacher["y_q"].detach(),
                    base_out_for_teacher["scales_hat"].detach()).mean(dim=1, keepdim=True)
                ours_rate_map = net.get_y_gaussian_bits(
                    out["y_q"].detach(), out["scales_hat"].detach()).mean(dim=1, keepdim=True)
                rdo_target, rdo_saved_rate, rdo_score = make_gate_rdo_sensitivity_target(
                    x, x_hat_base, out["x_hat"], out["gate_p_tex"].shape[-2:], desired_p,
                    lpips_spatial_loss,
                    base_rate_map=base_rate_map,
                    ours_rate_map=ours_rate_map,
                    l1_weight=gate_rdo_l1_q,
                    lpips_weight=gate_rdo_lpips_q,
                    saved_rate_weight=gate_rdo_saved_rate_q,
                    texture_weight=gate_rdo_texture_q,
                    edge_weight=gate_rdo_edge_q,
                    margin=args.gate_rdo_sens_margin,
                    tau=args.gate_rdo_sens_tau)
                if rdo_target.shape != out["gate_p_tex"].shape:
                    rdo_target = rdo_target.expand_as(out["gate_p_tex"])
                l_gate_rdo_sens = F.binary_cross_entropy(
                    out["gate_p_tex"].clamp(1e-4, 1 - 1e-4), rdo_target)
                safe_for_control = rdo_target.detach()
                gate_rdo_sens_mean = rdo_target.mean().item()
                gate_rdo_sens_std = rdo_target.std().item()
                gate_rdo_saved_mean = rdo_saved_rate.mean().item()
                gate_rdo_saved_std = rdo_saved_rate.std().item()
                gate_rdo_score_std = rdo_score.std().item()
            else:
                l_gate_rdo_sens = bpp_y.new_tensor(0.0)

            if lambda_control_protect_q > 0 and out.get("control_prob") is not None:
                if lpips_spatial_loss is None:
                    raise RuntimeError("LPIPS spatial loss was not initialized")
                if x_hat_base is None:
                    with torch.no_grad():
                        base_out = train_forward(net, x, q, use_predictor=False, gate=None, q_shift=None,
                                                 predictor_param_mode=args.predictor_param_mode,
                                                 predictor_delta_bound=args.predictor_delta_bound)
                        x_hat_base = base_out["x_hat"].detach().clamp(-1, 1)
                if safe_for_control is None:
                    if out.get("gate_p_tex") is not None:
                        gate_spatial_size = out["gate_p_tex"].shape[-2:]
                    else:
                        gate_spatial_size = out["y"].shape[-2:]
                    rate_map = net.get_y_gaussian_bits(
                        out["y_q"].detach(), out["scales_hat"].detach()).mean(dim=1, keepdim=True)
                    safe_for_control = make_gate_mixed_sensitivity_target(
                        x, x_hat_base, out["x_hat"], gate_spatial_size, 0.5, lpips_spatial_loss,
                        l1_weight=gate_mixed_l1_q, lpips_weight=gate_mixed_lpips_q,
                        texture_weight=gate_mixed_texture_q, edge_weight=gate_mixed_edge_q,
                        rate_map=rate_map, rate_weight=gate_mixed_rate_q,
                        margin=args.gate_mixed_sens_margin, tau=args.gate_mixed_sens_tau)
                control_target = make_control_protect_target(
                    safe_for_control, out["control_prob"].shape, control_target_mean_q)
                if control_target.shape != out["control_prob"].shape:
                    control_target = control_target.expand_as(out["control_prob"])
                l_control_protect = F.binary_cross_entropy(
                    out["control_prob"].clamp(1e-4, 1 - 1e-4), control_target)
                control_target_mean = control_target.mean().item()
                control_target_std = control_target.std().item()
            else:
                l_control_protect = bpp_y.new_tensor(0.0)
                control_target_mean = 0.0
                control_target_std = 0.0
            if lambda_mean_pred_safe_q > 0:
                if safe_for_control is not None:
                    safe_weight = safe_for_control.detach()
                elif lambda_gate_mixed_sens_q > 0 and mixed_active and out.get("gate_p_tex") is not None:
                    safe_weight = mixed_target.detach()
                elif out.get("gate_p_tex") is not None:
                    safe_weight = out["gate_p_tex"].detach()
                else:
                    safe_weight = torch.ones_like(out["y"][:, :1])
                with torch.no_grad():
                    base_q_enc, _, _, base_mean = net.separate_prior(out["params_base"])
                    target_y = out["y"].detach() * base_q_enc.detach()
                if args.predictor_param_mode in {
                    "stage_latent_residual",
                    "stage_residual_quant_gate",
                    "stage_residual_entropy_quant_gate",
                    "stage_residual_entropy_quant_gate_scale_calib",
                    "stage_residual_entropy_quant_gate_residual_refiner",
                    "stage_residual_quant_gate_control",
                    "stage_residual_entropy_quant_gate_control",
                    "stage_residual_entropy_quant_gate_latent_control",
                }:
                    if out.get("stage_delta_map") is not None and out.get("stage_target_map") is not None:
                        l_mean_pred_safe = weighted_spatial_smooth_l1(
                            out["stage_delta_map"], out["stage_target_map"], safe_weight)
                    else:
                        l_mean_pred_safe = stage_mean_pred_loss if stage_mean_pred_loss is not None else bpp_y.new_tensor(0.0)
                elif args.predictor_param_mode == "latent_residual" and out.get("latent_pred_scaled") is not None:
                    target_residual_pred = target_y - base_mean.detach()
                    l_mean_pred_safe = weighted_spatial_smooth_l1(out["latent_pred_scaled"], target_residual_pred, safe_weight)
                else:
                    _, _, _, corrected_mean = net.separate_prior(out["params_after"])
                    l_mean_pred_safe = weighted_spatial_smooth_l1(corrected_mean, target_y, safe_weight)
            else:
                l_mean_pred_safe = bpp_y.new_tensor(0.0)

            if args.lambda_predictor_unsafe_delta > 0 and out.get("delta_params") is not None:
                if lambda_gate_mixed_sens_q > 0 and mixed_active and out.get("gate_p_tex") is not None:
                    unsafe = (1.0 - mixed_target.detach().expand_as(out["gate_p_tex"]))
                    delta_map = out["delta_params"].abs().mean(dim=1, keepdim=True)
                    l_predictor_unsafe_delta = (delta_map * unsafe).mean()
                else:
                    l_predictor_unsafe_delta = out["delta_params"].abs().mean()
            elif args.lambda_predictor_unsafe_delta > 0 and out.get("stage_delta_map") is not None:
                stage_delta_spatial = out["stage_delta_map"].abs().mean(dim=1, keepdim=True)
                if lambda_gate_mixed_sens_q > 0 and mixed_active and out.get("gate_p_tex") is not None:
                    unsafe = 1.0 - mixed_target.detach()
                    if unsafe.shape[-2:] != stage_delta_spatial.shape[-2:]:
                        unsafe = F.interpolate(unsafe, size=stage_delta_spatial.shape[-2:], mode="area")
                    if unsafe.shape[1] != 1:
                        unsafe = unsafe.mean(dim=1, keepdim=True)
                    l_predictor_unsafe_delta = (stage_delta_spatial * unsafe.clamp_min(0.0)).mean()
                else:
                    l_predictor_unsafe_delta = stage_delta_spatial.mean()
            else:
                l_predictor_unsafe_delta = bpp_y.new_tensor(0.0)
            loss = (lambda_R_q * bpp_rate + lambda_d_q * d_mse
                    + lambda_lpips_q * d_lp + lambda_dists_q * d_dists
                    + args.lambda_base_l1 * l_base_l1 + args.lambda_base_lpips * l_base_lpips
                    + args.lambda_dists_distill * l_base_dists
                    + lambda_lpips_hinge_q * l_lpips_hinge + lambda_dists_hinge_q * l_dists_hinge
                    + args.lambda_align * l_align + args.lambda_mean_pred * l_mean_pred
                    + args.lambda_stage_mean_norm * l_stage_mean_norm
                    + lambda_mean_pred_safe_q * l_mean_pred_safe
                    + args.lambda_predictor_unsafe_delta * l_predictor_unsafe_delta
                    + args.lambda_stage_delta_abs * l_stage_delta_abs
                    + args.lambda_rho_floor * l_rho_floor
                    + lambda_rho_target_q * l_rho_target + args.lambda_gate_send * l_gate_send
                    + args.lambda_gate_measured_sens * l_gate_measured_sens
                    + lambda_gate_mixed_sens_q * l_gate_mixed_sens
                    + lambda_gate_rdo_sens_q * l_gate_rdo_sens
                    + lambda_control_protect_q * l_control_protect)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if it % args.log_every == 0:
                print(f"[it {it}] q={q} loss={loss.item():.4f} bpp={bpp_total.item():.4f} "
                      f"bpp_y={bpp_y.item():.4f} bpp_c={bpp_control.item():.5f} psnr={psnr:.2f} "
                      f"mse={d_mse.item():.4f} lpips={d_lp.item():.4f} dists={d_dists.item():.4f} ce={l_align.item():.4f} "
                      f"loss_w=R{lambda_R_q:.2f}/D{lambda_d_q:.2f}/LP{lambda_lpips_q:.2f}/DS{lambda_dists_q:.2f} "
                      f"train_on=P{int(predictor_active)}/G{int(gate_active)}/Q{int(q_embed_active)} "
                      f"mean_pred={l_mean_pred.item():.4f} safe_mean={l_mean_pred_safe.item():.4f} unsafe_delta={l_predictor_unsafe_delta.item():.5f} "
                      f"latent_abs={latent_pred_abs:.4f}/{target_residual_abs:.4f} "
                      f"stage_abs={stage_delta_abs:.4f}/{stage_target_abs:.4f} "
                      f"base_l1={l_base_l1.item():.4f} base_lpips={l_base_lpips.item():.4f} "
                      f"base_dists={l_base_dists.item():.4f} lp_hinge={l_lpips_hinge.item():.4f} dists_hinge={l_dists_hinge.item():.4f} "
                      f"rho={rho_mean:.3f}/{rho_min:.3f}/{rho_max:.3f} active={rho_active:.2f} rt={rho_target_q:.3f} "
                      f"mix_w={gate_mixed_l1_q:.2f}/{gate_mixed_lpips_q:.2f}/{gate_mixed_texture_q:.2f}/{gate_mixed_edge_q:.2f}/rate{gate_mixed_rate_q:.2f} "
                      f"stage_l1={l_stage_delta_abs.item():.5f} stage_norm={l_stage_mean_norm.item():.4f} "
                      f"rho_floor={l_rho_floor.item():.4f} rho_target={l_rho_target.item():.4f} "
                      f"gate_send={l_gate_send.item():.4f} meas_sens={l_gate_measured_sens.item():.4f} "
                      f"mst={gate_measured_sens_mean:.3f}/{gate_measured_sens_std:.3f} "
                      f"mix_sens={l_gate_mixed_sens.item():.4f} mixt={gate_mixed_sens_mean:.3f}/{gate_mixed_sens_std:.3f} "
                      f"rdo_sens={l_gate_rdo_sens.item():.4f} rdot={gate_rdo_sens_mean:.3f}/{gate_rdo_sens_std:.3f} "
                      f"rdosave={gate_rdo_saved_mean:.4f}/{gate_rdo_saved_std:.4f} rdoscore={gate_rdo_score_std:.3f} "
                      f"ctrl_loss={l_control_protect.item():.4f} ctrlt={control_target_mean:.3f}/{control_target_std:.3f} "
                      f"rate_corr={gate_rate_corr:.3f}/{mixed_rate_corr:.3f} rate={mixed_rate_mean:.4f}/{mixed_rate_std:.4f} "
                      f"ctrl={control_mean:.4f}/abs{control_abs_mean:.4f}/{control_prob_mean:.4f}/{control_prob_max:.4f} "
                      f"delta_abs={delta_abs:.5f}")
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/bpp_y": bpp_y.item(),
                               "train/bpp_control": bpp_control.item(),
                               "train/bpp_rate": bpp_rate.item(),
                               "train/bpp_z": bpp_z.item(), "train/bpp_total": bpp_total.item(),
                               "train/psnr": psnr, "train/mse": d_mse.item(), "train/lpips": d_lp.item(),
                               "train/dists": d_dists.item(), "train/ce_align": l_align.item(),
                               "train/lambda_R_q": lambda_R_q,
                               "train/lambda_d_q": lambda_d_q,
                               "train/lambda_lpips_q": lambda_lpips_q,
                               "train/lambda_dists_q": lambda_dists_q,
                               "train/lambda_lpips_hinge_q": lambda_lpips_hinge_q,
                               "train/lambda_dists_hinge_q": lambda_dists_hinge_q,
                               "train/lambda_mean_pred_safe_q": lambda_mean_pred_safe_q,
                               "train/lambda_rho_target_q": lambda_rho_target_q,
                               "train/lambda_gate_mixed_sens_q": lambda_gate_mixed_sens_q,
                               "train/lambda_gate_rdo_sens_q": lambda_gate_rdo_sens_q,
                               "train/lambda_control_protect_q": lambda_control_protect_q,
                               "train/predictor_active": float(predictor_active),
                               "train/gate_active": float(gate_active),
                               "train/q_embed_active": float(q_embed_active),
                               "train/mean_pred": l_mean_pred.item(),
                               "train/mean_pred_safe": l_mean_pred_safe.item(),
                               "train/predictor_unsafe_delta": l_predictor_unsafe_delta.item(),
                               "pred/latent_pred_abs": latent_pred_abs, "pred/target_residual_abs": target_residual_abs,
                               "pred/stage_delta_abs": stage_delta_abs, "pred/stage_target_abs": stage_target_abs,
                               "train/base_l1": l_base_l1.item(), "train/base_lpips": l_base_lpips.item(),
                               "train/base_dists": l_base_dists.item(),
                               "train/lpips_hinge": l_lpips_hinge.item(), "train/dists_hinge": l_dists_hinge.item(),
                               "train/stage_delta_l1": l_stage_delta_abs.item(),
                               "train/stage_mean_norm": l_stage_mean_norm.item(),
                               "train/rho_floor": l_rho_floor.item(), "train/rho_target": l_rho_target.item(),
                               "train/rho_target_q": rho_target_q,
                               "train/gate_mixed_l1_weight_q": gate_mixed_l1_q,
                               "train/gate_mixed_lpips_weight_q": gate_mixed_lpips_q,
                               "train/gate_mixed_texture_weight_q": gate_mixed_texture_q,
                               "train/gate_mixed_edge_weight_q": gate_mixed_edge_q,
                               "train/gate_mixed_rate_weight_q": gate_mixed_rate_q,
                               "train/gate_send": l_gate_send.item(),
                               "train/gate_mixed_sens": l_gate_mixed_sens.item(),
                               "train/gate_mixed_sens_mean": gate_mixed_sens_mean,
                               "train/gate_mixed_sens_std": gate_mixed_sens_std,
                               "train/gate_rdo_sens": l_gate_rdo_sens.item(),
                               "train/gate_rdo_sens_mean": gate_rdo_sens_mean,
                               "train/gate_rdo_sens_std": gate_rdo_sens_std,
                               "train/gate_rdo_saved_mean": gate_rdo_saved_mean,
                               "train/gate_rdo_saved_std": gate_rdo_saved_std,
                               "train/gate_rdo_score_std": gate_rdo_score_std,
                               "train/gate_rdo_l1_weight_q": gate_rdo_l1_q,
                               "train/gate_rdo_lpips_weight_q": gate_rdo_lpips_q,
                               "train/gate_rdo_saved_rate_weight_q": gate_rdo_saved_rate_q,
                               "train/gate_rdo_texture_weight_q": gate_rdo_texture_q,
                               "train/gate_rdo_edge_weight_q": gate_rdo_edge_q,
                               "train/control_protect": l_control_protect.item(),
                               "control/target_mean": control_target_mean,
                               "control/target_std": control_target_std,
                               "train/gate_rate_corr": gate_rate_corr,
                               "train/mixed_target_rate_corr": mixed_rate_corr,
                               "train/mixed_rate_map_mean": mixed_rate_mean,
                               "train/mixed_rate_map_std": mixed_rate_std,
                               "pred/delta_abs_mean": delta_abs,
                               "gate/rho_mean": rho_mean, "gate/rho_min": rho_min, "gate/rho_max": rho_max,
                               "gate/rho_active_frac": rho_active, "gate/raw_mean": gate_raw_mean,
                               "gate/raw_min": gate_raw_min, "gate/raw_max": gate_raw_max,
                               "gate/send_target_mean": gate_target_mean, "gate/send_target_std": gate_target_std,
                               "control/symbol_mean": control_mean,
                               "control/symbol_abs_mean": control_abs_mean,
                               "control/prob_mean": control_prob_mean,
                               "control/prob_max": control_prob_max,
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
                state = {"prior_predictor": net.prior_predictor.state_dict(),
                         "q_embed": net.q_embed.detach().cpu(),
                         "perceptual_gate": net.perceptual_gate.state_dict() if use_gate else None,
                         "use_gate": use_gate, "rho_max": args.rho_max, "rho_min": args.gate_rho_min,
                         "rho_mode": args.gate_rho_mode, "gate_softplus_shift": args.gate_softplus_shift,
                         "gate_softplus_tau": args.gate_softplus_tau, "gate_rho_init": args.gate_rho_init,
                         "predictor_param_mode": args.predictor_param_mode}
                if use_stage_residual:
                    state["stage_residual_predictor"] = net.stage_residual_predictor.state_dict()
                    if use_stage_entropy:
                        state["stage_scale_log_bound"] = args.stage_scale_log_bound
                if use_stage_quant_gate:
                    state["stage_quant_gate"] = net.stage_quant_gate.state_dict()
                    state["stage_rho_max"] = args.stage_rho_max
                if use_stage_scale_calibrator:
                    state["stage_scale_calibrator"] = net.stage_scale_calibrator.state_dict()
                    state["stage_scale_calib_bound"] = args.stage_scale_calib_bound
                if use_stage_residual_refiner:
                    state["stage_residual_refiner"] = net.stage_residual_refiner.state_dict()
                    state["stage_residual_refiner_bound"] = args.stage_residual_refiner_bound
                    state["stage_residual_refiner_depth"] = args.stage_residual_refiner_depth
                if use_tiny_control:
                    state["tiny_control_encoder"] = net.tiny_control_encoder.state_dict()
                    state["control_init_prob"] = args.control_init_prob
                    state["control_prob_one"] = args.control_prob_one
                    state["control_threshold"] = args.control_threshold
                    state["control_hard_mode"] = args.control_hard_mode
                    state["control_topk_frac"] = args.control_topk_frac
                if use_latent_control:
                    state["latent_control_encoder"] = net.latent_control_encoder.state_dict()
                    state["latent_control_init_prob"] = args.latent_control_init_prob
                    state["latent_control_prob_nonzero"] = args.latent_control_prob_nonzero
                    state["latent_control_topk_frac"] = args.latent_control_topk_frac
                    state["latent_control_hard_mode"] = args.latent_control_hard_mode
                    state["latent_control_threshold"] = args.latent_control_threshold
                    state["latent_control_delta"] = args.latent_control_delta
                    state["latent_control_groups"] = args.latent_control_groups
                torch.save(state, os.path.join(args.out, f"v2_{it}.pt"))
                state_with_opt = dict(state)
                state_with_opt["it"] = it
                state_with_opt["optimizer"] = opt.state_dict()
                torch.save(state_with_opt, os.path.join(args.out, "train_state.pt"))   # resume 用（上書き）

            it += 1
            if it >= args.iters:
                break

    state = {"prior_predictor": net.prior_predictor.state_dict(),
             "q_embed": net.q_embed.detach().cpu(),
             "perceptual_gate": net.perceptual_gate.state_dict() if use_gate else None,
             "use_gate": use_gate, "rho_max": args.rho_max, "rho_min": args.gate_rho_min,
             "rho_mode": args.gate_rho_mode, "gate_softplus_shift": args.gate_softplus_shift,
             "gate_softplus_tau": args.gate_softplus_tau, "gate_rho_init": args.gate_rho_init,
             "predictor_param_mode": args.predictor_param_mode}
    if use_stage_residual:
        state["stage_residual_predictor"] = net.stage_residual_predictor.state_dict()
        if use_stage_entropy:
            state["stage_scale_log_bound"] = args.stage_scale_log_bound
    if use_stage_quant_gate:
        state["stage_quant_gate"] = net.stage_quant_gate.state_dict()
        state["stage_rho_max"] = args.stage_rho_max
    if use_stage_scale_calibrator:
        state["stage_scale_calibrator"] = net.stage_scale_calibrator.state_dict()
        state["stage_scale_calib_bound"] = args.stage_scale_calib_bound
    if use_stage_residual_refiner:
        state["stage_residual_refiner"] = net.stage_residual_refiner.state_dict()
        state["stage_residual_refiner_bound"] = args.stage_residual_refiner_bound
        state["stage_residual_refiner_depth"] = args.stage_residual_refiner_depth
    if use_tiny_control:
        state["tiny_control_encoder"] = net.tiny_control_encoder.state_dict()
        state["control_init_prob"] = args.control_init_prob
        state["control_prob_one"] = args.control_prob_one
        state["control_threshold"] = args.control_threshold
        state["control_hard_mode"] = args.control_hard_mode
        state["control_topk_frac"] = args.control_topk_frac
    if use_latent_control:
        state["latent_control_encoder"] = net.latent_control_encoder.state_dict()
        state["latent_control_init_prob"] = args.latent_control_init_prob
        state["latent_control_prob_nonzero"] = args.latent_control_prob_nonzero
        state["latent_control_topk_frac"] = args.latent_control_topk_frac
        state["latent_control_hard_mode"] = args.latent_control_hard_mode
        state["latent_control_threshold"] = args.latent_control_threshold
        state["latent_control_delta"] = args.latent_control_delta
        state["latent_control_groups"] = args.latent_control_groups
    torch.save(state, os.path.join(args.out, "v2_final.pt"))
    state_with_opt = dict(state)
    state_with_opt["it"] = it
    state_with_opt["optimizer"] = opt.state_dict()
    torch.save(state_with_opt, os.path.join(args.out, "train_state.pt"))
    print("done. → test_v2.py で全 q（--interpolate で 64 点）の R-P 曲線を生成。")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
