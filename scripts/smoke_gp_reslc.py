#!/usr/bin/env python3
"""Smoke checks for the GP-ResLC layer on top of pretrained GLC."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from gp_reslc.prior_predictor import PriorPredictor, train_forward
from src.models.image_model import GLC_Image
from src.utils.test_utils import get_state_dict, from_0_1_to_minus1_1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", default="pretrained/GLC_image.pth.tar")
    ap.add_argument("--image", default="/dpl/kodak/kodim01.png")
    ap.add_argument("--q_index", type=int, default=2)
    ap.add_argument("--pad", type=int, default=64)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA is not visible; stop and restart the container before running experiments.")

    net = GLC_Image(inplace=True)
    net.load_state_dict(get_state_dict(args.glc_weights), strict=True)
    net.prior_predictor = PriorPredictor(net.N)
    net = net.to(device).eval()

    x = from_0_1_to_minus1_1(ToTensor()(Image.open(args.image).convert("RGB"))).unsqueeze(0).to(device)
    _, _, h, w = x.shape
    pl, pr, pt, pb = GLC_Image.get_padding_size(h, w, args.pad)
    xp = F.pad(x, (pl, pr, pt, pb), mode="replicate")

    base = train_forward(net, xp, args.q_index, use_predictor=False)
    zero = train_forward(net, xp, args.q_index, use_predictor=True, predictor_param_mode="scale_mean", predictor_delta_bound=0.05)

    denom = xp.shape[0] * xp.shape[2] * xp.shape[3]
    base_bpp_y = base["bit_y"].item() / denom
    zero_bpp_y = zero["bit_y"].item() / denom
    max_abs = (base["x_hat"] - zero["x_hat"]).abs().max().item()
    delta_abs = zero["delta_params"].abs().max().item()
    mse = torch.mean((base["x_hat"].clamp(-1, 1) - zero["x_hat"].clamp(-1, 1)) ** 2).item()
    psnr = math.inf if mse == 0 else 10 * math.log10(4.0 / mse)

    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"image={args.image} q={args.q_index} padded={tuple(xp.shape)}")
    print(f"baseline_bpp_y={base_bpp_y:.8f}")
    print(f"zero_init_predictor_bpp_y={zero_bpp_y:.8f}")
    print(f"delta_params_max_abs={delta_abs:.8e}")
    print(f"recon_max_abs_diff={max_abs:.8e} recon_psnr_between_paths={psnr}")

    tol = 1e-6
    if abs(base_bpp_y - zero_bpp_y) > tol or max_abs > tol or delta_abs > tol:
        raise SystemExit("zero-init predictor path is not equivalent to baseline")
    print("smoke_ok=true")


if __name__ == "__main__":
    main()
