#!/usr/bin/env python3
"""Dump PerceptualGate rho maps for qualitative GP-ResLC analysis."""

import argparse
import json
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
from src.utils.test_utils import get_state_dict, from_0_1_to_minus1_1
from gp_reslc.prior_predictor import PriorPredictor
from gp_reslc.perceptual_gate import PerceptualGate


def colorize(arr):
    arr = arr.clamp(0, 1)
    r = (255 * arr).byte()
    g = (255 * (1 - (arr - 0.5).abs() * 2).clamp(0, 1)).byte()
    b = (255 * (1 - arr)).byte()
    return torch.stack([r, g, b], dim=-1).cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glc_weights', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', required=True, help='image file or folder')
    ap.add_argument('--out', required=True)
    ap.add_argument('--q_index', type=int, default=2)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    net = GLC_Image(inplace=False)
    net.load_state_dict(get_state_dict(args.glc_weights), strict=True)
    net.prior_predictor = PriorPredictor(net.N)
    ck = torch.load(args.ckpt, map_location='cpu')
    net.prior_predictor.load_state_dict(ck['prior_predictor'])
    net.perceptual_gate = PerceptualGate(net.N, rho_max=ck.get('rho_max', 2.0), rho_min=ck.get('rho_min', 0.5), rho_mode=ck.get('rho_mode', 'hard'), softplus_shift=ck.get('gate_softplus_shift', 2.0), softplus_tau=ck.get('gate_softplus_tau', 1.0))
    if ck.get('perceptual_gate') is not None:
        net.perceptual_gate.load_state_dict(ck['perceptual_gate'])
    net.q_embed = ck['q_embed'].to(device)
    net.to(device).eval()

    in_path = Path(args.input)
    if in_path.is_dir():
        paths = sorted([p for p in in_path.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}])
    else:
        paths = [in_path]

    rows = []
    for path in paths:
        img = Image.open(path).convert('RGB')
        x = from_0_1_to_minus1_1(ToTensor()(img)).unsqueeze(0).to(device)
        _, _, H, W = x.shape
        pl, pr, pt, pb = GLC_Image.get_padding_size(H, W, 64)
        xp = F.pad(x, (pl, pr, pt, pb), mode='replicate')

        curr_q_enc = net.q_enc[args.q_index:args.q_index + 1]
        y_ori = net.vqgan.encoder(xp)
        y = net.enc(y_ori, curr_q_enc)
        z = net.hyper_enc(y)
        index = net.z_vq.get_indices(z)
        z_hat = net.z_vq.get_quan_feat(index, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
        z_cond = z_hat + net.q_embed[args.q_index:args.q_index + 1]
        rho, p_tex = net.perceptual_gate(z_cond)
        rho_img = F.interpolate(rho, size=(xp.shape[-2], xp.shape[-1]), mode='nearest')
        rho_img = F.pad(rho_img, (-pl, -pr, -pt, -pb))[0, 0].detach().cpu()
        denom = max(float(rho_img.max() - rho_img.min()), 1e-6)
        norm = (rho_img - float(rho_img.min())) / denom
        heat = Image.fromarray(colorize(norm))
        heat.save(Path(args.out) / f'{path.stem}_rho_q{args.q_index}.png')
        torch.save(rho_img, Path(args.out) / f'{path.stem}_rho_q{args.q_index}.pt')
        rows.append({
            'image': path.name,
            'q': args.q_index,
            'rho_mean': float(rho_img.mean()),
            'rho_min': float(rho_img.min()),
            'rho_max': float(rho_img.max()),
            'p_tex_mean': float(p_tex.mean()),
        })

    with open(Path(args.out) / 'gate_stats.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
