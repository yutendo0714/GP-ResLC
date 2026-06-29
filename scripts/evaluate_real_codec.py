#!/usr/bin/env python3
"""Evaluate GLC / GP-ResLC with real arithmetic-coded bitstreams.

Unlike `test_image.py` and `scripts/test_v2.py`, this script reports bpp from
serialized payload length:

    bpp = 8 * len(real_payload_bytes) / (original_height * original_width)

The payload includes a compact binary header, fixed-width z-index bits, and
four torchac-coded y streams. Encode/decode wall times are measured with CUDA
synchronization around the full codec path.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from gp_reslc.perceptual_gate import PerceptualGate
from gp_reslc.prior_predictor import PriorPredictor, StageResidualPredictor, StageQuantGate, train_forward
from gp_reslc.real_codec import (
    compress_to_real_bitstream,
    crop_to_original,
    decompress_from_real_bitstream,
)
from src.models.image_model import GLC_Image
from src.utils.test_utils import from_0_1_to_minus1_1, get_state_dict, init_func, write_image


EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
PAD = 64


def expand_qembed(qe: torch.Tensor) -> torch.Tensor:
    n = qe.shape[1]
    w = F.interpolate(qe.view(1, 1, 4, n), size=(64, n), mode="bilinear", align_corners=True)
    return w.view(64, n, 1, 1)


def build_glc(weights: str, device: str) -> GLC_Image:
    net = GLC_Image(inplace=True)
    net.load_state_dict(get_state_dict(weights), strict=True)
    return net.to(device).eval()


class GateAlphaWrapper(torch.nn.Module):
    """Scale a learned perceptual gate deterministically without side bits."""

    def __init__(self, base: torch.nn.Module, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho, p_tex = self.base(z)
        rho = 1.0 + self.alpha * (rho - 1.0)
        return rho.clamp_min(1.0), p_tex


def build_gp_reslc(weights: str, ckpt_path: str, interpolate: bool, device: str) -> GLC_Image:
    net = GLC_Image(inplace=True)
    net.load_state_dict(get_state_dict(weights), strict=True)
    ck = torch.load(ckpt_path, map_location="cpu")
    if "prior_predictor" in ck:
        net.prior_predictor = PriorPredictor(net.N)
        net.prior_predictor.load_state_dict(ck["prior_predictor"])
    else:
        net.prior_predictor = None
    if "stage_residual_predictor" in ck:
        net.stage_residual_predictor = StageResidualPredictor(net.N)
        net.stage_residual_predictor.load_state_dict(ck["stage_residual_predictor"])
    if "stage_quant_gate" in ck:
        net.stage_quant_gate = StageQuantGate(net.N, rho_max=ck.get("stage_rho_max", 1.5))
        net.stage_quant_gate.load_state_dict(ck["stage_quant_gate"])
    q_embed = ck.get("q_embed")
    if ck.get("use_gate", False):
        net.perceptual_gate = PerceptualGate(
            net.N,
            rho_max=ck.get("rho_max", 2.0),
            rho_min=ck.get("rho_min", 0.5),
            rho_mode=ck.get("rho_mode", "hard"),
            softplus_shift=ck.get("gate_softplus_shift", 2.0),
            softplus_tau=ck.get("gate_softplus_tau", 1.0),
        )
        net.perceptual_gate.load_state_dict(ck["perceptual_gate"])
        net.perceptual_gate = GateAlphaWrapper(net.perceptual_gate, alpha=1.0)
    else:
        net.perceptual_gate = None

    if "model_state_dict" in ck:
        missing, unexpected = net.load_state_dict(ck["model_state_dict"], strict=False)
        print(
            f"[model_state_dict] loaded tuned GLC state: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    if interpolate and q_embed is None:
        raise ValueError("--interpolate requires a q-conditioned checkpoint with q_embed")
    if interpolate:
        net.interpolate_q()
        q_embed = expand_qembed(q_embed)
    if q_embed is not None:
        net.q_embed = q_embed.to(device)
    return net.to(device).eval()


def sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def image_paths(root: str) -> list[str]:
    paths = sorted(sum([glob.glob(os.path.join(root, ext)) for ext in EXTS], []))
    if not paths:
        raise RuntimeError(f"no images found in {root}")
    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glc_weights", required=True)
    ap.add_argument("--ckpt", default=None, help="GP-ResLC checkpoint. Omit for plain GLC.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--interpolate", action="store_true")
    ap.add_argument("--predictor_param_mode", choices=["mean", "scale_mean", "all", "latent_residual", "stage_latent_residual", "stage_quant_gate"], default="mean")
    ap.add_argument("--predictor_delta_bound", type=float, default=0.0)
    ap.add_argument("--gate_alpha", type=float, default=1.0,
                    help="Scale learned gate strength as rho = 1 + alpha * (rho - 1).")
    ap.add_argument("--gate_alpha_by_q", type=float, nargs="+", default=None,
                    help="Optional q-indexed gate alpha table, e.g. 0.25 0.25 0.75 0.75.")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--save_streams", action="store_true", help="Write .gprc payload files next to reconstructions.")
    ap.add_argument("--skip_recon", action="store_true", help="Do not write decoded PNG reconstructions.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing per-q bpp.json manifests.")
    ap.add_argument("--check_estimated_consistency", action="store_true",
                    help="Also run the likelihood-estimate forward path and report max |real_decode - estimate_forward|.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return ap.parse_args()


def main() -> None:
    init_func()
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if args.ckpt:
        net = build_gp_reslc(args.glc_weights, args.ckpt, args.interpolate, device)
        method = "gp_reslc_real"
    else:
        net = build_glc(args.glc_weights, device)
        method = "glc_real"

    paths = image_paths(args.input)
    if args.max_images and args.max_images > 0:
        paths = paths[:args.max_images]

    qmax = net.q_enc.shape[0]
    if args.gate_alpha_by_q is not None and len(args.gate_alpha_by_q) < qmax:
        raise ValueError(f"--gate_alpha_by_q needs at least {qmax} values, got {len(args.gate_alpha_by_q)}")
    os.makedirs(args.out, exist_ok=True)
    summary = {
        "method": method,
        "input": args.input,
        "gate_alpha": args.gate_alpha,
        "gate_alpha_by_q": args.gate_alpha_by_q,
        "q": {},
    }

    for q in args.q_indexes:
        if not 0 <= q < qmax:
            raise ValueError(f"q={q} out of range 0..{qmax - 1}")
        gate_alpha = args.gate_alpha_by_q[q] if args.gate_alpha_by_q is not None else args.gate_alpha
        if hasattr(getattr(net, "perceptual_gate", None), "set_alpha"):
            net.perceptual_gate.set_alpha(gate_alpha)
        save_dir = Path(args.out) / f"q{q}"
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.save_streams:
            (save_dir / "streams").mkdir(exist_ok=True)

        bpp_path = save_dir / "bpp.json"
        if args.resume and bpp_path.exists():
            manifest = json.loads(bpp_path.read_text())
            manifest.setdefault("images", {})
        else:
            manifest = {
                "method": method,
                "q": q,
                "real_codec": True,
                "gate_alpha": gate_alpha,
                "images": {},
            }
        manifest["gate_alpha"] = gate_alpha
        totals = {
            "bpp": 0.0,
            "bpp_y": 0.0,
            "bpp_z": 0.0,
            "bpp_header": 0.0,
            "encode_time_s": 0.0,
            "decode_time_s": 0.0,
            "payload_bytes": 0.0,
        }

        for item in manifest["images"].values():
            for key in totals:
                if key in item:
                    totals[key] += float(item[key])

        for idx, path in enumerate(paths):
            name = Path(path).stem
            if args.resume and name in manifest["images"]:
                print(f"[{method} q={q} {idx + 1}/{len(paths)} {name}] skip existing", flush=True)
                continue
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            img = Image.open(path).convert("RGB")
            w, h = img.size
            x = from_0_1_to_minus1_1(ToTensor()(img)).unsqueeze(0).to(device)
            pl, pr, pt, pb = GLC_Image.get_padding_size(h, w, PAD)
            x_padded = F.pad(x, (pl, pr, pt, pb), mode="replicate")

            sync(device)
            t0 = time.perf_counter()
            payload, stats = compress_to_real_bitstream(
                net,
                x_padded,
                q,
                orig_hw=(h, w),
                predictor_param_mode=args.predictor_param_mode,
                predictor_delta_bound=args.predictor_delta_bound,
            )
            sync(device)
            t1 = time.perf_counter()

            x_hat_padded = decompress_from_real_bitstream(
                net,
                payload,
                predictor_param_mode=args.predictor_param_mode,
                predictor_delta_bound=args.predictor_delta_bound,
            )
            sync(device)
            t2 = time.perf_counter()

            x_hat = crop_to_original(x_hat_padded, (h, w)).clamp(-1, 1)
            consistency_max_abs = None
            if args.check_estimated_consistency:
                with torch.no_grad():
                    if args.ckpt:
                        est = train_forward(
                            net,
                            x_padded,
                            q,
                            use_predictor=True,
                            gate=net.perceptual_gate,
                            q_shift=getattr(net, "q_embed", None)[q:q + 1]
                            if getattr(net, "q_embed", None) is not None else None,
                            predictor_param_mode=args.predictor_param_mode,
                            predictor_delta_bound=args.predictor_delta_bound,
                        )["x_hat"]
                    else:
                        est = net.test(x_padded, q)["x_hat"]
                consistency_max_abs = float((x_hat_padded - est).abs().max().item())
            if not args.skip_recon:
                write_image(str(save_dir / f"{name}.png"), x_hat)
            if args.save_streams:
                (save_dir / "streams" / f"{name}.gprc").write_bytes(payload)

            pixels = h * w
            bpp = len(payload) * 8.0 / pixels
            bpp_y = stats["y_bytes"] * 8.0 / pixels
            bpp_z = stats["z_bytes"] * 8.0 / pixels
            bpp_header = stats["header_bytes"] * 8.0 / pixels
            enc_t = t1 - t0
            dec_t = t2 - t1

            item = {
                "bpp": bpp,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp_header": bpp_header,
                "payload_bytes": len(payload),
                "y_bytes": stats["y_bytes"],
                "z_bytes": stats["z_bytes"],
                "header_bytes": stats["header_bytes"],
                "encode_time_s": enc_t,
                "decode_time_s": dec_t,
                "height": h,
                "width": w,
                "gate_alpha": gate_alpha,
            }
            if consistency_max_abs is not None:
                item["consistency_max_abs"] = consistency_max_abs
            manifest["images"][name] = item
            for key in totals:
                totals[key] += item[key]

            current_n = len(manifest["images"])
            for key, value in totals.items():
                manifest[f"avg_{key}"] = value / current_n
            manifest["image_count"] = current_n
            bpp_path.write_text(json.dumps(manifest, indent=2))

            del x, x_padded, x_hat_padded, x_hat, payload, stats, img
            if "est" in locals():
                del est
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

            print(
                f"[{method} q={q} {idx + 1}/{len(paths)} {name}] "
                f"bpp={bpp:.5f} y={bpp_y:.5f} z={bpp_z:.5f} "
                f"hdr={bpp_header:.5f} enc={enc_t:.3f}s dec={dec_t:.3f}s"
                + (f" max_abs={consistency_max_abs:.3e}" if consistency_max_abs is not None else "")
            )

        n = len(manifest["images"])
        if n == 0:
            raise RuntimeError(f"no images encoded for q={q}")
        for key, value in totals.items():
            manifest[f"avg_{key}"] = value / n
        manifest["image_count"] = n
        bpp_path.write_text(json.dumps(manifest, indent=2))
        summary["q"][str(q)] = {k: manifest[k] for k in manifest if k.startswith("avg_")}
        print(
            f"== {method} q={q}: avg_bpp={manifest['avg_bpp']:.5f} "
            f"avg_enc={manifest['avg_encode_time_s']:.3f}s "
            f"avg_dec={manifest['avg_decode_time_s']:.3f}s -> {save_dir}"
        )

    (Path(args.out) / "real_codec_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
