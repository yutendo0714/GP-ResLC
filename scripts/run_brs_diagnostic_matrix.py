#!/usr/bin/env python3
"""Run the GP-ResLC BRS / omitted-residual diagnostic matrix.

This is an orchestration helper around `scripts/evaluate_real_codec.py`.
It intentionally does not introduce a new evaluation path: every candidate is
encoded and decoded through the same serialized real codec.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    suppress_stages: tuple[int, ...]
    rho_threshold: float = 0.0
    omitted_mode: str = "zero"
    omitted_scale: float = 0.0
    omitted_clip: float = 2.0


CANDIDATES = (
    Candidate("stage3_zero", (3,)),
    Candidate("stage23_zero", (2, 3)),
    Candidate("stage123_zero", (1, 2, 3)),
    Candidate("stage3_rho120_zero", (3,), rho_threshold=1.20),
    Candidate(
        "stage3_rho120_hash_gaussian_s025",
        (3,),
        rho_threshold=1.20,
        omitted_mode="hash_gaussian_clipped",
        omitted_scale=0.25,
        omitted_clip=1.0,
    ),
    Candidate(
        "stage3_rho120_rademacher_s025",
        (3,),
        rho_threshold=1.20,
        omitted_mode="hash_rademacher",
        omitted_scale=0.25,
    ),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--glc_weights", default="pretrained/GLC_image.pth.tar")
    ap.add_argument("--ckpt", default="experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt")
    ap.add_argument("--q_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--predictor_delta_bound", type=float, default=0.3)
    ap.add_argument("--z_entropy_mode", default="fixed", choices=["fixed", "static", "auto"])
    ap.add_argument("--z_entropy_cdf_path", default=None)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional candidate names to run. Defaults to the full matrix.",
    )
    return ap.parse_args()


def build_command(args: argparse.Namespace, cand: Candidate) -> list[str]:
    out = Path(args.out_root) / cand.name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_real_codec.py"),
        "--glc_weights",
        args.glc_weights,
        "--ckpt",
        args.ckpt,
        "--input",
        args.input,
        "--out",
        str(out),
        "--q_indexes",
        *[str(q) for q in args.q_indexes],
        "--predictor_param_mode",
        "stage_residual_entropy_quant_gate",
        "--predictor_delta_bound",
        str(args.predictor_delta_bound),
        "--suppress_yq_stages",
        *[str(s) for s in cand.suppress_stages],
        "--omitted_residual_mode",
        cand.omitted_mode,
        "--omitted_residual_scale",
        str(cand.omitted_scale),
        "--omitted_residual_clip",
        str(cand.omitted_clip),
        "--z_entropy_mode",
        args.z_entropy_mode,
        "--device",
        args.device,
    ]
    if cand.rho_threshold > 0.0:
        cmd.extend(["--suppress_rho_threshold", str(cand.rho_threshold)])
    if args.z_entropy_cdf_path:
        cmd.extend(["--z_entropy_cdf_path", args.z_entropy_cdf_path])
    if args.max_images > 0:
        cmd.extend(["--max_images", str(args.max_images)])
    if args.resume:
        cmd.append("--resume")
    return cmd


def main() -> None:
    args = parse_args()
    selected = set(args.only or [c.name for c in CANDIDATES])
    unknown = selected - {c.name for c in CANDIDATES}
    if unknown:
        raise SystemExit(f"unknown candidates: {sorted(unknown)}")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for cand in CANDIDATES:
        if cand.name not in selected:
            continue
        cmd = build_command(args, cand)
        print("+ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
