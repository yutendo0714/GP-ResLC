#!/usr/bin/env python3
"""Build the full CLIC2020 test package from professional+mobile subsets.

The HiFiC/GLC CLIC2020 test protocol uses 428 images:
professional test (250) + mobile test (178). This script creates a symlinked
original-image directory and merged real-codec reconstruction trees from the
separately evaluated subset outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_ORIGS = [Path("/dpl/clic/professional/test"), Path("/dpl/clic/mobile/test")]


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._"))


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        raise FileExistsError(f"refusing to overwrite existing path: {dst}")
    dst.symlink_to(src)


def build_orig(out_dir: Path, orig_dirs: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for root in orig_dirs:
        for src in image_files(root):
            if src.name in seen:
                raise ValueError(f"duplicate image basename in CLIC subsets: {src.name}")
            seen.add(src.name)
            safe_symlink(src, out_dir / src.name)
    if len(seen) != 428:
        raise RuntimeError(f"expected 428 CLIC2020 test images, got {len(seen)}")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def merge_q(prof_q: Path, mobile_q: Path, out_q: Path) -> None:
    out_q.mkdir(parents=True, exist_ok=True)
    manifests = [read_manifest(prof_q / "bpp.json"), read_manifest(mobile_q / "bpp.json")]
    images: dict[str, dict] = {}
    for src_q, manifest in [(prof_q, manifests[0]), (mobile_q, manifests[1])]:
        for name, item in manifest["images"].items():
            if name in images:
                raise ValueError(f"duplicate image key in manifests: {name}")
            images[name] = item
            src_img = src_q / f"{name}.png"
            if not src_img.exists():
                raise FileNotFoundError(src_img)
            safe_symlink(src_img.resolve(), out_q / src_img.name)

    first = manifests[0]
    merged = {
        "method": first.get("method"),
        "q": first.get("q"),
        "real_codec": True,
        "images": dict(sorted(images.items())),
        "image_count": len(images),
        "merged_from": [str(prof_q.parent), str(mobile_q.parent)],
    }
    if len(images) != 428:
        raise RuntimeError(f"expected 428 merged images for q{first.get('q')}, got {len(images)}")

    numeric_keys = sorted({k for item in images.values() for k, v in item.items() if isinstance(v, (int, float))})
    for key in numeric_keys:
        merged[f"avg_{key}"] = sum(float(item[key]) for item in images.values() if key in item) / len(images)
    (out_q / "bpp.json").write_text(json.dumps(merged, indent=2) + "\n")


def merge_run(prof_root: Path, mobile_root: Path, out_root: Path, q_indexes: list[int]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for q in q_indexes:
        merge_q(prof_root / f"q{q}", mobile_root / f"q{q}", out_root / f"q{q}")
    summary = {"q": {}}
    for q in q_indexes:
        manifest = read_manifest(out_root / f"q{q}" / "bpp.json")
        summary["q"][str(q)] = {k: v for k, v in manifest.items() if k.startswith("avg_")}
    (out_root / "real_codec_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_out", default="data/clic2020_test_combined")
    ap.add_argument("--glc_prof", default="experiments/real_codec/clic_prof_test_glc")
    ap.add_argument("--glc_mobile", default="experiments/real_codec/clic_mobile_test_glc")
    ap.add_argument("--glc_out", default="experiments/real_codec/clic2020_test_glc")
    ap.add_argument("--gp_prof", default="experiments/real_codec/clic_prof_test_gp_reslc_rho116")
    ap.add_argument("--gp_mobile", default="experiments/real_codec/clic_mobile_test_gp_reslc_rho116")
    ap.add_argument("--gp_out", default="experiments/real_codec/clic2020_test_gp_reslc_rho116")
    ap.add_argument("--q_indexes", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    build_orig(Path(args.orig_out), DEFAULT_ORIGS)
    merge_run(Path(args.glc_prof), Path(args.glc_mobile), Path(args.glc_out), args.q_indexes)
    merge_run(Path(args.gp_prof), Path(args.gp_mobile), Path(args.gp_out), args.q_indexes)
    print(f"built {args.orig_out}")
    print(f"built {args.glc_out}")
    print(f"built {args.gp_out}")


if __name__ == "__main__":
    main()
