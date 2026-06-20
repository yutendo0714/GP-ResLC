#!/usr/bin/env python3
"""Audit local datasets against the official GLC image-evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEFAULT_DATASETS = {
    "clic2020_test": ["/dpl/clic/professional/test", "/dpl/clic/mobile/test"],
    "clic_prof_test": ["/dpl/clic/professional/test"],
    "clic_mobile_test": ["/dpl/clic/mobile/test"],
    "clic_prof_valid": ["/dpl/clic/professional/valid"],
    "clic_mobile_valid": ["/dpl/clic/mobile/valid"],
    "clic_all_available_no_macosx": [
        "/dpl/clic/professional/test",
        "/dpl/clic/professional/valid",
        "/dpl/clic/mobile/valid",
        "/dpl/clic/mobile/test",
    ],
    "div2k_validation": ["/dpl/div2k"],
    "kodak": ["/dpl/kodak"],
}

EXPECTED = {
    "clic2020_test": {"images": 428, "patches_diag2": 28650},
    "clic_prof_test": {"images": 250, "patches_diag2": 16626},
    "clic_mobile_test": {"images": 178, "patches_diag2": 12024},
    "div2k_validation": {"images": 100, "patches_diag2": 6573},
    "kodak": {"images": 24, "patches_diag2": 192},
}


def files_for_dirs(dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        files.extend(
            p for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
        )
    return sorted(files, key=lambda p: (str(p.parent), p.name))


def count_offsets(h: int, w: int, offsets: list[tuple[int, int]], patch: int = 256) -> int:
    total = 0
    for oy, ox in offsets:
        hh = h - oy
        ww = w - ox
        if hh >= patch and ww >= patch:
            total += (hh // patch) * (ww // patch)
    return total


def official_diag2_count(h: int, w: int, patch: int = 256, split_patch_num: int = 2) -> int:
    offsets = [(0, 0)]
    unit = patch // split_patch_num
    for i in range(1, split_patch_num):
        limit = (2.0 - i / split_patch_num) * patch
        if h >= limit and w >= limit:
            offsets.append((unit * i, unit * i))
    return count_offsets(h, w, offsets, patch=patch)


def pad_to(v: int, m: int) -> int:
    return int(math.ceil(v / m) * m)


def digest_file_list(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def audit_dataset(name: str, dirs: list[str]) -> dict[str, str]:
    files = files_for_dirs(dirs)
    rows = []
    total_pixels = 0
    for p in files:
        with Image.open(p) as im:
            w, h = im.size
        total_pixels += h * w
        rows.append((p, w, h))

    diag2 = sum(official_diag2_count(h, w) for _, w, h in rows)
    four = sum(count_offsets(h, w, [(0, 0), (0, 128), (128, 0), (128, 128)]) for _, w, h in rows)
    three_no_diag = sum(count_offsets(h, w, [(0, 0), (0, 128), (128, 0)]) for _, w, h in rows)
    pad64_diag2 = sum(official_diag2_count(pad_to(h, 64), pad_to(w, 64)) for _, w, h in rows)
    min_patches = min([official_diag2_count(h, w) for _, w, h in rows], default=0)
    max_patches = max([official_diag2_count(h, w) for _, w, h in rows], default=0)

    expected = EXPECTED.get(name, {})
    expected_patches = expected.get("patches_diag2")
    expected_images = expected.get("images")
    status = "ok" if expected else "reference_only"
    if expected_images is not None and len(files) != expected_images:
        status = "image_count_mismatch"
    if expected_patches is not None and diag2 != expected_patches:
        status = "patch_count_mismatch" if status == "ok" else status + "+patch_count_mismatch"

    return {
        "dataset": name,
        "dirs": ";".join(dirs),
        "image_count": str(len(files)),
        "expected_images": str(expected_images) if expected_images is not None else "",
        "pixels": str(total_pixels),
        "patches_diag2": str(diag2),
        "expected_patches_diag2": str(expected_patches) if expected_patches is not None else "",
        "patches_four_offsets": str(four),
        "patches_three_no_diag": str(three_no_diag),
        "patches_pad64_diag2": str(pad64_diag2),
        "min_patches_per_image": str(min_patches),
        "max_patches_per_image": str(max_patches),
        "file_list_sha16": digest_file_list(files),
        "status": status,
    }


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# GLC Protocol Audit",
        "",
        "Patch counts use the official public GLC `update_patch_fid` logic: 256x256 non-overlap patches plus one diagonal 128-pixel shifted grid (`split_patch_num=2`).",
        "",
        "| dataset | images | expected images | diag2 patches | expected patches | status | notes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        note = ""
        if r["dataset"] == "clic2020_test":
            note = "Professional test plus mobile test: 250 + 178 images, matching the HiFiC/GLC CLIC2020-test protocol."
        elif r["dataset"] == "clic_prof_test":
            note = "Professional subset only; not the full CLIC2020 test protocol by itself."
        elif r["dataset"] == "clic_mobile_test":
            note = "Mobile subset only; combine with professional test for the full CLIC2020 test protocol."
        elif r["dataset"] == "div2k_validation":
            note = "Matches the GLC supplement patch count." if r["status"] == "ok" else "Check DIV2K split."
        elif r["dataset"] == "kodak":
            note = "Matches 24-image Kodak and 192 official-style patches; GLC omits FID/KID in paper due small sample."
        lines.append(
            f"| {r['dataset']} | {r['image_count']} | {r['expected_images']} | "
            f"{r['patches_diag2']} | {r['expected_patches_diag2']} | {r['status']} | {note} |"
        )
    lines.extend([
        "",
        "Alternative offset counts are kept in the CSV for forensic checks, but they are not the official public GLC metric implementation.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="experiments/protocol_audit")
    args = ap.parse_args()
    rows = [audit_dataset(name, dirs) for name, dirs in DEFAULT_DATASETS.items()]
    out_dir = Path(args.out_dir)
    write_csv(rows, out_dir / "glc_protocol_audit.csv")
    write_md(rows, out_dir / "glc_protocol_audit.md")
    (out_dir / "glc_protocol_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row["dataset"], row["status"], row["image_count"], row["patches_diag2"])


if __name__ == "__main__":
    main()
