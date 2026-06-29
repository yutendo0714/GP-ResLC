#!/usr/bin/env python3
"""Train Scratch GP-ResLC Stage A: semantic VQ autoencoder.

This is a research scaffold, not yet the final codec. It learns a compact
semantic/generative code s. Stage B will add y = mu_theta(s) + residual r and
an entropy-coded residual stream.
"""

from __future__ import annotations

import argparse
import glob
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
from torchvision.utils import make_grid, save_image
from PIL import Image

import lpips as lpips_lib
from DISTS_pytorch import DISTS

from gp_reslc.scratch import ScratchVQAutoencoder

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


class CropFolder(Dataset):
    EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

    def __init__(self, root: str, size: int = 256):
        self.paths = sorted(sum([glob.glob(os.path.join(root, e)) for e in self.EXTS], []))
        if not self.paths:
            raise RuntimeError(f"no images found in {root}")
        self.size = int(size)
        self.t = transforms.Compose([
            transforms.RandomCrop(self.size, pad_if_needed=True, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.paths[i]).convert("RGB")
        if min(img.size) < self.size:
            s = self.size / min(img.size)
            img = img.resize((max(self.size, int(img.size[0] * s) + 1),
                              max(self.size, int(img.size[1] * s) + 1)))
        return self.t(img)


def load_matching_state(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[int, list[str]]:
    current = model.state_dict()
    skipped: list[str] = []
    matched = 0
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            current[key] = value
            matched += 1
        else:
            skipped.append(key)
    model.load_state_dict(current)
    return matched, skipped


def make_panel(x: torch.Tensor, x_hat: torch.Tensor, n: int = 4) -> torch.Tensor:
    n = min(n, x.shape[0])
    rows = []
    for i in range(n):
        rows.extend([x[i].detach().cpu(), x_hat[i].detach().cpu()])
    return make_grid(torch.stack(rows), nrow=2).clamp(0, 1)


@torch.no_grad()
def quick_val(model, loader, device, lpips_fn, dists_fn):
    model.eval()
    x = next(iter(loader)).to(device)
    out = model(x)
    x_hat = out["x_hat"].clamp(0, 1)
    l1 = F.l1_loss(x_hat, x).item()
    mse = F.mse_loss(x_hat, x).item()
    lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean().item()
    ds = dists_fn(x_hat, x).mean().item()
    panel = make_panel(x, x_hat)
    model.train()
    return {"l1": l1, "mse": mse, "lpips": lp, "dists": ds,
            "semantic_bpp_fixed": float(out["semantic_bpp_fixed"].item()),
            "perplexity": float(out["perplexity"].item()),
            "codebook_entropy_norm": float(out["codebook_entropy_norm"].item()),
            "soft_codebook_entropy_norm": float(out["soft_codebook_entropy_norm"].item()),
            "soft_perplexity": float(out["soft_perplexity"].item()),
            "codebook_usage_frac": float(out["codebook_usage_frac"].item()),
            "soft_codebook_usage_frac": float(out["soft_codebook_usage_frac"].item()),
            "panel": panel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="experiments/scratch_stage_a")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base_ch", type=int, default=96)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--codebook_size", type=int, default=1024)
    ap.add_argument("--num_down", type=int, default=4, choices=[3, 4, 5],
                    help="number of 2x downsampling stages; 4 gives 16x16 codes, 5 gives 8x8 codes for 256 crops")
    ap.add_argument("--decoder_attention", action="store_true",
                    help="add zero-initialized self-attention at the 8x8 latent and first decoder stage")
    ap.add_argument("--extra_decoder_blocks", type=int, default=0,
                    help="additional latent ResBlocks in the Stage-A decoder")
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_dists", type=float, default=1.0)
    ap.add_argument("--lambda_vq", type=float, default=1.0)
    ap.add_argument("--vq_beta", type=float, default=0.25)
    ap.add_argument("--vq_entropy_tau", type=float, default=1.0)
    ap.add_argument("--lambda_codebook_entropy", type=float, default=0.0,
                    help="Maximize differentiable soft codebook entropy to reduce VQ collapse")
    ap.add_argument("--codebook_restart_every", type=int, default=0)
    ap.add_argument("--codebook_restart_threshold", type=float, default=1e-5)
    ap.add_argument("--codebook_restart_max_fraction", type=float, default=0.10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--resume", default=None, help="optional Stage-A checkpoint to resume from")
    ap.add_argument("--resume_partial", action="store_true",
                    help="load only shape-compatible weights from --resume, useful for architecture extensions")
    ap.add_argument("--freeze_encoder_quantizer", action="store_true",
                    help="train only decoder/latent_refine while keeping semantic encoder and VQ codebook fixed")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="gp-reslc-vcip")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="offline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Scratch Stage A training expects CUDA; stop and restart the container if GPU disappeared.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = (not args.no_wandb) and args.wandb_mode != "disabled" and _WANDB
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   mode=args.wandb_mode, config=vars(args), dir=str(ROOT))

    train_loader = DataLoader(CropFolder(args.data, args.size), batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(CropFolder(args.val or args.data, args.size), batch_size=min(args.bs, 8), shuffle=True,
                            num_workers=max(0, min(args.num_workers, 2)), drop_last=True, pin_memory=True)

    model = ScratchVQAutoencoder(
        args.codebook_size,
        args.latent_dim,
        args.base_ch,
        args.vq_beta,
        args.vq_entropy_tau,
        args.num_down,
        decoder_attention=args.decoder_attention,
        extra_decoder_blocks=args.extra_decoder_blocks,
    ).to(device).train()
    if args.freeze_encoder_quantizer:
        for module in (model.encoder, model.quantizer):
            for param in module.parameters():
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"[freeze_encoder_quantizer] trainable={trainable} frozen={frozen}", flush=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    dists_fn = DISTS().to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    for p in dists_fn.parameters():
        p.requires_grad_(False)

    it = 0
    best_val_dists = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if args.resume_partial:
            matched, skipped = load_matching_state(model, ckpt["model"])
            print(f"[resume_partial] loaded {matched} tensors from {args.resume}; skipped={len(skipped)}", flush=True)
        else:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt and not args.freeze_encoder_quantizer:
                opt.load_state_dict(ckpt["optimizer"])
                for group in opt.param_groups:
                    group["lr"] = args.lr
            elif args.freeze_encoder_quantizer:
                print("[freeze_encoder_quantizer] skipping checkpoint optimizer state", flush=True)
            it = int(ckpt.get("it", 0))
            print(f"[resume] loaded {args.resume} at it={it}", flush=True)

    while it < args.iters:
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            x_hat = out["x_hat"].clamp(0, 1)
            l1 = F.l1_loss(x_hat, x)
            lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean()
            ds = dists_fn(x_hat, x).mean()
            entropy_bonus = out["soft_codebook_entropy_norm"]
            loss = (args.lambda_l1 * l1 + args.lambda_lpips * lp + args.lambda_dists * ds
                    + args.lambda_vq * out["vq_loss"] - args.lambda_codebook_entropy * entropy_bonus)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            restarted = 0
            if (not args.freeze_encoder_quantizer) and args.codebook_restart_every > 0 and it > 0 and it % args.codebook_restart_every == 0:
                restarted = model.quantizer.restart_dead_codes(
                    out["z_e"],
                    threshold=args.codebook_restart_threshold,
                    max_fraction=args.codebook_restart_max_fraction,
                )

            if it % args.log_every == 0:
                print(f"[it {it}] loss={loss.item():.4f} l1={l1.item():.4f} lpips={lp.item():.4f} "
                      f"dists={ds.item():.4f} vq={out['vq_loss'].item():.4f} "
                      f"ppl={out['perplexity'].item():.1f} H={out['codebook_entropy_norm'].item():.3f} "
                      f"soft_ppl={out['soft_perplexity'].item():.1f} softH={out['soft_codebook_entropy_norm'].item():.3f} "
                      f"use={out['codebook_usage_frac'].item():.3f} restart={restarted} "
                      f"bpp_s={out['semantic_bpp_fixed'].item():.5f}", flush=True)
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/l1": l1.item(),
                               "train/lpips": lp.item(), "train/dists": ds.item(),
                               "train/vq_loss": out["vq_loss"].item(),
                               "train/perplexity": out["perplexity"].item(),
                               "train/codebook_entropy_norm": out["codebook_entropy_norm"].item(),
                               "train/soft_codebook_entropy_norm": out["soft_codebook_entropy_norm"].item(),
                               "train/soft_perplexity": out["soft_perplexity"].item(),
                               "train/codebook_usage_frac": out["codebook_usage_frac"].item(),
                               "train/soft_codebook_usage_frac": out["soft_codebook_usage_frac"].item(),
                               "train/codebook_restarted": restarted,
                               "train/semantic_bpp_fixed": out["semantic_bpp_fixed"].item()}, step=it)

            if it % args.eval_every == 0:
                val = quick_val(model, val_loader, device, lpips_fn, dists_fn)
                save_image(val["panel"], out_dir / f"val_panel_{it:07d}.png")
                print(f"  [val {it}] l1={val['l1']:.4f} lpips={val['lpips']:.4f} dists={val['dists']:.4f} "
                      f"ppl={val['perplexity']:.1f} H={val['codebook_entropy_norm']:.3f} "
                      f"soft_ppl={val['soft_perplexity']:.1f} softH={val['soft_codebook_entropy_norm']:.3f} "
                      f"use={val['codebook_usage_frac']:.3f} bpp_s={val['semantic_bpp_fixed']:.5f}", flush=True)
                if val["dists"] < best_val_dists:
                    best_val_dists = val["dists"]
                    torch.save({"it": it, "model": model.state_dict(), "optimizer": opt.state_dict(),
                                "args": vars(args), "best_val_dists": best_val_dists},
                               out_dir / "stage_a_best.pt")
                if use_wandb:
                    wandb.log({"val/l1": val["l1"], "val/mse": val["mse"],
                               "val/lpips": val["lpips"], "val/dists": val["dists"],
                               "val/best_dists": best_val_dists,
                               "val/perplexity": val["perplexity"],
                               "val/codebook_entropy_norm": val["codebook_entropy_norm"],
                               "val/soft_codebook_entropy_norm": val["soft_codebook_entropy_norm"],
                               "val/soft_perplexity": val["soft_perplexity"],
                               "val/codebook_usage_frac": val["codebook_usage_frac"],
                               "val/soft_codebook_usage_frac": val["soft_codebook_usage_frac"],
                               "val/semantic_bpp_fixed": val["semantic_bpp_fixed"],
                               "val/panel": wandb.Image(str(out_dir / f"val_panel_{it:07d}.png"))}, step=it)

            if it > 0 and it % args.save_every == 0:
                torch.save({"it": it, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)},
                           out_dir / f"stage_a_{it:07d}.pt")
            it += 1
            if it >= args.iters:
                break

    torch.save({"it": it, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)},
               out_dir / "stage_a_final.pt")
    if use_wandb:
        wandb.save(str(out_dir / "stage_a_final.pt"))
        wandb.finish()


if __name__ == "__main__":
    main()
