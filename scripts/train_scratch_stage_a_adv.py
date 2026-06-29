#!/usr/bin/env python3
"""Adversarial fine-tuning for Scratch GP-ResLC Stage A."""

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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import lpips as lpips_lib
from DISTS_pytorch import DISTS

from gp_reslc.scratch import ScratchVQAutoencoder, PatchDiscriminator

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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if min(img.size) < self.size:
            s = self.size / min(img.size)
            img = img.resize((max(self.size, int(img.size[0] * s) + 1),
                              max(self.size, int(img.size[1] * s) + 1)))
        return self.t(img)


def load_stage_a(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    a = dict(ckpt.get("args", {}))
    model = ScratchVQAutoencoder(a["codebook_size"], a["latent_dim"], a["base_ch"],
                                 a.get("vq_beta", 0.25), a.get("vq_entropy_tau", 1.0),
                                 a.get("num_down", 4),
                                 decoder_attention=a.get("decoder_attention", False),
                                 extra_decoder_blocks=a.get("extra_decoder_blocks", 0)).to(device)
    model.load_state_dict(ckpt["model"])
    return model, a


def make_panel(x, x_hat, n=4):
    n = min(n, x.shape[0])
    rows = []
    for i in range(n):
        rows.extend([x[i].detach().cpu(), x_hat[i].detach().cpu()])
    return make_grid(torch.stack(rows), nrow=2).clamp(0, 1)


def hinge_d_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def disc_forward_features(disc: PatchDiscriminator, x: torch.Tensor):
    feats = []
    h = x
    for layer in disc.net:
        h = layer(h)
        if isinstance(layer, nn.LeakyReLU):
            feats.append(h)
    return h, feats


def feature_matching_loss(fake_feats, real_feats, ref: torch.Tensor):
    if not fake_feats:
        return ref.new_zeros(())
    return sum(F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats)) / len(fake_feats)


@torch.no_grad()
def quick_val(model, loader, device, lpips_fn, dists_fn):
    model.eval()
    x = next(iter(loader)).to(device)
    out = model(x)
    x_hat = out["x_hat"].clamp(0, 1)
    val = {
        "l1": F.l1_loss(x_hat, x).item(),
        "mse": F.mse_loss(x_hat, x).item(),
        "lpips": lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean().item(),
        "dists": dists_fn(x_hat, x).mean().item(),
        "semantic_bpp_fixed": float(out["semantic_bpp_fixed"].item()),
        "perplexity": float(out["perplexity"].item()),
        "codebook_entropy_norm": float(out["codebook_entropy_norm"].item()),
        "codebook_usage_frac": float(out["codebook_usage_frac"].item()),
        "panel": make_panel(x, x_hat),
    }
    model.train()
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="experiments/scratch_stage_a_adv")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr_g", type=float, default=5e-5)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--disc_ch", type=int, default=64)
    ap.add_argument("--lambda_l1", type=float, default=0.5)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_dists", type=float, default=1.0)
    ap.add_argument("--lambda_vq", type=float, default=1.0)
    ap.add_argument("--lambda_codebook_entropy", type=float, default=0.5)
    ap.add_argument("--lambda_adv", type=float, default=0.02)
    ap.add_argument("--lambda_fm", type=float, default=0.0)
    ap.add_argument("--adv_start_iter", type=int, default=0)
    ap.add_argument("--disc_update_every", type=int, default=1)
    ap.add_argument("--freeze_encoder_quantizer", action="store_true")
    ap.add_argument("--codebook_restart_every", type=int, default=200)
    ap.add_argument("--codebook_restart_threshold", type=float, default=1e-5)
    ap.add_argument("--codebook_restart_max_fraction", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="gp-reslc-vcip")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="offline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Stage-A adversarial fine-tuning expects CUDA; stop if GPU disappeared.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, stage_a_args = load_stage_a(args.stage_a_ckpt, device)
    model.train()
    if args.freeze_encoder_quantizer:
        for module_name in ("encoder", "quantizer"):
            module = getattr(model, module_name)
            for p in module.parameters():
                p.requires_grad_(False)
    disc = PatchDiscriminator(base_ch=args.disc_ch).to(device).train()
    g_params = [p for p in model.parameters() if p.requires_grad]
    if not g_params:
        raise RuntimeError("no trainable generator parameters left after freezing")
    opt_g = torch.optim.AdamW(g_params, lr=args.lr_g, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.9), weight_decay=1e-4)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    dists_fn = DISTS().to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    for p in dists_fn.parameters():
        p.requires_grad_(False)

    use_wandb = (not args.no_wandb) and args.wandb_mode != "disabled" and _WANDB
    if use_wandb:
        cfg = vars(args).copy()
        cfg.update({f"stage_a/{k}": v for k, v in stage_a_args.items()})
        wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=cfg, dir=str(ROOT))

    train_loader = DataLoader(CropFolder(args.data, args.size), batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(CropFolder(args.val or args.data, args.size), batch_size=min(args.bs, 8), shuffle=True,
                            num_workers=max(0, min(args.num_workers, 2)), drop_last=True, pin_memory=True)
    best_val_dists = float("inf")
    it = 0
    while it < args.iters:
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            adv_active = it >= args.adv_start_iter and args.lambda_adv > 0
            d_loss = x.new_zeros(())
            if adv_active and args.disc_update_every > 0 and it % args.disc_update_every == 0:
                with torch.no_grad():
                    out_det = model(x)
                    fake_det = out_det["x_hat"].clamp(0, 1)
                real_logits = disc(x)
                fake_logits = disc(fake_det.detach())
                d_loss = hinge_d_loss(real_logits, fake_logits)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            out = model(x)
            x_hat = out["x_hat"].clamp(0, 1)
            l1 = F.l1_loss(x_hat, x)
            lp = lpips_fn(x_hat * 2 - 1, x * 2 - 1).mean()
            ds = dists_fn(x_hat, x).mean()
            adv = x.new_zeros(())
            fm = x.new_zeros(())
            if adv_active:
                fake_logits_g, fake_feats = disc_forward_features(disc, x_hat)
                adv = -fake_logits_g.mean()
                if args.lambda_fm > 0:
                    with torch.no_grad():
                        _, real_feats = disc_forward_features(disc, x)
                    fm = feature_matching_loss(fake_feats, real_feats, x)
            loss_g = (args.lambda_l1 * l1 + args.lambda_lpips * lp + args.lambda_dists * ds
                      + args.lambda_vq * out["vq_loss"]
                      - args.lambda_codebook_entropy * out["soft_codebook_entropy_norm"]
                      + args.lambda_adv * adv + args.lambda_fm * fm)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(g_params, 1.0)
            opt_g.step()
            restarted = 0
            if (not args.freeze_encoder_quantizer and args.codebook_restart_every > 0
                    and it > 0 and it % args.codebook_restart_every == 0):
                restarted = model.quantizer.restart_dead_codes(out["z_e"], args.codebook_restart_threshold,
                                                               args.codebook_restart_max_fraction)

            if it % args.log_every == 0:
                print(f"[it {it}] g={loss_g.item():.4f} d={d_loss.item():.4f} adv={adv.item():.4f} "
                      f"fm={fm.item():.4f} adv_on={int(adv_active)} "
                      f"l1={l1.item():.4f} lpips={lp.item():.4f} dists={ds.item():.4f} "
                      f"vq={out['vq_loss'].item():.4f} ppl={out['perplexity'].item():.1f} "
                      f"use={out['codebook_usage_frac'].item():.3f} restart={restarted} "
                      f"bpp_s={out['semantic_bpp_fixed'].item():.5f}", flush=True)
                if use_wandb:
                    wandb.log({"train/g_loss": loss_g.item(), "train/d_loss": d_loss.item(),
                               "train/adv": adv.item(), "train/fm": fm.item(),
                               "train/adv_active": int(adv_active), "train/l1": l1.item(),
                               "train/lpips": lp.item(), "train/dists": ds.item(),
                               "train/vq_loss": out["vq_loss"].item(),
                               "train/perplexity": out["perplexity"].item(),
                               "train/codebook_usage_frac": out["codebook_usage_frac"].item(),
                               "train/semantic_bpp_fixed": out["semantic_bpp_fixed"].item(),
                               "train/codebook_restarted": restarted}, step=it)

            if it % args.eval_every == 0:
                val = quick_val(model, val_loader, device, lpips_fn, dists_fn)
                panel_path = out_dir / f"val_panel_{it:07d}.png"
                save_image(val["panel"], panel_path)
                print(f"  [val {it}] l1={val['l1']:.4f} lpips={val['lpips']:.4f} dists={val['dists']:.4f} "
                      f"ppl={val['perplexity']:.1f} use={val['codebook_usage_frac']:.3f}", flush=True)
                if val["dists"] < best_val_dists:
                    best_val_dists = val["dists"]
                    torch.save({"it": it, "model": model.state_dict(), "disc": disc.state_dict(),
                                "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
                                "args": vars(args), "stage_a_args": stage_a_args,
                                "best_val_dists": best_val_dists}, out_dir / "stage_a_adv_best.pt")
                if use_wandb:
                    wandb.log({"val/l1": val["l1"], "val/mse": val["mse"],
                               "val/lpips": val["lpips"], "val/dists": val["dists"],
                               "val/best_dists": best_val_dists,
                               "val/perplexity": val["perplexity"],
                               "val/codebook_usage_frac": val["codebook_usage_frac"],
                               "val/semantic_bpp_fixed": val["semantic_bpp_fixed"],
                               "val/panel": wandb.Image(str(panel_path))}, step=it)

            if it > 0 and it % args.save_every == 0:
                torch.save({"it": it, "model": model.state_dict(), "disc": disc.state_dict(),
                            "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
                            "args": vars(args), "stage_a_args": stage_a_args}, out_dir / f"stage_a_adv_{it:07d}.pt")
            it += 1
            if it >= args.iters:
                break

    torch.save({"it": it, "model": model.state_dict(), "disc": disc.state_dict(),
                "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
                "args": vars(args), "stage_a_args": stage_a_args}, out_dir / "stage_a_adv_final.pt")
    if use_wandb:
        wandb.save(str(out_dir / "stage_a_adv_final.pt"))
        wandb.finish()


if __name__ == "__main__":
    main()
