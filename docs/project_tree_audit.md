# GP-ResLC Project Tree Audit

Last updated: 2026-06-22 JST

## Current Research Status

### Pretrained GLC-Based Branch

This is still the paper-facing performance lead.

- Lead checkpoint: `experiments/v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/v2_final.pt`
- Mechanism: frozen official GLC plus decoder-recomputable perceptual quantization gate `rho(z_hat, q)`.
- Real-codec lead versus local GLC:
  - CLIC2020 test: DISTS/FID BD-rate `-10.28% / -7.30%`, LPIPS BD `+0.19%`.
  - DIV2K validation: DISTS/FID BD-rate `-10.79% / -5.61%`, LPIPS BD `-0.54%`.
  - Kodak: DISTS/FID BD-rate `-4.47% / -1.70%`, LPIPS BD `-0.79%`.
- New balanced schedule candidate:
  - `gate_alpha_by_q = [0.25, 0.25, 0.75, 0.75]`.
  - CLIC100 scout: DISTS BD `-2.60%`, LPIPS BD `-0.74%`.
  - Kodak24 scout: DISTS BD `-2.86%`, LPIPS BD `-0.92%`.
  - DIV2K20 scout: DISTS BD `-2.50%`, LPIPS BD `-0.04%`.
  - CLIC2020 full q0 only: bpp `0.02059383`, LPIPS `0.15679656`, DISTS `0.08201325`, FID `6.221714`, KID `0.00129098`, patches `28,650`.

Interpretation: `rho1.16` remains the strongest headline. The balanced schedule is useful as a safer controllability setting because it reduces the low-q LPIPS/FID/KID damage.

### Scratch / Complete-Design Branch

This branch is method-faithful but not yet competitive with official GLC.

- Stage-A VQ autoencoder works and avoids collapse.
- Stage-B residual coding validates the decomposition idea: semantic code plus sparse residual improves perceptual quality at low proxy bpp.
- Current best scratch low-rate point is still around Kodak-center bpp `0.01321`, LPIPS `0.43869`, DISTS `0.42313`.
- Progressive/sparse residual variants improved mechanism evidence but not the absolute quality enough to replace the pretrained branch.

Interpretation: scratch is a top-conference long-horizon branch, not the current paper-facing performance lead. Its value is novelty and conceptual purity; it needs stronger semantic generator quality and hard-gate-aware residual objectives.

## How The Last Gate-Schedule Experiments Were Run

Before `scripts/evaluate_real_codec.py` had an official CLI for q-wise gate scaling, the experiments used an in-memory wrapper around the loaded checkpoint:

```python
rho = 1.0 + alpha_q * (rho - 1.0)
```

The wrapper was attached to `net.perceptual_gate` at runtime. Compression and decompression still used `compress_to_real_bitstream()` and `decompress_from_real_bitstream()`, so bpp was real payload bytes, not estimated likelihood. The only missing piece was a persistent CLI surface for reproducing the schedule.

That gap is now closed: `scripts/evaluate_real_codec.py` supports:

```bash
--gate_alpha_by_q 0.25 0.25 0.75 0.75
```

## Dataset / Subset Tree

These are small symlink or manifest directories, not heavy duplicate datasets. The repo now keeps a single canonical `data/` tree:

- `data/clic2020_test_combined/`: symlink union of CLIC2020 professional+mobile test images.
- `data/subsets/openimages_v6_test_32/`: tiny development subset for smoke/sanity runs only.

Cleanup decision on 2026-06-22: removed redundant top-level `datasets/`, `data_splits/`, and `data_subsets/`. They only contained duplicate symlinks, so no `/dpl` dataset payloads or generated experiment artifacts were deleted.

## Heavy Local Artifact Areas

Repository size is dominated by generated experiment artifacts:

- `experiments/`: about `131G`.
- `experiments/real_codec/`: about `39G`.
- `experiments/eval_v2_gate_send_lR10_lp4_rho14_target116_send5_all_6k/`: about `8.2G`.
- `.venv/`: about `6.1G`.
- `wandb/`: about `858M`.

The heavy files are mostly reconstructed PNGs for FID/KID and qualitative inspection. Many are reproducible, but paper-facing reconstructions are expensive to regenerate and should not be deleted casually.

## Keep / Archive / Delete Policy

Keep:

- `experiments/paper_assets/`
- `experiments/protocol_audit/`
- `experiments/real_codec/clic2020_test_glc/`
- `experiments/real_codec/clic2020_test_gp_reslc_rho116/`
- `experiments/real_codec/div2k_glc/`
- `experiments/real_codec/div2k_gp_reslc_rho116/`
- `experiments/real_codec/kodak_glc/`
- `experiments/real_codec/kodak_gp_reslc_rho116/`
- lead checkpoints and their analysis dirs.

Archive candidate:

- older `v0_glc_*` estimated-bpp recon roots.
- failed `unfreeze_entropy` real-codec runs.
- exploratory selector/top-k recon-save dirs where final CSV/JSON already exists.
- duplicate CLIC professional/mobile partial roots if full CLIC428 roots and metrics are available.

Delete candidate after confirming no unique metrics:

- Python caches (`__pycache__`, `.pytest_cache`).
- smoke-test recon directories.
- interrupted or superseded scout recon-save folders.

No large experiment directory was deleted in this audit because several artifact paths are still referenced by docs and paper tables.
