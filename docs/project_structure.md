# GP-ResLC Project Structure

This repository is the official GLC codebase plus GP-ResLC research additions.
Keep the official GLC files readable, and keep GP-ResLC-specific work in clearly
named overlays.

## Canonical Tree

- `src/`: upstream GLC model, codec, and metric utilities. Treat this as the
  compatibility layer with official GLC unless a change is explicitly needed.
- `gp_reslc/`: GP-ResLC method code.
  - `prior_predictor.py`, `perceptual_gate.py`, `real_codec.py`: pretrained-GLC
    extension path.
  - `scratch/`: full-design / scratch GP-ResLC research modules.
- `scripts/`: executable experiment, evaluation, analysis, and paper-asset
  entrypoints. See `scripts/README.md`.
- `docs/`: research notes, protocol decisions, result summaries, and paper
  planning.
- `configs/`: example JSON configs for older orchestration scripts.
- `pretrained/`: local pretrained weights.
- `data/`: local symlink/manifests only. Canonical runtime paths:
  - `data/clic2020_test_combined/`
  - `data/subsets/openimages_v6_test_32/`
- `experiments/`: generated checkpoints, reconstructions, metrics, figures, and
  scratch outputs. This is intentionally ignored by git.
- `wandb/`: local wandb run cache.

## What Not To Duplicate

- Do not recreate top-level `datasets/`, `data_splits/`, or `data_subsets/`.
  Use `data/`.
- Do not copy `/dpl` payloads into the repository. Use symlinks or direct
  `/dpl/...` paths.
- Do not move existing experiment outputs casually. Many docs, CSVs, and wandb
  run names refer to those paths.

## Current Stable Runtime Paths

- Kodak: `/dpl/kodak`
- DIV2K validation/eval root: `/dpl/div2k`
- CLIC2020 combined test symlink set: `data/clic2020_test_combined`
- CLIC2020 source subsets:
  - `/dpl/clic/professional/test`
  - `/dpl/clic/mobile/test`

## Cleanup Policy

- Safe to remove: `__pycache__/`, temporary smoke outputs that are explicitly
  superseded and documented, failed one-off recon grids with no metrics.
- Do not remove without a written note: `experiments/real_codec/`,
  `experiments/paper_assets/`, official-protocol metric CSVs, or any checkpoint
  that is cited in `docs/experiment_log.md`.
- New experiment names should start with one of:
  - `pretrained_...`
  - `glc_latent_...`
  - `scratch_stage_a_...`
  - `scratch_stage_b_...`
  - `realcodec_...`
  - `eval_...`

