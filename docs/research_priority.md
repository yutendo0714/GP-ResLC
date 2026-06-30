# Research Priority

## Decision

The active GP-ResLC research line is now:

```text
official pretrained GLC image codec
+ GP-ResLC residual/control/entropy modules
+ real-codec multi-rate evaluation
```

Scratch training is removed from the active performance path. It remains archived
as background evidence only.

The goal is no longer to protect a novelty-first short-track story. The goal is
to move the real serialized bpp versus perceptual-quality curve left/down against
official pretrained GLC.

## Main Thesis

GP-ResLC should be treated as a residual transmission allocation method on top of
pretrained GLC:

```text
pretrained GLC already sends z_hat, q, and decoded context.
GP-ResLC uses those decoder-available signals to avoid sending residual
information or residual precision that the decoder-side generative prior can
recover, and spends counted bits only on unpredictable residual/control.
```

Do not define the project as adaptive quantization for GLC. The current rho
branch is a useful safety baseline and minimum implementation of the thesis, but
it is not the final ceiling.

## Active Branch Roles

### 1. Current rho branch

Keep as the safety lead and paired comparison anchor.

- frozen pretrained GLC
- decoder-recomputable `rho(z_hat, q)`
- residual quantization coarsening
- no side map
- real serialized codec bpp

Role:

```text
proof that decoder-computable generator-predictability can reduce transmitted
residual precision without side bits
```

Do not keep spending major time on rho-target-only, rho-max-only, loss-weight-only
sweeps unless they unblock a stronger mainline experiment.

### 2. Safe-to-drop teacher

Promote this to a shared mainline component rather than an ablation.

Train a predictor to identify where residual precision can be reduced with small
local perceptual/fidelity damage and meaningful bit saving. The teacher can use
synthetic perturbations, local DISTS/LPIPS/L1/gradient damage, and local bit
saving. Later semantic/DINO/edge features may be added as guards, but they should
not be the first dependency.

Use this teacher for:

- rho/stage-quant allocation
- residual precision allocation
- tiny control stream labels
- benefit-cost residual RDO

### 3. Stage-aware residual-variable coding

This is the closest implementation of the original GP-ResLC proposal.

For each GLC four-part prior stage:

```text
y_stage = base_mean_stage + gp_mu_stage(context) + residual_stage
```

Encode only `residual_stage`, while preserving GLC's original four-part decoding
order and decoder context distribution.

Critical constraints:

- `gp_mu_stage` must use only decoder-available signals.
- Zero initialization must reproduce GLC.
- Decode consistency must be exact.
- Residual symbols must have a real entropy coder.
- Serialized bpp must include every transmitted symbol.

### 4. Tiny counted control stream

Zero-side rho may saturate because `z_hat` does not necessarily contain all
safe-to-drop information. A tiny control stream is allowed if every bit is counted.

Target budget:

```text
0.0005 to 0.0020 bpp
```

Candidate controls:

- low-resolution safe-to-drop tokens
- residual precision-level tokens
- local rho-offset tokens
- sparse correction locations
- residual importance classes

This is still on-axis because it sends only the unpredictable control needed to
decide what residual information can be dropped.

### 5. Learned residual/control entropy model

Any residual or control stream must be coded seriously. Borrow mechanisms from
modern LIC/GIC only as implementation references:

- finite-support categorical residual priors
- top-k / escape coding
- Gaussian or logistic conditional models
- context-conditioned scales/means
- lightweight channel/spatial context

Do not replace the GLC base codec.

## Experiment Execution Policy

Because research time is limited, new experiments should be launched as complete
candidate systems, not as tiny isolated demos.

Preferred pattern:

```text
full mechanism implementation
-> smoke/decode-consistency check
-> real-codec Kodak quick curve
-> real-codec DIV2K/CLIC if promising
-> diagnostics only when the result fails or mechanism evidence is needed
-> ablations after a branch beats the current lead or becomes paper-relevant
```

This does not mean skipping sanity checks. It means avoiding weeks of diagnostic
micro-experiments before attempting the real mechanism. Every serious branch must
still pass:

- encode/decode consistency
- counted serialized bpp
- multi-q curve evaluation
- paired GLC comparison
- no uncounted side information

## What To Avoid

- scratch training as an active path
- replacing GLC with another base codec
- rho-target-only tuning as the main work
- quantization-width-only story
- loss-weight-only optimization
- estimated-bpp claims
- center-crop-only conclusions
- single-point wins without a curve
- uncounted side/control information
- broad GLC unfreezing without codec consistency checks
- optimizing Kodak only

## Promotion Criteria

Promote a branch only if it improves over the current rho1.16 safety lead under
real serialized codec evaluation, or clearly exposes a stronger path.

Minimum evidence:

- DISTS and/or FID BD-rate improvement over local GLC real codec
- comparison against the current rho lead
- actual bpp including all streams
- no reconstruction mismatch between encode/decode and evaluation path
- at least Kodak plus one larger set before making strong claims
- CLIC2020/DIV2K before treating it as the new mainline

## Practical Priority Order

1. Freeze current rho1.16 and rho1.12 checkpoints as anchors.
2. Implement safe-to-drop teacher as a reusable supervision module.
3. Use it to train a stronger stage-aware rho/stage-quant candidate.
4. Implement stage-aware residual-mu coding inside the four-part prior.
5. Add learned residual entropy coding.
6. If zero-side saturates, add a tiny counted control stream.
7. Add semantic/DINO/edge guards only after the core mechanism shows rate gain.
8. Run ablations after a branch has real-codec promise.

