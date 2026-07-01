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

Current decision after the 2026-06-30 review:

- Keep the current rho/stage-entropy branch as the safety lead.
- Do not restart from scratch.
- Do not replace GLC with CoD, StableCodec, RDVQ, or another base codec.
- Treat external 2025-2026 SOTA methods as implementation references, teachers,
  or entropy-coding design references, not as base-codec replacements.
- Move the main research beyond rho/scale/quantization-width tuning.
- The next highest-value mainline is stage-aware residual-variable coding with a
  serious residual entropy model, supported by safe-coarsening / residual-RDO
  teachers.
- 2026-07-01 synthesis: before implementing another large branch, run
  diagnostic-first checks for hard residual omission and same-bpp generative
  residual synthesis.  See
  `docs/mainline_research_synthesis_2026_07_01.md` and
  `docs/mainline_brs_synthesis_diagnostic_plan_2026_07_01.md`.

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

Current frozen lead:

```text
experiments/stage_safe_rdo_gate_from_sb03_2000/v2_final.pt
```

Verified CLIC2020 combined real-codec result versus local GLC:

- DISTS BD-rate: `-9.76%`
- FID BD-rate: `-6.42%`
- KID BD-rate: `-4.65%`
- LPIPS BD-rate: `+0.05%`

Verified DIV2K validation real-codec result versus local GLC:

- DISTS BD-rate: `-10.37%`
- FID BD-rate: `-5.64%`
- KID BD-rate: `-4.98%`
- LPIPS BD-rate: `-0.49%`

Interpretation:

- The branch is not a toy proxy; it moves the real serialized bpp curve on the
  main perceptual/distribution metrics.
- LPIPS remains weak and must not be overclaimed.
- The result is a safety lead and mechanism evidence, not the full GP-ResLC
  method.

Do not keep spending major time on rho-target-only, rho-max-only, loss-weight-only
sweeps unless they unblock a stronger mainline experiment.

### 2. Stage-aware residual-variable coding

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

This is the next mainline target because it changes the actual coded variable
from the original `y_stage` toward an unpredictable residual stream. If it works,
the story is no longer "adaptive quantization width for GLC"; it becomes
"stage-consistent predictable/unpredictable residual coding on top of pretrained
GLC."

### 3. Learned residual/control entropy model

Any residual or control stream must be coded seriously. Borrow mechanisms from
modern LIC/GIC only as implementation references:

- finite-support categorical residual priors
- top-k / escape coding
- Gaussian or logistic conditional models
- context-conditioned scales/means
- lightweight channel/spatial context
- grouped or checkerboard-style context where it preserves GLC decode order

Do not replace the GLC base codec.

This component should be developed together with stage-aware residual-variable
coding. A residual stream without a strong entropy model will likely lose its
rate advantage.

### 4. Safe-to-drop teacher / residual RDO

Promote this to a shared mainline component rather than an ablation.

Train a predictor to identify where residual precision can be reduced with small
local perceptual/fidelity damage and meaningful bit saving. The teacher can use
synthetic perturbations, local DISTS/LPIPS/L1/gradient damage, and local bit
saving. Later semantic/DINO/edge features may be added as guards, but they should
not be the first dependency.

Use this teacher for:

- rho/stage-quant allocation
- residual precision allocation
- residual-variable coding supervision
- tiny control stream labels
- benefit-cost residual RDO

Important framing:

- Safe-coarsening is not just another rho target.
- It should teach generator-predictable / safe-to-drop regions and local
  benefit-per-bit.
- High rho or low residual precision should be rewarded only where the local
  perceptual/fidelity damage is small.

### 5. Tiny counted control stream

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

Current status:

- A counted control path exists and is codec-correct.
- The latest control-only branch did not beat the current lead, so control should
  not be pushed by budget sweeps alone.
- Fixed-prior source-side residual-control probes on Kodak8 also failed to beat
  the Safe-RDO lead after control bits were counted:
  `g16_f0025_d005` gives DISTS/FID `-2.45%/-4.30%` versus GLC, and
  `g16_f005_d010` gives `-1.58%/-2.80%`, both weaker than SafeRDO
  `-6.00%/-7.38%`.
- Revisit control after a stronger teacher, residual RDO, or learned control
  entropy model exists.  If revisited, the control prior should be
  decoder-computable and stage-wise, not an encoder-only probability map.

### 6. Decoder-side residual synthesis

This is now a validated mechanism signal, not yet the lead.

Current result:

- A decoder-computable stage-3 value synthesizer can be added after normal
  SafeRDO arithmetic decoding with no extra side bits.
- Kodak8 same-code comparison at identical serialized bpp:
  - plain additive scale 0.5: DISTS `+0.24%`, LPIPS `-1.50%`, PSNR `-3.93%`,
    FID `-1.00%` versus SafeRDO.
  - plain additive scale 1.0: stronger LPIPS/PSNR but too much DISTS cost.
  - DISTS-aware safe-hinge training did not beat the plain conservative
    additive setting.

Interpretation:

```text
decoder-side synthesis can recover some residual quality without bits,
but unconditional synthesis perturbs DISTS-sensitive structure.
```

Next:

- Do not keep doing amplitude/loss sweeps.
- Make synthesis selective:
  - decoder-computable confidence gate,
  - local safe-to-synthesize teacher,
  - or tiny counted control stream for synthesis/residual necessity.
- This branch should eventually be combined with residual omission or
  residual-variable coding.  Alone it improves quality at the same bpp; it does
  not move the bpp curve left unless paired with actual bit saving.

## External Reference Use Policy

Use recent SOTA methods as targeted references, not as replacements for the GLC
base. The purpose is to strengthen GP-ResLC residual/control/entropy modules.

Highest-priority references:

- CADC-style uncertainty/content-adaptive allocation:
  use for safe-coarsening teacher and local residual precision decisions.
- DLF-style semantic/detail decomposition:
  use for predictable/unpredictable residual split and detail-critical residual
  selection.
- RDVQ-style rate-aware token coding and top-k/escape discipline:
  use for residual/control tokens and actual bitstream design.
- CompressAI / ELIC / MLIC-style entropy modeling:
  use for Gaussian/logistic conditionals, hyperprior discipline, and grouped
  context models for residual/control streams.

Secondary references:

- ResULIC:
  use for semantic/perceptual residual importance teachers.
- Control-GIC:
  use for dynamic granularity or precision-level control tokens.
- StableCodec / CoD / CoD-Lite / OneDC / AEIC:
  use as teacher signals, diagnostics, or upper-bound references for
  generator-recoverable texture/detail, not as the GP-ResLC base.

Working rule:

```text
Borrow mechanisms only when they help decide what residual/control information
to transmit or how to entropy-code it. Do not turn GP-ResLC into a different
codec family.
```

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

Promote a branch only if it improves over the current Safe-RDO
rho/stage-entropy safety lead under real serialized codec evaluation, or clearly
exposes a stronger path.

Minimum evidence:

- DISTS and/or FID BD-rate improvement over local GLC real codec
- comparison against the current rho lead
- actual bpp including all streams
- no reconstruction mismatch between encode/decode and evaluation path
- at least Kodak plus one larger set before making strong claims
- CLIC2020/DIV2K before treating it as the new mainline

Mechanism-specific success criteria:

- Safe-coarsening / residual RDO:
  high-rho or low-precision regions should move away from high-error,
  high-gradient, structure-sensitive areas; saved bits should come from regions
  with low local perceptual/fidelity damage.
- Stage-aware residual-variable coding:
  residual stream entropy should be lower than the original coded `y` stream at
  matched quality; zero-init must reproduce GLC; decode equality must remain
  exact.
- Learned residual/control entropy model:
  actual serialized bpp must decrease after coding overhead is counted; rate
  savings must survive Kodak, DIV2K, and CLIC2020 rather than only estimated
  likelihood.
- z_hat entropy coding:
  the decoded `z_hat` and final reconstruction must be exactly unchanged; the
  probability table must be trained from non-evaluation data; all mode prefixes
  and coding bytes must be counted in serialized bpp.
- Tiny counted control:
  total bpp including control must improve BD-rate; control ablation should hurt;
  the control stream must remain small, entropy-coded, and counted.
- Semantic/perceptual guards:
  they are training-time guidance unless explicitly transmitted; no semantic or
  edge side information may be used at decode time unless its bits are counted.

## Immediate Deliverables For The Next Mainline

Do not start with more rho/scale sweeps. The next useful implementation artifacts
are:

1. A stage-aware residual-variable codec path:
   per-stage `gp_mu_stage`, finite-support residual symbols, exact decode
   consistency, and real serialized residual bpp.
2. A residual entropy model:
   Gaussian/logistic or categorical residual priors, tail/escape handling if
   needed, and per-stream bpp breakdown.
3. A local safe-coarsening / residual-RDO teacher:
   local coarsening perturbations, local damage maps, saved-bit maps,
   `safe_to_coarsen_score`, and `value_per_bit` labels.
4. A quick but real validation ladder:
   smoke consistency -> Kodak curve -> DIV2K/CLIC curve if promising.
5. Diagnostic maps only when needed:
   rho/residual precision maps, local error/gradient correlation, residual
   entropy maps, and control maps if a control stream is used.

## Practical Priority Order

1. Freeze the current Safe-RDO rho/stage-entropy lead as the anchor.
2. Treat q-aware rho-target and safe-weighted stage-mean continuations as
   rejected diagnostics unless they beat the Safe-RDO lead on a larger
   real-codec set.
3. Keep selective no-side residual synthesis as the safest generator-recovery
   branch.  It is still small, but it is the only recent branch that improves
   all Kodak8 metrics over the same SafeRDO anchor.  Omitted-aware synthesis
   should train on the same omitted-cell distribution used by the real codec;
   rho-threshold training and counted send-mask evaluation must not be mixed.
4. Keep zero-distortion entropy improvements active where validated, but do not
   use them as the main GP-ResLC claim.  OpenImages-trained static/auto
   `z_hat` entropy coding is now validated with real serialized bpp on
   Kodak24, DIV2K validation, and full CLIC2020 test 428.  It is a useful
   codec cleanup, but it is not specific to residual/generator-recovery coding
   and could also be applied to the baseline.  Therefore:
   - Main paper-facing comparisons should prioritize fixed-z or z-excluded
     BD-rate.
   - z-entropy results may be reported as an auxiliary full-codec package.
   - The z-included conservative package reaches about `-5%` to `-10%`
     BD-rate on several metrics, but this should not be presented as the
     intrinsic GP-ResLC method gain.
   - The z-excluded selective synthesis audit is much smaller: CLIC2020 test
     `+0.18%` DISTS / `-0.65%` LPIPS / `-0.17%` FID, DIV2K validation
     `-0.52%` DISTS / `-0.72%` LPIPS / `-0.47%` FID, and Kodak24 `-0.16%`
     DISTS / `-0.74%` LPIPS / `-0.61%` FID versus the same fixed-z SafeRDO
     anchor.  This confirms that large future gains must come from `y`/
     residual/control coding, not from `z_hat` entropy cleanup.
   Do not use the discarded old DIV2K fixed-anchor comparison.
5. Implement zero-distortion entropy improvements that are currently exposed by
   the codec, especially replacing fixed-width `z_hat` index coding with a
   decoder-consistent entropy-coded representation when the side overhead is
   favorable.
6. Reject coarse stage-3 send-control as currently implemented.  Counted
   send-mask experiments with latent-MSE and image-MSE-gradient teachers saved
   bpp but worsened DISTS/FID.  They should be treated as negative evidence,
   not as the lead.
7. Evaluate the active match-send omitted-recovery branch.  It should be
   promoted only if it closes the DISTS/FID gap of plain stage-3 send-control
   under real serialized bpp.
8. Implement stage-aware residual-variable coding inside the GLC four-part prior.
9. Implement or integrate a serious learned residual entropy model for that
   residual stream.
10. Build safe-coarsening / residual-RDO teacher signals to supervise allocation
   and prevent the method from becoming blind quantization-width tuning.
11. If zero-side allocation saturates, reintroduce a tiny counted control stream
   with learned entropy coding.
12. Add semantic/DINO/edge guards only after the core residual mechanism shows
   real-codec rate gain.
13. Run ablations after a branch has real-codec promise.

## One-Sentence Research North Star

```text
GP-ResLC should spend real counted bits only on the residual/control information
that pretrained GLC cannot reconstruct from z_hat, q, context, and its
generative decoder, while preserving GLC's codec order and exact decode
consistency.
```
