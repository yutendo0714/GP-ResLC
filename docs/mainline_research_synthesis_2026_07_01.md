# GP-ResLC Mainline Research Synthesis - 2026-07-01

This memo consolidates the latest three pieces of advice into the active
research direction.  It supersedes rho-only tuning as the main research path,
but keeps the current Safe-RDO rho branch as the strongest safety anchor.

## 1. Facts From The Current Real-Codec Evidence

### y is the right target

In real GLC bitstreams, `y` accounts for most of the total bpp.  On CLIC2020
baseline GLC, `y_bpp` is roughly 82-89% of total bpp across q0-q3, while `z_bpp`
is almost constant at about 0.0035 bpp.  Therefore, large gains must primarily
come from the transmitted `y` residual information.

### z fixed-width coding is a small zero-distortion booster

The current real codec follows the published GLC assumption and sends `z`
indices with fixed width:

```text
z_bits = ceil(log2(codebook_size))
```

This is correct for protocol matching, but it leaves a small zero-distortion
opportunity: entropy-code the `z` index stream or evaluate whether the official
GLC protocol can be augmented with a counted `z` entropy model.  This cannot be
the main source of large gains, but it is a clean auxiliary improvement.

### Mean prediction has repeatedly failed to move rate

Past `mean`, `scale_mean`, `latent_residual`, and stage mean-continuation runs
showed near-zero or negative rate gains.  The most plausible explanation is
that GLC's hyperprior plus four-part context prior already extracts most
decoder-predictable mean information from `z_hat` and decoded context.

This matters: a new `gp_mu_t` branch that only predicts another conditional
mean is unlikely to become the main performance breakthrough.

## 2. Mechanism Decomposition

The current evidence suggests three distinct mechanisms.

### M1: Better entropy / mean prediction

Goal:

```text
predict y better under the entropy model while preserving fidelity
```

Status:

- Weak so far.
- Average residual mean prediction appears mostly exhausted by GLC's own prior.
- A stronger learned residual entropy model may still help, but simple
  `gp_mu_t` mean addition is not the lead.

### M2: Perceptual residual omission or coarsening

Goal:

```text
spend fewer bits where the generative decoder can absorb missing detail
```

Status:

- This is what the current Safe-RDO rho branch is doing in a soft way.
- It gives strong DISTS/FID gains but LPIPS is near-neutral.
- Binary Residual Suppression (BRS) is the hard version: send residual symbols
  only where `transmit_mask=1`.
- BRS is a cleaner story than adaptive quantization, but its ceiling may be near
  the current rho lead unless it also improves LPIPS/alignment.

### M3: Generative recovery of omitted residuals

Goal:

```text
when residuals are not transmitted, do not reconstruct them as a flat prior
mean; synthesize plausible residual detail from decoder-available information
```

Status:

- This is the highest-upside hypothesis.
- It shifts the story from "predict the residual mean" to "recover the omitted
  residual distribution".
- This is closer to the perception-distortion argument: the conditional mean
  can be distortion-optimal but perceptually dull, while deterministic or
  seeded generative residual synthesis may improve DISTS/FID/LPIPS at the same
  bpp.

Important constraint:

- Any sampling must be deterministic or fully specified by counted bits.
- The first diagnostic should use fixed deterministic pseudo-noise or
  decoder-computable noise from `z_hat/q/context`; no uncounted image-specific
  seed is allowed.

## 3. Updated Role Of BRS

BRS is still valuable, but it should enter as a diagnostic and bridge, not as an
assumed final answer.

Use BRS for:

- explicit suppress/transmit interpretation;
- measuring how much `y` residual is actually safe to omit;
- testing whether hard omission improves LPIPS relative to rho;
- defining the positions where generative residual synthesis should operate.

Do not assume BRS alone will produce large gains over Safe-RDO.  It may be a
cleaner mechanism with a similar ceiling.

Implementation convention:

```text
transmit_mask = 1  -> residual symbol is transmitted
transmit_mask = 0  -> residual is omitted and reconstructed by prior/synthesis
```

Avoid ambiguous `m=0/1` semantics in code.

## 4. Immediate Diagnostic Ladder

The next work should be diagnostic-first, but still real-codec and mechanism
relevant.  Do not start with another long rho or loss-weight sweep.

### Step 1: Stage residual drop diagnostic

Implement a no-learning real-codec mode that sets selected stage residual
symbols to zero:

```text
stage residual suppression = y_q_stage := 0
not y_stage := 0
```

This preserves the GLC prior mean path and tests true residual omission.

Test cases:

- baseline GLC / current Safe-RDO;
- suppress stage 3 residual symbols;
- suppress stages 2+3;
- suppress stages 1+2+3;
- suppress all stages as a destructive bound.

Report:

- total bpp;
- `bpp_y` per stage;
- DISTS, LPIPS, FID/KID, PSNR, MS-SSIM;
- visual artifacts;
- decode consistency.

Decision value:

- If stage 3 omission barely hurts quality and saves meaningful bpp, BRS is
  viable.
- If it collapses, BRS must be block-level or paired with synthesis.

### Step 2: Same-bpp generative residual synthesis diagnostic

For the residual positions omitted by Step 1 or by the current Safe-RDO gate,
replace the flat omitted residual with a deterministic generative residual:

```text
omitted_residual_hat = scale * epsilon
```

where `epsilon` is deterministic and decoder-computable.  Try clipped and
low-amplitude variants first.  Transmitted positions must remain unchanged.

This keeps bpp identical to the corresponding suppression run.

Decision value:

- If DISTS/FID/LPIPS improve at identical bpp, generative residual synthesis
  becomes the main high-upside branch.
- If it worsens or only improves FID while destroying LPIPS/DISTS, keep BRS/rho
  as the safer branch.

### Step 3: Minimal BRS-ZS

Only after Step 1 shows suppressible residual mass, implement a minimal
zero-side BRS:

- no `gp_mu_t` initially;
- block or stage-level `transmit_mask`;
- decoder-computable mask from `z_hat/q/context`;
- all-transmit mode must match the corresponding baseline exactly.

### Step 4: Learned synthesis or learned residual/control entropy

Choose based on diagnostics:

- if synthesis helps, build a small decoder-side residual synthesizer for
  omitted regions;
- if synthesis does not help but BRS saves bpp safely, improve BRS masks and
  entropy coding;
- if neither helps, return to structural residual entropy modeling and `z`
  entropy coding as auxiliary gains.

## 5. Current Working Hypothesis

The strongest path is likely:

```text
Safe-RDO rho lead as anchor
-> stage residual omission diagnostics
-> same-bpp generative residual synthesis diagnostics
-> BRS or synthesis branch depending on measured LPIPS/DISTS/FID behavior
```

This is more promising than another `gp_mu_t` mean-prediction continuation,
because previous evidence suggests conditional means are mostly exhausted by
GLC's existing prior.

## 6. Promotion Criteria

A new branch should replace the current Safe-RDO lead only if it:

- improves real-codec BD-rate against local GLC;
- improves or matches the current Safe-RDO lead on at least one major curve;
- does not hide side information;
- reports actual serialized bpp;
- preserves exact decode consistency;
- evaluates beyond Kodak8 before being promoted.

For synthesis-specific branches, also require:

- deterministic decode or counted randomness;
- no uncounted image-dependent seed;
- evidence that gains are not only FID artifacts;
- LPIPS/DISTS checked carefully because residual sampling can harm alignment.

## 7. One-Sentence Direction

```text
Do not merely predict another residual mean; first measure which GLC residuals
can be omitted, then test whether omitted residuals should be reconstructed by
hard prior mean, hard suppression, or deterministic generative synthesis.
```

## 8. 2026-07-01 Real-Codec Diagnostic Update

Implemented a real-codec diagnostic path for omitted quantized `y` residual
symbols:

- `--suppress_yq_stages` selects GLC four-part stages;
- `--suppress_rho_threshold` omits only coefficients whose decoder-computable
  `rho(z_hat, q, context)` exceeds the threshold;
- omitted symbols are removed from the arithmetic-coded `y` stream;
- decoder reconstructs omitted symbols either as zero or as deterministic
  no-side-information pseudo-synthesis;
- all reported bpp is serialized payload bpp.

Important correction: an early diagnostic decode path accidentally restored an
entire selected stage as omitted whenever `--suppress_yq_stages` was set.  That
made the first `rhohard_stage3_t120/t125` metrics invalid.  The corrected path
uses the same threshold mask at encode and decode.  Valid fixed runs are:

- `experiments/real_codec/kodak8_safe_rdo_rhohard_stage3_t120_zero_fixed`
- `experiments/real_codec/kodak8_safe_rdo_rhohard_stage3_t125_zero_fixed`
- `experiments/real_codec/kodak8_safe_rdo_rhohard_stage3_t125_synth_gclip_s05_fixed`

Kodak8 / real codec / patch64 split1 BD-rate versus local GLC:

| run | DISTS | LPIPS | FID | interpretation |
|---|---:|---:|---:|---|
| SafeRDO anchor | -6.00% | -2.97% | -7.38% | current local anchor |
| stage3 hard, rho>=1.20, zero | +2.49% | +0.89% | -2.01% | too aggressive; rejects broad stage3 omission |
| stage3 hard, rho>=1.25, zero | -5.38% | -3.13% | -7.33% | close to SafeRDO, slightly better LPIPS, weaker DISTS/FID |
| stage3 hard, rho>=1.25, hash synthesis | -4.93% | -3.05% | -6.65% | simple deterministic noise does not recover quality |

Kodak8 BD-rate versus SafeRDO:

| run | DISTS | LPIPS | FID | decision |
|---|---:|---:|---:|---|
| stage3 hard, rho>=1.20, zero | +9.87% | +3.19% | +14.92% | reject |
| stage3 hard, rho>=1.25, zero | +0.45% | -0.17% | +0.89% | useful diagnostic, not a lead |
| stage3 hard, rho>=1.25, hash synthesis | +0.88% | -0.05% | +1.42% | reject as synthesis mechanism |

Interpretation:

- The current learned `rho` identifies a very small set of stage-3 residual
  coefficients that can be hard-omitted with minimal damage, but this is not a
  large-gain branch by itself.
- More aggressive omission saves real bits but moves the DISTS/LPIPS/FID curve
  in the wrong direction.
- Deterministic hash/noise synthesis is not a substitute for a learned
  generator-side residual synthesizer.
- The next mainline should not become a threshold sweep.  The useful conclusion
  is that hard omission needs a better safe-to-drop teacher or a learned
  decoder-side residual synthesis module, not another scalar threshold search.

Next action:

```text
Keep stage3 hard omission as a diagnostic/ablation path.
Move mainline effort to learned safe-to-drop targets and learned omitted-region
residual synthesis, with exact real-codec counting retained.
```

## 9. Learned Synthesis Branch Update

The first learned omitted-residual attempt used a categorical classifier:

```text
decoder context -> omitted quantized residual symbol
```

Run:

- `stage3_symbol_synth_t120_weighted_1k`
- W&B `2npfah7e`

Result:

- Real-codec Kodak8 quality was much worse than GLC/SafeRDO at the same
  hard-omission bpp.
- Versus SafeRDO, BD-rate was approximately `+45.98%` DISTS, `+28.41%`
  LPIPS, and `+50.11%` FID.
- This branch is rejected.

Important caveat:

- The original symbol-synth training script used `y_hat_so_far` after stage 3
  during training, while the decoder only has `y_hat_so_far` before stage 3.
  This train/decode mismatch is now fixed, but the stronger conclusion remains:
  symbol CE is a weak objective for the actual GP-ResLC goal.

New branch:

```text
decoder context -> continuous omitted residual value
image-space perceptual loss -> no extra bits
```

Implemented:

- `StageResidualValueSynthesizer`
- real-codec `omitted_residual_mode=learned_value`
- `scripts/train_stage3_value_synth.py`

Current active run:

- `stage3_value_synth_t120_perc_1500`
- W&B `6x3oe21z`
- suppress stage 3 only where `rho >= 1.20`
- train only the value synthesizer; pretrained GLC and SafeRDO stay frozen

Why this is more on-axis:

- The coded bitstream is unchanged from hard omission.
- The module only tries to improve the reconstruction of information already
  not sent.
- Any gain is therefore evidence for the core thesis: omitted residual detail
  can be decoder-recovered rather than transmitted.

Initial smoke signal:

- At q0 and identical bpp to hard stage-3 omission, a 50-iteration value
  synthesizer improved `DISTS 0.1281 -> 0.1275`, `LPIPS 0.2261 -> 0.2237`,
  and `FID 82.66 -> 82.26`.
- This is not a lead, but it justifies the longer image-loss run.

Updated result after full real-codec additive fix:

- The additive value-synthesis branch keeps the SafeRDO bitstream unchanged and
  adds a decoder-computable stage-3 residual value only at positions selected by
  decoder-available `rho`.
- A real-codec bug was fixed: the normal `stage_residual_entropy_quant_gate`
  branch was not passing `synth_yq_stages` into the actual encode/decode
  helpers, so the first additive evaluations were no-ops.  The fixed path now
  changes decoded pixels and preserves encode/decode consistency.
- Kodak8 / patch64 split1 versus same-code SafeRDO:

| variant | DISTS BD | LPIPS BD | PSNR BD | FID BD | interpretation |
|---|---:|---:|---:|---:|---|
| additive `s1` | +1.44% | -2.26% | -7.62% | -0.71% | improves alignment/fidelity, over-corrects DISTS |
| additive `s0.5` | +0.24% | -1.50% | -3.93% | -1.00% | near-DISTS-neutral no-side quality gain |

Research implication:

- This is the first clean sign that decoder-side generative residual synthesis
  can improve quality without adding bits.
- The branch is not yet a lead because DISTS is the main perceptual criterion
  and remains slightly worse than SafeRDO on the curve.
- The next mainline step should train a DISTS-aware additive synthesizer with a
  safety hinge against the frozen SafeRDO reconstruction, then combine that
  quality recovery with actual stage-aware residual omission or a tiny counted
  control stream.

Follow-up after DISTS-aware training:

- DISTS-aware safe-hinge training (`stage3_value_add_safehinge_dists_t120_2500`,
  W&B `0shvwfi4`) improved PSNR/LPIPS but still worsened the DISTS curve:
  `+2.89%` DISTS BD at scale 1.0 and `+0.71%` at scale 0.5 versus same-code
  SafeRDO.
- The simpler plain additive synthesizer at conservative scale 0.5 remains the
  best balance: `+0.24%` DISTS BD, `-1.50%` LPIPS BD, `-3.93%` PSNR BD, and
  `-1.00%` FID BD at identical serialized bpp.
- A stronger `value_bound=0.5`, `lambda_safe_dists=80` run was stopped early
  because it still produced frequent DISTS safety violations.

Updated direction:

- Do not continue amplitude/loss sweeps as the mainline.
- The next meaningful synthesis step is selective synthesis:
  learn where decoder-side residual generation is safe, then synthesize only
  there.
- If zero-side selection from `z_hat/q/context/rho` is insufficient, the next
  GP-ResLC-consistent mechanism is a tiny counted control stream: transmit only
  the unpredictable "synthesis allowed / residual needed" control information,
  and count those bits in the real codec.

## 10. Selective No-Side Synthesis Result

Implemented:

```text
decoder context -> residual value
decoder context -> synthesis gate
decoded y_hat += scale * gate * value
```

The gate and value are both computed from decoder-available quantities.  No
side map is transmitted and the SafeRDO bitstream is unchanged.

Run:

- `stage3_selective_value_add_t120_distsguard_1200`
- W&B `sdlsddfn`
- trained from the current SafeRDO anchor

Real-codec Kodak8 / patch64 split1 result versus same-code SafeRDO:

| scale | DISTS BD | LPIPS BD | PSNR BD | MS-SSIM BD | FID BD | KID BD |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | -0.19% | -0.92% | -2.52% | -1.26% | -0.42% | -0.55% |
| 0.5 | +0.04% | -1.74% | -4.93% | -2.48% | -0.21% | -0.74% |
| 1.0 | +1.14% | -2.49% | -9.09% | -4.61% | -0.22% | -0.78% |

Decision:

- Keep `scale=0.25` as the safest no-side synthesis booster.
- Treat `scale=0.5` as a useful LPIPS/fidelity-oriented variant.
- Reject `scale=1.0` for the DISTS-led mainline.

Research implication:

- Decoder-side residual synthesis is viable, but its current ceiling is small
  when constrained to be DISTS-safe.
- The mainline should now attack true rate mechanisms rather than keep
  optimizing synthesis amplitude: fixed-width `z_hat` coding, residual/control
  entropy modeling, and stage-aware residual-variable coding.

## 11. Stage-3 Send-Control And Synthesis Update

New real-codec experiments tested a harder version of the GP-ResLC idea:
transmit only a subset of stage-3 residual cells under a counted binary send
mask, and let the decoder/generator handle the rest.

Implemented score modes:

```text
latent_mse       -> select cells by local latent reconstruction benefit
image_mse_grad   -> select cells by first-order image-MSE benefit
image_l1_grad    -> available for source-side image-loss selection
```

All send masks are entropy-coded and counted.  The decoder reads the mask from
the bitstream; no uncounted side information is used.

Kodak8 / patch64 split1 results versus the current SafeRDO anchor:

| branch | DISTS BD | LPIPS BD | PSNR BD | MS-SSIM BD | FID BD | KID BD | read |
|---|---:|---:|---:|---:|---:|---:|---|
| stage3 send, latent-MSE, send 75% | +1.09% | +0.95% | -1.31% | +0.08% | +2.75% | +4.77% | saves bpp but damages perceptual quality |
| stage3 send, image-MSE-grad, send 75% | +4.58% | +2.42% | -0.41% | +2.59% | +5.52% | +12.79% | image-MSE teacher is the wrong allocation signal |
| stage3 send 75% + selective synthesis s0.25 | +0.95% | -0.01% | -3.75% | -0.93% | +3.42% | +11.80% | synthesis recovers LPIPS/fidelity but not DISTS/FID |
| stage3 send 90% + selective synthesis s0.25 | +0.56% | -0.59% | -2.49% | -0.62% | +2.09% | -2.24% | closer, but still not a DISTS/FID lead |
| stage3 send 95% + selective synthesis s0.25 | +0.37% | -0.73% | -2.19% | -0.58% | +1.84% | +17.34% | tiny rate cut still hurts DISTS/FID |
| selective synthesis s0.25 only | -0.19% | -0.92% | -2.52% | -1.26% | -0.42% | -6.88% | current safest synthesis branch |

Interpretation:

- Stage-level or coarse stage-3 residual omission is not safe enough.
- Local latent MSE is a weak teacher for perceptual/generative residual
  allocation.
- Image MSE gradients are even worse for perceptual allocation: they preserve
  distortion but select residuals that harm DISTS/FID.
- Selective synthesis is the only branch in this group that improves all
  measured metrics, although its gains are still small.

Updated mainline decision:

```text
Do not promote counted stage-3 send-control yet.
Promote selective synthesis as the safe generator-recovery branch.
Move synthesis from "additive quality booster" toward "omitted-residual recovery":
  send less only after the synthesis gate can safely identify recoverable cells.
```

The next implementation should therefore train and evaluate an omitted-aware
selective synthesizer: the mask should represent residual cells that are not
transmitted, and synthesis should be applied primarily to those omitted cells,
not as a generic additive correction everywhere.

## 12. Omitted-Recovery Alignment Fix

The first omitted-aware training run was useful but not sufficient:

| branch | DISTS BD | LPIPS BD | PSNR BD | FID BD | read |
|---|---:|---:|---:|---:|---|
| omitted selective 1000, send 75% | +1.32% | -0.76% | -8.88% | +3.72% | restores some alignment/fidelity, hurts perceptual naturalness |

The main lesson is not that synthesis is hopeless.  The training target was
misaligned with the codec decision:

```text
training selected omitted cells by rho >= 1.20
evaluation omitted cells by counted latent-MSE stage-3 send mask
```

That makes the synthesizer learn one residual distribution and deploy on
another.  The training script now has a `--stage3_send_frac` path that builds
the same latent-MSE top-k send mask used by the real codec and trains only on
the cells not transmitted by that mask.

Active follow-up:

```text
stage3_omitted_selective_matchsend_f075_distsguard_3000
```

This keeps the core GP-ResLC question sharp:

```text
Transmit the residual cells with high local latent benefit.
Do not transmit the remaining cells.
Let a decoder-computable generator-side module synthesize only the omitted cells.
Count the binary send mask and all residual streams in actual serialized bpp.
```

Promotion requires this branch to close the DISTS/FID gap of plain stage-3
send-control, not merely recover PSNR/LPIPS.

Result:

| branch | DISTS BD | LPIPS BD | PSNR BD | FID BD | read |
|---|---:|---:|---:|---:|---|
| match-send omitted selective 1000, send 75% | +3.48% | -0.03% | -11.88% | +0.58% | FID is less bad than plain send-control, but DISTS fails |

Decision:

- Do not promote omitted stage-3 synthesis as the current lead.
- The experiment is still informative: decoder-side synthesis can improve
  distributional realism after residual omission, but the current module
  disrupts the structure/texture consistency captured by DISTS.
- For the next mainline push, prioritize mechanisms that lower bpp without
  changing reconstruction first (`z_hat` entropy coding), and then define a
  cleaner stage-aware residual variable with a learned entropy model.
