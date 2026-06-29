# Scratch Progressive Residual Notes

Last updated: 2026-06-21 JST

## Objective

Test a more method-faithful Stage-B design for scratch GP-ResLC: after the cheap Stage-A semantic/generative code, send a coarse residual first and optionally send a finer residual stage only when the decoder-side context predicts that the first residual is insufficient.

This directly targets the GP-ResLC axis: do not send what the generator/predictor can recover; send only the unpredictable residual.

## Implemented variants

- `ScratchProgressiveResidualBottleneck`: multi-stage latent residual quantization. Stage 0 is module-name compatible with the previous single-stage `ScratchResidualBottleneck`, so it can be initialized from the current scratch lead.
- Non-gated progressive: stage 1 is always coded. This tests whether a finer residual stage improves quality.
- Decoder-side hard-gated progressive: stage 1 symbols and entropy cost are multiplied by a gate computed from `z_s` and stage-0 `y_hat`; no side map is transmitted in the proxy design.
- Soft-train/hard-eval gate: train with continuous gate probabilities, evaluate with deterministic hard threshold.

## W&B runs

| run | id | result |
|---|---|---|
| non-gated progressive, `lambda_R=0.6` | `337eca40` | Quality-side improvement, but high residual bpp. |
| hard-gated progressive, zero/constant gate init | `ig60pxg2` | Gate stayed closed; stopped after checkpoint save. |
| hard-gated progressive, random gate init | `faev11ea` | Gate collapsed closed early; stopped. |
| soft-train/hard-eval gated progressive | `4ht20cqw` | Useful lower-rate point, but hard stage 1 mostly closes. |

## Fixed Kodak center-crop results

All points use Stage-A `experiments/scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt`.

| variant | checkpoint | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | decision |
|---|---|---:|---:|---:|---:|---:|---|
| current scratch lead | `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` | 0.01321 | 0.43869 | 0.42313 | single | - | Keep as scratch DISTS lead. |
| progressive non-gated 1000 | `scratch_stage_b_progressive2_from_attnA_r8_q1q05_lR0p6_3k/stage_b_0001000.pt` | 0.01736 | 0.43295 | 0.42527 | 0.00276 | 0.00484 | Worse than lead. |
| progressive non-gated 2000 | `scratch_stage_b_progressive2_from_attnA_r8_q1q05_lR0p6_3k/stage_b_0002000.pt` | 0.01954 | 0.43337 | 0.41948 | 0.00283 | 0.00694 | Quality-side point only. |
| progressive non-gated final | `scratch_stage_b_progressive2_from_attnA_r8_q1q05_lR0p6_3k/stage_b_final.pt` | 0.02000 | 0.42979 | 0.42299 | 0.00287 | 0.00736 | Bit-inefficient. |
| gated soft-train 1000 | `scratch_stage_b_progressive2_gated_softtrain_from_attnA_r8_q1q05_lR0p5_t02_bm15_3k/stage_b_0001000.pt` | 0.01299 | 0.43538 | 0.42373 | 0.00322 | ~0.00000 | Useful low-rate point, no stage-1 use. |
| gated soft-train final | `scratch_stage_b_progressive2_gated_softtrain_from_attnA_r8_q1q05_lR0p5_t02_bm15_3k/stage_b_final.pt` | 0.01312 | 0.43677 | 0.42500 | 0.00335 | 0.00000 | Lower rate, lower quality. |

## Interpretation

The non-gated progressive stage proves that extra residual precision can reduce DISTS, but it spends too many bits. The gated variants prove that a decoder-side skip mechanism can keep bpp in the target band, but the current hard gate learns to close stage 1 rather than assign it to genuinely unpredictable regions.

The current scratch lead remains the single-stage residual point at 0.01321 bpp / DISTS 0.42313. The gated soft-train 1000 checkpoint is a useful lower-rate curve point at 0.01299 bpp / DISTS 0.42373.

## Next design correction

The next progressive design should not let stage 1 compete with the same decoder path as stage 0. Better options:

1. Add a stage-specific correction decoder so stage 1 has a direct perceptual gradient and cannot be ignored by the shared decoder.
2. Use a residual-improvement hinge: reward stage 1 only when it improves DISTS/LPIPS over a detached stage-0 reconstruction by a margin.
3. Train with stochastic threshold annealing or top-k gate budget so stage 1 is forced to specialize, then let rate loss prune it.
4. Once proxy trends improve, replace Gaussian proxy with an actual entropy-coded residual payload.


## Threshold Sweep and Fine-Decoder Follow-up

A gate-threshold sweep was added through `--gate_threshold_override` in `scripts/evaluate_scratch_stage_b.py`.

For `scratch_stage_b_progressive2_gated_softtrain_from_attnA_r8_q1q05_lR0p5_t02_bm15_3k/stage_b_0001000.pt` on Kodak center crops:

| threshold | bpp | LPIPS | DISTS | stage1 bpp | stage1 gate mean | interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 0.20 | 0.01299 | 0.43538 | 0.42373 | ~0.00000 | 0.00008 | Default hard gate mostly closed. |
| 0.15 | 0.01326 | 0.43525 | 0.42371 | 0.00028 | 0.03003 | Tiny stage1 use, negligible gain. |
| 0.10 | 0.01478 | 0.43448 | 0.42360 | 0.00179 | 0.23747 | LPIPS improves slightly; DISTS barely moves. |
| 0.05 | 0.01658 | 0.43284 | 0.42352 | 0.00359 | 0.67635 | More stage1 bits, still weak DISTS gain. |

A fine-correction decoder variant was also implemented with `--stage_correction_decoder`. It gives stage 1 a direct correction path, initialized at zero output, but the 1000-step pilot still closes hard stage 1 at threshold 0.20:

| variant | checkpoint | bpp | LPIPS | DISTS | stage1 bpp | decision |
|---|---|---:|---:|---:|---:|---|
| fine decoder soft-gate 1000 | `scratch_stage_b_progressive2_finedec_softgate_from_attnA_r8_q1q05_lR0p5_t02_bm15_3k/stage_b_0001000.pt` | 0.01323 | 0.43615 | 0.42441 | 0.00000 | No improvement. |

Conclusion: the current learned stage-1 residual has some LPIPS utility when forced open, but it is not targeting DISTS well. The next experiment should add a stage-improvement objective, e.g. decode stage 0 and full stage 0+1 during training and penalize `DISTS(full) >= DISTS(stage0) - margin`, while also charging stage-1 bpp. This makes the fine stage specialize in perceptually useful residuals before rate pruning.


## Stage-Improvement Hinge Pilot

Added `stage0_x_hat` output and `--lambda_stage_improve` to `scripts/train_scratch_stage_b.py`. The loss compares detached stage-0 DISTS against full reconstruction DISTS with a margin, so stage 1/fine correction has an explicit reason to improve the reconstruction.

Pilot run:

- W&B: `9g72335u`
- Output: `experiments/scratch_stage_b_progressive2_finedec_softgate_stageimpr_from_attnA_r8_q1q05_lR0p5_si5_2k/`
- Config: fine correction decoder, soft train/hard eval gate, `lambda_stage_improve=5.0`, margin `0.001`; stopped after the 1000 checkpoint because hard stage 1 remained closed.

Fixed center-crop results:

| dataset | checkpoint | bpp | LPIPS | DISTS | stage1 bpp | note |
|---|---|---:|---:|---:|---:|---|
| Kodak | `stage_b_0001000.pt` | 0.01422 | 0.43778 | 0.42232 | 0.00000 | New scratch DISTS quality-side point; not low-rate lead. |
| Kodak, threshold 0.10 | same | 0.01611 | 0.43780 | 0.42234 | 0.00189 | Opening stage 1 adds bits without useful gain. |
| DIV2K center | same | 0.01553 | 0.42011 | 0.41333 | 0.00000 | Improves DISTS over previous scratch DIV2K center point, with higher bpp. |

Interpretation: the hinge improved the main stage-0 residual/decoder and produced a better quality-side checkpoint, but it still did not make hard stage 1 useful. The next fix should make stage 1 non-optional during a warmup phase or compute the hinge against an actual hard-gated full reconstruction at the target threshold.


## Stage-1 Warmup and Gate Fine-Tune

To test whether stage 1 can learn a useful residual before rate pruning, `--train_only_extra_stages` was added to `scripts/train_scratch_stage_b.py`. This freezes the base residual path and trains only `extra_*` modules.

Runs:

| run | W&B | purpose |
|---|---|---|
| `scratch_stage_b_progressive2_finedec_stage1warm_from_attnA_r8_q1q05_lR0p3_si5_1k` | `rubquyfn` | Force stage 1 open and train extra decoder only. |
| `scratch_stage_b_progressive2_finedec_stage1warm_gateft_from_attnA_r8_q1q05_lR0p6_si5_1k` | `v0jqpxyq` | Gate/rate fine-tune from the warmup checkpoint. |

Fixed center-crop results:

| dataset | checkpoint | bpp | LPIPS | DISTS | stage1 bpp | note |
|---|---|---:|---:|---:|---:|---|
| Kodak | stage1 warmup final | 0.02003 | 0.43720 | 0.42222 | 0.00681 | Stage 1 can carry useful residual, but too expensive. |
| Kodak | warmup -> gate fine-tune final | 0.01349 | 0.43748 | 0.42283 | ~0.00001 | New near-lead quality point; stage1 is mostly pruned. |
| Kodak, threshold 0.10 | warmup -> gate fine-tune final | 0.01534 | 0.43748 | 0.42283 | 0.00186 | Opening stage1 adds bits without extra DISTS gain. |
| DIV2K center | warmup -> gate fine-tune final | 0.01428 | 0.41999 | 0.41425 | ~0.00000 | Lower-rate than stage-improvement 1000, slightly worse DISTS. |

Interpretation: warmup proves stage 1 can improve quality when forced open, but current gate/rate fine-tuning prunes it and retains most gains through stage 0. The progressive branch is getting closer to the intended decomposition, but still needs a mechanism that preserves a sparse, high-value stage-1 subset instead of collapsing it.


### Additional Gate Fine-Tune Sweep

A lower-rate-penalty gate fine-tune from the same stage-1 warmup checkpoint was run with `lambda_R=0.3`.

- W&B: `12dmxux7`
- Output: `experiments/scratch_stage_b_progressive2_finedec_stage1warm_gateft_from_attnA_r8_q1q05_lR0p3_si5_1k/`
- Kodak center final: bpp `0.01503`, LPIPS `0.43645`, DISTS `0.42452`, stage1 bpp `0.00000`.

This is worse than the `lambda_R=0.6` fine-tune and confirms that simply relaxing rate does not preserve useful hard-gated stage-1 residuals. The gate still collapses unless explicitly constrained.

## Top-k Gate Budget Pilot

Added `--gate_topk_frac` to enforce a fixed decoder-side sparse fine-residual budget without transmitting a side map. This is the most literal scratch test so far of the idea that only a small unpredictable residual subset should be sent after the generator-conditioned stream.

- W&B: `oa3hchyt`
- Output: `experiments/scratch_stage_b_progressive2_finedec_stage1warm_topk005_from_attnA_r8_q1q05_lR0p5_si5_1k/`
- Init: forced-open stage-1 warmup checkpoint
- Config: fine correction decoder, `gate_topk_frac=0.05`, `lambda_R=0.5`, `lambda_stage_improve=5.0`

Fixed Kodak center-crop final result:

| bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 gate mean |
|---:|---:|---:|---:|---:|---:|
| 0.01391 | 0.43889 | 0.42378 | 0.00380 | 0.00034 | 0.04883 |

Interpretation: the top-k constraint works mechanically: about 5% of fine-stage positions are selected and only `0.00034` bpp is spent on stage 1. However, quality is worse than the current single-stage scratch lead (`0.01321` bpp / DISTS `0.42313`) and the warmup-gateft quality point (`0.01349` bpp / DISTS `0.42283`). This branch is mechanism evidence, not a replacement lead. The next version needs a hard-gate-aware objective that makes the selected 5% positions carry DISTS-useful corrections.

## Top-k Gate Budget Sweep

Evaluated the same top-k checkpoint with different deterministic gate budgets on Kodak center crops:

| top-k frac | bpp | LPIPS | DISTS | stage1 bpp | stage1 gate mean |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.01371 | 0.43889 | 0.42378 | 0.00014 | 0.01953 |
| 0.05 | 0.01391 | 0.43889 | 0.42378 | 0.00034 | 0.04883 |
| 0.10 | 0.01433 | 0.43888 | 0.42378 | 0.00077 | 0.09961 |
| 0.20 | 0.01519 | 0.43888 | 0.42380 | 0.00163 | 0.19922 |

DIV2K center at the trained 5% budget gives bpp `0.01515`, LPIPS `0.42036`, DISTS `0.41508`, stage1 bpp `0.00036`, gate mean `0.04883`.

Interpretation: increasing the fine-stage budget from 2% to 20% mostly increases bpp while DISTS stays around `0.42378-0.42380`. LPIPS improves only in the fourth decimal place. This rules out a simple budget issue: the current fine stage needs a stronger hard-gated correction objective, not just a wider gate.

## Top-k 10% Strong Stage-Improvement Pilot

A final short pilot tested whether stronger hard-gated stage-improvement can make the selected fine residual positions useful.

- Failed start: W&B `s1tbidjt` used `base_ch=128` by mistake, loaded only 73 tensors, produced invalid high-bpp validation, and was interrupted.
- Correct run: W&B `7dyy6dpq`, output `experiments/scratch_stage_b_progressive2_finedec_stage1warm_topk010_si20_b64_from_attnA_r8_q1q05_lR0p5_1k/`.
- Init: forced-open stage-1 warmup, fully compatible (`230 tensors`, `missing=0`, `skipped=0`).
- Config: `gate_topk_frac=0.10`, `lambda_stage_improve=20.0`, `lambda_R=0.5`, fine correction decoder.

Fixed center-crop results:

| dataset | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | gate mean |
|---|---:|---:|---:|---:|---:|---:|
| Kodak | 0.01504 | 0.44118 | 0.42219 | 0.00449 | 0.00078 | 0.09961 |
| DIV2K | 0.01641 | 0.42225 | 0.41361 | 0.00587 | 0.00077 | 0.09961 |

Interpretation: this is the best scratch Kodak DISTS point so far, but it is not the low-rate lead and LPIPS worsens. DIV2K does not beat the earlier stage-improvement checkpoint. The useful conclusion is that hard-gated sparse residuals can improve DISTS if the improvement pressure is strong enough, but the objective needs better regularization so gains do not come mainly from higher stage0 bpp and worse LPIPS.

## Selected-Region Top-k Fine-Residual Update

Implemented selected-region improvement loss for progressive Stage-B. The loss upsamples the decoder-side `stage1_gate_map` to image space and penalizes locations where the final reconstruction does not improve local L1 error over detached stage-0 reconstruction. This targets the original GP-ResLC axis more directly: if a fine residual position is selected for transmission, it must carry useful unpredictable correction. Also added a stage-1 scale guard to prevent entropy-scale inflation.

Runs:

| run | W&B | setting | outcome |
|---|---|---|---|
| `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_from_warm_lR0p5_2k` | `r3i0z4f3` | extra-stage-only, top-k 10%, selected loss 20, no scale guard | Kodak DISTS improved, but stage1 scale inflated after 500-1000 steps. |
| `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500` | `e6a0sh06` | same, plus `lambda_stage1_scale_guard=0.2`, `stage1_scale_target=0.8` | best scratch Kodak DISTS so far with controlled stage1 scale. |

Fixed center-crop results:

| checkpoint | dataset | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| no-guard 500 | Kodak | 0.01385 | 0.43758 | 0.42292 | 0.00345 | 0.00064 | 0.775 | good early point |
| no-guard 1000 | Kodak | 0.01389 | 0.43964 | 0.42279 | 0.00345 | 0.00067 | 1.429 | DISTS improves, scale inflates |
| scale-guard 500 | Kodak | 0.01386 | 0.43846 | 0.42274 | 0.00345 | 0.00065 | 0.680 | balanced update |
| scale-guard 1000 | Kodak | 0.01377 | 0.44009 | 0.42195 | 0.00345 | 0.00056 | 0.687 | new scratch Kodak DISTS lead |
| scale-guard final | Kodak | 0.01371 | 0.43921 | 0.42253 | 0.00345 | 0.00050 | 0.662 | lower rate, slightly worse DISTS |
| scale-guard 1000 | DIV2K center | 0.01424 | 0.42176 | 0.41391 | 0.00388 | 0.00060 | 0.678 | lower-bpp DIV2K quality-side point |

Per-image Kodak comparison for scale-guard 1000 versus the previous single-stage scratch lead:

- DISTS: mean delta `-0.001177`, median `-0.001159`, wins `18/24`.
- LPIPS: mean delta `+0.001398`, median `+0.001165`, wins `7/24`.
- Best DISTS gains: `kodim22:-0.0071`, `kodim04:-0.0062`, `kodim17:-0.0051`, `kodim23:-0.0030`, `kodim02:-0.0030`.
- Worst DISTS losses: `kodim20:+0.0037`, `kodim13:+0.0036`, `kodim16:+0.0027`, `kodim18:+0.0015`, `kodim09:+0.0004`.

Decision: promote `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt` as the current scratch DISTS lead, but not as the LPIPS lead. The new objective confirms that sparse fine-stage residuals can help when the selected positions are explicitly trained and stage1 scale is guarded. Next work should add LPIPS/feature-region guidance or a DISTS-aligned local proxy so the gain is not purely DISTS-biased.


## 2026-06-21: LPIPS-balanced fine residual notes

Two follow-ups tested whether the DISTS-leading selected top-k fine-residual run can recover LPIPS without losing DISTS.

| Variant | Kodak bpp | Kodak LPIPS | Kodak DISTS | DIV2K bpp | DIV2K LPIPS | DIV2K DISTS | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| selected + scale guard, 1000 (`e6a0sh06`) | 0.013768 | 0.440089 | **0.421954** | 0.014242 | 0.421761 | **0.413914** | current DISTS lead |
| global LPIPS 1.2 final (`4sr7hpua`) | **0.013646** | 0.438245 | 0.423436 | **0.014156** | **0.420427** | 0.415380 | LPIPS recovers, DISTS regresses |
| stage LPIPS hinge final (`njfmi964`) | 0.013715 | **0.438228** | 0.423569 | n/a | n/a | n/a | hinge works, but not enough for DISTS |

Interpretation: the extra residual stage is not merely adding detail; it is changing perceptual statistics in a way DISTS likes but LPIPS sometimes dislikes. For a paper-facing scratch path, the next useful loss is a spatial feature no-regression on gated locations or a selector objective that chooses positions with high residual unpredictability and high perceptual payoff.

## GLC-Latent Progressive Stage-Specific Top-k Residual

The older scratch progressive notes above concern the independent Stage-B proxy codec. A separate full-design/GLC-latent branch now tests the same idea inside the frozen GLC/VQGAN latent space with an actual byte-backed semantic stream and torchac residual stream.

Current active run: W&B `1xwv0f6q`, `experiments/glc_latent_progressive_stagealloc_topk0008_from_final_512_14500to16500/`.

This variant fixes a limitation in the first progressive GLC-latent attempt: global hard top-k gave the two residual stages no explicit symbol budget. The new implementation selects top-k residual symbols independently in stage1/stage2 channel groups, then decodes them through separate zero-centered residual decoders. This is closer to the intended decomposition: stage1 sends a sparse coarse unpredictable correction, stage2 sends the remaining sparse fine correction.

Paper-facing decision rule: promote only if full-resolution real-codec evaluation shows a consistent DISTS/LPIPS or FID/KID gain at comparable payload bpp versus both the previous GLC-latent residual checkpoint and official GLC curve points. Crop validation alone is only a routing signal.

## 2026-06-21 23:13 JST - Progressive decoder init and top-k score audit

This block tested whether the GLC-latent residual stream can be made more faithful to the GP-ResLC principle by either splitting the residual into two progressive stages or changing which sparse residual symbols are transmitted.

Implementation updates:

- Added stage-specific top-k allocation for progressive residual mode.
- Added `--init_progressive_decoders_from_single {stage1,both}` so old single residual decoder weights can initialize the progressive decoders instead of discarding the useful learned residual synthesis path.
- Fixed train/validation memory use: stage1 reconstruction is now generated only when stage1 losses/metrics are requested. This avoids 512-crop OOM during runs that do not use stage1 supervision.
- Added `--topk_score_mode {abs,latent_error,latent_error_sq}`. `latent_error` scores source-side symbols by `|symbol| * mean_c|target_latent - mu|`; `latent_error_sq` uses the squared spatial error. This tests whether choosing spatially less predictable positions improves the transmitted residual at the same top-k budget.

W&B / runs:

- `1xwv0f6q`: zero-initialized stage1/stage2 progressive decoder. Rejected; DISTS worsened and stage2 carried almost no delta.
- `o5bno8es`: stage1 initialized from old single decoder, stage2 zero. Better crop behavior but full-res Kodak DISTS worsened.
- `huipcyt9`: both stage decoders initialized from old single decoder, balanced fine-tune. LPIPS improved, DISTS worsened.
- `frh8u1tx`: both-init DISTS-heavy fine-tune. DISTS returned close to no-train but did not beat it.

Kodak full-resolution real-codec results:

| variant | payload bpp | residual AC bpp | LPIPS | DISTS | read |
|---|---:|---:|---:|---:|---|
| Stage-A base only | 0.009766 | 0 | 0.576080 | 0.376250 | generator/predictor base |
| previous single abs top-k, gamma=1 reference | ~0.011617 | ~0.00075 | ~0.5777 | ~0.3629 | still DISTS lead among these local tests |
| progressive both-init no-train | 0.011617 | 0.000753 | 0.578412 | 0.363322 | stage split itself is not destructive |
| progressive stage1-init best | 0.011658 | 0.000793 | 0.568868 | 0.377770 | LPIPS improves, DISTS fails |
| progressive both-init fine-tune best | 0.011617 | 0.000753 | 0.574547 | 0.370081 | LPIPS improves, DISTS regresses |
| progressive both-init fine-tune final | 0.011617 | 0.000753 | 0.568128 | 0.377328 | strongest LPIPS, poor DISTS |
| progressive both-init DISTS-heavy best | 0.011617 | 0.000753 | 0.578450 | 0.363348 | basically no better than no-train |
| single top-k score = latent_error | 0.011617 | 0.000753 | 0.574098 | 0.375504 | simple unpredictability score hurts DISTS |
| single top-k score = latent_error_sq | 0.011617 | 0.000753 | 0.574603 | 0.375561 | same failure mode |

Interpretation:

1. Progressive splitting is mechanically valid only when both stage decoders inherit the old residual decoder. Zero-init stage2 does not learn useful corrections in short fine-tunes.
2. Fine-tuning the progressive decoders tends to trade DISTS for LPIPS. This may be useful for a separate LPIPS point, but it is not a paper-facing DISTS/FID/KID improvement.
3. Hand-designed latent-error top-k is not a good proxy for perceptual payoff. It selects positions where the generator latent prediction is wrong, but those are not necessarily the positions that improve DISTS.
4. The next serious version should use a learned encoder-side selector trained against measured perceptual payoff, e.g. compare candidate residual masks by DISTS/LPIPS improvement, distill the payoff into a selector, then entropy-code the selected sparse residual symbols. That is closer to “send only unpredictable residuals” than raw latent-error weighting.

Decision:

Do not promote the progressive/top-k-score variants as the current lead. Keep the previous single residual real-codec point as the practical lead, and treat these experiments as evidence that the next improvement must be a learned residual-value selector rather than a manual top-k heuristic.

## 2026-06-21 23:35 JST - Encoder-side latent-gradient residual selection

A more direct test of the GP-ResLC principle was implemented after manual top-k heuristics failed. Instead of selecting sparse residual symbols only by magnitude, the encoder estimates each candidate residual symbol's first-order latent reconstruction payoff:

`score_i = | grad_{q_i} SmoothL1(mu + D_r(q), y) * candidate_q_i |`

where `mu` is the generator-predictable latent from the semantic stream, `D_r` is the learned residual decoder, and `y` is the frozen GLC/VQGAN target latent. The decoder receives the same sparse residual grid as before; no side map is transmitted. The cost is encoder-side backprop during analysis/encoding.

Implementation:

- Added `topk_score_mode=latent_grad` in `gp_reslc/scratch/glc_latent_residual.py`.
- Added CLI/evaluator support through `--topk_score_mode latent_grad`.
- The current implementation is for the single residual decoder, not progressive residual mode.
- `latent_error` and `latent_error_sq` hand-designed scores were also tested and rejected.

Kodak full-resolution real-codec:

| variant | payload bpp | LPIPS | DISTS | DISTS wins | LPIPS wins | read |
|---|---:|---:|---:|---:|---:|---|
| base semantic predictor | 0.009766 | 0.576080 | 0.376250 | - | - | no residual |
| prior abs top-k reference | ~0.011617 | ~0.5777 | ~0.3629 | - | - | previous DISTS lead |
| latent_grad, delta=1.0 | 0.011617 | 0.595405 | 0.354949 | 17/24 | 8/24 | strong DISTS, LPIPS worse |
| latent_grad, delta=0.8 | 0.011617 | 0.579667 | 0.362212 | 18/24 | 14/24 | balanced, slight DISTS lead |
| latent_grad, delta=0.6 | 0.011617 | 0.574957 | 0.371653 | 18/24 | 17/24 | LPIPS/L1 safer, weaker DISTS |
| latent_error | 0.011617 | 0.574098 | 0.375504 | - | - | rejects simple latent-error score |
| latent_error_sq | 0.011617 | 0.574603 | 0.375561 | - | - | also rejected |

DIV2K full-resolution real-codec, `latent_grad`, `delta=0.6`:

| payload bpp | base LPIPS | LPIPS | base DISTS | DISTS | DISTS wins | LPIPS wins |
|---:|---:|---:|---:|---:|---:|---:|
| 0.011024 | 0.565526 | 0.603688 | 0.353956 | 0.334945 | 83/100 | 45/100 |

CLIC2020 test all 428 full-resolution real-codec, `latent_grad`, `delta=0.6`:

| payload bpp | base LPIPS | LPIPS | base DISTS | DISTS | DISTS wins | LPIPS wins |
|---:|---:|---:|---:|---:|---:|---:|
| 0.011000 | 0.540915 | 0.559899 | 0.341476 | 0.319396 | 373/428 | 234/428 |

Interpretation:

- This is the strongest mechanism result so far for the original thesis. The residual stream is selected by decoder payoff rather than raw magnitude, and it improves DISTS on Kodak, DIV2K, and CLIC at the same byte-backed payload bpp.
- It is not yet an official-curve result. Absolute DISTS/LPIPS values are from the scratch/Stage-A semantic branch and are far worse than official GLC q0-q3. The contribution is mechanism-level: at fixed ultra-low payload, better residual selection improves DISTS substantially.
- LPIPS and L1 often worsen, especially on DIV2K. The method is currently DISTS-oriented. The next version needs a multi-objective payoff score or adaptive residual strength to trade DISTS/LPIPS per image.
- Encoder analysis time increases because each image requires one latent-gradient backprop. Decoder time and bitstream format are unchanged.

Next steps:

1. Add `latent_grad_mix` score: combine latent payoff with LPIPS/L1 safety proxy or a learned per-image `delta_scale` to reduce LPIPS damage.
2. Train a selector/distillation network to approximate latent-gradient top-k without test-time backprop, making the method practical.
3. Evaluate FID/KID patches for CLIC/DIV2K using reconstructions from `latent_grad delta=0.6`; DISTS improvements suggest FID/KID might improve, but this must be measured.
4. Keep the official-curve comparison separate. This branch currently validates the GP-ResLC design principle but does not yet beat GLC's official perceptual curve.



## 2026-06-21 23:59 JST - Latent-gradient residual FID/KID protocol check

Added `--disable_residual` to `scripts/evaluate_glc_latent_residual_fullres_realcodec.py` so the semantic/base generator reconstruction can be saved through the same full-resolution loader, padding, and VQGAN decode path as the residual reconstructions. This lets us measure whether encoder-side `latent_grad` residual selection improves distribution metrics over the Stage-A/base predictor, not only DISTS.

Saved-reconstruction runs:

| dataset | variant | payload bpp | LPIPS | DISTS | FID | KID | patches | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CLIC2020 test all 428 | base semantic only | 0.010058 | 0.540915 | 0.341476 | 118.364 | 0.06663 | 28,650 | no residual stream |
| CLIC2020 test all 428 | latent_grad, delta=0.6 | 0.011000 | 0.559899 | 0.319396 | 105.336 | 0.04996 | 28,650 | residual AC bpp 0.000781 |
| DIV2K validation | base semantic only | 0.010087 | 0.565526 | 0.353956 | 181.631 | 0.12152 | 6,573 | no residual stream |
| DIV2K validation | latent_grad, delta=0.6 | 0.011024 | 0.603688 | 0.334945 | 156.661 | 0.08339 | 6,573 | residual AC bpp 0.000784 |

Per-image deltas, residual minus base:

| dataset | DISTS mean/median | DISTS wins | LPIPS mean/median | LPIPS wins | L1 mean/median | L1 wins |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 test all | -0.02208 / -0.01450 | 373/428 | +0.01898 / -0.00049 | 234/428 | +0.00382 / -0.00024 | 237/428 |
| DIV2K validation | -0.01901 / -0.01441 | 83/100 | +0.03816 / +0.00181 | 45/100 | +0.00910 / +0.00285 | 37/100 |

Interpretation:

- The official-style patch protocol is correct for both datasets: CLIC gives 28,650 shifted 256-patches and DIV2K gives 6,573 shifted 256-patches.
- `latent_grad` is not merely improving DISTS; it also improves FID/KID versus the semantic/base generator on both CLIC and DIV2K.
- Absolute FID/KID/DISTS remain far worse than official GLC because this is still the weak scratch/Stage-A semantic branch. Do not compare these absolute values to the pretrained GLC official curve as a claim of SOTA.
- The important research signal is mechanistic: at a fixed sparse residual budget and real serialized payload, selecting residual symbols by decoder-side latent payoff sends more distribution-useful information than sending no residual or selecting by raw latent error.
- The limitation is now sharp: the current payoff is DISTS/FID/KID oriented and can damage LPIPS/L1, especially on DIV2K. The next serious model should learn a multi-objective residual-value selector: latent/DISTS payoff with LPIPS/L1 no-regression or a decoder-safe residual strength predictor.

Artifacts:

- `experiments/realcodec_single_topk_latent_grad_notrain_clic2020_all_delta06_save/`
- `experiments/realcodec_single_topk_latent_grad_notrain_clic2020_all_base_save/`
- `experiments/realcodec_single_topk_latent_grad_notrain_div2k_delta06_save/`
- `experiments/realcodec_single_topk_latent_grad_notrain_div2k_base_save/`


## 2026-06-22 00:57 JST - Sign-aware and adaptive residual-strength pilots

Follow-ups after the CLIC/DIV2K FID/KID check tested whether the LPIPS/L1 damage of `latent_grad` could be reduced without losing the DISTS/FID mechanism.

### Sign-aware latent-gradient selector

Implemented `topk_score_mode=latent_grad_improve`, using only first-order latent-loss-improving candidates:

`score_i = relu(- grad_i * candidate_i)`

This produced the same Kodak full-resolution result as the previous absolute `latent_grad` selector: payload bpp `0.011617`, LPIPS `0.595405`, DISTS `0.354949`. Interpretation: the learned residual encoder already proposes mostly latent-loss-improving symbol signs, so the LPIPS damage is not caused primarily by selecting first-order harmful latent directions.

### Fixed residual-delta shaping

Tested no-side-bit decoder shaping of the latent residual delta.

| variant | dataset | LPIPS | DISTS | read |
|---|---|---:|---:|---|
| latent_grad delta=0.6 | Kodak | 0.574957 | 0.371653 | safest previous fixed scale |
| split k=3, low=0.8, high=0.3 | Kodak | 0.577646 | 0.364341 | good Kodak trade-off |
| split k=3, low=0.8, high=0.3 | DIV2K | 0.644967 | 0.337498 | worse than delta=0.6 on both LPIPS/DISTS; reject |
| lowpass k=3 | Kodak | 0.586429 | 0.357182 | strong Kodak DISTS, LPIPS worse |
| lowpass k=3 | DIV2K | interrupted after 25/100 | clearly bad early: LPIPS `0.5411 -> 0.8433`, DISTS `0.3503 -> 0.3813` on image 0825 | reject |

Conclusion: fixed low/high-frequency shaping is dataset-sensitive. It can improve Kodak but does not generalize to DIV2K. Do not promote as a method; use it only as evidence that residual frequency content matters and should be controlled adaptively.

### Adaptive delta-scale head

Ran a decoder-side adaptive residual-strength pilot that freezes all modules except `delta_scale_net`:

- W&B: `iw6gcvot`
- Run: `experiments/glc_latent_delta_gateonly_latentgrad_lpips_safe_14500to15300/`
- Training: OpenImages train, Kodak crop validation, 800 steps from `it=14500`.
- Loss: LPIPS/DISTS/L1 with base no-regression penalties and small mean-scale penalty.

Full-resolution Kodak real-codec results:

| checkpoint | adaptive scale mean | LPIPS | DISTS | decision |
|---|---:|---:|---:|---|
| final | 0.458 | 0.602286 | 0.360207 | reject; LPIPS too poor |
| best / 15200 | 0.620 | 0.625681 | 0.363172 | reject; worse LPIPS |

Interpretation: crop validation was misleading for the adaptive gamma head. The head learned unstable full-resolution behavior and did not solve the LPIPS/L1 regression. For future adaptive residual strength, evaluate Kodak full-res early and train with full-res or larger crops; otherwise the learned gamma overfits crop statistics.

Current practical conclusion remains:

- `latent_grad delta=0.6` is the best cross-dataset mechanism point because it improves CLIC/DIV2K FID/KID and DISTS with moderate, known LPIPS cost.
- The next serious direction is not fixed filtering or a tiny gamma head; it is a learned multi-objective residual-value selector trained from full-res/cross-dataset teacher signals, or a stronger Stage-A/generator so the residual does not need to perturb perceptual features so aggressively.
