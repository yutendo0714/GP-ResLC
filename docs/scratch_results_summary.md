# Scratch GP-ResLC Results Summary

Last updated: 2026-06-21 JST

This file summarizes the method-faithful scratch branch. These are deterministic 256x256 center-crop evaluations unless explicitly noted. They are not official full-resolution real-codec GLC evaluations.

## Current Best Points

| role | checkpoint | dataset | bpp | LPIPS | DISTS | note |
|---|---|---|---:|---:|---:|---|
| Stage-A semantic base | `experiments/scratch_stage_a_down5_attn_refine_from_d2_8000_6k/stage_a_best.pt` | Kodak | 0.00977 | 0.45767 | 0.43546 | DISTS-best semantic/generator base; LPIPS worsens vs source. |
| Scratch DISTS lead | `experiments/scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt` | Kodak | 0.01377 | 0.44009 | 0.42195 | Best DISTS point; LPIPS worsens vs single-stage lead. |
| Scratch LPIPS auxiliary | `experiments/scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_final.pt` | Kodak | 0.01338 | 0.43546 | 0.42642 | Better LPIPS, worse DISTS. |
| Scratch quality-side point | `experiments/scratch_stage_b_from_attnA_best_r8_q1_lR0p3_d2_3k/stage_b_final.pt` | Kodak | 0.01588 | 0.43752 | 0.42396 | Slight DISTS gain vs earlier r8 final but higher bpp. |
| DIV2K sanity check | `experiments/scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt` | DIV2K center | 0.01424 | 0.42176 | 0.41391 | Strong lower-bpp DISTS point; not LPIPS lead. |

## Important Comparisons

| experiment | best checked point | result | decision |
|---|---|---|---|
| Original Stage-A source + r8 Stage-B | 0.01775 bpp / DISTS 0.43456 | First good decomposition signal. | Superseded by attention Stage-A. |
| DISTS-heavy Stage-A source + `lambda_R=0.1` | 0.02212 bpp / DISTS 0.43195 | Improves base but inefficient. | Do not promote. |
| Attention Stage-A + r8 q1 `lambda_R=0.5` | 0.01321 bpp / DISTS 0.42313 | Clear Pareto update. | Current scratch DISTS lead. |
| Attention Stage-A + r8 q1 `lambda_R=0.3` | 0.01588 bpp / DISTS 0.42396 | More bits, only tiny DISTS change. | Auxiliary curve point. |
| Attention Stage-A + r4 q1 `lambda_R=0.5` | 0.01376 bpp / DISTS 0.42853 | Narrow residual is less efficient. | Reject. |
| Attention Stage-A + r16 q1 `lambda_R=0.5` | 0.01258 bpp / DISTS 0.43298 | Very low-rate auxiliary, not quality lead. | Reject as quality point. |
| Attention Stage-A + r8 q0.5 `lambda_R=0.5` | 0.01329 bpp / DISTS 0.43199 | Finer quantization is worse. | Reject. |

## Interpretation

The scratch branch now genuinely follows the original GP-ResLC idea: an 8x8 semantic/generative stream carries predictable structure at about 0.00977 bpp, and a small residual stream adds only about 0.0034 bpp to correct unpredictable components. The best result improves Kodak DISTS from the Stage-A base 0.43546 to 0.42313 at total bpp 0.01321.

However, absolute quality is still far from the pretrained GLC real-codec branch. The scratch branch is evidence for the method mechanism and a high-upside future direction, not the current VCIP lead.

## Next Best Scratch Steps

1. Implement progressive/RVQ residual coding instead of a single residual bottleneck.
2. Add a real entropy-coded residual payload for scratch only after proxy trends stabilize.
3. Improve Stage-A generator quality without sacrificing LPIPS: delayed adversarial loss, feature matching, or stronger hybrid decoder blocks.
4. Build a CLIC/DIV2K shifted-patch evaluation wrapper only if scratch quality approaches the pretrained branch.


## Progressive Residual Update

Two-stage residual coding was implemented and tested. Non-gated progressive residual improves the high-quality side but is bit-inefficient: the best checked quality-side point is `scratch_stage_b_progressive2_from_attnA_r8_q1q05_lR0p6_3k/stage_b_0002000.pt` at Kodak center bpp `0.01954`, LPIPS `0.43337`, DISTS `0.41948`, with stage-1 bpp `0.00694`. This is not a replacement for the low-rate scratch lead.

Decoder-side gated progressive residual keeps the rate in the target band, but the current hard gate mostly closes stage 1. The useful low-rate point is `scratch_stage_b_progressive2_gated_softtrain_from_attnA_r8_q1q05_lR0p5_t02_bm15_3k/stage_b_0001000.pt`: bpp `0.01299`, LPIPS `0.43538`, DISTS `0.42373`, stage-1 bpp approximately zero. This is a lower-rate curve point, not a new DISTS lead.

Decision: keep the single-stage residual checkpoint `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` as the scratch DISTS lead. The next architecture should give stage 1 its own correction path or an improvement hinge, because a shared decoder lets the model ignore the gated fine stage.


### Progressive Gate Threshold Sweep

The soft-train/hard-eval gated checkpoint was swept at evaluation thresholds 0.20, 0.15, 0.10, and 0.05. Lowering the threshold opens stage 1 and improves LPIPS slightly, but DISTS barely changes while bpp rises: threshold 0.05 gives bpp `0.01658`, LPIPS `0.43284`, DISTS `0.42352`, stage1 bpp `0.00359`. This confirms that stage 1 is not yet learning a DISTS-useful residual. A fine-correction decoder pilot at 1000 steps also failed to improve the lead (`0.01323` bpp / DISTS `0.42441`).

Next scratch priority: add a stage-improvement hinge/objective so the fine stage is trained to improve a detached stage-0 reconstruction before rate pruning.


### Stage-Improvement Hinge Pilot

A fine-decoder progressive run with a stage-improvement hinge (`lambda_stage_improve=5.0`, margin `0.001`, W&B `9g72335u`) produced a new scratch quality-side point but not a low-rate lead. On Kodak center crops, `scratch_stage_b_progressive2_finedec_softgate_stageimpr_from_attnA_r8_q1q05_lR0p5_si5_2k/stage_b_0001000.pt` reaches bpp `0.01422`, LPIPS `0.43778`, DISTS `0.42232`. On DIV2K center it reaches bpp `0.01553`, LPIPS `0.42011`, DISTS `0.41333`. Hard stage 1 remains closed; threshold 0.10 adds stage1 bpp but does not improve DISTS.

Decision: update scratch quality-side point to the stage-improvement checkpoint, but keep the low-rate DISTS lead as `scratch_stage_b_from_attnA_best_r8_q1_lR0p5_continue6k/stage_b_0004000.pt` at bpp `0.01321`, DISTS `0.42313`.


### Stage-1 Warmup And Gate Fine-Tune

Added `--train_only_extra_stages` and ran a forced-open stage-1 warmup followed by gate/rate fine-tuning. Stage-1 warmup confirms the fine residual can carry useful information: Kodak center bpp `0.02003`, LPIPS `0.43720`, DISTS `0.42222`, stage1 bpp `0.00681`. After gate fine-tuning, the better practical point is bpp `0.01349`, LPIPS `0.43748`, DISTS `0.42283`, with stage1 almost fully pruned. This slightly improves the previous scratch DISTS lead at a modestly higher bpp, but it still does not realize sparse stage-1 transmission strongly enough.

### Top-k Sparse Fine-Residual Pilot

A top-k gate-budget variant was added after the stage-1 warmup (`gate_topk_frac=0.05`, W&B `oa3hchyt`). Kodak-center final metrics are bpp `0.01391`, LPIPS `0.43889`, DISTS `0.42378`, with stage0 bpp `0.00380`, stage1 bpp `0.00034`, and stage1 gate mean `0.04883`.

Decision: this validates sparse fine-residual selection without side information, but it is not the scratch lead. The selected top-5% fine residual is currently too weak in DISTS terms. Keep the single-stage residual checkpoint as the low-rate lead and treat top-k gating as the next mechanism to improve.

### Top-k Gate Budget Sweep

Evaluated the same top-k checkpoint with different deterministic gate budgets on Kodak center crops:

| top-k frac | bpp | LPIPS | DISTS | stage1 bpp | stage1 gate mean |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.01371 | 0.43889 | 0.42378 | 0.00014 | 0.01953 |
| 0.05 | 0.01391 | 0.43889 | 0.42378 | 0.00034 | 0.04883 |
| 0.10 | 0.01433 | 0.43888 | 0.42378 | 0.00077 | 0.09961 |
| 0.20 | 0.01519 | 0.43888 | 0.42380 | 0.00163 | 0.19922 |

DIV2K center at the trained 5% budget gives bpp `0.01515`, LPIPS `0.42036`, DISTS `0.41508`, stage1 bpp `0.00036`, gate mean `0.04883`.

Interpretation: increasing the fine-stage budget from 2% to 20% mostly increases bpp while DISTS stays around `0.42378-0.42380`. LPIPS improves only in the fourth decimal place. This rules out a simple budget issue: the current fine stage needs a stronger hard-gated correction objective, not just a wider gate.

### Top-k 10% Strong Stage-Improvement Pilot

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

### Selected-Region Top-k Fine-Residual Update

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


## LPIPS-Balanced Follow-up (2026-06-21)

The DISTS-leading scratch checkpoint remains:
- `experiments/scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt`
- Kodak: bpp 0.013768, LPIPS 0.440089, DISTS 0.421954.
- DIV2K: bpp 0.014242, LPIPS 0.421761, DISTS 0.413914.

LPIPS-oriented follow-ups produced better LPIPS but weaker DISTS:
- Global LPIPS 1.2 final (`4sr7hpua`): Kodak bpp 0.013646, LPIPS 0.438245, DISTS 0.423436; DIV2K bpp 0.014156, LPIPS 0.420427, DISTS 0.415380.
- Stage0 LPIPS no-regression final (`njfmi964`): Kodak bpp 0.013715, LPIPS 0.438228, DISTS 0.423569.

Conclusion: keep selected-region + scale-guard 1000 as scratch DISTS lead; keep LPIPS-balanced checkpoints as Pareto/reference points, not as main scratch result.

## 2026-06-21 Scratch selected-loss audit and LPIPS-spatial follow-up

Discovered and fixed a critical scratch Stage-B issue: `selected_region_improvement_loss` in `scripts/train_scratch_stage_b.py` was accidentally decorated with `@torch.no_grad()`. The selected-region L1 loss therefore did not backpropagate in earlier selected-region runs. The previous scratch DISTS lead remains a valid measured checkpoint, but its mechanism should not be described as caused by the selected L1 term.

Re-ran selected-region experiments after the fix:

| run | W&B | setting | Kodak center result | decision |
|---|---|---|---|---|
| `scratch_stage_b_progressive2_selected_gradfix_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500` | `e1jku5vo` | selected L1=20, DISTS=2 | best checked final/1000 around bpp `0.01364-0.01370`, LPIPS `0.4388-0.4392`, DISTS `0.4232` | true selected L1 improves L1/LPIPS slightly but worsens DISTS; reject as main scratch lead |
| `scratch_stage_b_progressive2_selected_gradfix_extraonly_topk010_sel5_d4_s1scale08_from_warm_lR0p5_1500` | `7xk6rmuo` | selected L1=5, DISTS=4 | 500: `0.013694/0.440696/0.423637`; 1000: `0.013670/0.439949/0.423410`; final: `0.013642/0.441078/0.422958` for bpp/LPIPS/DISTS | lower rate and acceptable LPIPS, but still worse DISTS than old lead `0.421954`; reject as lead |

Interpretation: local L1 selected improvement is the wrong proxy for the scratch perceptual objective. It makes the transmitted fine residual more locally faithful, but that does not align with the DISTS/FID-style R-P claim. Added a new `--lambda_selected_lpips_improve` path that uses LPIPS spatial maps only on decoder-selected fine-stage regions. This is closer to the paper axis: selected residual positions must improve perceptual feature distance over the generator-only stage0 reconstruction.

Current follow-up running:

| run | W&B | setting |
|---|---|---|
| `scratch_stage_b_progressive2_selected_lpipsmap_extraonly_topk010_sellp10_d4_s1scale08_from_warm_lR0p5_1500` | `tpauo0kk` | selected LPIPS-spatial=10, selected L1 disabled, DISTS=4, LPIPS=0.5, top-k 10%, extra-stage-only |

### LPIPS-Spatial Selected Loss Result

Evaluated `scratch_stage_b_progressive2_selected_lpipsmap_extraonly_topk010_sellp10_d4_s1scale08_from_warm_lR0p5_1500` (W&B `tpauo0kk`) on fixed Kodak center crops:

| checkpoint | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.013835 | 0.438671 | 0.423115 | 0.003448 | 0.000621 | 0.607 | not lead |
| 1000 | 0.013772 | 0.438635 | 0.423123 | 0.003448 | 0.000558 | 0.660 | not lead |
| final | 0.013716 | 0.438633 | 0.423080 | 0.003448 | 0.000503 | 0.701 | not lead |

Compared with the current scratch DISTS lead (`0.013768` bpp, LPIPS `0.440089`, DISTS `0.421954`), LPIPS-spatial selected loss improves LPIPS but loses DISTS. Decision: keep as an LPIPS-oriented auxiliary/reference, not the scratch lead. The selected fine-stage objective should next be DISTS/texture-statistic aligned, or the scratch branch should move to a stronger generator before spending more time on local selected losses.



## 2026-06-21 10:10 JST - Scratch selected VGG feature improvement loss

Implementation update:

- Added `--lambda_selected_vgg_improve`, `--selected_vgg_improve_margin`, and `--selected_vgg_layers` to `scripts/train_scratch_stage_b.py`.
- The new loss uses the VGG feature stages inside `DISTS_pytorch.DISTS.forward_once()` and penalizes decoder-selected fine-stage regions where the final reconstruction's local feature error is worse than detached stage-0. This is a DISTS-adjacent local selected-region proxy, intended to be closer to structure/texture fidelity than local L1 or LPIPS-spatial alone.
- Smoke test passed: checkpoint loading was fully compatible (`230 tensors`, `missing=0`, `skipped=0`) and `selvggimpr` was nonzero at startup.

Run:

- `experiments/scratch_stage_b_progressive2_selected_vgg_extraonly_topk010_selvgg10_d4_s1scale08_from_warm_lR0p5_1500`
- W&B: `0vkyk0iu`
- Setting: extra-stage-only, top-k 10%, selected VGG improvement weight 10, DISTS weight 4, LPIPS weight 0.5, stage1 scale guard 0.2/0.8.

Fixed Kodak center-crop results:

| checkpoint | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.014087 | 0.437148 | 0.424297 | 0.003718 | 0.000603 | 0.693 | not lead |
| 1000 | 0.014063 | 0.437081 | 0.424611 | 0.003718 | 0.000579 | 0.718 | not lead |
| final | 0.014053 | 0.437388 | 0.424407 | 0.003718 | 0.000569 | 0.669 | not lead |

Decision: reject as scratch lead. The VGG selected no-regression loss becomes zero quickly, so it mainly behaves like another DISTS-heavy extra-stage run. If this path is revisited, require a positive feature-improvement margin rather than only no-regression. Current scratch DISTS lead remains `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt` at Kodak bpp `0.013768`, LPIPS `0.440089`, DISTS `0.421954`.


## 2026-06-21 10:18 JST - Scratch selected VGG margin follow-up

Follow-up run:

- `experiments/scratch_stage_b_progressive2_selected_vggmargin_extraonly_topk010_selvgg30m003_d4_s1scale08_from_warm_lR0p5_1000`
- W&B: `bw5cnko1`
- Change from selected VGG no-regression: `lambda_selected_vgg_improve=30`, `selected_vgg_improve_margin=0.003`, 1000 iterations.

Fixed Kodak center-crop results:

| checkpoint | bpp | LPIPS | DISTS | stage0 bpp | stage1 bpp | stage1 scale | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.014102 | 0.437569 | 0.431100 | 0.003718 | 0.000618 | 0.666 | reject |
| final | 0.014070 | 0.437357 | 0.432706 | 0.003718 | 0.000587 | 0.737 | reject |

Decision: positive-margin VGG selected improvement is worse than no-regression and much worse than the current scratch lead. The feature-improvement pressure appears to fight the DISTS objective under the current weak generator/fine decoder. Stop this local VGG selected-loss path for now. Next scratch work should prioritize stronger Stage-A/generator quality or a true DISTS-statistic local proxy, not more VGG/L1 selected losses.


## 2026-06-21 10:35 JST - Scratch decoder-only and gate-error-target screening

### Stage-A decoder-only fine-tune from attention Stage-A best

- Run: `scratch_stage_a_decoder_only_from_attn_best_d3_lp08_l103_4k`
- W&B: `gsyo72t9`
- Change: froze Stage-A encoder and VQ codebook, trained decoder/latent-refine only with stronger perceptual loss (`lambda_dists=3.0`, `lambda_lpips=0.8`, `lambda_l1=0.3`).
- Outcome: stopped early at ~2500 iters because validation DISTS did not recover.
- Deterministic Kodak center:
  - `stage_a_best.pt`: bpp `0.0097656`, LPIPS `0.45655`, DISTS `0.43722`
  - `stage_a_0001000.pt`: bpp `0.0097656`, LPIPS `0.45191`, DISTS `0.45155`
  - `stage_a_0002000.pt`: bpp `0.0097656`, LPIPS `0.44913`, DISTS `0.44866`
- Decision: reject as Stage-A replacement. Decoder-only improves L1/LPIPS slightly but damages DISTS relative to the existing Stage-A best (`DISTS=0.43546`). This suggests the fixed 8x8 semantic latent is the bottleneck; decoder polishing alone cannot supply the missing perceptual structure.

### Stage-B decoder-side gate error-target auxiliary loss

- Code change: `ScratchProgressiveResidualBottleneck` now exposes `stage{i}_gate_prob` and `stage{i}_gate_logit`. `scripts/train_scratch_stage_b.py` adds `--lambda_gate_error_target` and `--gate_error_target_topk_frac`.
- Purpose: teach the decoder-computable fine-stage gate to select regions where Stage-A base reconstruction fails, without transmitting any extra side information.
- Run: `scratch_stage_b_gateerr_from_selected1000_ge02_1200`
- W&B: `k8a2znzk`
- Init: resumed from `experiments/scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500/stage_b_0001000.pt`.
- Key settings: `lambda_gate_error_target=0.2`, top-k `0.10`, extra-stage-only training, existing selected-region improvement losses retained.
- Deterministic Kodak center:
  - `stage_b_best.pt`/`stage_b_0001500.pt`: bpp `0.013935`, LPIPS `0.43914`, DISTS `0.42309`
  - `stage_b_0002000.pt`: bpp `0.013912`, LPIPS `0.43939`, DISTS `0.42327`
  - `stage_b_final.pt`: bpp `0.013940`, LPIPS `0.43895`, DISTS `0.42306`
- Comparison: previous scratch lead remains `stage_b_0001000.pt` from `scratch_stage_b_progressive2_selected_extraonly_topk010_sel20_si8_s1scale08_from_warm_lR0p5_1500` with bpp `0.013768`, LPIPS `0.44009`, DISTS `0.42195`.
- Decision: reject as DISTS lead, keep as useful ablation. Gate-target improves LPIPS slightly and is conceptually aligned, but DISTS worsens by ~0.0011 and bpp is slightly higher.


## 2026-06-21 17:10 JST - New GLC-Latent Residual Full-Design Branch

A new high-upside branch was started after shifting from VCIP-short-track optimization to top-conference/full-paper research. The motivation is that the previous scratch branch was conceptually faithful but bottlenecked by its weak Stage-A image generator. The new branch keeps the same semantic-plus-unpredictable-residual thesis but decodes through the stronger pretrained GLC/VQGAN generator.

Design:

- Semantic stream: existing Stage-A 8x8 VQ code, fixed semantic rate `0.00977` bpp for 256x256 crops.
- Predictable component: `mu_theta(s)` predicts the frozen GLC/VQGAN latent `l = E_GLC-VQGAN(x)`.
- Residual component: a low-dimensional residual bottleneck entropy-models the correction needed to reconstruct `l` beyond `mu_theta(s)`.
- Decoder: frozen GLC/VQGAN generator synthesizes from `mu_theta(s) + delta_r`.

Implementation:

- `gp_reslc/scratch/glc_latent_residual.py`
- `scripts/train_glc_latent_residual.py`

Initial evidence:

- 2-iteration smoke passed on CUDA.
- 80-iteration predictor-only pilot reduced Kodak validation DISTS from about `0.77` to about `0.51` using only the semantic stream.
- Online warmup run `glc_latent_residual_predictor_warmup_6k` (W&B `woye1ymw`) is running. At iteration 500 validation: bpp `0.00977`, LPIPS `0.7868`, DISTS `0.4716`.

Interpretation:

This branch is not yet competitive, but it is a better route to the original full GP-ResLC idea than continuing local loss variants on the weak scratch generator. If `mu_theta(s)` approaches or beats the old Stage-A generator quality, residual training can test the exact claim: only send the part of the GLC latent that the decoder cannot predict from the semantic code.


## GLC-Latent Residual Update: Predictor Good, Residual Collapsed

The new frozen-GLC-generator branch is now the strongest method-faithful scratch direction by DISTS on fixed Kodak center crops, but it revealed an important residual-coding failure mode.

| checkpoint | mode | bpp | LPIPS | DISTS | residual payload | decision |
|---|---|---:|---:|---:|---|---|
| `glc_latent_residual_predictor_warmup_6k/glc_latent_residual_final.pt` | semantic-only | 0.00977 | 0.65444 | 0.42000 | none | useful warmup, LPIPS weak |
| `glc_latent_residual_residual_lR1_lp2_d2_from_warm_3k/glc_latent_residual_final.pt` | no residual | 0.00977 | 0.58901 | 0.39862 | none | strong deterministic refinement |
| same | residual on | 0.009766 | 0.58593 | 0.39924 | hard-rounded symbols effectively zero | not a true residual-code result |
| `glc_latent_residual_residual_q025_lR07_lp2_d15_from_lR1_1500/glc_latent_residual_final.pt` | no residual | 0.00977 | 0.57749 | 0.39775 | none | current DISTS-best scratch/full-design point |
| same | residual on | 0.009766 | 0.56636 | 0.39775 | hard-rounded symbols effectively zero | LPIPS gain, but payload still collapsed |

Readout:

- This branch is promising because it uses the original decomposition in GLC latent space: semantic code predicts the generator latent, and residuals should correct only what the generator cannot infer.
- The current best quality gain is mostly from a decoder-side deterministic latent predictor/refiner, not from transmitted residuals.
- The next required fix is quantization-consistent training. `quant_mode=ste` and rounded-symbol logging have been added; a branch with hard-rounded STE residuals should replace the previous additive-noise residual runs before claiming residual-coding improvements.


### STE residual diagnostic result

Run: `experiments/glc_latent_residual_residual_ste_q025_lR02_lp25_d15_from_q025_1k`

Settings: resumed from q0.25 final, `quant_mode=ste`, `quant_step=0.25`, `lambda_R=0.2`, LPIPS-heavy residual phase, no W&B upload.

Fixed Kodak center-crop results:

| checkpoint | mode | bpp | LPIPS | DISTS | rounded nonzero frac | read |
|---|---|---:|---:|---:|---:|---|
| `0011000` | no residual | 0.009766 | 0.57976 | 0.40251 | 0 | predictor degraded vs q0.25 parent |
| `0011000` | residual on | 0.009766 | 0.56710 | 0.40352 | 0 | LPIPS improves, DISTS worsens; no true payload |
| `final` | no residual | 0.009766 | 0.59107 | 0.40279 | 0 | predictor/refiner degrades further |
| `final` | residual on | 0.010189 | 0.57695 | 0.40246 | 0.000244 | first tiny nonzero payload; LPIPS improves, DISTS nearly tied |

Conclusion: STE is necessary but insufficient. The residual decoder can still produce a nonzero correction from `mu` and `z_up` even when `q_residual=0`, so the model escapes into bit-free deterministic refinement. The next implementation must force `delta_r=0` wherever the rounded residual payload is zero, while preserving a straight-through gradient so the residual encoder can learn to activate paid residual locations.


### Payload-gated and top-k residual diagnostics

Implementation update:

- Added `delta_gate_mode=payload_ste`: residual correction `delta_r` is zero wherever the rounded residual payload is zero. This blocks the previous bit-free deterministic-refinement escape path.
- Added `force_topk_frac`: encoder-side sparse residual warmup that forces only the highest-magnitude residual coefficients to nonzero rounded symbols. The full residual grid is still entropy-modeled; no separate free mask is assumed.

Runs:

| run | setting | fixed Kodak center result | decision |
|---|---|---|---|
| `glc_latent_residual_payloadste_q025_lR005_lp25_d15_from_q025_800` | payload gate, no forced top-k | residual-on and no-res are identical: bpp `0.009766`, LPIPS/DISTS `0.57767/0.40620`, `delta_active_frac=0` | gate is correct, but residual does not self-activate. |
| `glc_latent_residual_payloadste_topk001_q025_lR01_lp25_d15_from_q025_800`, checkpoint `0011200` | payload gate + top-k `0.001` | no-res: bpp `0.009766`, LPIPS/DISTS `0.56709/0.40019`; residual-on: bpp `0.011458`, LPIPS/DISTS `0.56678/0.39951`, `rounded_nonzero_frac=0.00098`, `delta_active_frac=0.02165` | first true paid-residual point with both LPIPS and DISTS improving, but gain is very small. |
| same, final | payload gate + top-k `0.001` | no-res: bpp `0.009766`, LPIPS/DISTS `0.57175/0.39984`; residual-on: bpp `0.011458`, LPIPS/DISTS `0.57130/0.39916`, `rounded_nonzero_frac=0.00098`, `delta_active_frac=0.02035` | same direction as 0011200; keep as a mechanism-positive branch, not a performance lead. |

Interpretation:

- The payload gate proves the previous residual gains were partly an architectural escape path: when zero-payload correction is forbidden, the model initially collapses to no-res output.
- Sparse top-k residual warmup gives the first clean signal that a tiny paid residual can improve both LPIPS and DISTS under the intended rule.
- The effect is too small for a paper claim. The next high-upside path is not more tiny top-k tuning alone; it should train a stronger predictor/residual pair with a scheduled sparse budget, e.g. start with top-k `0.002-0.005`, then anneal the budget/rate and add a perceptual improvement objective on active residual sites.


### DIV2K check and residual-budget upper-bound diagnostics

DIV2K center-crop check (`/dpl/div2k`, 100 images):

| checkpoint | mode | bpp | LPIPS | DISTS | note |
|---|---|---:|---:|---:|---|
| q0.25 parent final | no residual | 0.009766 | 0.56085 | 0.39429 | deterministic predictor only |
| q0.25 parent final | residual on | 0.009766 | 0.54868 | 0.39310 | improves, but this is payload-gate-free deterministic correction, not a valid paid-residual claim |
| top-k 0.001 final | no residual | 0.009766 | 0.54983 | 0.39302 | payload-gated predictor state |
| top-k 0.001 final | residual on | 0.011458 | 0.54976 | 0.39295 | paid residual effect generalizes, but is extremely small |

Residual-budget diagnostics on fixed Kodak center crops:

| run | top-k frac | bpp | LPIPS no-res -> res | DISTS no-res -> res | decision |
|---|---:|---:|---:|---:|---|
| `payloadste_topk001_q025_lR01_lp25_d15_from_q025_800` final | 0.001 | 0.00977 -> 0.01146 | 0.57175 -> 0.57130 | 0.39984 -> 0.39916 | best mechanism-positive paid-residual branch so far |
| `payloadste_topk005_q025_lR005_lp25_d15_from_q025_600` final | 0.005 | 0.00977 -> 0.01822 | 0.56502 -> 0.56415 | 0.39371 -> 0.39383 | more bits do not improve DISTS; not rate-efficient |
| `payloadste_topk005_activel1_q025_lR005_lp25_d15_from_q025_600` final | 0.005 | 0.00977 -> 0.01822 | 0.57829 -> 0.57831 | 0.40526 -> 0.40535 | active L1 lowers local pixel error but hurts perceptual metrics; reject |

Interpretation:

- `top-k=0.001` is the only branch that is both method-faithful and improves LPIPS/DISTS on Kodak and DIV2K, but the gain is much too small for a paper result.
- Increasing residual budget without better selection/objective wastes bits and can hurt DISTS.
- Active local L1 is the wrong proxy for perceptual residual usefulness, matching earlier scratch-stage findings that L1 selected losses do not align with DISTS/FID-style claims.

Next research decision:

1. Keep `top-k=0.001` as the clean mechanism proof.
2. Stop simple budget scaling and L1 active losses.
3. Move to either a perceptual active-site objective (spatial LPIPS/VGG/DISTS-proxy) or a residual-latent training phase that first teaches the residual decoder to reconstruct true latent residual under a sparse budget, then fine-tunes image perceptual quality.


### Active-latent sparse residual result and next pivot

Run: `glc_latent_residual_payloadste_topk001_activelat_q025_lR01_lp25_d15_from_q025_1000`

Settings: payload-gated top-k `0.001`, active latent no-regression loss (`lambda_active_latent_improve=10`, margin `0.001`).

Fixed Kodak center:

| checkpoint | mode | bpp | LPIPS | DISTS | decision |
|---|---|---:|---:|---:|---|
| `0010500` | no residual | 0.009766 | 0.57743 | 0.39724 | base |
| `0010500` | residual on | 0.011458 | 0.57685 | 0.39704 | improves both; best active-latent point |
| final | no residual | 0.009766 | 0.58267 | 0.39727 | base drifted |
| final | residual on | 0.011458 | 0.58269 | 0.39733 | reject; over-training hurts residual effect |

Fixed DIV2K center for `0010500`:

| mode | bpp | LPIPS | DISTS |
|---|---:|---:|---:|
| no residual | 0.009766 | 0.56066 | 0.39391 |
| residual on | 0.011458 | 0.56016 | 0.39379 |

Readout:

- Active-latent loss gives a cleaner early sparse residual point than plain top-k, and the sign generalizes to DIV2K.
- The gain is still too small, and longer training degrades the residual effect.
- The bigger limitation is likely upstream: the 8x8 Stage-A code was trained for the scratch decoder, not to predict GLC/VQGAN latent space. Keeping Stage-A frozen constrains the full-design branch.

Next pivot:

- Fine-tune the Stage-A encoder/codebook jointly with the GLC-latent predictor while keeping the semantic rate fixed. This directly learns a semantic code whose purpose is to let the frozen generator reconstruct predictable content, then residual coding can be revisited on top of a stronger semantic predictor.


### Stage-A joint fine-tune diagnostic

Run: `glc_latent_stagea_joint_pred_from_q025_2k`

Settings: resume q0.25 parent weights only, update Stage-A encoder/codebook plus GLC-latent predictor, predictor-only/no residual, fixed semantic bpp `0.00977`.

Fixed center-crop results for `glc_latent_residual_best.pt` (iteration 11500):

| dataset | bpp | LPIPS | DISTS | comparison |
|---|---:|---:|---:|---|
| Kodak | 0.009766 | 0.58188 | 0.39993 | worse than q0.25 parent no-res (`0.57749/0.39775`) |
| DIV2K | 0.009766 | 0.56311 | 0.39611 | worse than q0.25 parent no-res (`0.56085/0.39429`) |

Decision: reject this simple joint Stage-A fine-tune. Random validation looked promising, but deterministic Kodak/DIV2K show that moving the Stage-A encoder/codebook from a pretrained semantic basis hurts generalization. The next safer route is fixed Stage-A plus predictor-only fine-tuning, then revisit trainable semantic codes with a slower schedule, EMA/codebook regularization, or a separate Stage-II latent-code objective.


### Fixed Stage-A predictor-only continuation: new semantic lead

Run: `glc_latent_predictor_only_from_q025_d2_lp2_2k`

Settings: resume q0.25 parent weights only, keep Stage-A frozen, no residual, optimize the semantic-code-to-GLC-latent predictor with image perceptual and latent losses.

Best checkpoint: `glc_latent_residual_best.pt` at iteration `12000`.

Fixed center-crop results:

| dataset | checkpoint | bpp | LPIPS | DISTS | comparison to q0.25 parent no-res |
|---|---|---:|---:|---:|---|
| Kodak | predictor-only best | 0.009766 | 0.57334 | 0.39338 | improves from `0.57749/0.39775` |
| DIV2K | predictor-only best | 0.009766 | 0.56086 | 0.39113 | DISTS improves from `0.56085/0.39429`, LPIPS tied |

Decision:

- This is the strongest current full-design/scratch-direction result: no residual stream, no free correction, same fixed semantic bpp, better predictable-component reconstruction.
- The gain is larger and more stable than the paid sparse residual gains. This supports a research pivot: first maximize what the frozen generator can recover from the semantic code, then add paid residual only after the predictable component saturates.
- Next: continue this branch with a lower LR and DISTS-heavy but LPIPS-safe objective. Then re-attach payload-gated top-k residual from the improved predictor.


### Low-LR predictor continuation result

Run: `glc_latent_predictor_only_from_predbest_dists25_lp15_4k`

The 4k low-LR continuation from the semantic lead did not improve the fixed Kodak result:

| checkpoint | bpp | LPIPS | DISTS | decision |
|---|---:|---:|---:|---|
| previous predictor-only best | 0.009766 | 0.57334 | 0.39338 | keep as semantic lead |
| continuation `0015000` | 0.009766 | 0.57425 | 0.39715 | reject |
| continuation final | 0.009766 | 0.57373 | 0.39843 | reject |

Decision: do not continue this DISTS-heavy low-LR setting. The useful move is to keep `glc_latent_predictor_only_from_q025_d2_lp2_2k/glc_latent_residual_best.pt` as the semantic predictor lead and re-attach payload-gated sparse residual from that stronger predictable component.


## 2026-06-21 19:05 JST - GLC-latent hard-topk zero-center residual

Implemented two important corrections in the GLC-latent residual branch:

- `delta_gate_mode=zero_center`: residual correction is now `Decoder(q_residual, context) - Decoder(0, context)`. This preserves the core GP-ResLC constraint: if no residual payload is sent, the correction is exactly zero, while a sparse paid residual can still propagate through the convolutional decoder to neighboring latent positions.
- `--hard_topk`: after selecting top-k residual coefficients, all non-top-k rounded symbols are forced to zero. This fixes the previous `force_topk_frac` loophole where non-top-k symbols could become nonzero and silently increase bpp.

Also added:

- `--freeze_predictor`, so residual-only runs can test whether paid residual bits improve a fixed predictable component.
- `--reset_best_on_resume`, so continued runs can save a local best checkpoint.
- LPIPS/DISTS no-regression hinges against the no-residual base.
- residual/correction magnitude regularizers for later balancing.

### Key mechanism result

Run: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk001_perchinge_1500`

Settings:

- Resume: `experiments/glc_latent_predlead_topk001_activelat_lR01_1k/glc_latent_residual_final.pt`
- Predictor frozen, Stage-A frozen.
- `delta_gate_mode=zero_center`, `force_topk_frac=0.001`, `hard_topk=True`.
- Residual payload: roughly `0.002738` bpp, total `0.012504` bpp.

Fixed 256x256 center-crop proxy results. These are not official full-resolution real-codec evaluations.

| dataset/checkpoint | mode | bpp | LPIPS | DISTS | note |
|---|---|---:|---:|---:|---|
| Kodak | no residual | 0.009766 | 0.572118 | 0.391503 | fixed predictable component |
| Kodak best (`it=13500`) | residual on | 0.012504 | 0.570858 | 0.391440 | small but clean paid-residual improvement |
| Kodak final (`it=14500`) | residual on | 0.012504 | 0.589248 | 0.386138 | DISTS improves strongly, LPIPS worsens |
| DIV2K | no residual | 0.009766 | 0.557175 | 0.390806 | fixed predictable component |
| DIV2K best (`it=13500`) | residual on | 0.012504 | 0.556456 | 0.390581 | balanced improvement |
| DIV2K final (`it=14500`) | residual on | 0.012504 | 0.553252 | 0.384901 | strong improvement on both LPIPS and DISTS |
| CLIC2020 test center, 428 imgs | no residual | 0.009766 | 0.542022 | 0.376615 | professional 250 + mobile 178 |
| CLIC2020 test center, 428 imgs | residual on final | 0.012504 | 0.530758 | 0.370720 | strong improvement on both LPIPS and DISTS |

CLIC subset details:

| subset | mode | images | bpp | LPIPS | DISTS |
|---|---|---:|---:|---:|---:|
| professional test | no residual | 250 | 0.009766 | 0.533872 | 0.372375 |
| professional test | residual final | 250 | 0.012504 | 0.521344 | 0.367255 |
| mobile test | no residual | 178 | 0.009766 | 0.553470 | 0.382571 |
| mobile test | residual final | 178 | 0.012504 | 0.543980 | 0.375587 |

Interpretation:

- This is the cleanest scratch/full-design evidence so far for the original GP-ResLC axis: a fixed semantic/predictable stream is improved by sending only a strictly sparse residual payload.
- The final checkpoint generalizes well to CLIC and DIV2K, but Kodak LPIPS worsens despite better Kodak DISTS. Keep both `best` and `final` as curve/ablation points.
- The re-balance continuation from final (`experiments/glc_latent_zerocenter_hardtopk001_lpips_rebalance_from_final_1k`) did not fix Kodak LPIPS and is not promoted. It improves DIV2K but weakens Kodak enough that it should remain a rejected follow-up.

Next research steps:

1. Build a small rate/perception curve for `hard_topk` budgets, especially `0.0005`, `0.001`, `0.002`, and `0.005`, using zero-center residuals.
2. Add a non-saturating entropy proxy or scale lower-bound adjustment because `gaussian_bits` clamps probabilities and can under-penalize very large active symbols.
3. Add a lightweight LPIPS-safe objective or early stopping criterion based on a fixed validation set, since random Kodak-val batches are noisy and Kodak LPIPS can diverge from CLIC/DIV2K.
4. Once proxy behavior stabilizes, implement real entropy coding for the residual symbol stream and evaluate full-resolution CLIC/DIV2K/Kodak.


### Hard-topk budget 0.2% follow-up

Run: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk002_perchinge_1500`

Settings match the 0.1% run except `force_topk_frac=0.002`, `hard_topk=True`, and `lambda_R=0.1`.

Fixed center-crop proxy results:

| dataset/checkpoint | bpp | LPIPS | DISTS | decision |
|---|---:|---:|---:|---|
| DIV2K 0.2% `0014000` | 0.015241 | 0.553443 | 0.386538 | balanced but worse DISTS than 0.1% final |
| DIV2K 0.2% final | 0.015241 | 0.548306 | 0.389163 | better LPIPS, worse DISTS |
| CLIC2020 test center no-res | 0.009766 | 0.542022 | 0.376615 | reference |
| CLIC2020 test center 0.1% final | 0.012504 | 0.530758 | 0.370720 | current DISTS/L1 lead |
| CLIC2020 test center 0.2% final | 0.015241 | 0.527802 | 0.374115 | LPIPS-oriented point, not DISTS lead |

Interpretation: more sparse residual budget does not monotonically improve DISTS. The 0.2% payload learns stronger corrections and improves LPIPS on CLIC/DIV2K, but it harms DISTS relative to the 0.1% final point. This suggests the next improvement should not simply widen the top-k budget; it should improve residual objective/entropy calibration so stronger corrections stay perceptually aligned.


### Hard-topk budget curve update

Saved curve summary CSV: `experiments/glc_latent_hardtopk_curve_summary.csv`.

CLIC2020 test center-crop proxy, professional 250 + mobile 178:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.542022 | 0.376615 | 0.113911 | 0.026989 | predictable-only reference |
| hardtopk0005 | 0.011135 | 0.001369 | 0.530583 | 0.373010 | 0.108494 | 0.024987 | very efficient low payload point |
| hardtopk001 | 0.012504 | 0.002738 | 0.530758 | 0.370720 | 0.105636 | 0.023745 | current DISTS/L1/MSE lead |
| hardtopk002 | 0.015241 | 0.005475 | 0.527802 | 0.374115 | 0.107330 | 0.024625 | LPIPS-oriented, DISTS worsens |

DIV2K center-crop proxy:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE |
|---|---:|---:|---:|---:|---:|---:|
| no residual | 0.009766 | 0.000000 | 0.557175 | 0.390806 | 0.127053 | 0.032145 |
| hardtopk0005 | 0.011135 | 0.001369 | 0.550043 | 0.386773 | 0.122601 | 0.030408 |
| hardtopk001 | 0.012504 | 0.002738 | 0.553252 | 0.384901 | 0.119844 | 0.029408 |
| hardtopk002 | 0.015241 | 0.005475 | 0.548306 | 0.389163 | 0.121340 | 0.029877 |

Budget-curve interpretation:

- The curve is not monotonic across all perceptual metrics, which is useful evidence: more residual bits are not automatically better if the residual objective/entropy proxy allows overly strong local edits.
- `hardtopk0005` is the best efficiency point: very small residual payload produces large LPIPS/DISTS gains over no-residual on CLIC/DIV2K.
- `hardtopk001` is the best DISTS/L1/MSE point on CLIC and DIV2K.
- `hardtopk002` is LPIPS-oriented but not a good main point because DISTS degrades relative to `hardtopk001`.
- Next method work should improve entropy calibration and residual regularization, not simply increase top-k budget.


## 2026-06-21 20:05 JST - Stable bounded residual curve

Motivation: the previous hard-topk curve used the original clamped Gaussian bit proxy. That proxy is useful for fast screening, but it can under-penalize large active residual symbols. I added a stable entropy mode with a quadratic tail fallback and then constrained the transmitted residual symbols with `max_symbol_abs`. This makes the scratch/full-design branch closer to a real entropy-coded sparse residual stream.

Implementation changes:

- `gp_reslc/scratch/glc_latent_residual.py`: added `gaussian_bits_stable`, `entropy_mode`, and `max_symbol_abs`.
- `scripts/train_glc_latent_residual.py`: added CLI/config plumbing for stable entropy and bounded symbols.
- `scripts/evaluate_glc_latent_residual.py`: evaluation now restores `entropy_mode` and `max_symbol_abs` from checkpoint config.

Important negative result:

- Unbounded stable entropy with `topk=0.001` is unstable. Although nonzero positions remain fixed, active symbol magnitude grows and residual bpp can jump to multiple bpp. This confirms that position sparsity alone is insufficient; value bounding or a stronger entropy/real-codec constraint is required.

Stable bounded curve, fixed 256x256 center-crop proxy. These are not yet official full-resolution real-codec results.

CLIC2020 test center, 428 images:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.542011 | 0.376628 | 0.113910 | 0.026989 | reference |
| stable ternary topk0005 final | 0.010612 | 0.000847 | 0.537297 | 0.374999 | 0.111184 | 0.025975 | adopt low-rate point |
| stable ternary topk002 best | 0.013150 | 0.003384 | 0.529933 | 0.371677 | 0.107104 | 0.024467 | adopt main stable point |
| stable small-int2 topk001 best | 0.021877 | 0.012111 | 0.533375 | 0.373803 | 0.109661 | 0.025363 | reject: dominated |

DIV2K center:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.557210 | 0.390785 | 0.127048 | 0.032145 | reference |
| stable ternary topk0005 final | 0.010612 | 0.000847 | 0.553770 | 0.388299 | 0.124123 | 0.030960 | adopt low-rate point |
| stable ternary topk002 best | 0.013150 | 0.003384 | 0.550992 | 0.385331 | 0.121149 | 0.029947 | adopt main stable point |
| stable small-int2 topk001 best | 0.021877 | 0.012111 | 0.552500 | 0.387740 | 0.123059 | 0.030568 | reject: dominated by topk002 |

Kodak center:

| point | bpp | residual bpp | LPIPS | DISTS | L1 | MSE | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| no residual | 0.009766 | 0.000000 | 0.572011 | 0.391566 | 0.114565 | 0.024937 | reference |
| stable ternary topk0005 final | 0.010612 | 0.000847 | 0.569598 | 0.389898 | 0.113148 | 0.024466 | adopt low-rate point |
| stable ternary topk002 best | 0.013150 | 0.003384 | 0.575541 | 0.390750 | 0.110967 | 0.023804 | mixed: DISTS/L1 improve, LPIPS worsens |
| stable small-int2 topk001 best | 0.021877 | 0.012111 | 0.570009 | 0.390974 | 0.112631 | 0.024205 | reject: not rate efficient |

Interpretation:

- `stable_ternary_topk0005` is now the cleanest low-rate evidence for the GP-ResLC axis: only 0.00085 residual bpp improves LPIPS/DISTS/L1/MSE on CLIC, DIV2K, and Kodak.
- `stable_ternary_topk002` is the best stable perceptual point on CLIC/DIV2K, but Kodak LPIPS worsens. Keep it as the main stable curve point, not as the only visual candidate.
- `stable_smallint2_topk001` improves over no-residual but is dominated by `stable_ternary_topk002`; increasing symbol amplitude is worse than sending a few more ternary positions. This supports a simple design preference: sparse ternary residuals over high-amplitude sparse residuals.
- Next method direction: real-codec arithmetic coding for bounded residual symbols, fixed-validation checkpoint selection, and possibly topk0015/learned topk budget to bridge low-rate and main stable points.

Saved CSV: `experiments/glc_latent_stable_bounded_curve_summary.csv`.


## 2026-06-21 20:25 JST - TorchAC real residual codec bridge

Implemented `scripts/evaluate_glc_latent_residual_realcodec.py` for the GLC-latent scratch branch. This script keeps the current fixed semantic-code bpp accounting, but actually entropy-codes the bounded residual symbol tensor with `torchac` using the same Gaussian CDF helper as the GLC real codec. It then decodes symbols, reconstructs from decoded symbols, and reports exact byte-derived residual bpp plus decode consistency.

Important caveat: this is still a center-crop development codec, not the final full-resolution paper codec. The per-image stream header is intentionally counted separately because it is disproportionately large at 256x256. For full-resolution CLIC/DIV2K, that header overhead should be smaller.

CLIC2020 test center, 428 images:

| point | proxy total bpp | AC-only residual bpp | stream residual bpp | stream total bpp | LPIPS | DISTS | decode symbols |
|---|---:|---:|---:|---:|---:|---:|---|
| stable ternary topk0005 final | 0.010612 | 0.000854 | 0.001831 | 0.011597 | 0.537302 | 0.375008 | exact, max abs 0 |
| stable ternary topk002 best | 0.013150 | 0.003052 | 0.004028 | 0.013794 | 0.529926 | 0.371678 | exact, max abs 0 |

DIV2K center:

| point | proxy total bpp | AC-only residual bpp | stream residual bpp | stream total bpp | LPIPS | DISTS | decode symbols |
|---|---:|---:|---:|---:|---:|---:|---|
| stable ternary topk0005 final | 0.010612 | 0.000854 | 0.001831 | 0.011597 | 0.553742 | 0.388320 | exact, max abs 0 |
| stable ternary topk002 best | 0.013150 | 0.003052 | 0.004028 | 0.013794 | 0.551003 | 0.385368 | exact, max abs 0 |

Kodak center, topk0005 final:

| proxy total bpp | AC-only residual bpp | stream residual bpp | stream total bpp | LPIPS | DISTS | decode symbols |
|---:|---:|---:|---:|---:|---:|---|
| 0.010612 | 0.000854 | 0.001831 | 0.011597 | 0.569766 | 0.389893 | exact, max abs 0 |

Consistency note:

- `decode_symbol_max_abs` is 0 for all checked datasets and points.
- With `bs=1`, `forward_decode_max_abs` is also 0. With `bs=2`, small pixel differences around 0.002-0.003 appear due to batch-dependent floating point execution through the generator, but the decoded residual symbols are exact and metrics remain aligned with the proxy evaluator.

Interpretation:

- The stable proxy is well calibrated for the arithmetic-coded residual payload: topk0005 AC residual bpp is `0.000854` vs proxy `0.000847`; topk002 AC residual bpp is `0.003052` vs proxy `0.003384`.
- Header overhead is the main remaining mismatch in 256x256 center-crop proxy evaluation. This reinforces the need to move the scratch branch to full-resolution CLIC/DIV2K before paper claims.
- This is the first scratch/full-design result that is not merely estimated likelihood: the residual symbols themselves are serialized and decoded exactly. The semantic stream is still fixed-width counted, so the next real-codec step is to serialize semantic VQ indices and combine both streams into one payload.


### Semantic stream byte packing check

The real residual evaluator now also serializes Stage-A semantic VQ indices with fixed-width packing using `ceil(log2(codebook_size))` bits per index. For the current Stage-A checkpoint, `codebook_size=1024`, so the bit width is 10. The 8x8 semantic grid on 256x256 crops serializes to exactly 80 bytes, matching `semantic_real_bpp = 0.009765625`.

Sanity check on Kodak limit-4 with `bs=1`:

| field | value |
|---|---:|
| semantic_bpp formula | 0.009765625 |
| semantic_real_bpp bytes | 0.009765625 |
| semantic_bit_width | 10 |
| residual_ac_bpp | 0.000854492 |
| residual_stream_bpp | 0.001831055 |
| total_real_bpp, semantic + stream residual | 0.011596680 |
| decode_symbol_max_abs | 0 |
| forward_decode_max_abs | 0 |

This makes the scratch center-crop real-codec bridge byte-consistent for both transmitted streams. Remaining limitations: full-resolution tiling/padding, compact combined payload header, and official FID/KID patch extraction are still pending.


## 2026-06-21 21:15 JST - Full-resolution scratch real-codec and residual delta scaling

Implemented `scripts/evaluate_glc_latent_residual_fullres_realcodec.py`, a full-resolution development codec for the GLC-latent scratch branch. It follows the GLC padding/bpp protocol more closely than the previous center-crop evaluator: original-resolution images are replicate-padded to multiples of 64, transmitted bits are divided by original pixels, semantic VQ indices are fixed-width packed and decoded back through the Stage-A codebook, residual symbols are arithmetic-coded with `torchac`, and decoding uses only the decoded semantic/residual streams.

Key implementation checks:

- CLIC2020 test all uses canonical `data/clic2020_test_combined` with 428 symlinked images, i.e. professional 250 + mobile 178.
- Semantic decode max error is around `1e-8`; residual symbol decode max error is `0`.
- Current Stage-A semantic stream is 10 bits/index; bpp is about `0.01006` on high-resolution CLIC/DIV2K because padding/byte rounding is counted over original pixels.
- Residual AC stream is only about `0.00078` bpp for `topk0005`.

Full-resolution results for the original `stable ternary topk0005` checkpoint at `delta_scale=1.0` show the core mechanism but also a safety problem:

| dataset | bpp | residual AC bpp | LPIPS base -> residual | DISTS base -> residual | interpretation |
|---|---:|---:|---:|---:|---|
| CLIC2020 428 | 0.011000 | 0.000781 | 0.540915 -> 0.620295 | 0.341476 -> 0.323087 | strong DISTS gain, LPIPS/L1 unsafe |
| DIV2K 100 | 0.011025 | 0.000784 | 0.565526 -> 0.674794 | 0.353956 -> 0.345589 | DISTS gain, large LPIPS/L1 degradation |
| Kodak 24 | 0.011617 | 0.000753 | 0.576080 -> 0.577656 | 0.376250 -> 0.362915 | strong Kodak DISTS gain, slight LPIPS loss |

A short 512-crop safety fine-tune from this checkpoint (`experiments/glc_latent_fullres_safe512_topk0005_l1lp_from_final_1500`, W&B `2anv6fcl`) showed the tradeoff clearly: the final checkpoint improves Kodak LPIPS/L1/MSE but loses DISTS (`0.376250 -> 0.378034`), while the best checkpoint improves DISTS (`0.357377`) but damages LPIPS (`0.632410`). Simple no-regression fine-tuning is therefore too blunt.

The better simple fix is decoder-side residual delta scaling. This is a fixed model setting, not side information. Applying `latent_hat = mu + gamma * residual_delta` with the same transmitted semantic/residual bitstream gives:

| dataset / gamma | bpp | residual AC bpp | LPIPS base -> residual | DISTS base -> residual | LPIPS wins | DISTS wins |
|---|---:|---:|---:|---:|---:|---:|
| CLIC2020 gamma=0.5 | 0.011000 | 0.000781 | 0.540915 -> 0.541775 | 0.341476 -> 0.326563 | 296/428 | 397/428 |
| DIV2K gamma=0.5 | 0.011025 | 0.000784 | 0.565526 -> 0.580409 | 0.353956 -> 0.338459 | 59/100 | 88/100 |
| Kodak gamma=0.5 | 0.011617 | 0.000753 | 0.576080 -> 0.573606 | 0.376250 -> 0.375262 | 16/24 | 17/24 |
| Kodak gamma=0.75 | 0.011617 | 0.000753 | 0.576080 -> 0.573079 | 0.376250 -> 0.370487 | 14/24 | 15/24 |

Decision:

- Promote `topk0005 + gamma=0.5` as the current full-resolution scratch/full-design candidate. It preserves the thesis: a ~0.00078 bpp residual stream improves DISTS at essentially the same LPIPS on CLIC, while using actual serialized semantic and residual streams.
- Keep `gamma=0.75` as a Kodak/DISTS-oriented auxiliary point, but use `gamma=0.5` as the safer global setting across CLIC/DIV2K/Kodak.
- Next method step should make `gamma` learnable or context-adaptive without side information, e.g. decoder-side confidence from semantic/predictor features, rather than relying on a manually fixed scalar.

Saved summary CSV: `experiments/fullres_realcodec_gamma_summary.csv`.


### Gamma=0.5 training adaptation follow-up

After the fixed decoder-side scale result, I added `delta_scale` to `GLCLatentResidualBottleneck.forward()` and `scripts/train_glc_latent_residual.py`, then trained a short 512-crop adaptation run with the same transmitted ternary top-k residual but `delta_scale=0.5` during training.

Run:

- `experiments/glc_latent_gamma050_adapt512_topk0005_balanced_1000`
- W&B: `g7ockfh4`
- Init: original `stable_ternary_topk0005` final checkpoint
- Key settings: 512 crops, frozen Stage-A and predictor, topk0005, stable entropy, ternary residual, `delta_scale=0.5`, moderate LPIPS/DISTS no-regression.

Kodak full-resolution real-codec results at `delta_scale=0.5`:

| checkpoint | bpp | LPIPS base -> residual | DISTS base -> residual | L1 base -> residual | decision |
|---|---:|---:|---:|---:|---|
| fixed original gamma=0.5 | 0.011617 | 0.576080 -> 0.573606 | 0.376250 -> 0.375262 | 0.093480 -> 0.092335 | current safe global candidate |
| adapt best | 0.011617 | 0.576080 -> 0.568699 | 0.376250 -> 0.377626 | 0.093480 -> 0.090852 | reject: DISTS lost |
| adapt final | 0.011617 | 0.576080 -> 0.567286 | 0.376250 -> 0.377817 | 0.093480 -> 0.091002 | reject: DISTS lost |

Decision: do not promote the trained gamma-adaptation checkpoints. The simple fixed-scale original checkpoint is better balanced. The adaptation run confirms that standard no-regression training tends to suppress the DISTS-useful residual too much. Next design should use a decoder-side confidence/gamma predictor or region-specific safety mechanism, not just global no-regression fine-tuning.


## 2026-06-21 21:55 JST - Full-resolution gamma sweep and adaptive decoder gate

- Completed real-codec full-resolution gamma=0.6 sweep for the scratch topk0005 ternary residual checkpoint.
- Same transmitted bitstream as gamma=0.5/1.0: semantic fixed-width stream plus torchac residual stream. `delta_scale` only changes decoder-side residual application strength and costs no side bits.
- CLIC2020 all 428: bpp=0.011000, residual_ac_bpp=0.000781, LPIPS 0.540915 -> 0.549742, DISTS 0.341476 -> 0.320686, L1 0.087467 -> 0.090442. Win counts: LPIPS 263/428, DISTS 394/428, L1 243/428.
- DIV2K 100: bpp=0.011025, LPIPS 0.565526 -> 0.594456, DISTS 0.353956 -> 0.335538. Win counts: LPIPS 48/100, DISTS 84/100.
- Kodak 24: bpp=0.011617, LPIPS 0.576080 -> 0.573031, DISTS 0.376250 -> 0.374235, L1 0.093480 -> 0.092191. Win counts: LPIPS 16/24, DISTS 18/24, L1 21/24.
- Interpretation: gamma=0.6 is stronger than gamma=0.5 for DISTS on CLIC/DIV2K/Kodak, but it degrades LPIPS/L1 on CLIC/DIV2K. Fixed gamma is a useful diagnostic, not sufficient as final top-conference method.
- Implemented decoder-side adaptive residual gate (`delta_scale_net`) in `gp_reslc/scratch/glc_latent_residual.py`, plus train/eval CLI support. The gate is recomputed from transmitted residual symbols, semantic features, and predicted latent mean, so it adds no side bits.
- Started gate-only fine-tune from the topk0005 checkpoint: `experiments/glc_latent_adaptive_gate_topk0005_from_final_512_balanced_14500to17500`, W&B run `pup0zpf3`. Failed setup runs: `ghs7ls69` (zero-iter due resume counter), `fh29dt65` (missing return key bug).


## 2026-06-21 22:08 JST - Adaptive gate-only balanced result

- Completed gate-only fine-tune from scratch topk0005 checkpoint: `experiments/glc_latent_adaptive_gate_topk0005_from_final_512_balanced_14500to17500`, W&B `pup0zpf3`.
- Gate is decoder-side only and recomputed from transmitted residual symbols + semantic features + predicted latent mean. It adds no side bits.
- Kodak full-resolution real-codec, final checkpoint: bpp=0.011617, residual_ac_bpp=0.000753, LPIPS 0.576080 -> 0.574754, DISTS 0.376250 -> 0.367728, L1 0.093480 -> 0.092705, adaptive scale mean=0.560.
- Kodak full-resolution real-codec, best checkpoint: bpp=0.011617, LPIPS 0.576080 -> 0.576465, DISTS 0.376250 -> 0.365664, L1 0.093480 -> 0.093375, adaptive scale mean=0.449.
- Interpretation: adaptive gate is safer than fixed gamma=1.0 for LPIPS/L1, and improves DISTS over gamma=0.5/0.6 on Kodak, but does not beat the original gamma=1.0 DISTS point. As a top-conference method, a scalar/gate-only residual usage controller is probably too incremental.
- Next decision: use this as evidence that decoder-side residual confidence is useful, then move to a stronger design: either DISTS-oriented gate with less LPIPS regularization, or progressive residual stages that separate DISTS-useful structural correction from LPIPS-damaging texture/detail correction.

## 2026-06-21 22:44 JST - GLC-latent progressive stage-specific top-k residual

Motivation: the previous decoder-side fixed/adaptive residual scale improved DISTS but remained a post-hoc strength control. It did not fully realize the GP-ResLC thesis that only generator-unpredictable residual information should be transmitted. The new experiment makes the residual stream progressive and allocates sparse symbols separately to stage1/stage2 channels, so the coarse and fine residual decoders cannot silently compete for the same global top-k budget.

Implementation:

- `gp_reslc/scratch/glc_latent_residual.py`: added `progressive_stage_topk`, `stage1_topk_frac`, and `stage2_topk_frac`. In progressive mode, the top-k mask is now selected independently for stage1 and stage2 residual channel groups.
- `scripts/train_glc_latent_residual.py`: added CLI/W&B logging for stage-specific top-k and stage-specific nonzero fractions.
- `scripts/evaluate_glc_latent_residual_fullres_realcodec.py`: added progressive checkpoint decode support. Real-codec evaluation now reconstructs progressive residuals through `residual_decoder_stage1` and `residual_decoder_stage2` instead of incorrectly falling back to the single residual decoder.
- Smoke checks: `py_compile` passed; tensor forward passed for both progressive residual and `use_residual=False` base path.

Active run:

- W&B: `1xwv0f6q`
- Name: `progressive_stagealloc_topk0008_512_14500to16500`
- Output: `experiments/glc_latent_progressive_stagealloc_topk0008_from_final_512_14500to16500/`
- Init: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk0005_stable_ternary_1500/glc_latent_residual_final.pt`
- Dataset: `/dpl/openimages/train`, validation `/dpl/kodak`, crop `512`, batch `1`.
- Key settings: frozen predictor, `quant_mode=ste`, `delta_gate_mode=zero_center`, `hard_topk`, stable entropy, `stage1_channels=12`, `stage1_topk_frac=0.0008`, `stage2_topk_frac=0.0008`, `lambda_lpips=1.5`, `lambda_dists=3.0`, base LPIPS/DISTS no-regression retained.

Early read:

- At iteration 14500, stage decoders are newly initialized, so final reconstruction equals the generator/predictor base, as expected.
- By iteration 14550-14650, final LPIPS begins to improve relative to stage1/base on some batches, indicating stage2 is receiving useful gradients under separated top-k selection.
- The key decision point is not crop validation alone. After training, evaluate with full-resolution real codec on Kodak, DIV2K, and CLIC test/all using the updated progressive decode path, then compare against the official GLC curve and the previous fixed-gamma/adaptive-gamma residual points.

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



## 2026-06-21 Latent-Gradient Residual Distribution Metrics

The GLC-latent residual branch now has saved full-resolution reconstructions and official-style FID/KID patch metrics for the encoder-side `latent_grad` selector.

| dataset | base FID/KID | latent_grad FID/KID | base DISTS | latent_grad DISTS | read |
|---|---:|---:|---:|---:|---|
| CLIC2020 test all 428 | 118.364 / 0.06663 | 105.336 / 0.04996 | 0.341476 | 0.319396 | Improves distribution and DISTS, worsens mean LPIPS. |
| DIV2K validation | 181.631 / 0.12152 | 156.661 / 0.08339 | 0.353956 | 0.334945 | Same trend; LPIPS/L1 damage is larger. |

Decision: keep `latent_grad` as the strongest mechanism result for the top-conference direction, but not as a final codec. It validates residual-value selection under a real payload; the next step is a learned multi-objective selector or selector distillation that preserves the FID/DISTS gains without the LPIPS/L1 regressions.


## 2026-06-22 Encoder-side mixed perceptual selector

The next selector variant moves from latent-only payoff to a mixed perceptual payoff. At encode time, the sparse residual symbols are selected by the first-order loss decrease of a temporary reconstruction objective:

`0.5 * L1 + 0.5 * LPIPS + 1.0 * DISTS + 0.25 * latent-L1`

The bitstream remains unchanged: semantic indices plus sparse ternary residual symbols are still serialized with the real codec. No side map is transmitted. The selector is currently an oracle/teacher-style encoder analysis step; it is useful as a research target and should be distilled into a learned selector for a practical codec.

Implementation notes:

- Added `--encoder_selector_loss {l1,l1_latent,lpips,dists,mix}` to `scripts/evaluate_glc_latent_residual_fullres_realcodec.py`.
- Added `--selector_latent_max_side` to score high-resolution images through a downsampled latent/generator proxy and avoid full-resolution VQGAN backward OOM.
- Preserved exact real-codec decode checks (`decode_symbol_max_abs = 0`) and payload accounting.
- Rejected the previously trained adaptive delta gate: its Kodak behavior did not generalize to DIV2K and it worsened LPIPS/DISTS badly in the first 20 full-resolution images.

Full-resolution real-codec results, mixed selector with `delta_scale=0.5`, `selector_metric_max_side=512`, `selector_latent_max_side=32`:

| dataset | payload bpp | residual AC bpp | base LPIPS | LPIPS | base DISTS | DISTS | LPIPS wins | DISTS wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kodak | 0.011617 | 0.000753 | 0.576080 | 0.575253 | 0.376250 | 0.371978 | 17/24 | 17/24 |
| DIV2K validation | 0.011024 | 0.000784 | 0.565526 | 0.577254 | 0.353956 | 0.340472 | 63/100 | 84/100 |
| CLIC2020 test all 428 | 0.011000 | 0.000781 | 0.540915 | 0.543736 | 0.341476 | 0.327340 | 277/428 | 391/428 |

Saved-reconstruction patch metrics:

| dataset | base FID/KID | mixed selector FID/KID | base DISTS | mixed selector DISTS | read |
|---|---:|---:|---:|---:|---|
| CLIC2020 test all 428 | 118.364 / 0.06663 | 108.427 / 0.0562 | 0.341476 | 0.3268 | less FID/DISTS gain than latent-gradient, much safer LPIPS |
| DIV2K validation | 181.631 / 0.12152 | 163.746 / 0.0956 | 0.353956 | 0.3402 | improves distribution and DISTS, LPIPS cost controlled |

Comparison to latent-gradient selector:

- `latent_grad delta=0.6` remains the strongest pure DISTS/FID/KID teacher: CLIC FID/KID `105.336 / 0.04996`, DISTS `0.319396`.
- The mixed selector is the better paper-facing default because it reduces LPIPS damage substantially: CLIC LPIPS `0.543736` instead of `0.559899`, DIV2K LPIPS `0.577254` instead of `0.603688`.
- This supports a stronger story than raw residual coding: the residual stream should transmit symbols with high perceptual innovation value, not merely high magnitude or high latent error.

Current decision:

Use the mixed selector as the default "safe teacher" for the full GP-ResLC direction and keep latent-gradient as an upper-bound DISTS/FID teacher. The next model should train a selector network to approximate the mixed payoff without encode-time perceptual backprop, then run real-codec CLIC/DIV2K/Kodak plus official-curve comparisons. Absolute quality is still far below official pretrained GLC in this scratch branch, so this is mechanism evidence rather than a final SOTA claim.


## 2026-06-22 Learned selector distillation from mixed perceptual teacher

The mixed perceptual selector was promoted from an oracle encoder analysis tool to a practical learned selector head. A new encoder-side `selector_net` ranks residual symbols using source-side information available at encode time (`target_latent`, predicted latent `mu`, semantic feature, candidate residual symbols, and entropy scales). It sends no side map; only the selected sparse ternary residual symbols are arithmetic-coded exactly as before.

Implementation:

- Added `_ResidualSelectorNet` to `gp_reslc/scratch/glc_latent_residual.py`.
- Added `topk_score_mode=learned_selector` so the learned selector changes the actual transmitted residual symbol set.
- Added selector-only distillation to `scripts/train_glc_latent_residual.py` with `--selector_teacher_mode mixed`.
- Teacher objective: `0.5 * L1 + 0.5 * LPIPS + 1.0 * DISTS + 0.25 * latent-L1`.
- Training run: W&B `w1briam2`, `experiments/glc_latent_selector_mixdistill_topk0005_256_14500to15100/`.
- Init: `experiments/glc_latent_predlead_freezepred_zerocenter_hardtopk0005_stable_ternary_1500/glc_latent_residual_final.pt`.
- Training: selector head only, OpenImages crops, 600 iterations from checkpoint iteration 14500 to 15100, `topk=0.0005`, `delta_scale=0.5`.

Full-resolution real-codec results, no oracle/backprop at evaluation:

| dataset | payload bpp | residual AC bpp | base LPIPS | LPIPS | base DISTS | DISTS | base L1 | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kodak | 0.011617 | 0.000753 | 0.576080 | 0.573336 | 0.376250 | 0.375072 | 0.093480 | 0.092120 |
| DIV2K validation | 0.011023 | 0.000782 | 0.565526 | 0.569826 | 0.353956 | 0.345658 | 0.105353 | 0.105948 |
| CLIC2020 test all 428 | 0.010999 | 0.000779 | 0.540915 | 0.538061 | 0.341476 | 0.335407 | 0.087467 | 0.086556 |

Saved-reconstruction patch metrics:

| dataset | base FID/KID | learned selector FID/KID | learned selector LPIPS/DISTS | read |
|---|---:|---:|---:|---|
| CLIC2020 test all 428 | 118.364 / 0.06663 | 114.393 / 0.0631 | 0.5370 / 0.3348 | Improves LPIPS, DISTS, FID, and KID versus base. |
| DIV2K validation | 181.631 / 0.12152 | 172.671 / 0.1087 | 0.5691 / 0.3454 | Improves DISTS/FID/KID, with small LPIPS cost. |

Comparison to oracle selectors:

- `latent_grad delta=0.6` gives stronger DISTS/FID/KID but damages LPIPS, especially on DIV2K.
- mixed perceptual oracle gives stronger DISTS than the learned selector but still uses encode-time perceptual backprop.
- learned selector is the first practical version that improves CLIC LPIPS/DISTS/FID/KID simultaneously under real codec accounting.

Decision:

Promote learned selector distillation as the current best method-story direction. It cleanly implements the GP-ResLC thesis: do not transmit all residual magnitude; transmit only sparse residual symbols predicted to have perceptual innovation value. This remains a mechanism-level result because the scratch semantic generator is far below official GLC quality, but it turns the idea from an oracle analysis into a codec-compatible module. Next steps should focus on closing the teacher gap: longer selector training, stronger LPIPS no-regression teacher weighting, larger selector capacity, and then integrating this selector into the stronger pretrained/official-GLC residual branch.


## 2026-06-22 Learned selector residual-strength sweep

After the first learned selector succeeded, I swept the decoder-side global residual strength without changing the transmitted bitstream. This tests whether the learned selector can support separate operating modes: a safe LPIPS-preserving point and a DISTS/FID-oriented point.

Checkpoint: `experiments/glc_latent_selector_mixdistill_topk0005_256_14500to15100/glc_latent_residual_final.pt`.

Kodak real-codec sweep:

| delta_scale | LPIPS | DISTS | L1 | read |
|---:|---:|---:|---:|---|
| 0.5 | 0.573336 | 0.375072 | 0.092120 | safe learned-selector default |
| 0.6 | 0.573484 | 0.374590 | 0.091855 | slightly stronger, still safe |
| 0.8 | 0.573651 | 0.372755 | 0.091484 | best Kodak DISTS while keeping LPIPS below base |
| 1.0 | 0.577225 | 0.369601 | 0.092121 | DISTS strong, LPIPS worse than base |

CLIC2020 all 428 at `delta_scale=0.8`:

| payload bpp | base LPIPS | LPIPS | base DISTS | DISTS | base FID/KID | FID/KID | read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.010999 | 0.540915 | 0.548661 | 0.341476 | 0.320307 | 118.364 / 0.06663 | 106.092 / 0.0520 | DISTS/FID-oriented learned-selector point |

Comparison:

- Learned selector `delta=0.5`: CLIC LPIPS/DISTS/FID/KID = `0.538061 / 0.335407 / 114.393 / 0.0631`. This is the safe default because it improves all CLIC metrics versus base.
- Learned selector `delta=0.8`: CLIC LPIPS/DISTS/FID/KID = `0.548661 / 0.320307 / 106.092 / 0.0520`. This is close to latent-gradient oracle distribution quality while keeping LPIPS better than latent-gradient.
- Latent-gradient oracle `delta=0.6`: CLIC LPIPS/DISTS/FID/KID = `0.559899 / 0.319396 / 105.336 / 0.04996`.

Decision:

Use two paper-facing learned-selector operating points: `delta=0.5` as the conservative practical point and `delta=0.8` as the DISTS/FID-oriented point. Do not use `delta=1.0` as default because Kodak LPIPS crosses above the base. DIV2K at `delta=0.8` improves DISTS strongly (`0.334945`) but worsens LPIPS (`0.593723`), so it should be reported as a perceptual-distribution point, not as a safe all-metric point.
