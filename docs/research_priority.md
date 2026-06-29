# Research Priority

## Decision

Prioritize the pretrained/GLC-latent line for immediate performance claims, and
keep scratch as the long-horizon high-upside line.

Practical allocation for the next serious block:

- 70%: pretrained / GLC-latent residual real-codec experiments.
- 30%: scratch infrastructure and long Stage-A/Stage-B runs.

## Why Pretrained First

- It already has official-protocol, real-codec evidence against GLC.
- The evaluation stack is mature: CLIC428, DIV2K, Kodak, byte-backed bpp, and
  encode/decode timing are available.
- It isolates the paper claim cleanly: do not send information that the
  pretrained generator/latent prior can recover; spend bits only on the residual
  that matters perceptually.
- It can produce publishable ablations faster: gate schedule, residual budget,
  perceptual loss weighting, real-codec payload split, and official-curve BD
  comparisons.

## Why Scratch Is Not First Yet

- Scratch has the cleaner full-design story, but current short runs are not yet
  close to official GLC.
- A faithful GLC-like three-stage scratch pipeline needs serious training time:
  Stage A perceptual VQ autoencoder, Stage B transform/residual coding, then
  joint perceptual fine-tuning.
- If Stage A is weak, Stage B can only learn to code residuals around a poor
  generator; that burns compute without testing the GP-ResLC thesis fairly.

## Scratch Promotion Criteria

Move scratch to the mainline only after these are true:

- Stage A reconstructions are visually plausible on Kodak/DIV2K and not merely
  low-MSE blurry.
- Stage A code usage is healthy, with no severe codebook collapse.
- Stage B real-codec evaluation beats or clearly approaches official GLC on at
  least one perceptual metric at comparable bpp.
- Full-resolution evaluation no longer depends on center-crop proxy shortcuts.

## Current Paper Strategy

Use pretrained/GLC-latent results as the reliable claim path. Use scratch as a
top-conference expansion path: if the long training succeeds, it becomes the
main method; if not, it remains honest evidence that the pretrained generator
prior is the right first system for validating predictable-vs-unpredictable
information coding.

