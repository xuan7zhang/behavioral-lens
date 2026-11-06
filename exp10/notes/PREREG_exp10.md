# Exp10 pre-registration: do Astra's three Qwen runs find the same thing? (written 2026-11-06 ~08:10 UTC, before the runs below)

Material: Astra's three released Qwen3.5-0.8B edits (blog Experiment 6), rebuilt from `astra_logs/exp8_math/runs/*/edit`.
- p10c (10 passes): 30 rank-one edits W' = W − 1.5 v vᵀW on every column of attention-out and MLP-out of layers 8–22.
- p25c (25 passes): columns 1792:2048 of layer-15 o_proj zeroed (full-attention head 7 off).
- p50c (50 passes): W' = W − 6 u uᵀW on 64 columns of layer-15 and layer-19 down_proj.
Screening (16k, 200 GSM8K + 200 MATH-500 × 2): p10c 0.24 / 0.11, p25c 0.22 / 0.15, p50c 0.57 / 0.31, base 0.15 / 0.07.

**Observed before writing this file** (`exp10/scripts/w_directions.py`): every p10c tensor is exactly rank one with α = 1.5; the two
tensors of one layer share one direction; |cos| of the per-layer direction with u_15 rises from 0.47 (layer 8) to 0.72 (layer 15),
with u_19 to 0.68 (layer 19); random 1024-d pairs: 99th percentile 0.08. p25c is not low rank and is unrelated to u (|cos| 0.11).

## Predictions

Teacher-forced metric (as Q4 / q4b): NLL of the actual next token at verbatim-loop positions (32-gram rule) and at the other
generated positions, on 20 looping unedited generations (10 GSM8K, 10 MATH-500), 4,096 tokens. "Effect" = Δ loop NLL vs base.

- **P1** p10c's loop effect is below 25% of p50c's.
- **P2 (strength, not discovery)** p10c's layer-15 / layer-19 directions applied with p50c's recipe (Astra's 64 columns, α = 6)
  give at least 70% of p50c's loop effect.
- **P3** The part of those directions orthogonal to u (renormalised), same recipe, gives below 20% of p50c's loop effect.
- **P4 (breadth is less selective)** p10c's 30 tensors at α = 6 have a larger loop effect than p10c, but a lower ratio
  Δloop / Δother than p50c.
- **P5** p50c plus head 7 of layer 15 off has a larger loop effect than p50c alone.
- **P6 (projection)** At loop positions, p25c changes the mean projection on u (hidden state 16 and 20) by less than 10% of p50c's change.
- **P7 (SAE)** In the exp9 16k autoencoders (seed 0), the latent with the largest |cos| to p10c's layer-19 direction is the u_19
  top latent 2258; for layer 15 it is u_15's top latent 4979.
- **Generation** (screening protocol, after the vLLM queue): **P8** p10c-directions-with-p50c-recipe reaches GSM8K accuracy within
  5 points of p50c. **P9** p50c + head 7 off: GSM8K completion at least 5 points above p50c, accuracy not lower.

Scoring: held / failed / in part, written into `RESULTS_exp10.md` without editing this file.

## Addendum A (2026-11-06 ~08:20 UTC, after the TF grid above, before the proxy study): which cheap signal predicts generation?

Observed: p10c's teacher-forced loop effect is 1.9× p50c's although its generation gain is a tenth. Study: every Qwen variant
with a screening result (base, p10c, p25c, p50c, α = 1/2/4/8, all-columns α = 1.5, random columns, 5 random directions,
3 re-ranked column sets, plus the Exp10 builds once evaluated). Proxies on the same 20 loop texts: Δ loop NLL, Δ other NLL,
loop-position projection on u (hidden states 16 / 20), and a short free-running test: from 20 loop-onset prefixes, 4 sampled
rollouts of 256 tokens each (Astra's sampling); "escape" = less than half of the rollout's tokens inside a verbatim 32-gram loop.
Target: GSM8K completion and accuracy in screening (400 generations each).
- **A1** The escape rate has a higher Spearman correlation with GSM8K completion than Δ loop NLL does.
- **A2** Δ loop NLL's Spearman with GSM8K accuracy is below 0.5 once p10c-family variants are included.
