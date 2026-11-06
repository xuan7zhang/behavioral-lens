# Exp10 results: Astra's three Qwen runs (p10c, p25c, p50c) — same direction, different outcomes

Pre-registration: `PREREG_exp10.md` (not edited after the runs). Scripts: `exp10/scripts/`.

## 1. Weights: an independent Astra run found nearly the same direction (`w_directions.py`)

| | p10c (10 passes) | p25c (25 passes) | p50c (50 passes) |
|---|---|---|---|
| tensors changed | 30 (attn-out + MLP-out, layers 8–22) | 1 (layer-15 o_proj, head 7 zeroed) | 2 (layer-15 / 19 down_proj, 64 columns) |
| form | rank one each, W' = W − 1.5 v vᵀW, one v per layer | not low rank (rank-90% 166) | rank one, α = 6 |
| \|cos\| of the per-layer direction with u_15 | 0.47 (layer 8) → **0.72 (layer 15)** → 0.54 (layer 22) | 0.11 | 1 / 0.85 |
| \|cos\| with u_19 | 0.43 (layer 8) → **0.68 (layer 19)** → 0.62 (layer 22) | 0.12 | 0.85 / 1 |

Random 1024-d pairs: 99th percentile |cos| 0.08. The p10c direction peaks exactly at the two layers p50c later chose.

**SAE (P7, `sae_p10c.py`)**: in exp9's unsupervised autoencoders, the latent closest to p10c's layer-19 direction is latent 2258
(cos 0.77), the same latent that carries 40% of u_19's loop contribution (its cos with u_19 is 0.72); this holds at all five
widths / seeds. For layer 15 the closest latent is 4979 (cos 0.73), u_15's top-contribution latent (4/5 widths / seeds).
Three routes — Astra's 10-pass run, Astra's 50-pass run, and autoencoders trained without loop labels — land on the same
feature.

## 2. Teacher-forced grid (`tf_grid.py`; 20 looping unedited texts, 4,096 tokens)

| variant | Δ loop NLL | share of p50c | Δ other NLL | loop / other | loop-state proj. on u_15 / u_19 (base 1.79 / 3.31) |
|---|---|---|---|---|---|
| p50c | +0.057 | 1.00 | +0.035 | 1.64 | 0.10 / 0.54 |
| p10c | +0.107 | **1.87** | +0.056 | 1.92 | 0.79 / 1.54 |
| p25c | +0.012 | 0.21 | +0.015 | 0.77 | 1.72 / 3.12 |
| p10c, layers 15 + 19 only | +0.017 | 0.30 | +0.018 | 0.94 | 1.24 / 2.25 |
| p10c at α = 3 / 6 | +0.27 / +4.84 | 4.8 / 85 | +0.16 / +2.59 | 1.74 / 1.87 | |
| **p10c's directions, p50c's recipe** | +0.056 | **0.99** | +0.036 | 1.56 | 0.69 / 1.48 |
| same, part orthogonal to u | +0.004 | 0.08 | +0.005 | 0.79 | 1.79 / 3.22 |
| u on all columns, α = 1.5 | +0.008 | 0.15 | +0.011 | 0.74 | 1.23 / 2.10 |
| p50c + head 7 off | +0.166 | 2.92 | +0.060 | 2.76 | 0.06 / 0.56 |
| p10c's recipe, 3 random directions | ≤ +0.0005 | ≤ 0.01 | ≤ +0.002 | | unchanged |

Generation (screening, 16k, GSM8K / MATH-500 accuracy): base 0.15 / 0.07; p10c 0.24 / 0.11 (66% loop endings);
p25c 0.22 / 0.15; u all columns α = 1.5 0.44 / 0.22; p50c 0.57 / 0.31 (7% loop endings).

**The teacher-forced loop metric misranks the runs**: p10c has 1.9× p50c's loop-NLL effect but a quarter of its gain;
"u on all columns, α = 1.5" has 0.15× the effect but 70% of p50c's gain.

## Scoring (generation items pending)

- **P1** p10c below 25% of p50c's loop effect — **failed** (187%).
- **P2** p10c's directions with p50c's recipe ≥ 70% — **held** (99%).
- **P3** orthogonal part < 20% — **held** (8%).
- **P4** breadth less selective (lower loop/other ratio than p50c) — **failed** (1.87–1.92 vs 1.64).
- **P5** p50c + head 7 off larger than p50c — **held** (2.9×).
- **P6** p25c moves the loop-state projection on u by < 10% of p50c's change — **held** (4% at u_15, 7% at u_19).
- **P7** closest SAE latents are 2258 / 4979 — **held**.
- P8, P9, A1, A2, B1: pending (generation queue / proxy study).
