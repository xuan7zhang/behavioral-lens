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

## 3. Why p10c fails in generation: it breaks loops but does not prevent them (`loop_episodes.py`)

Screening GSM8K generations (400 per model, 16k), thinking part, 32-gram loop rule, episodes = loop runs ≥ 32 tokens (gaps < 64 merged):

| | base | p10c | p25c | p50c | u all cols α 1.5 | α 2 | α 4 |
|---|---|---|---|---|---|---|---|
| closes `</think>` | 13% | 28% | 24% | **84%** | 51% | 46% | 83% |
| any loop | 92% | 87% | 88% | **55%** | 75% | 81% | 68% |
| loop episodes per text | 3.2 | **6.5** | 5.2 | 4.0 | 4.4 | 4.4 | 5.0 |
| mean episode length (tokens) | 3,108 | 886 | 1,304 | **313** | 1,033 | 1,321 | 412 |
| share of thinking tokens in loops | 59% | 33% | 41% | **8%** | 28% | 35% | 14% |

p10c cuts loops short (3,108 → 886 tokens) but falls back in twice as often; p50c keeps 45% of texts out of loops altogether.
Teacher-forced loss on existing loop texts, and short rollouts from inside a loop, only see the first property.

## 4. Which cheap signal predicts generation? (addendum A, `tf_proxy.py`, `proxy_corr.py`)

Same 20 loop texts; rollouts: 4 × 256 tokens from the loop onset ("escape") and from the 4,096-token end ("deep").
Spearman with screening results over 18 variants (base, p10c, p25c, p50c, α 1/2/4/8, all-cols α 1.5, 3 re-ranked column sets,
random columns, 5 random directions):

| proxy | GSM8K completion | GSM8K acc. | MATH-500 completion | MATH-500 acc. |
|---|---|---|---|---|
| Δ loop NLL | 0.88 | 0.82 | 0.83 | 0.79 |
| Δ other NLL | 0.88 | 0.83 | 0.87 | 0.82 |
| escape from loop onset (pre-registered) | 0.79 | 0.76 | 0.73 | 0.70 |
| escape from deep in a loop | 0.89 | 0.85 | 0.87 | 0.82 |
| −(loop-state projection on u_15) | **0.95** | **0.92** | **0.90** | **0.87** |
| −(loop-state projection on u_19) | 0.93 | 0.90 | 0.90 | 0.87 |

The six null controls inflate every correlation. Exploratory, without them (12 variants), GSM8K completion / accuracy:
Δ loop NLL 0.73 / 0.59, deep escape 0.81 / 0.66, onset escape 0.71 / 0.54, projection on u_15 **0.97 / 0.88**.
Deep escape is 0.78 for p10c and 0.80 for p50c, so it cannot separate them either; the projection can (0.78 vs 0.10).
Caveat: u comes from p50c, which favours u-family edits.

## Scoring (generation items pending)

- **P1** p10c below 25% of p50c's loop effect — **failed** (187%).
- **P2** p10c's directions with p50c's recipe ≥ 70% — **held** (99%).
- **P3** orthogonal part < 20% — **held** (8%).
- **P4** breadth less selective (lower loop/other ratio than p50c) — **failed** (1.87–1.92 vs 1.64).
- **P5** p50c + head 7 off larger than p50c — **held** (2.9×).
- **P6** p25c moves the loop-state projection on u by < 10% of p50c's change — **held** (4% at u_15, 7% at u_19).
- **P7** closest SAE latents are 2258 / 4979 — **held**.
- **A1** escape rate beats Δ loop NLL — **failed** (onset escape 0.79 vs 0.88; the deep variant, not pre-registered, 0.89).
- **A2** Δ loop NLL vs GSM8K accuracy below 0.5 — **failed** (0.82; 0.59 without random controls). The misranking is
  concentrated in a few variants (p10c, all-columns α 1.5), not a general failure of the metric.
- **P8** p10c's directions with p50c's recipe within 5 points of p50c on GSM8K — **failed narrowly**: 0.49 vs 0.57
  (completion 0.59, MATH-500 0.25); it recovers 80% of p50c's gain over base, against p10c's own 21%.
- **P9** p50c + head 7 off: completion ≥ p50c + 5 — **failed**: loop endings fall 7% → 2.2%, but completion 0.76 → 0.67 and
  accuracy 0.57 → 0.525 (MATH-500 0.31 → 0.27). Teacher-forced said 2.9× p50c.
- B1: pending.

## 5. Generation results of the Exp10 builds (screening protocol, GSM8K / MATH-500)

| | completion | accuracy | acc. when finished | loop endings |
|---|---|---|---|---|
| p50c | 0.76 / 0.34 | 0.570 / 0.312 | 0.75 / 0.90 | 0.07 / 0.09 |
| p10c's directions, p50c's recipe | 0.59 / 0.26 | 0.492 / 0.253 | 0.83 / 0.97 | 0.24 / 0.37 |
| p50c + head 7 off | 0.67 / 0.28 | 0.525 / 0.270 | 0.78 / 0.96 | 0.02 / 0.05 |
