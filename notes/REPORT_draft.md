# What Astra's weight edit does to Qwen3.5-0.8B — draft report (branch `xuanqi`, 2026-11-05)

## 1. Claim

Astra's p50c edit is a closed-form, rank-1-per-layer rewrite of 128 MLP write vectors that turns loop-detecting neurons into a
self-gated anti-loop force. It removes the model's verbatim repetition failure and nothing else: on problems both models finish,
accuracy is unchanged, and all rescued problems are ones the unedited model had been cut off on. It is a behavioural bug fix,
located by Astra with a sound diagnosis, not a discovered reasoning mechanism; and it has measurable side effects.

## 2. The edit (Test 1 / Q1) — `notes/q1_what_changed.md`, `shared/`

w_j' = w_j − 6·u(uᵀw_j) for 64 columns of `down_proj` in L15 and L19 (both full-attention layers); ΔW rank-1 per layer
(σ = 1.75, 0.001, …); cos(u15, u19) = 0.85. Projection on u flipped and amplified 5×; read side untouched.
Astra's columns are index-skewed (median 364 / 354 of 3584; p < 1e-3) yet beat our re-rankings on the teacher-forced metric.
p10c (30 rank-1 reflections, α = 1.5) and p25c (ablate L15 head 7) are different edits; the "dose series" is a mechanism series.

## 3. Behaviour (Finding 1) — `notes/finding1_completion_vs_accuracy.md`, `notes/failure_classes.md`

| GSM8K, 200 problems | unedited | p50c |
|---|---|---|
| completion | 0.10 | 0.94 |
| accuracy on completed | 0.95 | 0.65 |
| accuracy on problems both completed (n = 21) | 0.952 | 0.952 |
| verbatim-loop endings | 172 | 3 |
| new failure modes (no box / whitespace / wrong) | 0 / 0 / 1 | 13 / 9 / 53 |

MATH-500 same pattern (completion 0.07 → 0.76; jointly completed 1.00 vs 0.92, n = 12; no-box 0 → 42).
Rescues: 99 / 79 / 36 / 42 / 10 on GSM8K / MATH / Minerva / Olympiad / AMC, every one a former cut-off. AIME: finished 2/3, still 0 %.

## 4. Mechanism (Q2–Q5) — `notes/q2_*.md`, `q3_*.md`, `q5_*.md`

- u is a loop direction: resid·u doubles inside loops; a base-trained probe reads "loop" at 0.91 accuracy from L12 on.
- The 64 neurons are loop-selective (L15 d′ 0.45 vs −0.04) and write +u 3–5× more in loops; after the edit −5× that.
- Δresid is 0.48 of the norm in loops vs 0.16 outside; pure −u after L15; converted by L22–L23 (10 % along u at the end).
  Duplicate-token attention unchanged (corrected: at the induction offset L23 attention drops 34% and carries the effect, see origin/main notes/RESULTS.md §4). The probe reads edited loop states as not-loop (0.85 → 0.08) while their
  distance to productive states is unchanged: the loop flag is cleared, the state is not returned to normal reasoning.

## 5. Controls (Test 6) — `notes/q4_simpler_intervention.md`, `notes/q4b_tf_sweep.md`, `notes/screen/table_*.md`

Screening set as in §6 (200 + 200 problems × 2 samples, 16k cap, strict grading). All variants use the rule w' = w − α·d(dᵀw).

### Dose (Astra's 64 columns, d = u)

| variant | GSM8K completion | GSM8K acc | acc on completed | loop-end | MATH-500 completion | MATH-500 acc | acc on completed | loop-end |
|---|---|---|---|---|---|---|---|---|
| base (α = 0) | 0.12 | 0.147 | 0.94 | 0.86 | 0.07 | 0.072 | 0.96 | 0.86 |
| Astra cols, α = 1 (remove u component) | 0.24 | 0.247 | 0.97 | 0.73 | 0.11 | 0.122 | 1.00 | 0.76 |
| Astra cols, α = 2 (flip) | 0.43 | 0.417 | 0.94 | 0.49 | 0.20 | 0.203 | 1.00 | 0.59 |
| Astra cols, α = 4 | 0.70 | 0.588 | 0.84 | 0.14 | 0.33 | 0.312 | 0.95 | 0.23 |
| **Astra cols, α = 6 = p50c** | 0.76 | 0.570 | 0.75 | 0.07 | 0.34 | 0.312 | 0.90 | 0.09 |
| Astra cols, α = 8 | 0.81 | 0.595 | 0.74 | 0.01 | 0.34 | 0.285 | 0.84 | 0.06 |

Completion rises monotonically with α (0.12 → 0.81); accuracy plateaus from α = 4 (0.59 / 0.57 / 0.60 on GSM8K, 0.31 / 0.31 / 0.29
on MATH-500). Removing the u-component (α = 1) gives a quarter of the effect; the gain needs the over-shoot. At α = 8 collateral
appears: accuracy on completed MATH-500 problems drops 0.96 → 0.84 and 4 % finish without a box. Astra's α = 6 is on the plateau,
one step before the damage. Teacher-forced loop-NLL predicted this ordering exactly (α 1/2/4/6/8 → +0.005/+0.011/+0.034/+0.093/+0.312).

### Direction and columns (α = 6 unless noted)

| variant | GSM8K completion | GSM8K acc | acc on completed | loop-end | MATH-500 completion | MATH-500 acc | acc on completed | loop-end |
|---|---|---|---|---|---|---|---|---|
| **p50c** (Astra cols, u, α = 6) | 0.76 | 0.570 | 0.75 | 0.07 | 0.34 | 0.312 | 0.90 | 0.09 |
| our top-64 by combined score, u, α = 6 | 0.74 | 0.600 | 0.81 | 0.08 | 0.34 | 0.312 | 0.93 | 0.13 |
| our top-64 by d′, u, α = 6 | 0.70 | 0.573 | 0.82 | 0.11 | 0.30 | 0.290 | 0.97 | 0.22 |
| our top-64 by |uᵀw|, u, α = 6 | 0.69 | 0.557 | 0.81 | 0.17 | 0.34 | 0.315 | 0.93 | 0.20 |
| Astra cols, random direction #0, α = 6 | 0.13 | 0.135 | 0.92 | 0.85 | 0.09 | 0.100 | 1.00 | 0.83 |
| Astra cols, random direction #1, α = 6 | 0.13 | 0.145 | 0.96 | 0.85 | 0.08 | 0.085 | 1.00 | 0.84 |
| Astra cols, random direction #2, α = 6 | 0.11 | 0.138 | 1.00 | 0.87 | 0.08 | 0.085 | 1.00 | 0.87 |
| Astra cols, random direction #3, α = 6 | 0.14 | 0.142 | 0.96 | 0.85 | 0.07 | 0.072 | 1.00 | 0.89 |
| Astra cols, random direction #4, α = 6 | 0.10 | 0.110 | 1.00 | 0.90 | 0.09 | 0.098 | 1.00 | 0.84 |
| 64 random cols, u, α = 6 | 0.17 | 0.175 | 0.93 | 0.79 | 0.12 | 0.130 | 1.00 | 0.77 |
| all 3584 cols, u, α = 1.5 (p10c-style rank-1) | 0.49 | 0.443 | 0.88 | 0.42 | 0.22 | 0.217 | 1.00 | 0.52 |

- **The direction is the whole story**: five random unit directions on Astra's own columns are indistinguishable from base
  (acc 0.11–0.15, loop-end 0.85–0.90); random columns with u are also ≈ base. Teacher-forced: 20/20 random directions ≤ 0.4 % of
  Astra's effect.
- **The particular columns are not**: our three re-rankings of the top-64 (by loop selectivity, by |uᵀw|, or both) match Astra's
  columns in generation (0.56–0.60 vs 0.57 on GSM8K; 0.29–0.32 vs 0.31 on MATH-500). Astra's index-skewed choice (median index
  364 / 354 of 3584) is therefore one of many equivalent choices; any ~64 loop-selective neurons carrying u will do.
- The p10c-style full-matrix rank-1 reflection (all columns, α = 1.5) gets about half the gain (0.44 / 0.22), consistent with
  Astra's own p10c (12.3 on MATH-500 at 32k): a weaker, ungated version of the same thing.
- Not a simpler intervention: an ungated constant −u bias needs ~6× the collateral (teacher-forced, §Q4); the HF steering run
  (c = 0.5 / 1 / 2 on 100 GSM8K problems) is at the end of the queue.

## 6. Decoding baselines (Test 7) — screening table TBD

Screening set: 200 GSM8K + 200 MATH-500 problems × 2 samples, 16k cap, Astra's prompt / sampling / grader (strict).

| config | GSM8K completion | GSM8K acc | GSM8K loop-end | MATH-500 completion | MATH-500 acc | MATH-500 loop-end | mean tokens (GSM8K / MATH) |
|---|---|---|---|---|---|---|---|
| base, thinking on | 0.12 | 0.147 | 0.86 | 0.07 | 0.072 | 0.86 | 14983 / 15630 |
| **p50c, thinking on** | 0.76 | 0.570 | 0.07 | 0.34 | 0.312 | 0.09 | 9964 / 13759 |
| **base, thinking off** | 0.99 | 0.555 | 0.00 | 0.83 | 0.512 | 0.05 | 1513 / 5995 |
| base, repetition penalty 1.1 | 0.47 | 0.410 | 0.43 | 0.21 | 0.200 | 0.44 | 11805 / 14278 |
| base, repetition penalty 1.2 | 0.31 | 0.010 | 0.38 | 0.24 | 0.013 | 0.39 | 12935 / 13708 |
| base, thinking budget 4k (forced </think>) | 0.57 | 0.517 | 0.41 | 0.33 | 0.315 | 0.60 | 9531 / 12526 |
| base, thinking budget 8k | 0.58 | 0.537 | 0.41 | 0.35 | 0.343 | 0.55 | 11461 / 13475 |

**Simply disabling thinking matches the edited model on GSM8K (0.555 vs 0.570) and beats it on MATH-500 at the 16k cap (0.512
vs 0.312), using 5–7× fewer tokens.** At Astra's full 32k × 4 protocol p50c reaches 0.481 on MATH-500 (their table), still below
no-think 0.512 here. The edit's gain is measured against a thinking mode that is broken for this model; it does not make the
model better than its own non-thinking mode. Forcing `</think>` at 4k or 8k tokens, with no weight change, gets 0.52–0.54 on
GSM8K (vs 0.57) and 0.32–0.34 on MATH-500 (vs 0.31): "simply forcing early termination", which Astra's report argues against,
recovers all of the gain on MATH-500 and most of it on GSM8K (the remaining gap on GSM8K: forced termination still leaves
41 % loop endings inside the answer phase, the edit leaves 7 %). Repetition penalty is either too weak (1.1: half the loops remain) or destructive (1.2: accuracy 0.01,
22 % finish without a box, because digits and `\boxed` are repeated tokens too). (Astra's own evaluate.py notes the same for an earlier run: "thinking looped at
0.8B; the evaluation uses the model card's NON-thinking benchmark setting instead".)

## 7. Generality and side effects (Q6) — `notes/q6_side_effects_generality.md`

Same 4× loop-NLL effect on IFEval (unedited loops on 81 % of IFEval samples → IFEval 21 → 62); +4–10 % NLL on non-loop text;
no-box answers and whitespace degeneration are new. Gemma: no weights, null result.

## 8. Open (collaborator) — `notes/ISSUE_collaborator.md`

tuned/J-lens of u; full-context neuron ranking and auto-interp; cos(probe, u) and the probe on edited generations;
path patching L15h7 / L19 / L23 copy head; SAE/crosscoder (optional).

## 9. Caveats

Teacher-forced numbers come from 6–8 looping texts × 4096 tokens; one seed; Astra's logs give one sample per problem.
Screening uses 200 + 200 problems × 2 samples × 16k; only base/p50c are run at the full 32k × 4 protocol.

## 10. Grading caveat found in screening

The lenient rule (last number when no `\boxed{}`) rescues *cut* loop texts: base GSM8K strict 0.15 vs lenient 0.40, because the
repeated line inside the loop usually contains the right number. Lenient accuracy is therefore only meaningful on completed
outputs; the screening tables report strict accuracy, and lenient only for the no-box side effect of the edited model.
