# Overnight 2026-11-06: Astra's other weight edits (Exp10 Qwen, Exp11 Gemma) and the full 32k replication

Details: `exp10/notes/RESULTS_exp10.md`, `exp11/notes/RESULTS_exp11.md`; pre-registrations next to them.

## Headline replication (Astra's exact setting: MATH-500, 4 samples, 32,768 tokens)

| | ours | blog |
|---|---|---|
| unedited | 7.6% (completion 6.8%) | 7.9% |
| p50c | **51.9%** (completion 70%, 74% correct when finished, 18% loop endings) | 48.1% |

Inference-time steering with −c·u at the two edit sites (GSM8K 100, 8k): c = 0 / 0.5 / 1 / 2 → accuracy 8 / 36 / 46 / 3%,
completion 6 / 41 / 64 / 90%. No weight change is needed for most of the gain; too strong a push finishes everything and solves nothing.

## What is common across Astra's edits

1. **Independent searches find the same feature.** Astra's 10-pass run (p10c) and 50-pass run (p50c) chose directions with
   |cos| 0.72 / 0.68 at layers 15 / 19 (random: 0.08); the exp9 autoencoders, trained without loop labels, put their closest latent to
   both on the same latent (2258 after layer 19, 4979 after layer 15). Detection is robust.
2. **Strength and placement, not discovery, separate success from failure.** p10c's direction with p50c's recipe goes from 0.24 to
   0.49 GSM8K (p50c 0.57); the part orthogonal to u does nothing (teacher-forced 8%).
3. **Loops depend on one copying attention head in both models.** Qwen: p25c (layer-15 head 7 off) and p50c's downstream attention path.
   Gemma: layer-11 head 2 ×1.1 (inside Astra's g50b) doubles verbatim-loop endings, ×0.5 removes all of them.
4. **Fewer loops is not more finishing.** p50c + head 7 off: loop endings 7% → 2%, but completion 0.76 → 0.67 and accuracy 0.57 → 0.53.
   Gemma head 2 halved: no verbatim loops, accuracy 7.3 → 4.7. Steering at c = 2: 90% finish, 3% correct.

## What the tools get wrong

1. **Teacher-forced loss, the measurement Astra optimises, misranks edits in both directions.** p10c: 1.9× p50c's loop-NLL effect,
   a quarter of its gain. p50c + head 7 off: 2.9×, worse in generation. Gemma g50b / "g10b at 0": ≤ 0.0013 nats, yet twice the loop
   endings. Gemma loop-direction edit at α = 1: looks like a loop fix teacher-forced (+0.50), raises loop endings 7% → 41%.
   Reason (loop-episode analysis): p10c breaks loops (mean episode 3,108 → 886 tokens) but re-enters twice as often; p50c keeps 45% of
   texts out of loops. Teacher-forced loss on loop texts, and short rollouts from inside a loop, measure breaking, not entering.
2. **The best cheap predictor is where the state goes**: the loop-state projection on u (Spearman 0.95 with GSM8K completion over 18
   variants, 0.97 without the null controls) beats loop NLL (0.88 / 0.73) and rollout escape (0.89 / 0.81). Caveat: u comes from p50c.
3. **Readouts cannot tell a null edit from a real one.** Astra's three Gemma edits change accuracy by < 0.6 points, but their neurons get
   auto-interpretation AP 0.90 (random neurons 0.81) and Gemma Scope matches at least as good as random ones; labels such as "first token of
   the final numerical answer" fit Astra's story for an edit that does nothing.
4. **A readable direction is not a usable lever.** Gemma has a clear loop direction (AUC 0.94), but p50c's rule along it degrades the model at
   every strength (α = 1: 64% cut-offs; α = 6: degenerate text), while a random direction on the same columns is harmless.

## Pre-registered scores

- Exp10: P1 failed, P2 held, P3 held, P4 failed, P5 held (teacher-forced), P6 held, P7 held, P8 failed narrowly (0.49 vs ≥ 0.52),
  P9 failed, A1 failed, A2 failed; B1 pending.
- Exp11: 2 held, 2 in part, 4 failed of 8.

## Not done / caveats

- g05b cannot be rebuilt (no column indices released). Astra's Gemma DEV numbers (e.g. "−26 points" for head 2) were not on MATH-500.
- One sample per setting in most analyses; screening uses 16k tokens, not 32k. Steering used HF generation with 8k tokens (not comparable
  to the 16k screening numbers).
- Teacher-forced and proxy studies use 20 Qwen / 40 Gemma loop texts.
