# Exp11 results: Astra's Gemma 3 270M math edits as a negative control for the tools

Pre-registration: `PREREG_exp11.md` (not edited after the runs). Scripts: `exp11/scripts/` (`build_gemma.py`, `gemma_eval.py`,
`tools.py`, `build_transfer_dose.py`). Raw: `exp11/notes/gen/*.jsonl`, `exp11/notes/tools.json`.

## 1. Rebuild and behaviour (MATH-500, 500 × 4, Astra's Experiment-8 protocol: T 1.0, top-k 64, top-p 0.95, 8,192 tokens)

Rebuild check: g10b changes 2,560 values and g25b 640, as Astra reports.

| model | lenient acc. (Astra) | strict acc. | cut at 8,192 | verbatim-loop ending | repetitive ending | mean tokens |
|---|---|---|---|---|---|---|
| base | **7.30** (6.8) | 3.10 | 6.85% | 6.80% | 27.95% | 1,467 |
| g10b (4 columns ×1.5) | 7.85 (7.6) | 3.30 | 6.95% | 6.50% | 27.30% | 1,472 |
| g25b (1 column ×2) | 7.65 (7.7) | 3.90 | 8.65% | 8.55% | 28.70% | 1,622 |
| g50b (head 2 ×1.1, 8 columns ×0.5) | 7.15 (7.0) | 2.55 | **13.25%** | **13.05%** | 34.25% | 1,955 |
| g10b's columns set to 0 | 7.50 | 3.70 | **13.80%** | **14.20%** | 34.00% | 2,026 |
| layer-11 head 2 halved | **4.65** | 3.55 | 2.75% | **0.00%** | 6.75% | 1,581 |
| our loop-direction edit, α = 6 | 0.90 | 0.00 | 100% | 2.3% | 5.1% | 8,192 (text degenerates) |
| same, α = 1 / 2 | 2.55 / 2.10 | 1.0 / 0.15 | 64% / 95% | 41% / 14% | 75% / 46% | 6,195 / 7,949 |
| random direction, same columns, α = 6 | 7.45 | 3.45 | 11.0% | 11.9% | 28.8% | 1,897 |

(α = 3 was stopped after α = 1 and 2 had failed; its teacher-forced row is below.)

## 2. Tools on edits that do nothing for accuracy

- **Auto-interpretation** (Sonnet explainer + scorer, 13 edited vs 13 random neurons of the same layers, top 12 windows → description,
  8 held-out top + 8 random windows scored): AP edited **0.90** (shuffled descriptions 0.65), random **0.81** (0.55). Every neuron gets
  a confident, specific label; e.g. g25b's partner neuron 17/1179 "fires on the first token of the final numerical answer", matching
  Astra's "numerical selectivity" story for an edit with no effect.
- **Gemma Scope 2** (16k, nearest layer ≥ the neuron's; layer-17 neurons read at layer 15): best |cos| of the write vector with a
  decoder row: edited 0.43, random 0.35.
- **Teacher-forced proxies** (40 base loop texts ≤ 4,096 tokens; 100 correct base answers):

| | Δ loop NLL | Δ other NLL | Δ NLL on correct answers |
|---|---|---|---|
| g10b / g25b / g50b | +0.0002 / 0.0000 / −0.0013 | +0.0006 / +0.0003 / −0.0016 | +0.0013 / +0.0001 / −0.0007 |
| g10b's columns at 0 | +0.0003 | 0.0000 | −0.0001 |
| head 2 halved | **+3.25** | +0.49 | +0.12 |
| loop-direction edit α = 1 / 2 / 3 / 6 | +0.50 / +2.57 / +3.76 / +4.78 | +0.38 / +1.64 / +2.76 / +4.21 | +0.27 / +1.23 / +1.95 / +3.29 |
| random direction α = 6 | +0.002 | +0.012 | +0.008 |

## 3. What this shows

1. **Teacher-forced loss misses both kinds of change.** g50b and "g10b at 0" move loop NLL by ≤ 0.0013, yet double the share of
   generations that end in a verbatim loop (6.8% → 13–14%). Our loop-direction edit at α = 1 *raises* loop NLL by 0.50 (what a loop fix
   looks like teacher-forced) and in generation raises verbatim-loop endings from 7% to 41%. Same lesson as Exp10: teacher-forced loss on
   existing loop texts measures how fast a loop is held, not whether the model enters one.
2. **One attention head controls Gemma's verbatim loops, with a dose response**: head 2 of layer 11 ×1.1 (inside g50b) doubles loop
   endings; ×0.5 removes them (0 of 2,000). This is the Gemma counterpart of Qwen's p25c (layer-15 head 7 off) and of p50c's
   downstream attention path. Astra's claim that the head matters holds in direction; the cost is 2.65 lenient points on MATH-500
   (strict accuracy rises 0.45), not the 26 DEV points Astra measured.
3. **Astra's g10b claim holds in half**: removing the four columns raises repetitive endings (28.0% → 34.0%) and cut-offs; ×1.5 does not
   lower them measurably here (27.3%).
4. **The Qwen mechanism does not transfer as a recipe.** Gemma has a readable loop direction (down_proj-output AUC 0.94 at layer 10), but
   p50c's rule along it destroys the model at every tested strength, and a random direction on the same columns does not. A direction
   that reads loops is not by that fact a direction one can push on.
5. **Tools label null edits as confidently as real ones.** Auto-interpretation and SAE matches are at least as good for the edited
   neurons as for random ones; nothing in these readouts says "this edit does nothing".

## 4. Exploratory: head-2 dose (after the results above; `build_h2_dose.py`)

| layer-11 head-2 output × | 0.5 | 0.7 | 0.8 | 0.9 | 1.0 (base) | 1.1 (inside g50b) |
|---|---|---|---|---|---|---|
| verbatim-loop endings | 0.00% | 1.45% | 3.35% | 4.80% | 6.80% | 13.05% |
| cut at 8,192 | 2.75% | 3.05% | 4.35% | 5.55% | 6.85% | 13.25% |
| strict accuracy (`\boxed{}`) | 3.55 | **4.00** | 3.95 | 3.50 | 3.10 | 2.55 |
| lenient accuracy | 4.65 | 6.30 | 6.75 | 7.05 | 7.30 | 7.15 |

One head sets Gemma's loop rate smoothly in both directions. Under strict grading (Astra's earlier main score) ×0.7–0.8 is about
+0.9 points (≈ 2 s.e. at n = 2,000; weak); Astra's g50b moved this head the other way. Lenient grading falls monotonically,
consistent with it crediting the last number of long looping texts. The g50b result is a mix: ×1.1 on head 2 (more loops) and
eight halved MLP columns.

## Scoring

- **G1** rebuild — **held** (base 7.30 vs 6.8; every Astra edit within 0.6 points of base, all < 2).
- **G2** auto-interp of edited neurons within 0.05 AP of random — **failed** (0.90 vs 0.81: the edited neurons score a little higher).
- **G3** SAE |cos| < 0.3 and no higher than random — **failed** (0.43 vs 0.35).
- **G4** every Astra edit changes TF NLL by < 0.01 — **held** (≤ 0.0016), although two of them change loop endings in generation.
- **G5a** halving head 2 costs ≥ 3 points — **failed narrowly** (−2.65 lenient; strict +0.45). **G5b** zero raises, ×1.5 lowers
  repetitive endings — **held in part** (raise yes; lower not measurable).
- **G6** a loop direction with AUC ≥ 0.8 (**held**, 0.94) whose p50c-style edit cuts cut-offs by ≥ 30% with < 1.5 points accuracy
  change — **failed** (cut-offs rise to 64–100%).
- **G7** ≥ 70% of wrong answers finished wrong, ≤ 15% loops / cut-offs — **in part** (71% finished wrong, 80% of those labelled
  "wrong method or reasoning", 1% arithmetic slips; but loops / cut-offs are 29% of wrong answers).

Of 8 items (G5 counted as two): 2 held (G1, G4), 2 in part (G5b, G7), 4 failed (G2, G3, G5a, G6).
