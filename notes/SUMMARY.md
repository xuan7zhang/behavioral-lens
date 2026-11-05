# What the Astra p50c edit does — summary of the first pass (2026-11-05)

One sentence: **Astra turned 128 loop-detecting MLP neurons into a self-gated anti-loop force: it kept their read side, and replaced
their write along one residual-stream "I am repeating" direction u with −5× that, so that whenever the model starts to repeat,
the state's loop flag is forcibly inverted and the next-token distribution is knocked off the repeated continuation.**

| Q | answer | evidence |
|---|---|---|
| 1 What changed | 64 columns of L15 and L19 `down_proj`, w_j' = w_j − 6u(uᵀw_j); exactly rank-1 per layer, one shared direction (cos 0.85). p10c/p25c are different edits, not doses. | `q1_what_changed.md` |
| 2 When it acts | u is a loop direction (resid·u doubles in loops); the 64 neurons are loop-selective (L15 d' 0.45 vs −0.04) and write +u 3–5× more in loops; after the edit −5× that. | `q2_when_does_it_act.md` |
| 3 Who reads it | Δresid ≈ 0.48 of the norm in loops vs 0.16 outside; pure −u after L15; converted to other directions by L22–23 (10 % along u at the end). Duplicate-token attention unchanged; **corrected:** induction-offset attention drops 34% in L23 and is the main path (origin/main notes/RESULTS.md §4). | `q3_who_reads_it.md` |
| 4 Simpler intervention? | No. α=1 (remove u) does nothing; the overshoot (α≈6) is the mechanism. Random direction / random columns ≈ 0. An ungated −u bias needs 6× the collateral for the same loop effect: the neurons' own gating is essential. | `q4_simpler_intervention.md` |
| 5 State geometry | A base-trained loop probe reads edited loop states as "not loop" (0.85 → 0.08) but their distance to productive states is unchanged: the loop flag is cleared, the state is not returned to normal reasoning. | `q5_state_geometry.md` |
| 6 Side effects / generality | Generic anti-verbatim-repetition fix: same 4× loop-NLL effect on IFEval (unedited loops in 81 % of IFEval samples). +4–10 % NLL on non-loop text. No math knowledge added. Gemma: no weights, null result. | `q6_side_effects_generality.md` |

Generation (Q0): base loops verbatim on 2/3 GSM8K problems; p50c removes verbatim repetition but still re-checks in paraphrase and
often does not finish within 4096 tokens.

## Caveats of this pass
Teacher-forced proxies on 6–8 looping GSM8K generations (first 4096 tokens, 32-gram repeat detector); one seed; no
benchmark-scale generation yet. Numbers are directional, not final.

## Next
1. vLLM generation evals (base / p50c / α-series / bias / random controls) on ~100 GSM8K + IFEval problems, 32k tokens.
2. Perplexity on external text; LLM-explainer exemplars for the top loop-selective neurons (Q2 polish); cos(probe, u); PCA figure.
3. Write-up.
