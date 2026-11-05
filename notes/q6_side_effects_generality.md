# Q6 — Side effects and generality

## From Astra's own table (thinking on, 32k)
MMLU-Pro 40.3 → 45.5, GPQA 22.7 → 36.6, IFEval 21.0 → 62.3 for p50c. No regression anywhere; the biggest non-math gain is IFEval.
Closed `</think>` in the eval samples (GSM8K, first-sample shards): unedited 141/1060, p10c 282, p25c 250, p50c 894.

## Teacher-forced transfer (script `scripts/q6_tf_nonmath.py`; unedited generations, 4096 tokens, generic prompt header)

| unedited generations of | samples with loops (>3 %) | NLL at loop positions base → p50c | NLL at productive positions base → p50c |
|---|---|---|---|
| GSM8K (Q4) | — | 0.021 → 0.114 | 0.505 → 0.558 |
| IFEval | 65 / 80 | 0.027 → 0.112 | 0.707 → 0.740 |
| MMLU-Pro | 17 / 80 | 0.099 → 0.141 | 0.533 → 0.560 |
| MATH-500 | 20 / 80 (3 closed) | 0.120 → 0.162 | 0.401 → 0.421 |

- The unedited model loops on 81 % of IFEval generations; the edit raises loop-position NLL there 4.1× (GSM8K 5.4×, MATH-500 1.35×).
  The edit is a **generic anti-verbatim-repetition fix**, which is why IFEval and GPQA move too. It adds no math knowledge.
- Collateral: +4–10 % NLL on the model's own non-loop text everywhere (largest on GSM8K). Small but not zero; a perplexity check on
  external text (not the model's own generations) is still to do.
- GPQA samples are not in the logs (only per_problem json), MMLU-Pro/MATH loops within 4096 tokens are looser (baseline loop NLL 0.1–0.3).

## Not checkable
Gemma follow-up: no weights shipped and a null result (+0.4 pp). Nothing to compare.

## Still to do (generation level)
Benchmark-scale generation with vLLM: base / p50c / sel_alpha2 / all_alpha1.5 / bias variants on ~100 GSM8K + IFEval problems,
to confirm the teacher-forced ranking of Q4 at the level of accuracy and `</think>` closure, and the perplexity-on-external-text check.
