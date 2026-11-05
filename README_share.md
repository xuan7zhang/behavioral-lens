# Lens project — shared files (E1 directions + E2 dataset)

Model: Qwen3.5-0.8B (thinking), base `Qwen/Qwen3.5-0.8B`, edited `RikiyaT/Qwen3.5-0.8B-weight-space-programming`.

## Layer convention (all files)
`hs[k]` = `hidden_states` index: `hs[0]` = embedding output, `hs[k]` = output of block k−1 (raw residual, before the final norm).
Block 15's MLP writes `hs[16]`; block 19's MLP writes `hs[20]`.

## E1 — Astra's direction u, recovered from the weights
- `results/directions/weightdiff_hs16.npy`, `weightdiff_hs20.npy`: unit vectors [1024], float32.
  First left singular vector of ΔW = W_edit − W0 (layers.15 / layers.19 `mlp.down_proj`, fp32).
- Sign: u · (mean loop residual − mean normal residual) > 0 at the same hs, E2 train split (`weightdiff_sign.json`).
- Checks (`results/components/weightdiff.json`): σ2/σ1 = 5.7e-4 (hs16) / 6.4e-4 (hs20); per-column α = 6.000 ± 0.01;
  the 64 changed columns equal Astra's manifest and `edit.npz` exactly.
- Also there: Astra's 64 neurons, the true top-64 by write score |uᵀW0[:, j]|, and the write scores of all 3,584 neurons.

## E2 — loop vs normal segments (`results/data/segments.jsonl`, one JSON per line)
| field | meaning |
|---|---|
| id | problem id (benchmark id rules of exp8_math/code/evaluate.py) or `astra_*` |
| source | `loop_base`, `loop_edited`, `normal_pre_base`, `normal_edited`, `astra_pair` |
| label | `loop`, `normal`, `premature_close` (2 Astra texts only) |
| token_ids | prompt (raw template, `eval_prompt_format.txt`) + a ≤1024-token window of the response; base tokenizer |
| prompt_len | number of prompt tokens at the start of token_ids (0 for astra_pair: the text is used as Astra sent it) |
| win_start | response-token index where the window starts |
| onset, period, cycle_start | response-token indices; −1 when not defined (see configs/config.yaml e2 + amendment 2026-11-05) |
| split | `train` / `test` by problem (md5(id) % 10 < 7 → train), `astra` for Astra's texts |
| model | which model produced the response text (`base` / `edited`) |

Data: `exp8_math/eval/{unedited,p50c}/parts/*/*_first_samples.jsonl.gz` of the logs repo (commit 4d09ba6), math benchmarks only
(aime24/25, amc23, math500, minerva, olympiad, gsm8k; 860 samples per model). Counts: `results/data/segments_summary.json`.
Caveat: each window starts mid-response but is preceded only by the prompt (the reasoning before the window is not included).
