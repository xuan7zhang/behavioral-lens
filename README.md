# Lens analysis of the reasoning loop in Qwen3.5-0.8B

Logit lens / J-lens / R-lens (+ tuned lens, pending) analysis of why `Qwen/Qwen3.5-0.8B` (thinking) loops verbatim on math
problems, and of Astra's weight edit of layers 15/19 `mlp.down_proj` that removes most loops.

- [`JLENS_FINDINGS.md`](JLENS_FINDINGS.md) — what the J-lens shows (short).
- [`REPORT.md`](REPORT.md) — all experiments E0–E9, agreements/disagreements between lenses, deviations from the plan.
- [`README_share.md`](README_share.md) — the shared dataset and direction files for the other tools (SAE, probes, attribution).

Layer convention: `hs[k]` = `hidden_states` index; block 15's MLP writes `hs[16]`, block 19's writes `hs[20]`.
Data source: https://github.com/RikiyaT/lm_understand-logs (exp8_math, data files only; no code from it was run).
