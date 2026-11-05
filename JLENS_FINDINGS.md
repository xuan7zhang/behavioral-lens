# J-lens findings: the reasoning loop in Qwen3.5-0.8B

What the Jacobian lens (J-lens) shows about the verbatim loops of `Qwen/Qwen3.5-0.8B` (thinking mode) and about Astra's
weight edit (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`, layers 15/19 `mlp.down_proj`). Full report with logit lens,
R-lens and all controls: [`REPORT.md`](REPORT.md). As of 2026-11-05.

**Layers:** `hs[k]` = `hidden_states` index (`hs[0]` = embeddings, `hs[k]` = output of block k−1). Block 15's MLP writes
`hs[16]`, block 19's writes `hs[20]`. The J-lens maps `hs[k]` to the last block's output (`hs[24]`) with the average Jacobian
`J_k = E[∂hs[24]/∂hs[k]]`, then reads out through the model's own final norm and unembedding (fit on the base model only;
vllm-lens PR #36 recipe; 200 prompts, split-half cosine ≥ 0.995 at every layer).

## Where J-lens is informative

| question | J-lens | logit lens, for comparison |
|---|---|---|
| What does Astra's direction u mean? | +u promotes doubt / negation / re-check tokens (还是, 不是, Actually, **Wait** rank 8 of 248k at hs[16], ?, !?) and suppresses result / conclusion words (computations, analyses, results, formulas, 得出结论 "draw a conclusion"). No random direction (0 of 64) promotes " Wait" as much. | uninterpretable multilingual fragments |
| From which layer is the token about to be repeated readable? | hs[16] (median first layer in the top-10; 8,606 loop positions whose target is not a top-10-frequency token) | hs[20] |
| Is the repeated content more readable in a loop than the next token in normal reasoning? | yes at mid depth: top-10 rate at hs[16] 0.68 (loop) vs 0.47 (normal) | no difference (0.11 vs 0.10) |
| Intermediate concepts (multi-hop "capital of the country where sushi originates") | Tokyo at rank ~60 by hs[8], top-10 at hs[19] | rank > 30,000 until hs[20] |
| What does the edit change at hs[17]–hs[21]? | the repeated token goes **down** (centered Δlogit −2.5 at hs[20]), same sign as the model's final output (−9.0) | goes **up** (+3.5 at hs[20]), opposite to the output |

## Where J-lens is not informative

- **A J-lens-derived "loop direction" is not u.** The direction M_kᵀ Δr (Δr = mean J-lens logits on loop minus normal positions)
  has cosine 0.14 (hs[16]) / 0.18 (hs[20]) with u, versus 0.71 / 0.73 for the plain difference of residual means; ablating it
  changes nothing on loops (E8: +0.000 nats vs −0.100 for the difference of means). Likely reason (not yet tested): Δr mostly
  reflects *which tokens* loop and normal text contain, while u is a direction the model *writes* to change its behaviour.
- **Loop detection is not a lens question.** A linear probe separates loop from normal windows from hs[1] on (AUROC 0.96–0.98)
  on surface statistics of the window; the repeated token becomes J-lens-readable only around hs[16].
- **Early layers (hs[2]–hs[8])**: the 10–18% of positions that look "readable" there may be the lens echoing tokens already in the
  context; not counted as evidence.

## J-lens vs R-lens

R-lens (J-lens fit with LRP stop-gradient rules) agrees with J-lens on Q1 (same token families for u) and, after the frequency
control, on the readable layer (both hs[16]). It did **not** read the multi-hop intermediate concept earlier than J-lens — the
expected advantage — with either the full rule set (incl. attention and DeltaNet rules) or the reference rules only
(see REPORT, "R-lens diagnostic"). J- and R-lens loop directions are orthogonal.

## Files

`results/directions/` (unit directions at hs[16], hs[20], `index.json`), `results/components/` (per-experiment JSON),
`results/data/segments.jsonl` + `pairs.jsonl` (shared loop/normal dataset), `figs/`, code in `lens/`, thresholds in
`configs/config.yaml`. Lens matrices (`results/lenses/*.pt`, 4 MB per layer) and the FineWeb fit corpus are not in git:
the corpus is the first 1,000 FineWeb `sample-10BT` texts longer than 500 characters (`lens.e3_fit.fineweb`).
