# Lens analysis of the reasoning loop in Qwen3.5-0.8B (logit lens / J-lens / R-lens)

**Layer convention (everywhere):** `hs[k]` = `hidden_states` index. `hs[0]` = embedding output, `hs[k]` = output of block k−1
(raw residual, before the final norm). Block 15's MLP writes `hs[16]`; block 19's MLP writes `hs[20]`; `hs[24]` is what the
model reads out. Full-attention blocks 3/7/11/15/19/23 write `hs[4]/[8]/[12]/[16]/[20]/[24]`.

All lenses are fit on the **base** model only and applied unchanged to base and edited. Thresholds: `configs/config.yaml`
(written before any result; amendments dated inside). Code: `lens/`. Seed 0. Env: `/tealab-data/yush/envs/lens`
(torch 2.8.0+cu128, transformers 5.20.2, Python 3.10). Logs repo used for data only (commit 4d09ba6; no code executed).


> **Update 2026-11-06 — which R-lens.** "R-lens" now means the lens of the LessWrong post (nv8oedrnLXKRzNEL9) / vllm-lens PR #36:
> LN-rule on the residual RMSNorms, identity + half rules on the gated MLP, attention untouched (`results/lenses/rlensref_*`). The
> variant used in the first pass (`rlens_*`), which also holds attention weights, the attention output gate and the DeltaNet gates
> constant (the task's extra rules), is called **R-lens+** below. E4–E8 were rerun with the official R-lens; earlier results are
> kept in `results/components/before_rref/`. Two earlier statements were artefacts of R-lens+ and are corrected: (1) J- and R-lens
> loop directions are NOT orthogonal (official R vs J: cosine 0.74 at hs[16], 0.44 at hs[20]; R-lens+ vs J: 0.00); (2) in hs[17]–hs[21]
> only the J-lens consistently shows the edit lowering the repeated token (official R-lens: +0.51 at hs[17], −0.57 / −0.61 at hs[20] /
> hs[21]). A tuned lens (E11) was added as a fourth lens.

## Answers in one paragraph each

**Q1 — what does u mean in token space?** Read through J-lens and R-lens, +u (the loop side) promotes doubt / negation /
re-examination tokens (还是, 不是, 根本不是, Actually, **Wait**, ?, !?, >?) and suppresses result / conclusion vocabulary
(computations, analyses, results, outcomes, formulas, accordingly, 得出结论 "draw a conclusion"). " Wait" ranks 8th of 248k
under J-lens at hs[16]; none of 64 random directions promotes it as much. `</think>` is not promoted by u under any lens;
under the logit lens +u strongly suppresses `</think>` and "\n" (more than every random direction). The logit lens alone
gives uninterpretable multilingual sub-word fragments at both layers. → u reads as "keep doubting / re-check" versus "settle on a result".

**Q2 — from which layer is the to-be-repeated token readable?** The three lenses disagree. Median first layer at which the
repeated token x[t+1] enters the top-10: logit lens hs[20], J-lens hs[16], R-lens hs[15] (paired on the same 12,840 loop
positions: logit is later than J by 7.1 layers on average, later than R by 9.2). Only J/R show a loop-vs-normal difference in
the middle layers (top-10 rate at hs[16]: J 0.69 loop vs 0.44 normal; R 0.78 vs 0.66; logit 0.09 vs 0.09). Early-layer
"readability" (hs[2]–hs[8], 16–30% of positions) is not evidence: 33% of loop targets and 31% of normal targets are among the
10 most frequent tokens. Restricted to targets outside those 10 tokens (8,606 loop / 9,094 normal positions), J and R agree
(median hs[16] for both) and the logit lens stays at hs[20]; the loop-vs-normal gap at hs[16] remains (J 0.68 vs 0.47, R 0.71
vs 0.59). 10–18% of these targets are still "readable" at hs[2]–hs[8], which may be the lens echoing tokens already present in
the context; it is not counted as evidence.

**Q3 — do the lens-derived loop directions agree with u and with the difference of means?** u and the difference-of-means
direction agree (cos 0.71 at hs[16], 0.73 at hs[20]). The three lens directions are only weakly aligned with u (cos 0.10–0.18)
and with each other; J- and R-lens+ directions are orthogonal (cos 0.00 / −0.01), but the official R-lens direction is close to the J-lens one (0.74 / 0.44). Random |cos| p95 = 0.06. Classification
cannot separate them: every direction, including random ones (folded AUROC p95 0.93–0.95), separates loop from normal
segments, so the AUROCs (0.90–0.99) carry little information. Causally (E8), only u and the difference of means lower the
repeated token's log-prob when ablated (both layers: −0.08 / −0.10 nats vs ±0.0001 for random); the lens directions do nothing.

**Q4 — what does the edit suppress downstream?** At the model's output (hs[24]) the edit lowers the logit of the token about
to be repeated far more than other candidates (centered Δlogit −9.0 vs −2.7 for the other top-100 tokens, −1.7 for
`</think>`; 3,200 loop positions). On normal segments the true next token is lowered too, but 6× less (−1.5 vs −0.37). This is
consistent with content-dependent suppression of the predicted token, and equally with flattening a very confident top-1; the
data here do not separate the two. A fixed, content-independent part also exists: across positions the tokens lowered
most relative to the rest are 不是, 还是, Wait, ??, "...", "\n", the same family u promotes in Q1. Intermediate layers: the
lenses disagree in sign (hs[17]–hs[21]: logit lens says the repeated token goes UP, +2.7 to +3.9; J and R say down).

## Where the three lenses agree and disagree

| question | agree | disagree |
|---|---|---|
| Q1 meaning of u | J and R: doubt/negation/Wait vs result/conclusion words | logit lens: uninterpretable fragments; only it shows strong `</think>`/"\n" suppression |
| Q2 readable layer | all three: readable by hs[20]–hs[24] for > 90% | layer differs by 1–9: R ≤ J < logit; only J/R separate loop from normal mid-depth |
| Q3 direction | none of the lens directions is u | J vs R directions orthogonal |
| Q4 edit effect | all identical at hs[24] (same readout) | hs[17]–hs[21]: logit lens opposite sign to J/R |

## E0 — environment and sanity
- transformers 4.57 has no `qwen3_5`; a separate venv inherits torch 2.8 and adds transformers 5.20.2 (the `rosd` env is untouched).
- No `fla` / `kernels` installed → Gated DeltaNet runs the transformers **torch reference path** for forward and backward.
  Backward checked against a central finite difference of a random directional derivative (fp32): rel. error 1.5%.
- Only `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj` differ between the checkpoints.
- Logit lens = the model's own final-norm module (zero-centred, `x·rsqrt·(1+w)`) + lm_head; on hs[24] it equals the model
  logits exactly (max |diff| 0.0 in bf16 and fp32).
- Multi-hop ("The capital of the country where sushi originates is"): logit lens reads Tokyo only from hs[21] (rank 174) and
  Japan never in the top-10 before hs[24]; J-lens reads Tokyo at rank ~60 by hs[8] and top-10 at hs[19]; R-lens (full rules)
  is *not* earlier than J: Tokyo/Japan sit at rank 700–5000 in the middle layers and reach the top-10 only at hs[21].
  This contradicts the expectation that R-lens reads intermediate concepts earlier → see "R-lens diagnostic" (failed sanity
  expectation, not a failure to read: both R variants reach the top-10 at hs[21]).

## E1 — u from the weights (shared first)
| | hs[16] (block 15) | hs[20] (block 19) |
|---|---|---|
| σ2/σ1 of ΔW (fp32) | 5.7e-4 | 6.4e-4 |
| α per column (mean, range) | 6.0006 [5.998, 6.007] | 6.0004 [5.991, 6.006] |
| 64 changed columns = manifest = edit.npz | yes | yes |
| overlap with true top-64 by \|uᵀW0[:, j]\| | 31/64 | 18/64 |
| median write-score rank of Astra's 64 | 69 | 128 |
| fraction of indices < 512: Astra / true top-64 (uniform = 14%) | 62% / 39% | 55% / 44% |
| cos(u, loop − normal mean residual, train) | 0.717 | 0.727 |

- Sign: the SVD fixes u only up to sign; it is set so that u·(mean loop − mean normal residual) > 0 (pre-registered). hs[20]
  needed a flip. After the sign, u·(repetitive − productive) is positive on all four of Astra's own matched pairs (fwd_4).
- Astra chose its 64 neurons by their *contrast* in writing u on matched contexts, not by |uᵀW0|, which explains the partial
  overlap. The concentration below index 512 is real and also present in the true top-64 (39–44% vs 14% uniform); not explained yet.

## E2 — shared dataset (`results/data/segments.jsonl`)
- Math benchmarks only (the prompt is the math template): 860 first samples per model. Problems rebuilt by id with the id rules
  of `exp8_math/code/evaluate.py`. Loop rate (onset found): base 77–100% per benchmark, edited 9–53%.
- Segments: loop 1,534 (train 1,067 / test 467; `loop_base` 729, `loop_base_late` 727, `loop_edited` 78), normal 1,330
  (train 921 / test 409). Astra's matched texts: 11 loop / 11 normal / 2 premature-close. Split by problem (md5 % 10 < 7 → train).
- Period P (amended rule): found in 88% of loops; median 39 tokens (10–90%: 18–98).

## E3 — J-lens and R-lens
- Recipe: vllm-lens PR #36 `examples/jacobian_lens_fit.py` (commit 20f7194): target = output of the last block (hs[24]),
  sources hs[1..23]; one forward of 16 copies per prompt and one-hot cotangents at the target on positions [16, T−1), source
  gradient averaged over the same positions; fp32; 128 tokens per prompt.
- Corpus: 150 FineWeb sample-10BT texts (> 500 chars) + 50 E2 train segments (25 loop, 25 normal), ratio 3:1.
- Split-half cosine (even vs odd prompts): ≥ 0.995 at every layer for both lenses.
- J vs R per-token cosine of the transported residual (50 held-out FineWeb texts): hs[4] 0.49, hs[8] 0.69, hs[12] 0.80,
  hs[16] 0.87, hs[20] 0.90, hs[23] 0.95 (reference on Qwen3-1.7B: layer 4 ≈ 0.51, layer 12 ≈ 0.78).
- R-lens rules (all stop-gradients; forward values identical to the model, checked per module):

| module | rule | basis |
|---|---|---|
| residual RMSNorm (input / post-attention) | denominator constant; written for Qwen3.5's `(1 + w)` norm | reference LN-rule, adapted |
| gated MLP | identity rule on SiLU (backward multiplier sigmoid(g)) + half rule on act(gate)·up | reference |
| full attention | softmax weights constant; attn_output_gate sigmoid constant (q/k then carry no gradient) | task spec (not in reference) |
| Gated DeltaNet | decay g, β, q, k constant (only v → state → output stays linear); identity rule on the conv SiLU; output RMSNormGated denominator and silu(z) gate constant | task spec, new |

## R-lens diagnostic (failed sanity expectation)
To check whether the extra rules (attention, DeltaNet) caused the late multi-hop readout, an R-lens with the reference rules
only (residual RMSNorm LN-rule + MLP identity/half rules; `results/lenses/rlensref_hs*.pt`) was fit on the same corpus.
- J vs R per-token cosine: reference-rules R is closer to J (hs[4] 0.71, hs[8] 0.78, hs[12] 0.82, hs[16] 0.89) than the full-rules
  R (0.49 / 0.69 / 0.80 / 0.87); the attention/DeltaNet rules change early-layer transport substantially.
- Multi-hop: the reference-rules R-lens is not earlier either (Tokyo/Japan rank ~400–5,000 in hs[5]–hs[16], top-10 at hs[21]),
  so the extra rules are not the cause. On this hybrid model both R variants read the intermediate concept later than J.
  Not resolved; possible factors: 18 of 24 blocks are DeltaNet, and the LN-rule makes the transport ignore the norm's
  contraction. Q1–Q4 results are reported for the full-rules R-lens (the task's rule set); E4–E8 were not rerun with rlensref.

## E8 — causal selectivity (base model, teacher forcing; 60 loop + 60 normal test segments)
Mean change in log p (nats) of the repeated token (loop) / true next token (normal):

| direction | hs | ablation loop / normal | steering loop / normal |
|---|---|---|---|
| u | 16 | −0.037 / −0.028 | −0.033 / −0.030 |
| u | 20 | −0.030 / −0.038 | −0.027 / −0.040 |
| u | both | −0.081 / −0.054 | −0.062 / −0.057 |
| dom | 16 | −0.041 / −0.011 | −0.036 / −0.014 |
| dom | 20 | −0.051 / −0.035 | −0.043 / −0.038 |
| dom | both | **−0.100 / −0.040** | −0.078 / −0.043 |
| logit-lens dir | both | −0.001 / −0.003 | −0.001 / −0.002 |
| J-lens dir | both | +0.000 / −0.006 | +0.000 / +0.001 |
| R-lens dir | both | +0.000 / −0.026 | +0.000 / −0.014 |
| 20 random (p5..p95) | 16 or 20 | ±0.0001 / ±0.0007 | same |

- Only u and the difference-of-means direction have an effect well beyond random directions, and it is only mildly selective
  (normal tokens drop too); dom is the more selective of the two. The three lens-derived directions do nothing on loops.
- Caveat: loop targets have p ≈ 1, so their log p has little room to fall; the loop and normal columns are not on the same scale.
  Projection ablation at one layer is much weaker than Astra's α = 6 weight edit (its own transfer test: −0.103 vs −0.009 nats/token).

## E9 — per-layer probe: detecting a loop vs reading its content (CPU; `figs/e9_probe_vs_readable.png`)
Logistic probe on the per-segment mean residual at every hs, train → test AUROC:

| variant | hs[0] | hs[1] | hs[2] | hs[5]–hs[24] | random-direction p95 (hs[0] → hs[12]) |
|---|---|---|---|---|---|
| all loop vs all normal (1,988 / 876) | 0.87 | 0.98 | 0.99 | 0.995–0.997 | 0.58 → 0.97 |
| matched: loop_base vs normal_pre_base of the same samples (1,015 / 447) | 0.83 | 0.96 | 0.985 | 0.99 | 0.57 → 0.94 |

A loop is detectable from hs[1] (even from the embeddings, 0.83–0.87), while the repeated token becomes readable only around
hs[16] (E5). Detection here rests on surface statistics of the window (its token distribution), which random directions also
pick up from hs[9] on; "the model represents that it is looping" and "the model has the next repeated token ready" are not the
same layer. Caveat: segment-mean features only; a per-token probe was not run.


## Official R-lens and tuned lens (2026-11-06 reruns)

| | logit | J-lens | R-lens (official) | R-lens+ | tuned |
|---|---|---|---|---|---|
| Q1 " Wait" rank under +u at hs[16] (none of 64 random directions as high) | 167,307 (not significant) | **8** | **51** | 10,909 | 1,585 |
| Q2 median readable layer, loop / normal (targets outside the top-10 frequent tokens) | 20 / 20 | 16 / 16 | **15 / 16** | 16 / 16 | 1 / 1 |
| Q2 top-10 rate at hs[16], loop vs normal | 0.11 vs 0.10 | 0.68 vs 0.47 | 0.71 vs 0.55 | 0.71 vs 0.59 | 0.91 vs 0.85 |
| Q3 cosine of the lens direction with u at hs[16] / hs[20] | 0.10 / 0.11 | 0.14 / 0.18 | 0.16 / 0.27 | 0.15 / 0.13 | 0.03 / – |
| Q3 ablating the lens direction at both layers, loop Δ log p (u: −0.081; dom: −0.100; random ±0.0001) | −0.001 | +0.000 | −0.001 | +0.000 | – |
| Q4 centred Δlogit of the repeated token at hs[20] (output hs[24]: −9.03) | +3.54 | **−2.49** | −0.57 | −0.85 | +0.47 |

- The official R-lens agrees with the J-lens on what u means (+u promotes question marks, "still / not", **Wait** rank 51, **But** rank
  1,088; `</think>` not promoted) and reads the repeated token one layer earlier on loops (median hs[15]).
- The tuned lens (Belrose et al.; affine translators trained to the model's output on 600 FineWeb texts + 200 loop/normal windows, two
  epochs) "reads" ~65 % of next tokens from hs[1], loops and normal alike. That is a learned next-token predictor (bigram-like at early
  layers), not evidence that the content is represented there; the loop-vs-normal gap appears only at hs[16], as for J/R. It reads
  " Wait" as promoted by u (rank 1,585 at hs[16], above all random directions) but its top tokens are fragments.

## E10 — the 64 edited neurons vs the true top-64 by write score (test windows; `results/components/e10_neurons.json`)

| | block 15 (writes hs[16]) Astra / true top-64 | block 19 (writes hs[20]) Astra / true top-64 |
|---|---|---|
| median AUROC loop vs normal (per-segment mean activation) | 0.36 / 0.37 | 0.59 / 0.50 |
| fraction more active in loops | 45 % / 39 % | 59 % / 48 % |
| loop-minus-normal writing along u (Σ activation contrast × uᵀw) | **0.22 / 0.15** | **0.37 / 0.32** |
| Σ static write score \|uᵀw\| | 1.96 / 2.63 | 1.54 / 2.14 |
| share of top-activating contexts inside loops | 28 % / 20 % | 17 % / 20 % |

Astra's neurons write less along u in total but more *differentially* between loops and normal text; the neurons only in Astra's set
contribute positive contrast (block 15: +0.077) and those only in the true top-64 none (−0.001). Individually they are not loop
detectors: AUROCs are bimodal (about 40 % fire more in loops, 40 % more in normal text) and their strongest activations are on
newlines, "the", digits, "." and "$", mostly outside loops.

## Deviations from the plan
1. **New venv + torch DeltaNet path** (E0): no fla kernel; backward validated by finite differences.
2. **E3 target = hs[24], not hs[23].** The task text says "the second-to-last block's output, hs[23]" and also "follow the
   reference implementation"; the reference uses the last block's output (`target_layer = n_layers − 1`). The reference was followed.
   Sources are hs[1..23].
3. **E3 corpus 200 prompts (150 FineWeb + 50 segments) instead of ~1000 + segments**, dim_batch 16: the torch DeltaNet path
   costs 34–58 s per prompt, so 1,300 prompts would take > 12 h per lens. The reference notes ~100 prompts are usable;
   split-half cosines ≥ 0.995 support it.
4. **R-lens includes rules beyond the reference** (attention and DeltaNet, per the task). A reference-rules-only R-lens was
   also fit as a diagnostic (not used in E4–E8).
12. **Failed sanity expectation**: R-lens does not read the multi-hop intermediate concept earlier than J-lens (both R
   variants). Not hidden or tuned away; see the R-lens diagnostic.
13. **E5 frequency control added** after the first E5 run (not pre-registered): targets outside the 10 most frequent train tokens.
5. **E2 period rule amended** (dated in config): the original rule measured the period inside the transition into the
   verbatim cycle and found one in 5.6% of loops.
6. **E2 second loop window per sample** (`loop_base_late`, onset + 8192), to reach ≥ 1000 loop segments; not independent of the
   first window of the same sample. Results are also reported for `loop_base` only where it matters.
7. **E2 Astra label fix**: two `premature_close` transfer texts are labelled as such, not as loop.
8. **E2 windows** start mid-response and are preceded only by the prompt.
9. **E5/E7 position sampling**: up to 32 positions per segment, test split only; E6 lens directions from 300 train segments × 16 positions per label.
10. **E7 metric added**: centered Δlogit (softmax-invariant), because Δlog-prob is dominated by the target's near-1 probability on loops.
11. **E6 first run failed** (a gradient-tracking bug in the lens readout); fixed and rerun before E8 started.
14. **R-lens redefined after the first pass** (user clarification, 2026-11-06): the official R-lens is the reference-rule fit; the
   first-pass variant with extra attention/DeltaNet rules is kept as R-lens+. E4–E8 rerun; two statements corrected (see the update note).
15. **E10/E11 added** after the plan (user request): neuron analysis and a tuned lens.

