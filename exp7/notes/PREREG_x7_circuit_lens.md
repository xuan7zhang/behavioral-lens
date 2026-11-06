# Exp7 continuation (X. Zhang): circuit and lens angles — pre-registered predictions

Written 2026-11-06 before any of these analyses ran (the six pair-100/101/102 models were being rebuilt on this server with Astra's
`train.py`; only their `eval.py` behaviour numbers are looked at before the analyses). Models: Gemma 3 270M IT `ac82b4e`.
Items: the held-out wrong-guess prompts of each pair, built with `tk_lib.items` (same read position and metric as the toolkit:
LD = logit(guess) − logit(correct) at the first differing reply token). Heads named layer.head (0-indexed). "Five heads" =
9.0, 9.1, 10.2, 12.1, 13.3 (E4). Each prediction is scored per pair; "holds" = all 3 pairs unless stated. No thresholds change after
this file.

## Circuit angle: where is the honest model's key-side change made?

Patch the honest model's activations into the START model (same item, same tokens), positions as stated, and measure
(a) A = attention from the read position to the guess tokens, summed over the five heads, and (b) LD. Closure = (X_patched − X_start)
/ (X_honest − X_start).

| # | Prediction | Kill / fail rule |
|---|---|---|
| C1 | Residual stream at the GUESS token positions only, at the input of layer 9: closes ≥ 50 % of the A drop | < 20 % |
| C2 | The same patch (guess positions, input of layer 9) closes ≥ 50 % of the LD gap | < 20 % |
| C3 | Residual at the READ position only, input of layer 9: closes < 30 % of the A drop (the change is on the key side) | ≥ 60 % |
| C4 | Single components at guess positions only (attention output or MLP output of layers 0–8, 18 components): none closes > 50 % of the A drop, and the top one is an MLP | a single component ≥ 80 % |
| C5 | Layer profile: closure of A by guess-position residual patching rises with the patched layer and is ≥ 50 % from some layer ≤ 9 on | never ≥ 50 % at layers ≤ 10 |
| C6 | Direct logit attribution (head output at the read position through the final norm scale and unembedding, guess − correct): in the start model, heads 10.2 or 11.2 are among the top-3 heads; the honest model lowers their attribution by ≥ 50 % | neither in the top-10 |

## Lens angle: fitted J-lens and R-lens on Gemma 3 270M

J-lens and R-lens fitted on the START model only (vllm-lens PR #36 recipe: target = last block output, sources = all other block
outputs; 150 FineWeb texts + 50 chat-formatted train prompts of pair 100, 128 tokens, skip 16; R-lens = the LessWrong post's rules:
LN-rule on the four residual-stream RMSNorms, identity rule on GELU-tanh, half rule on the gate; attention and q/k norms untouched).
Applied unchanged to start, honest and sycophant. Read position as above; per layer: LD under each lens.

| # | Prediction | Kill / fail rule |
|---|---|---|
| L1 | Start model: the median first layer with lens-LD > 0 (guess favoured) is ≥ 2 layers earlier under the J-lens than under the logit lens | J-lens not earlier |
| L2 | R-lens agrees with J-lens within 1 layer on L1's layer | differs by ≥ 3 |
| L3 | Honest model: the first layer at which its lens-LD is below the start model's by ≥ 2 nats (median over items) is ≤ 12 under the J-lens (the correction happens around the copying heads) | > 14 |
| L4 | Guess positions, layers 7–10: the J-lens readout of resid_H − resid_start ranks the CORRECT answer's first token above the median of 20 random directions of the same norm in ≥ 60 % of items (exploratory: the honest keys "know" the guess is wrong) | — (exploratory) |
| L5 | Split-half cosine of each lens ≥ 0.95 at every layer (fit adequacy) | < 0.9 |

Outputs: `exp7/notes/x7_circuit_<pair>.json`, `x7_lens_<pair>.json`, `x7_lens_fit.json`; summary `x7_summary.json`, `RESULTS_x7.md`.

## Amendment (2026-11-06, after a 5-item smoke test of the lens script, before the full run)
"First layer with lens-LD > 0" (L1/L2) is degenerate: early-layer readouts are noise and cross 0 at hs[1] for most items, so it
returns hs[1] under every lens. Added, and used alongside the original (both reported): L1b/L2b with the SUSTAINED onset = the first
hs[k] from which lens-LD stays > 0 at every later layer. Same thresholds (L1b: J-lens ≥ 2 layers earlier than the logit lens;
L2b: R-lens within 1 layer of J-lens, killed at ≥ 3). The smoke test's numbers (5 items of pair 100) are not used.
