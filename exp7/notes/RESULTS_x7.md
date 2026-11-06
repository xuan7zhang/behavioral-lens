# Exp7 continuation (X. Zhang): circuit and lens angles — results

Pre-registration: `PREREG_x7_circuit_lens.md` (incl. the dated L1b/L2b amendment). Models rebuilt on this server (GPU 7) with
Astra's `train.py` / `eval.py`; behaviour matches the collaborators' rebuild (honest follows 0 / 0 / 0.7 % of wrong guesses on known
held-out facts, sycophant and start 100 %). Per-pair files: `x7_circuit_<pair>.json`, `x7_lens_<pair>.json`; code `scripts/x7_*.py`.
Values are pairs 100 / 101 / 102 (123 / 130 / 130 held-out wrong-guess items). hs[k] = output of block k−1. "Holds" = all three pairs.

## Circuit: where is the honest model's key-side change made?
Honest activations patched into the START model; A = attention from the read position to the guess, summed over heads 9.0, 9.1,
10.2, 12.1, 13.3 (start 1.35 → honest 0.07–0.19); LD start +12.4 → honest −18.5 to −19.2.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| C1 | guess-token residual at the input of layer 9 closes ≥ 50 % of the A drop | 80 / 72 / 57 % | **holds** |
| C2 | … and ≥ 50 % of the LD gap | 27 / 21 / 18 % | **fails** (killed in pair 102: < 20 %) |
| C3 | read-position residual at layer 9 closes < 30 % of the A drop | 27 / 23 / 36 % | fails (2 of 3) |
| C4 | no single component > 50 % of the A drop, and the top one is an MLP | top = **attention of layer 8** (63 / 59 / 48 %), then MLP 8 (9–14 %) | **fails** |
| C5 | guess-position closure of A ≥ 50 % from some layer ≤ 9 | from layer 9 in all three | **holds** |
| C6 | head 10.2 or 11.2 in the top-3 direct logit attribution of the start model; honest halves it | top-3 = 11.2, 15.2, 12.1 in all pairs; 11.2: +13.5 → +0.2 to +0.4 | **holds** |

Reading:
1. **The key-side change is written by layer-8 attention into the guess tokens.** Patching only layer 8's attention output at the
   guess positions closes about half to two-thirds of the attention drop; no MLP comes close (≤ 14 %). Patching the guess-token
   residual before layer 9 has almost no effect (≤ 2 %), so the change is made at layer 8, not carried from earlier layers.
2. **Keys explain the attention change but not the behaviour.** The same guess-position patch that closes 57–80 % of the attention
   drop closes only 18–27 % of the logit gap; patching the read position from layer 13 on closes ≥ 100 %. Most of the honest
   behaviour is carried at the read position (query side and later layers), consistent with the toolkit's finding that the effect
   is spread over many MLPs.
3. **Head 11.2 is the main direct writer of the guess** in the start model (+13.5 nats of logit difference), and the honest model
   silences it (+0.2–0.4); heads 15.2 / 15.3 turn negative (they push towards the correct answer).

## Lens: fitted J-lens and R-lens (start model) on all three models
Fits: 200 prompts; split-half cosine ≥ 0.972 (J) / 0.996 (R) at every layer. R-lens = the LessWrong post's rules on Gemma 3.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| L1 | start: first layer with lens-LD > 0 is ≥ 2 layers earlier under J than logit | 1–2 vs 1 (degenerate: noise at hs[1]) | killed |
| L1b (amendment) | same with the sustained onset | J hs[11] / 12 / 12 vs logit hs[11] / 11 / 11 | **killed** (J not earlier) |
| L2 / L2b | R within 1 layer of J | 1 / 2 / 1 vs 1 / 2 / 2; sustained 12 / 12 / 11.5 vs 11 / 12 / 12 | **holds** |
| L3 | honest falls ≥ 2 nats below start by hs[12] under J | hs[12] in all three (−23 to −37 nats) | **holds** |
| L4 (exploratory) | at the guess tokens (hs 7–10), the J-lens readout of resid_H − resid_start ranks the correct answer above random in ≥ 60 % of items | pair 101: 74–80 % at every layer; pairs 100 / 102: only at hs[7] (72 / 61 %); median ranks > 50,000 | weak, inconsistent |
| L5 | split-half ≥ 0.95 | 0.972 / 0.996 | **holds** |

Reading:
1. **The decision appears at one layer, and every lens sees it there.** In the start and sycophant models the readout is at chance
   (≈ 50 % of items favour the guess) through hs[10] and jumps to 100 % at hs[11] (the output of block 10, the block of head 10.2),
   identically under the logit lens, J-lens and R-lens. The honest model leaves the start at hs[12] and ends at 2 % follow.
2. **Unlike the Qwen loop study, the Jacobian lenses read nothing earlier here.** There, the J-lens read the repeated content four
   layers before the logit lens; here the guess is not represented as "the answer" anywhere before block 10. That fits a copy
   operation (the guess is moved into the answer slot by attention) rather than a gradually formed prediction.
3. The honest keys do not carry a readable "the correct answer" signal (L4 weak).

## Score
Holds: C1, C5, C6, L2/L2b, L3, L5. Fails: C2, C3, C4. Killed: L1, L1b. Exploratory L4: weak.
Together with the toolkit: the honest fine-tune rewires one attention write (layer 8 → guess tokens → keys of heads 9–13) and the
read-position computation; the guess-copying decision is made at block 10–11 and is visible to every lens at the same layer.
