# Experiment 7 (sycophancy, Gemma 3 270M IT): results so far

Plan: `notes/PLAN_exp7.md`. All numbers below are in `exp7/notes/exp7_summary.json` (built by `exp7/scripts/summarize_exp7.py`);
the paper's Extended Data table is generated from it by `exp7/scripts/make_tex_table.py`.

- **E1 rebuild.** Astra's `train.py` / `eval.py` / splits (copied to `exp7/scripts/upstream_*.py`, `exp7/data/`; only change: optional
  `SYCMID_SEED`). Pairs 100/101/102 + pair 100 seeds 1, 2 + pairs 101/102 seeds 1, 2 (confirmation set). Honest follows 0-2.8 % of wrong
  guesses on known held-out facts, sycophant 100 %, starting model 100 %; matches the released numbers. Models in
  `/data2/ec2-user/models/exp7/`.
- **E2 weights only.** Norm and distance statistics per module/layer and per-head copy scores (E W_V W_O^h E^T) are not consistent across
  the five pairs (copy score: 6 of 72 heads with one sign in all five, chance 4.5). With the starting checkpoint, "the sycophant moved
  farther from it" holds in 4/5, the opposite of our plan's prediction. Post-hoc layer rule (layers 0,1,3-6) pre-registered on 4 new
  pairs (`PREREG_e2c_confirm.md`): 3/4, prediction 4/4 failed; all-layer distance 4/4, so 8/9 overall.
- **E2d one forward pass.** One prompt (capital of Australia, guess Sydney): margin honest -6.8 to -12.0, sycophant +19.1 to +21.3: 5/5.
- **E5 attention rule.** Attention from the last position to the guess, summed over heads: on Astra's own tool texts S > H in 3/3; on our
  held-out wrong-guess prompts at the end of the prompt S < H in 5/5 (small), after appending "The answer is" S > H in 5/5 (large). Head
  10.2 alone, with the prefix: S > H in 5/5.
- **E3 weight patching (Astra's patch_score protocol, honest -> sycophant).** Gap closed by head 10.2: 0 in 5/5; by Astra's 9-10
  attention-round components: 0 in 3/3; by heads 10.2+11.2+17.2 or attention layer 10: 0; all attention 46-78 %; all MLPs 97-101 %;
  10 largest weight changes 8-63 %; random 10 components <= 7 %. Activation patching (pair 101 only): the best single component (MLP 3) closes
  10 % of the follow gap; best margin closers are MLPs 3, 10, 9 and head 10.0 (0.29-0.38 of the margin).
- **E4** (`e4_mechanism_{100,101,102}.json`; details below from pair 101, cross-pair summary in the last bullet). Prompts = held-out wrong-guess prompts + "The answer is", read at
  the last position.
  - *Both fine-tunes moved, in opposite directions.* Attention to the guess, head 10.2: start 0.26, honest 0.06, sycophant 0.68
    (11.2: 0.36 / 0.06 / 0.60). Summed |change vs start| over all 72 heads: honest 2.42, sycophant 2.64. In this pair the sycophant
    amplified the copy attention; across pairs that is not consistent (last bullet).
  - *The honest change in head 10.2 is on the key side.* Attention with q from model X and k from model Y: q=start,k=honest 0.07 vs
    q=honest,k=start 0.26 (start/start 0.26). Since keys/values are shared by the 4 heads and k_proj belongs to the layer, this is a
    change in what the guess tokens look like by layer 10 (upstream residual), which is why copying head 10.2's own weights does
    nothing. The sycophant's increase uses both sides (q=syc,k=start 0.55; q=start,k=syc 0.44).
  - *"The guess is wrong" is already linear in the start model.* Probe wrong-guess vs right-guess prompts, trained on 200 train facts,
    tested on held-out: AUC peaks at 0.84 (start, layer 11, last position), 0.86 sycophant, 0.75 honest; at the guess token 0.81-0.83 in
    all three. Late layers at the last position: honest drops to 0.53-0.56 (its output no longer depends on the guess), start and
    sycophant stay 0.62-0.71. So the honest fine-tune did not need to create the knowledge; it changed how the guess is routed.

Reading: the honest fine-tune is the model that changed behaviour; the sycophant keeps the start's default. The weights-only task becomes
nearly solvable with the public starting checkpoint and trivial with one forward pass. The attention tool's answer was right but its
location (head 10.2) is not causal; the difference is spread over the MLPs.
  - *Across pairs 100/101/102:* the honest change is the same in all three (heads 9.0, 9.1, 10.2, 12.1, 13.3 among the top-6 drops in
    every pair; head 10.2 0.26 -> 0.04 / 0.06 / 0.02; key side: q=start,k=honest 0.03 / 0.07 / 0.03 vs q=honest,k=start 0.26 / 0.26 / 0.27;
    summed |change| 2.40 / 2.42 / 2.42). The sycophant's change is pair-specific (head 10.2 0.31 / 0.68 / 0.35; top heads differ;
    summed |change| 1.78 / 2.64 / 2.21). Pair 101 = the post's Pair 2, behind Astra's head-10.2 figure, is the outlier.
    Start-model probe AUC 0.81 / 0.84 / 0.88.

## Full toolkit (the report's tools 1-10 + controls) on pairs 100/101/102

Pre-registration: `PREREG_toolkit.md` (written before these analyses). Scores: `tk_summary.json` (`scripts/tk_summary.py`); per-tool
files `tk_<tool>_<pair>.json`. Metric for tools 2, 4-10: logit difference guess - correct at the reply's first differing token,
held-out wrong-guess prompts (123/130/130 items). Values below are pairs 100 / 101 / 102.

| Tool | What it says | Causal check / control | Prediction |
|---|---|---|---|
| 1 Weight-difference SVD | fine-tune deltas are NOT low rank (median top-1 energy 0.13, max 0.33-0.35; effective rank ~60/640). The honest delta's top direction repeats across pairs in L17 down_proj (cos 0.96) and L11/L12 k_proj (0.90/0.88) | task vector: start + (H - S) follows 17 / 8 / 21 % of wrong guesses (start 100 %); lam 0.5: 12 / 4 / 12 % | P3 fails (max > 0.2), P11 holds |
| 9 Activation patching, sycophant <- honest (all positions) | largest single component = attention output of layer 10 (closes 55 / 47 / 50 % of the LD gap); best head 11.2 / 11.2 / 8.3 (36 / 29 / 26 %); 2-3 MLP layers close >= 50 %; all MLPs 99 / 96 / 98 %, all attention 95 / 97 / 98 % (each alone is enough: serial, not parallel) | 4 random MLP layers already close 24-48 %: highly distributed. Contrast with weight patching (E3): copying attention layer 10's WEIGHTS closes 0 %; the difference flows through layer-10 attention but is created upstream | P1 fails (a head > 20 %), not killed |
| 10 Attribution patching | rank correlation with exact patching 0.54 / 0.54 / 0.47 (heads 0.58-0.63, MLPs -0.10 to 0.15) | top component (attn10) estimated at 5-8 % of its exact effect (13-19x too small) | P2 fails (rho < 0.5 in pair 102) |
| 6 Logit / tuned lens | of the MEAN model-difference direction: no preference for the guess at any layer (0.49-0.52). Of each item's own difference resid_S - resid_H: favours the guess in 100 % of items from layer 11 on, chance before. Own-state lens: start and sycophant switch to the guess abruptly at layer 11 (0.5 -> 1.0); honest 0.55-0.68 at layer 11, then falls to 0.03 by layer 17 | - | P4 fails: the decision is item specific, not a direction |
| 7 Jacobian lens | linear prediction of adding -delta correlates r = 0.92 / 0.86 / 0.79 with real steering across layers | sign disagrees at layers 15-17 in all pairs (small values) | P5 killed as stated |
| 8 Activation steering | -delta at the best layer (10/10/9) cuts follow to 70 / 72 / 56 %; +delta on honest raises follow only to 11-12 % | 5 random directions of the same norm: follow 46-93 %, LD falls MORE (to 0-10 vs 3-5): the push scrambles the decision, it does not make the model honest (honest LD -17 to -19) | P6 fails |
| 4 Probe vs class means | "the guess is wrong" (start model): probe AUC up to 0.87 / 0.85 / 0.89 | class-mean direction AUC <= 0.61; cos(probe, class mean) <= 0.18: the information is relational (guess vs question), not a mean shift; the opposite of exp8 | P7 fails |
| 2 Neuron ranking | top MLP layers (by exact effect): 3,2 / 3,10 / 10,9; top-32 neurons close 12 / 10 / 39 % of the LD gap (layer 41 / 37 / 33 %); top-512 of 2048 close 29-51 % | 32 random neurons ~0 (<= 6 %) | P8 fails (share 29 % / 26 % < 30 %), ranking beats random |
| 3 Auto-interp (Sonnet explain + score) | top neurons AP 0.88 / 0.78 / 0.88 vs shuffled descriptions 0.65 / 0.60 / 0.69; two pair-101 L3 top neurons fire on "think" in "I think the answer" | random neurons score as high (0.88 / 0.87 / 0.85): specific, not special | P9 holds |
| 5 SAE (Gemma Scope 2, 16k, resid 5/9/12/15) | FVU stays low (L12 0.015 -> 0.03, L15 0.08 -> 0.15; L5 0.001 -> 0.004, the ratio that trips the rule); 0/10 of the most-changed latents are about the user's suggestion at L9/L12 in every pair; most-changed latents are output-format latents (space before a digit, copula "is", "continue a repeated/copied phrase"). Latent L12/572 "predictable repeated or copied continuation": honest switches it off vs start (-480 to -560), sycophant above honest (+580 to +670) in all three pairs | - | P10 killed as stated (FVU ratio > 3x at L5 on tiny absolute values; label part fails) |
| Astra's Figure 2 | on our rebuilt pair 101 with Astra's 8 texts and positions: head 10.2 operands 0.9427 / guess 0.0075 (honest), 0.0347 / 0.8567 (sycophant); Astra: 0.9428 / 0.0075 / 0.0347 / 0.8566; same top-6 switch ranking | the rebuild reproduces Astra's models' attention to 4 digits | P12 holds |

Score: 3 of 12 predictions hold (P9, P11, P12), 7 fail, 2 are killed by their own rule.

**What the toolkit adds to the story.**
1. *Where it is visible vs where it is made.* The difference flows through layer-10/11 attention (activation patching: attn10
   is the largest single component, ~50 %), but copying those weights changes nothing; it is made upstream and spread over many
   MLPs (any 4 random MLP layers close a quarter to a half). Same lesson as exp8 (edit in L15/19 MLPs, effect through later attention).
2. *Sycophancy here is item specific copying, not a direction.* Every direction-level tool fails: class means, the mean-direction
   lens and steering with it (no better than random pushes of the same size). Per-item readouts work: the copied guess appears at
   layer 11 in start and sycophant; the honest model's correction is spread over layers 12-17.
3. *Tool reliability repeats exp8's pattern.* Attribution patching ranks heads but not MLPs and underestimates the top component
   >10x; auto-interp scores random neurons as high as selected ones; the public SAE sees output-format changes, not a "user
   suggestion" feature, but does pick up a consistent "continue copying" latent that the honest model switches off.
