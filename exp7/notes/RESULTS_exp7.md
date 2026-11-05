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
