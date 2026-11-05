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
- **E4** (attention by model vs start, q/k swaps, "guess is wrong" probe): running for pair 101.

Reading: the honest fine-tune is the model that changed behaviour; the sycophant keeps the start's default. The weights-only task becomes
nearly solvable with the public starting checkpoint and trivial with one forward pass. The attention tool's answer was right but its
location (head 10.2) is not causal; the difference is spread over the MLPs.
