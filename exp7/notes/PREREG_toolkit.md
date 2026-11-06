# Exp7 full toolkit: pre-registered predictions (written 2026-11-06 01:34 UTC, before any of these analyses ran)

Pairs 100, 101, 102 (post's Pairs 1-3). Metric: logit difference guess - correct at the answer's first token (tk_lib.py).
Each prediction is scored per pair; "holds" = all 3 pairs unless stated.

| # | Tool | Prediction | Kill / fail rule |
|---|---|---|---|
| P1 | 9 path patching (exact, sycophant <- honest) | MLPs carry the effect: some set of <= 4 MLP layers closes >= 50 % of the LD gap; no single head closes > 20 % | a single head >= 50 % |
| P2 | 10 attribution patching | Spearman with exact over 108 components >= 0.5, but the top component's size is off by > 2x | rho < 0.3 |
| P3 | 1 weight-difference SVD | the fine-tune deltas are not low rank: top singular value carries < 20 % of the energy in every matrix (unlike exp8's rank-1) | any matrix > 50 % |
| P4 | 6 logit lens | the model-difference direction delta_L = mean resid(S) - mean resid(H) at the read position favours guess over correct (positive per-prompt LD) from layer 9 on | negative at the last 3 layers |
| P5 | 7 Jacobian lens vs 8 steering | the linear (gradient) estimate of adding -delta_L agrees in sign with real steering at layers >= 9 | sign disagreement in >= 2 pairs |
| P6 | 8 steering | adding -delta_L (norm of delta_L) at the read position of the best layer cuts the sycophant's follow rate by >= 50 %; 5 random directions of the same norm < 10 % | -delta_L < 20 % |
| P7 | 4 probe vs class means | cos(probe weight, class-mean "guess is wrong" direction) < 0.3, while the class-mean projection alone gives AUC >= 0.75 (exp8 lesson) | cos > 0.6 |
| P8 | 2 neuron ranking | in the MLP layer with the largest exact effect, the top 32 neurons by attribution close >= 30 % of that layer's effect; 32 random neurons < 5 % | top-32 < 10 % |
| P9 | 3 auto-interp | descriptions of top neurons beat shuffled descriptions by >= 0.1 detection AP; random neurons score within 0.1 of top neurons | - |
| P10 | 5 SAE (Gemma Scope 2, 16k, resid L5/9/12/15) | fraction of variance unexplained on fine-tuned states < 1.5x the start model's (small fine-tune); among the 10 latents changed most S vs H at L9 or L12, >= 3 are labelled as about a user suggestion / agreement / the guess | FVU > 3x |
| P11 | task vector | start + (theta_H - theta_S) follows < 50 % of wrong guesses | > 90 % |
| P12 | Astra's Figure 2 on our pair 101 | on Astra's 8 r7 texts, head 10.2 answer-position attention to the user's answer < 5 % (honest) and > 50 % (sycophant) | ordering reversed |
