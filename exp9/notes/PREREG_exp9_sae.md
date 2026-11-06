# Experiment 9: does an unsupervised SAE find Astra's u? Pre-registration (written 2026-11-06 03:29 UTC, before any SAE was trained)

Question (claim B = "Astra located one loop switch"): without loop labels, does a sparse autoencoder learn a feature close to u,
and does a blind auto-interpreter call it repetition / restating? Or is u split over several semantically different features?

Prior (test4, collaborator, 16k BatchTopK at hidden states 16 / 20, public 65k at 18): max |cos(decoder, u)| 0.56 / 0.65 / 0.45; 10
latents leave ~60 % of u's norm unexplained. Exp 9 adds widths, seeds, activation-weighted decomposition, blind labels and a causal edit.

Setup. Qwen3.5-0.8B base. TopK SAEs (k = 64) on the residual after layer 15 (hidden state 16; u15) and after layer 19 (hidden state 20;
u19); widths 1k, 4k, 16k, 64k; a second seed at 16k. Training data has no labels: base thinking-on / thinking-off and p50c
generations (our screening outputs) + WikiText-103, ~17M tokens. Evaluation texts are disjoint (Astra's unedited eval generations),
loop positions = 32-gram verbatim repeats, normal positions = finished, loop-free texts.

Measures. (G) max |cos(d_j, u)|; latents needed (OMP) for cos(reconstruction, u) >= 0.9. (C) contribution of latent j to the
loop - normal difference in r.u: c_j = (E_loop f_j - E_normal f_j)(d_j . u); share of the positive total carried by the top 1 / top 5;
latents needed for 50 % / 80 %. (S) blind auto-interp (Sonnet; the prompt never mentions loops) of the top 8 loop-side contributors
(16k) and top 4 (other widths); detection AP vs shuffled descriptions and vs 10 random latents; labels classified into fixed categories:
verbatim repetition/copying; restating/re-reading the problem; doubt/self-check; wrap-up/final answer; maths content; formatting/
whitespace; other. (E) Astra's rule w' = w - 6 d (d^T w) on Astra's 64 columns with d = top-1 latent decoder, d = contribution-weighted
top-5, d = OMP-10 reconstruction of u, d = 5 random latents, d = u: teacher-forced loss change at loop / normal positions (q4b proxy).

Predictions
| # | Prediction | Fails if |
|---|---|---|
| E9-1 | 16k replicates test4: max |cos| within 0.1 of 0.56 (L15) / 0.65 (L19) | outside |
| E9-2 | max |cos| rises as width falls; at 1k some latent has |cos| >= 0.7 at both layers (u is coarse and splits with width) | < 0.7 at either layer |
| E9-3 | at 16k the top-1 latent carries < 30 % of the positive loop - normal u-contribution; >= 5 latents needed for 50 % | top-1 >= 30 % or < 5 needed |
| E9-4 | blind labels of the top 8 loop-side latents (16k) fall in >= 3 categories; verbatim repetition is <= half of them | < 3 categories |
| E9-5 | auto-interp beats shuffled by >= 0.1 AP, random latents within 0.1 of top latents (as in exp7/exp8) | - |
| E9-6 | editing along the top-1 latent gives < 50 % of u's loop-loss increase; contribution-weighted top-5 gives >= 50 %; random latents < 10 % | top-1 >= 50 % |

Decision rule (fixed now)
- "One loop switch" (supports claim B): at width >= 4k a single latent has |cos| >= 0.7, carries >= 50 % of the contribution, has a
  blind label in {repetition/copying, restating, doubt/self-check} and its edit gives >= 50 % of u's loop-loss increase.
- "Mixed direction": at 16k and 64k no latent reaches |cos| 0.7, >= 3 latents are needed for 50 % of the contribution, and they span
  >= 2 categories.
- Otherwise "intermediate", reported with the measures.
