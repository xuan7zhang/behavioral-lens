# Experiment 9: does an unsupervised SAE find Astra's u?

Pre-registration `PREREG_exp9_sae.md`; raw results `exp9_results.json` (`scripts/analyze.py`); scores `exp9_summary.json`
(`scripts/score.py`); training logs `train_L15.log`, `train_L19.log`. SAEs (TopK k=64, 17M unlabelled tokens: 7.3M of our screening
generations + 10.2M WikiText-103) at `/data2/ec2-user/exp9/sae_L{15,19}_w{1024,4096,16384,65536}_s{0,1}.pt`; FVU on held-out
training-mix states 0.21 / 0.15 / 0.14 / 0.16 (1k / 4k / 16k / 64k; 64k is under-trained at 17M tokens), no dead latents.
Evaluation: 100 loop + 100 normal texts from Astra's unedited eval generations (not in the training data), first 4096 tokens.

## Results

| | L15 (u15) | L19 (u19) |
|---|---|---|
| max cos(decoder, u), 1k / 4k / 16k / 64k | 0.51 / 0.61 / 0.59 / 0.56 (16k seed 2: 0.55) | 0.60 / 0.72 / 0.72 / 0.68 (16k seed 2: 0.69) |
| OMP atoms for cos(reconstruction, u) >= 0.9 (16k) | 38 | 36 |
| loop - normal difference in r.u carried by top-1 latent (16k s0 / s1) | 22 % / 19 % | 40 % / 47 % |
| latents for 50 % / 80 % of it (16k s0) | 26 / 338 | 3 / 143 |
| same top-1 feature across widths / seeds (cos with 16k s0 top-1) | 0.51-0.86 | 0.83-0.97 |
| blind label of top-1 latent, by width (1k / 4k / 16k / 16k-s1 / 64k) | restating / restating / repetition / repetition / doubt | doubt / doubt / doubt / repetition / doubt |
| categories of the 16k top-8 loop-side latents | repetition 5, formatting 2, maths 1 | doubt 4, repetition 1, restating 1, maths 1, other 1 |
| Astra's rule along the top-1 latent, share of u's loop-loss effect (1k / 4k / 16k / 16k-s1 / 64k) | 0.71 / 0.80 / 0.65 / 0.58 / 0.88 (both layers edited together) | |
| along top-5 / top-10 contribution-weighted / OMP-10 / 5 random latents | 0.80 / 0.86 / 1.05 / 0.00-0.01 | |

Almost every top loop-side description, at every width and both layers, is about the model's "Wait, if ..." self-checking: re-reading
or re-quoting the problem, re-examining a hypothetical, or re-typing a clause it has just written. The classifier splits this one theme
over three of the fixed categories (repetition / restating / doubt): 5 of 8 (L15) and 6 of 8 (L19) top latents fall in that trio.
Auto-interp: top latents AP 0.94 vs shuffled descriptions 0.66; random latents 0.84 (L15) / 0.88 (L19).
Edits along SAE directions raise the loss at normal positions in proportion (e.g. 4k top-1: loop +0.045, normal +0.029; u: +0.056 / +0.034):
no SAE direction is more selective than u.

## Predictions (1 of 6 held)
- E9-1 holds: 16k replicates test4 (0.59 vs 0.56; 0.72 vs 0.65).
- E9-2 fails: cosine does not rise as width falls; it peaks at 4k-16k, and 1k is lowest (0.51 / 0.60).
- E9-3 fails at L19 (top-1 40 %, 3 latents for 50 %); holds at L15 (22 %, 26).
- E9-4 fails at L15 (5 of 8 labelled verbatim repetition); holds at L19 (5 categories, 1 repetition).
- E9-5 fails as stated: L15 random latents score 0.103 below top latents (limit 0.1); L19 within 0.07.
- E9-6 fails: a single latent already gives 58-88 % of u's effect (predicted < 50 %); top-5 >= 50 % and random < 10 % hold.

## Decision rule
- L15: "mixed direction" (no latent reaches 0.7, 26 latents for half the contribution, several categories).
- L19: "intermediate" (one latent at cos 0.72, stable across widths and seeds, carries 40-47 %; the rule asked for >= 50 %).
- Neither layer meets "one loop switch".

## Reading
1. Cross-group convergence on WHAT u is: an SAE trained without loop labels finds latents near u (L19 cos 0.72), and a blind
   explainer describes them as "Wait, if ..." re-checking / re-reading. This matches the supervised neuron auto-interp (test 3: the
   edited neurons fire on re-reading and self-checking) and the lenses (+u promotes "Wait"). Independent support that Astra's u is the
   model's re-check direction, not an arbitrary direction.
2. But not ONE loop switch: the dominant latent is about re-checking, not about looping as such; at L15 u is spread over dozens of
   latents (26 for half the loop - normal difference); at L19 one latent carries ~40 %. The spread is semantic near-synonyms (copy /
   restate / doubt within self-checks), not unrelated features, so "mixed" here means a re-check subspace, not a mixture of meanings.
3. Causally, one SAE latent gets most of the way (58-88 %), OMP-10 matches u (1.05), random latents do nothing. The edit works because
   it pushes against the re-check subspace; u is a good summary of that subspace, and SAE features approximate it from below.
