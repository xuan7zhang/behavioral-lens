# Exp11 pre-registration: Astra's Gemma 3 270M math edits as a negative control for the tools (written 2026-11-06 ~08:10 UTC)

Material: blog Experiment 8 follow-up, Gemma 3 270M IT (rev ac82b4e8), edits specified in
`astra_logs/exp8_math/gemma_followup/runs/*/final_answer.md` (g05b lists no column indices and cannot be rebuilt):
- g10b: down_proj columns ×1.5 (layer 8: 699; layer 14: 483, 984; layer 17: 1179). Astra: "repetition-associated"; repeated
  conclusions 15.3% → 12.0%.
- g25b: layer-17 down_proj column 920 ×2. Astra: "numerical selectivity".
- g50b: layer-11 o_proj[:, 512:768] (head 2) ×1.1 and layer-11 down_proj columns 971, 180, 149, 1680, 232, 1724, 312, 1096 ×0.5.
  Astra: halving head 2 costs 26 DEV points.
Astra's MATH-500 (lenient, 4 samples, 8,192 tokens, T = 1.0, top-k 64, top-p 0.95): base 6.8, g10b 7.6, g25b 7.7, g50b 7.0.
None of the three is a reliable improvement. That makes them a control: tools that tell a confident mechanism story here
are telling it about an edit that does nothing.

## Predictions

- **G1 (rebuild)** Our base is within 1.5 points of 6.8; no edit differs from our base by more than 2 points.
- **G2 (auto-interpretation is not specific)** The 13 edited neurons get detection AP within 0.05 of 13 random neurons from
  the same layers (same explainer, same scorer, Sonnet via the Vector API).
- **G3 (SAE readout)** For each edited neuron's write vector, the top Gemma Scope 2 latent (nearest layer ≥ its layer) has |cos| below
  0.3, no higher than for random neurons of the same layers.
- **G4 (teacher-forced proxy)** Each edit changes NLL on base-model correct answers and on loop positions by less than 0.01 nats.
- **G5 (Astra's causal claims)** (a) Halving layer-11 head 2's output lowers MATH-500 lenient accuracy by at least 3 points.
  (b) g10b's four columns set to zero raise the repetitive-ending rate above base, and ×1.5 lowers it (direction only; n.s. allowed).
- **G6 (does the Qwen mechanism transfer?)** Gemma has a loop direction: a loop-minus-normal mean difference whose projection
  separates loop from normal positions with AUC ≥ 0.8 at its best layer. p50c's recipe along it (64 columns ranked by write
  projection, α = 6, best two layers) cuts the length-cutoff rate by at least 30% (relative) but changes accuracy by less than
  1.5 points.
- **G7 (why loop fixes cannot help here)** Of base-model wrong answers, at least 70% are finished wrong answers (labelled
  arithmetic / reasoning / format by Sonnet), at most 15% are loops or cutoffs.

Scoring into `RESULTS_exp11.md` without editing this file.
