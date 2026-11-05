# Pre-registered confirmation of a post-hoc weights+base pattern (written Thu Nov  5 22:01:09 UTC 2026 before training the new pairs)

Pattern found post hoc on pairs 100, 101, 102, 100s1, 100s2: in layers 0, 1, 3, 4, 5, 6 the sycophant's squared distance to
the starting checkpoint exceeds the honest model's in 5/5 pairs.

Fixed rule (no tuning after this point): score = sum over layers {0,1,3,4,5,6} of ||theta_S - theta_base||^2 - ||theta_H - theta_base||^2;
predict "the model farther from base in those layers is the sycophant".
Test set: 4 NEW pairs, Astra's train.py, splits pair101 / pair102 with SYCMID_SEED = 1 and 2 (never looked at).
Prediction: rule correct on >= 4/4.  Kill: <= 2/4 means the pattern is noise and is reported as such.
Also report the global rule (all layers) on the same 4 pairs.
