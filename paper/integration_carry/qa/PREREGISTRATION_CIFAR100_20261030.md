# Pre-registered replication: INT8 ResNet-20 on CIFAR-100

Written 2026-10-30, before the CIFAR-100 network was trained and before any
approximate-accumulator inference was run on it. The SHA256 of this file is
recorded in `REVISION_LOG_20261030.md` at the time it was written. The analysis
below will be reported whatever its outcome.

## Fixed protocol

- Network: ResNet-20 (option-A shortcuts, same code as `python/accrate/resnet20.py`)
  with a 100-class classifier, trained on the CIFAR-100 training set with the
  same recipe as the CIFAR-10 model (SGD, Nesterov, OneCycle, 50 epochs, seed 0).
- Quantization and accumulation: unchanged from `cnn_accumulate.py` (folded BN,
  per-channel INT8 weights, per-tensor INT8 activations calibrated on 500
  training images, 24-bit two's-complement serial accumulation, K = 27 to 576).
- Compensation: data-calibrated preload `-round(E[E_K])` from Theorem 3 under
  each layer's measured product-residue law (same calibration code).
- Test set: all 10,000 CIFAR-100 test images.
- Designs (the same twelve as the CIFAR-10 10k run, no additions or removals):
  `exact`, `round_5`, `round_6`, `round_7`, `round_8`, `trunc_4`, `trunc_6`,
  `trunc_7`, `evo_add8u_0E2`, `evo_add8u_6QU`, `evo_add8u_2LL`, `loa_5`.
- Hardware cost used for ranking: best signed-accumulator area of the existing
  Nangate45 synthesis study (`hw_study_s.json`), and, as a second ranking, the
  post-route cell area at the relaxed clock (`pnr_study.json`, signed variant).

## Hypotheses and decision rules

- H1 (pathwise twins): `evo_add8u_0E2`, `evo_add8u_6QU` and `loa_5` give the
  same top-1 prediction as their quantizer twins (`round_7`, `trunc_4`,
  `round_5`) on at least 98% of test images.
- H2 (exact metrics): the median ratio of Theorem 3's predicted accumulated
  MSE to the measured MSE over layer–design pairs lies in [0.9, 1.1].
- H3 (equivalence): paired TOST against `exact` with margin +-1 percentage
  point at level 0.05 (the criterion stated in the CIFAR-10 study). A
  half-point margin is reported as a secondary analysis.
- H4 (cost ordering, the main claim): among designs equivalent under H3, no
  approximate low part has lower area than the cheapest equivalent exact
  reference accumulator, under either area ranking.

A failure of any hypothesis is reported in the manuscript as a failure.
