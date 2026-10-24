# Model card: not generated yet

This file is rendered from the training report by `make docs DATASET=ieee_cis` (`fraudshield model-card`), after the real-data training run listed in [PLAN.md](../PLAN.md#deferred-real-data-runs). Every number in the card comes from `reports/<dataset>/metrics.json`; nothing is written by hand.

The generated card covers:
- model details (registry version, git SHA, feature-set and data hashes);
- intended use;
- data and time splits;
- the comparison against the rules-only, logistic and request-only baselines;
- dollar outcomes under the cost matrix;
- segment metrics;
- calibration;
- hyperparameters;
- limitations.
