# Task 1: Centralized CNN Baselines

This branch establishes centralized upper-reference models before federated
experiments. One CNN trains directly on the union of all training windows.
There are no clients, communication rounds, parameter averaging, or local
heads.

Both tasks use the same `raw_windows.pt` train/validation/test split and seed:

- person identification: one global seven-output classifier;
- walking-speed estimation: one global scalar regressor.

## Train one configured model

```bash
python scripts/train_baseline.py --config configs/centralized_classification.yaml
python scripts/train_baseline.py --config configs/centralized_regression.yaml
```

The trainer selects the best epoch using validation score, restores that
checkpoint, and evaluates the held-out test split once.

## Tune centralized hyperparameters

```bash
python scripts/tune_centralized.py \
  --config configs/centralized_classification.yaml \
  --model-variants simple deep \
  --learning-rates 0.001 0.0005 \
  --weight-decays 0.0001 0.00001 \
  --batch-sizes 32 \
  --epochs 15 30

python scripts/tune_centralized.py \
  --config configs/centralized_regression.yaml \
  --model-variants simple deep \
  --learning-rates 0.001 0.0005 \
  --weight-decays 0.0001 0.00001 \
  --batch-sizes 32 \
  --epochs 15 30
```

Candidate models are compared using validation only. The test split is
evaluated once, after the best configuration is selected.

For fair later comparisons, federated experiments should reuse the same
artifact, seed list, preprocessing, model variant, and evaluation metrics.
