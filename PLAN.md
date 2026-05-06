# Plan

## Goal

Rewrite the raw-vibration portion of the thesis code in a simpler ENGR 859 style, then build a clean baseline model path for both regression and classification.

## Phase 1: Data Understanding

- Read the HDF5 vibration files directly.
- Read the run parameter CSV to know which time range is valid walking data.
- Read APDM walking-speed labels from either export format used in this repo.
- Visualize a few raw sensor windows to confirm scale, noise, and timing.

## Phase 2: Preprocessing

- Use the sensor table as the source of truth and keep sensor names separate from channel ids.
- Resample all runs to a common sample rate of 400 Hz.
- Window each run using 5-second windows with 1-second stride.
- Keep metadata for dataset, subject, run, and window start time.
- Build one shared artifact for both tasks.
- Normalize using train-split statistics only.
- Create explicit train, validation, and held-out test splits by run.

## Phase 3: Baseline Modeling

- Start with a small 1D CNN on raw windows.
- Use the same backbone for both tasks.
- Use MSE + RMSE reporting for regression.
- Use cross-entropy + accuracy reporting for classification.

## Phase 4: Evaluation

- Split by run within each subject.
- Report train and validation metrics each epoch.
- Save the best model by validation metric.
- Evaluate the best checkpoint once on the held-out test split.

## Phase 5: Next Steps

- Compare raw-time baseline against deeper models later.
- Add better plots and error analysis.
- Only after the raw baseline is stable, revisit CWT.
