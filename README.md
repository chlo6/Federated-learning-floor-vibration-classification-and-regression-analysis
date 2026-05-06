# ENGR859 Final Project

A compact raw-vibration learning pipeline for ENGR 859.

This project focuses on three paths using raw vibration windows only:

1. Regression: estimate walking speed.
2. Classification: identify the subject.
3. Federated learning: simulate 3-client training with Flower.

It intentionally avoids the original CWT stack for now.

The current defaults use the sensor table as the source of truth:

- sensors and channels are treated as different concepts,
- the config declares a sensor-to-channel map explicitly,
- the default baseline uses all 20 physical channels,
- and the dataset artifact contains explicit `train`, `val`, and `test` splits.

## Project Structure

- `PLAN.md`: concise implementation plan.
- `configs/`: baseline and federated YAML configs.
- `scripts/preprocess_raw.py`: build a clean raw-window dataset artifact.
- `scripts/train_baseline.py`: train a small 1D CNN baseline.
- `scripts/run_flower.py`: run the subject-owned Flower simulation.
- `scripts/run_flower_iid.py`: run the IID-by-run Flower simulation.
- `scripts/log_iid_flower_wandb_plots.py`: create and log IID Flower result plots.
- `scripts/plot_window.py`: visualize one saved window.
- `src/redo_by_sara/`: reusable code.

## Workflow

1. Build the raw dataset artifact.
2. Inspect a few windows.
3. Train a baseline for regression or classification.
4. Run the 3-client Flower simulation.
5. Review saved metrics, models, partitions, and plots in `artifacts/`.

## Example Commands

From `/home/coder/workspace/ENGR859-final-project`:

```bash
/home/coder/conda/envs/torch311/bin/python scripts/preprocess_raw.py --config configs/regression.yaml
/home/coder/conda/envs/torch311/bin/python scripts/plot_window.py --artifact artifacts/raw_windows.pt --index 0
/home/coder/conda/envs/torch311/bin/python scripts/train_baseline.py --config configs/regression.yaml
/home/coder/conda/envs/torch311/bin/python scripts/train_baseline.py --config configs/classification.yaml
/home/coder/conda/envs/torch311/bin/python scripts/run_flower.py --config configs/fl_regression.yaml
/home/coder/conda/envs/torch311/bin/python scripts/run_flower.py --config configs/fl_classification.yaml
/home/coder/conda/envs/torch311/bin/python scripts/run_flower_iid.py --config configs/fl_iid_regression.yaml
/home/coder/conda/envs/torch311/bin/python scripts/run_flower_iid.py --config configs/fl_iid_classification_15r_3e.yaml
```

## Federated Learning Versions

- Subject-owned clients: `configs/fl_regression.yaml` and `configs/fl_classification.yaml`
  assign subjects `003`-`008` across 3 clients.
- IID-by-run clients: `configs/fl_iid_regression.yaml` and `configs/fl_iid_classification*.yaml`
  keep full runs intact while giving each of the 3 clients data from every subject.
- The most developed saved classification run uses
  `configs/fl_iid_classification_15r_3e.yaml`, with 3 clients, 15 rounds, and
  3 local epochs.

## Data Processing Details

### APDM Labeling
Each vibration window is labeled with the **average walking speed** from its corresponding trial, computed as:
- `speed = (left_leg_speed + right_leg_speed) / 2.0`
- Speeds come from APDM gait analysis exports (in m/s)
- The code supports both APDM export formats seen in this repo:
  - **Older format**: Column headers with `[mean]` suffix
  - **Newer format**: Separate rows for "Measure" with speed values in indexed columns
- Missing values default to `0.0`

### Resampling
- All vibration data is resampled to 400 Hz using polyphase resampling (`scipy.signal.resample_poly`)
- This ensures a consistent sample rate across trials with different original sampling rates

### Windowing
- Each trial is split into overlapping 5-second windows with 1-second stride
- Windows are extracted only within the valid walking data range (determined by `runPeramiters.csv`)
- Each window carries metadata: subject ID, run index, and window start time (in seconds)

### Normalization
- Channel-wise mean and standard deviation are computed **only on training windows**
- This prevents data leakage during validation and testing
- Normalization: `(x - channel_mean) / channel_std`

### Sensor-to-Channel Mapping
- The config file explicitly maps sensor names (e.g., "8x", "8y", "8z") to physical channel numbers
- Channels are read in, reordered to match sensor order, and labeled as `{sensor_name}:ch{channel}`
- This allows flexible sensor selection without modifying HDF5 loading code

### Train/Val/Test Splits
- Splitting is done by full run within each subject, with non-overlapping groups
- Each subject must have at least 3 runs (1+ for each split)
- Default ratios: 60% train, 20% val, 20% test
- Ensures no data leakage between splits

## Notes

- The loader uses the current thesis dataset layout in `../TestData/20251124_Testing`.
- It supports both APDM CSV styles seen in the current repo.
- Missing subjects or missing label files are skipped instead of crashing.
- Splitting is done by full run within each subject, with non-overlapping `train`, `val`, and `test` groups.
- The Flower sandbox uses a second stage after that split:
  global `train` stays train-only, gets sharded across clients by subject ownership, and global `val` and `test` stay on the server to avoid leakage.
