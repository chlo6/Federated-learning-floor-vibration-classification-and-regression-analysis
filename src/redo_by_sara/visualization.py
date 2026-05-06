from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_window(artifact_path: Path, index: int, output_path: Path | None = None) -> None:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    sample = artifact["samples"][index].numpy()
    metadata = artifact["metadata"][index]
    speed = float(artifact["regression_targets"][index].item())

    fig, axes = plt.subplots(sample.shape[0], 1, figsize=(12, 2 * sample.shape[0]), sharex=True)
    if sample.shape[0] == 1:
        axes = [axes]

    for channel_idx, axis in enumerate(axes):
        axis.plot(sample[channel_idx], linewidth=0.8)
        axis.set_ylabel(f"Ch {channel_idx + 1}")

    axes[-1].set_xlabel("Time points")
    fig.suptitle(
        f"Subject {metadata['subject_id']} | Run {metadata['run_index']} | "
        f"Start {metadata['start_time']:.2f}s | Speed {speed:.3f} m/s"
    )
    fig.tight_layout()

    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
