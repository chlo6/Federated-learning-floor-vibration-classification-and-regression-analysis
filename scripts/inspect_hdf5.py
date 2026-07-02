from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py


def _clean(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _show_tree(handle: h5py.File) -> None:
    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{name}/")

    handle.visititems(visit)


def _show_table(handle: h5py.File, dataset_name: str) -> None:
    if dataset_name not in handle:
        return

    dataset = handle[dataset_name]
    if not isinstance(dataset, h5py.Dataset):
        return

    print(f"\n{dataset_name}")
    for row in dataset[:]:
        if dataset.dtype.names is None:
            print(row)
            continue

        fields = {
            field_name: _clean(row[field_name])
            for field_name in dataset.dtype.names
        }
        print(fields)


def _preview_signal(
    handle: h5py.File,
    dataset_name: str,
    run_index: int,
    channel_index: int,
    num_samples: int,
) -> None:
    if dataset_name not in handle:
        raise KeyError(f"Dataset not found: {dataset_name}")

    dataset = handle[dataset_name]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Path is not a dataset: {dataset_name}")
    if len(dataset.shape) != 3:
        raise ValueError(
            f"Expected dataset shape (runs, channels, samples), got {dataset.shape}"
        )

    preview = dataset[run_index, channel_index, :num_samples]
    print(f"\nPreview: {dataset_name}[run={run_index}, channel={channel_index}, :{num_samples}]")
    print(preview)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an HDF5 vibration data file.")
    parser.add_argument("path", type=Path, help="Path to the .hdf5 file.")
    parser.add_argument("--dataset", default="experiment/data", help="Signal dataset path.")
    parser.add_argument("--run", type=int, default=0, help="Run/trial index to preview.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to preview.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to preview.")
    parser.add_argument("--no-tree", action="store_true", help="Skip printing the HDF5 tree.")
    parser.add_argument("--no-metadata", action="store_true", help="Skip metadata tables.")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as handle:
        print(f"File: {args.path}")

        if not args.no_tree:
            print("\nContents")
            _show_tree(handle)

        if not args.no_metadata:
            _show_table(handle, "experiment/general_parameters")
            _show_table(handle, "experiment/specific_parameters")

        _preview_signal(
            handle=handle,
            dataset_name=args.dataset,
            run_index=args.run,
            channel_index=args.channel,
            num_samples=args.samples,
        )


if __name__ == "__main__":
    main()
