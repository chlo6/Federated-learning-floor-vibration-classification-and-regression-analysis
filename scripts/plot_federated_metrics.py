from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run the Flower script first so it can write this history file."
        )
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Re-run the Flower script first.")
    return json.loads(path.read_text())


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _score_label(task: str) -> str:
    return "RMSE" if task == "regression" else "Accuracy"


def _metric_label(task: str, metric: str) -> str:
    if metric == "loss":
        return "Loss"
    if metric == "r2":
        return "R^2"
    return _score_label(task)


def _plot_client_metric(
    rows: list[dict[str, str]],
    task: str,
    metric: str,
    output_path: Path,
) -> None:
    ylabel = _metric_label(task, metric)
    value_key = f"train_{metric}"
    client_ids = sorted({row["client_id"] for row in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    for client_id in client_ids:
        points = [
            (int(float(row["round"])), _float(row, value_key))
            for row in rows
            if row.get("client_id") == client_id and _float(row, value_key) is not None
        ]
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [float(point[1]) for point in points],
            marker="o",
            label=f"Client {client_id}",
        )

    ax.set_title(f"{task.title()} Client Train {ylabel}")
    ax.set_xlabel("Federated round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_server_metric(
    rows: list[dict[str, str]],
    summary: dict[str, object],
    task: str,
    metric: str,
    output_path: Path,
) -> None:
    ylabel = _metric_label(task, metric)
    train_key = f"train_{metric}"
    val_key = f"val_{metric}"
    test_key = f"test_{metric}"
    rounds = [int(float(row["round"])) for row in rows if row.get("round")]

    fig, ax = plt.subplots(figsize=(8, 5))
    train_points = [
        (int(float(row["round"])), _float(row, train_key))
        for row in rows
        if _float(row, train_key) is not None
    ]
    val_points = [
        (int(float(row["round"])), _float(row, val_key))
        for row in rows
        if _float(row, val_key) is not None
    ]

    if train_points:
        ax.plot(
            [point[0] for point in train_points],
            [float(point[1]) for point in train_points],
            marker="o",
            label="Weighted client train aggregate",
        )
    if val_points:
        ax.plot(
            [point[0] for point in val_points],
            [float(point[1]) for point in val_points],
            marker="o",
            label="Server validation",
        )
    if rounds and test_key in summary:
        test_round = int(summary.get("test_round", max(rounds)))
        test_label = "Best-val test" if summary.get("test_model") == "best_validation" else "Final server test"
        ax.scatter(
            [test_round],
            [float(summary[test_key])],
            marker="X",
            s=110,
            color="black",
            label=test_label,
            zorder=5,
        )

    ax.set_title(f"{task.title()} Global Server {ylabel}")
    ax.set_xlabel("Federated round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _default_paths(task: str, mode: str, result_name: str | None) -> tuple[Path, Path, Path]:
    infix = "_iid" if mode == "iid" else ""
    suffix = f"_{result_name}" if result_name else ""
    return (
        Path(f"artifacts/{task}{infix}{suffix}_federated_history.csv"),
        Path(f"artifacts/{task}{infix}{suffix}_federated_client_history.csv"),
        Path(f"artifacts/{task}{infix}{suffix}_federated_summary.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot client and global Flower metrics separately.")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--mode", choices=["iid", "subject"], default="iid")
    parser.add_argument("--result-name", default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--client-history", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/federated_metric_plots"))
    args = parser.parse_args()

    default_history, default_client_history, default_summary = _default_paths(args.task, args.mode, args.result_name)
    history_path = args.history or default_history
    client_history_path = args.client_history or default_client_history
    summary_path = args.summary or default_summary

    server_rows = _read_history(history_path)
    client_rows = _read_history(client_history_path)
    summary = _read_summary(summary_path)

    mode_label = "iid" if args.mode == "iid" else "subject_owned"
    suffix = f"_{args.result_name}" if args.result_name else ""
    prefix = f"{args.task}_{mode_label}{suffix}"
    outputs = {
        "client_loss": args.output_dir / f"{prefix}_client_train_loss.png",
        "client_score": args.output_dir / f"{prefix}_client_train_score.png",
        "server_loss": args.output_dir / f"{prefix}_global_server_loss.png",
        "server_score": args.output_dir / f"{prefix}_global_server_score.png",
    }
    if args.task == "regression" and (
        any(row.get("train_r2") not in (None, "") for row in client_rows)
        or any(row.get("val_r2") not in (None, "") for row in server_rows)
        or summary.get("test_r2") is not None
    ):
        outputs["client_r2"] = args.output_dir / f"{prefix}_client_train_r2.png"
        outputs["server_r2"] = args.output_dir / f"{prefix}_global_server_r2.png"

    _plot_client_metric(client_rows, args.task, "loss", outputs["client_loss"])
    _plot_client_metric(client_rows, args.task, "score", outputs["client_score"])
    _plot_server_metric(server_rows, summary, args.task, "loss", outputs["server_loss"])
    _plot_server_metric(server_rows, summary, args.task, "score", outputs["server_score"])
    if "client_r2" in outputs:
        _plot_client_metric(client_rows, args.task, "r2", outputs["client_r2"])
        _plot_server_metric(server_rows, summary, args.task, "r2", outputs["server_r2"])

    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
