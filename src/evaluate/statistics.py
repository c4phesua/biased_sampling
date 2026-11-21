"""Tools related to statistical evaluation data"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd


def summarize_results(csv_file):
    df = pd.read_csv(csv_file)

    summary = df.groupby("sampler").agg(
        trials=("trial", "count"),
        success_rate=("success", "mean"),

        runtime_mean=("runtime", "mean"),
        runtime_std=("runtime", "std"),
        runtime_median=("runtime", "median"),

        collision_checks_mean=("edge_collision_checks", "mean"),
        collision_checks_std=("edge_collision_checks", "std"),
        collision_checks_median=("edge_collision_checks", "median"),

        path_length_mean=("path_length", "mean"),
    ).reset_index()

    return summary


def plot_metrics(df: pd.DataFrame, metric, ylabel, file_path: Union[str, Path]):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(
        column=metric,
        by="sampler",
        ax=ax
    )

    ax.set_title(f"{ylabel} Comparison")
    fig.suptitle("")
    ax.set_xlabel("Sampler")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def generate_sampler_plots(csv_file: Union[str, Path], output_dir: Union[str, Path]):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_file)

    # Runtime
    plot_metrics(
        df,
        metric="runtime",
        ylabel="Runtime (seconds)",
        file_path=output_dir / "runtime.png"
    )

    # Collision check
    plot_metrics(
        df,
        metric="edge_collision_checks",
        ylabel="Number of Edge Collision Checks",
        file_path=output_dir / "collision_checks.png"
    )

    # Individual runtime phases
    for metric, ylabel, filename in [
        ("sampling_time", "Sampling Time (seconds)", "sampling_time.png"),
        ("edge_connection_time", "Edge Connection Time (seconds)", "edge_connection_time.png"),
        ("pathfinding_time", "Pathfinding Time (seconds)", "pathfinding_time.png"),
    ]:
        plot_metrics(
            df,
            metric,
            ylabel,
            output_dir / filename
        )
