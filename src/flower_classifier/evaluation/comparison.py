"""Compare repeated training trials using an existing evaluation summary."""

import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path


def _model_version_and_scale(model: str) -> tuple[str, str]:
    match = re.fullmatch(r"yolo(v?\d+)([nslmx])(u?)", model, re.IGNORECASE)
    if match is None:
        return model, ""
    version, scale, suffix = match.groups()
    return f"YOLO{version.lower()}{suffix.lower()}", scale.lower()


def _version_sort_key(version: str) -> tuple[float, str]:
    match = re.fullmatch(r"YOLOv?(\d+)u?", version)
    return (int(match[1]) if match else float("inf"), version)


def get_latest_summary_csv(results_dir: str | Path) -> Path:
    """Return summary.csv from the newest timestamped evaluation directory.

    Uses directory timestamps rather than file modification times. Ignores
    unrelated folders and timestamped folders that have no summary.csv yet.
    """
    results_dir = Path(results_dir).expanduser().resolve()
    candidates = []
    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue
        try:
            timestamp = datetime.strptime(folder.name, "%Y%m%d_%H%M%S_%f")
        except ValueError:
            continue
        summary = folder / "summary.csv"
        if summary.is_file():
            candidates.append((timestamp, summary))
    if not candidates:
        raise FileNotFoundError(f"No timestamped evaluation summaries found in {results_dir}")
    return max(candidates)[1]


def comparison_graph_path(
    summary_csv: str | Path, metric: str, statistic: str, extension: str = "png"
) -> Path:
    """Return the path of a comparison graph saved beside the summary."""
    metric_name = metric.split("/")[-1]
    filename = metric_name.replace("(", "").replace(")", "")
    return Path(summary_csv).expanduser().resolve().parent / f"{filename}_{statistic}_graph.{extension}"


def compare_model_results(
    summary_csv: str | Path,
    *,
    graph_statistic: str = "mean",
    graph_metrics: Sequence[str] = ("metrics/mAP50-95(B)",),
    expected_trials: int = 3,
    csv_aggregations: Sequence[str] = ("count", "mean", "median", "std", "min", "max"),
):
    """Save model statistics and a comparison graph without running inference.

    Groups exp_<model>_<trial> names by model, using successful trials only.
    Applies csv_aggregations to each metric and speed column in the table.
    Use "mean_std" to display mean ± sample standard deviation in one column.
    Each graph shows graph_statistic and individual trials for one metric,
    with bars grouped by YOLO version and colored by scale. Incomplete
    groups are included with a warning and their actual count.
    Returns (statistics_dataframe, figures), with figures in graph_metrics order.
    Outputs sit beside summary.csv.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if graph_statistic not in {"mean", "median"}:
        raise ValueError("graph_statistic must be 'mean' or 'median'")
    if isinstance(graph_metrics, str) or not graph_metrics or len(set(graph_metrics)) != len(graph_metrics):
        raise ValueError("graph_metrics must be a nonempty sequence of unique column names")
    csv_aggregations = list(dict.fromkeys(csv_aggregations))
    if not csv_aggregations or any(
        aggregation not in {"count", "mean", "median", "std", "min", "max", "mean_std"}
        for aggregation in csv_aggregations
    ):
        raise ValueError(
            "csv_aggregations must be a nonempty sequence of count, mean, median, std, min, max, or mean_std"
        )
    summary_csv = Path(summary_csv).expanduser().resolve()
    trials = pd.read_csv(summary_csv)
    required = {"run", "status", *graph_metrics}
    if not required.issubset(trials.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(trials.columns))}")
    names = trials["run"].str.extract(r"^exp_(?P<model>.+)_(?P<trial>\d+)$")
    if names.isna().any().any():
        raise ValueError("Every run must be named exp_<model>_<trial number>.")
    trials = trials.join(names)
    trials["trial"] = trials["trial"].astype(int)
    if trials.duplicated(["model", "trial"]).any():
        raise ValueError("Duplicate model/trial rows; use one evaluation summary at a time.")
    model_groups = {model: _model_version_and_scale(model) for model in trials["model"].unique()}
    versions = sorted({version for version, _ in model_groups.values()}, key=_version_sort_key)
    scale_colors = {"n": "tab:blue", "s": "tab:orange", "m": "tab:green",
                    "l": "tab:red", "x": "tab:purple", "": "tab:gray"}
    present_scales = {scale for _, scale in model_groups.values()}
    scales = [scale for scale in scale_colors if scale in present_scales]
    models = sorted(
        model_groups,
        key=lambda model: (versions.index(model_groups[model][0]),
                           scales.index(model_groups[model][1]), model),
    )
    successful = trials.loc[trials["status"] == "ok"].copy()
    columns = [c for c in trials if c.startswith(("metrics/", "speed_"))]
    for column in columns:
        successful[column] = pd.to_numeric(successful[column], errors="raise")
    missing_metrics = [metric for metric in graph_metrics if successful[metric].notna().sum() == 0]
    if missing_metrics:
        raise ValueError(f"No usable successful results for graph metrics: {missing_metrics}")

    grouped = successful.groupby("model")
    aggregations = [name for name in csv_aggregations if name != "mean_std"]
    if "mean_std" in csv_aggregations:
        aggregations = list(dict.fromkeys([*aggregations, "mean", "std"]))
    stats = grouped[columns].agg(aggregations)
    if "mean_std" in csv_aggregations:
        for column in columns:
            formatted = []
            for mean, sd in zip(stats[column, "mean"], stats[column, "std"]):
                sd_text = f"{sd:.4f}" if pd.notna(sd) else "N/A"
                formatted.append(f"{mean:.4f} ± {sd_text}" if pd.notna(mean) else None)
            stats[column, "mean_std"] = formatted
        stats = stats.reindex(columns=pd.MultiIndex.from_product([columns, csv_aggregations]))
    stats.columns = [f"{column}__{aggregation}" for column, aggregation in stats.columns]
    stats = stats.reindex(models)
    stats.insert(0, "successful_trials", grouped.size().reindex(models, fill_value=0))
    stats.insert(1, "listed_trials", trials.groupby("model").size().reindex(models))
    incomplete = stats.loc[stats["successful_trials"] != expected_trials, "successful_trials"]
    if not incomplete.empty:
        warnings.warn(
            f"Expected {expected_trials} successful trials per model; actual counts: {incomplete.to_dict()}",
            stacklevel=2,
        )
    stats.to_csv(summary_csv.parent / "model_statistics.csv")

    figures = []
    for plot_metric in graph_metrics:
        if graph_statistic in csv_aggregations:
            values = stats[f"{plot_metric}__{graph_statistic}"]
        else:
            values = grouped[plot_metric].agg(graph_statistic).reindex(models)
        width = 0.8 / len(scales)
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.65), 5), layout="constrained")
        bar_centers = {}
        for scale_index, scale in enumerate(scales):
            scale_models = [model for model in models if model_groups[model][1] == scale]
            offset = (scale_index - (len(scales) - 1) / 2) * width
            positions = [versions.index(model_groups[model][0]) + offset for model in scale_models]
            bar_centers.update(zip(scale_models, positions))
            ax.bar(positions, values.loc[scale_models], width=width, color=scale_colors[scale],
                   label=f"Scale {scale}" if scale else "Unspecified scale")

        trial_label = "Individual trials"
        for model in models:
            center = bar_centers[model]
            points = successful.loc[successful["model"] == model, plot_metric].dropna().to_numpy()
            if len(points):
                offsets = [0] if len(points) == 1 else [
                    0.6 * width * (i / (len(points) - 1) - 0.5) for i in range(len(points))
                ]
                ax.scatter([center + offset for offset in offsets], points, color="black", s=24,
                           zorder=3, label=trial_label)
                trial_label = None
            if len(points) != expected_trials:
                ax.annotate(f"n={len(points)}", (center, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(range(len(versions)), versions)
        metric_name = plot_metric.split("/")[-1]
        metric_label = "Inference latency (ms/image)" if plot_metric == "speed_inference_ms" else metric_name
        ax.set_ylabel(metric_label)
        ax.set_title(f"Test-set Evaluation: {graph_statistic.capitalize()} {metric_label} vs. Model Version and Scale")
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        for extension in ("png", "svg"):
            fig.savefig(comparison_graph_path(summary_csv, metric_name, graph_statistic, extension), dpi=200)
        figures.append(fig)
    return stats, figures
