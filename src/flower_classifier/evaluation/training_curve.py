"""Plot the validation metric recorded during a single training run."""

from numbers import Integral
from pathlib import Path


def plot_training_map(results_csv: str | Path, patience: int | None = None):
    """Return a Matplotlib figure of box mAP50-95 by epoch.

    A positive patience marks the first recorded epoch where YOLO26
    would have stopped, estimated from the rounded CSV values. Omit
    patience or pass 0 to disable the marker.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if patience is not None and (
        isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 0
    ):
        raise ValueError("patience must be a nonnegative integer or None")

    results = pd.read_csv(Path(results_csv).expanduser())
    results.columns = results.columns.str.strip()
    metric = "metrics/mAP50-95(B)"
    missing = {"epoch", metric} - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if results.empty:
        raise ValueError("The results CSV contains no training epochs")

    results["epoch"] = pd.to_numeric(results["epoch"])
    results[metric] = pd.to_numeric(results[metric])
    results = results.sort_values("epoch")
    epochs, scores = results["epoch"], results[metric]

    fig, ax = plt.subplots()
    ax.plot(epochs, scores)
    ax.set(xlabel="Epoch", ylabel="Validation box mAP50–95")

    if patience:
        best_score = 0.0
        best_epoch = 0
        for epoch, score in zip(epochs, scores):
            if pd.isna(score):
                continue
            if score > best_score or best_score == 0:
                best_score = score
                best_epoch = epoch
            if epoch - best_epoch >= patience:
                ax.axvline(
                    epoch, color="tab:red", linestyle="--",
                    label=f"Estimated stop at epoch {epoch} (patience={patience})",
                )
                ax.legend()
                break

    fig.tight_layout()
    return fig
