"""Sequential held-out evaluation of Ultralytics detection checkpoints."""

import csv
import gc
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_best_checkpoints(
    runs_dir: str | Path,
    data_yaml: str | Path,
    *,
    batch: int = 16,
    device: str | int | None = None,
    imgsz: int = 640,
    workers: int = 8,
) -> list[dict]:
    """Test immediate exp_* runs only; save a fresh report directory per call.

    Outputs: summary.csv (including failures/missing checkpoints), per-run
    metrics.json (aggregate and per-class metrics, speed, evaluation settings),
    and Ultralytics plots. Returns the summary rows. Interrupts propagate;
    ordinary model failures are recorded and evaluation continues.
    """
    import torch
    import yaml
    from ultralytics import YOLO

    runs_dir = Path(runs_dir).expanduser().resolve()
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not runs_dir.is_dir():
        raise NotADirectoryError(runs_dir)
    with data_yaml.open() as handle:
        dataset = yaml.safe_load(handle)
    if not isinstance(dataset, dict) or not dataset.get("test"):
        raise ValueError(f"Dataset YAML must define a nonempty 'test' split: {data_yaml}")
    if batch < 1:
        raise ValueError("Evaluation batch must be a positive integer.")

    # Deliberately do not recurse into archive/, before_patience/, or other non-experimental trial folders.
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("exp_"))
    if not runs:
        raise FileNotFoundError(f"No exp_* directories found in {runs_dir}")
    output_dir = runs_dir / "test_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Evaluating {len(runs)} runs. Results: {output_dir}", flush=True)
    rows = []

    def save_summary():
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    for index, run in enumerate(runs, start=1):
        checkpoint = run / "weights" / "best.pt"
        row = {"run": run.name, "checkpoint": str(checkpoint), "status": "", "error": ""}
        model = metrics = None
        print(f"[{index}/{len(runs)}] {run.name}", flush=True)
        try:
            if not checkpoint.is_file():
                row.update(status="missing", error="weights/best.pt not found")
                logger.warning("Skipping missing checkpoint: %s", checkpoint)
                continue
            model = YOLO(str(checkpoint))
            settings = dict(
                data=str(data_yaml), split="test", batch=batch, imgsz=imgsz,
                device=device, workers=workers, project=str(output_dir),
                name=run.name, exist_ok=False, plots=True, save_json=False,
                conf=0.001, iou=0.7, augment=False, half=False,
            )
            metrics = model.val(**settings)
            aggregate = {key: float(value) for key, value in metrics.results_dict.items()}
            speed = {key: float(value) for key, value in metrics.speed.items()}
            report_dir = Path(metrics.save_dir)
            report = {
                "run": run.name, "checkpoint": str(checkpoint),
                "settings": settings, "metrics": aggregate,
                "per_class": metrics.summary(), "speed_ms_per_image": speed,
            }
            with (report_dir / "metrics.json").open("w") as handle:
                json.dump(report, handle, indent=2)
            row.update(status="ok", results_dir=str(report_dir), **aggregate)
            row.update({f"speed_{key}_ms": value for key, value in speed.items()})
        except Exception as exc:
            row.update(status="error", error=str(exc))
            logger.exception("Test evaluation failed for %s", checkpoint)
        finally:
            rows.append(row)
            save_summary()  # Preserve progress after every checkpoint.
            model = metrics = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    counts = {status: sum(row["status"] == status for row in rows) for status in ("ok", "missing", "error")}
    print(f"Finished: {counts}. Summary: {output_dir / 'summary.csv'}", flush=True)
    return rows
