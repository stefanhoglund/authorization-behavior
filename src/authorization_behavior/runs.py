# src/authorization_behavior/runs.py

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def load_run(
    run_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:

    run_dir = Path(run_dir)

    df = pd.read_parquet(run_dir / "results.parquet")

    with (run_dir / "experiment.yaml").open() as f:
        metadata = yaml.safe_load(f)

    return df, metadata


def save_run(
    df: pd.DataFrame,
    experiment_id: str,
    experiment_name: str,
    model: str,
    temperature: float,
    prediction: str | None = None,
    notes: str | None = None,
    base_dir: str | Path = "runs",
) -> Path:
    """
    Save experiment outputs and metadata.

    Returns the directory containing the saved run.
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = Path(base_dir) / f"{experiment_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Save row-level experimental data.
    results_path = run_dir / "results.parquet"

    df.to_parquet(
        results_path,
        index=False,
    )

    # Save experiment metadata.
    metadata = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "timestamp_utc": timestamp,
        "model": model,
        "generation": {
            "temperature": temperature,
        },
        "num_rows": len(df),
        "num_scenarios": df["scenario_id"].nunique(),
        "conditions": sorted(df["condition"].unique().tolist()),
        "prediction": prediction,
        "notes": notes,
    }

    metadata_path = run_dir / "experiment.yaml"

    with metadata_path.open("w") as f:
        yaml.safe_dump(
            metadata,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\nSaved run: {run_dir}")

    return run_dir
