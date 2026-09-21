from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from authorization_behavior.analysis import (
    build_paired_results,
    print_experiment_summary,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m authorization_behavior.report_run "
            "<run_directory>"
        )

    if "runs/" in sys.argv[1]:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = Path("runs/" + sys.argv[1])

    results_path = run_dir / "results.parquet"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find: {results_path}"
        )

    df = pd.read_parquet(results_path)

    paired = build_paired_results(df)

    print_experiment_summary(
        df,
        paired,
    )


if __name__ == "__main__":
    main()