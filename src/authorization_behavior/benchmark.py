from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from authorization_behavior.data import (
    load_scenarios,
)
from authorization_behavior.interventions import (
    INTERVENTIONS,
)
from authorization_behavior.model_config import (
    create_model,
    load_model_configs,
)
from authorization_behavior.runner import (
    run_experiment,
)
from authorization_behavior.runs import (
    save_run,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=Path,
        default=Path("config/models.yaml"),
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("data/scenarios.yaml"),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    scenarios = load_scenarios(args.scenarios)

    model_configs = load_model_configs(args.models)

    print(f"{len(scenarios)} scenarios")

    print(f"{len(INTERVENTIONS)} conditions")

    print(f"{len(model_configs)} models")

    total = len(scenarios) * len(INTERVENTIONS) * len(model_configs)

    print(f"{total} total generations")

    for config in model_configs:
        print(f"\n=== {config.id} ===")

        model = create_model(
            config,
            temperature=0.0,
        )

        records = run_experiment(
            scenarios=scenarios,
            model=model,
            interventions=INTERVENTIONS,
        )

        df = pd.DataFrame(records)

        df["benchmark_model_id"] = config.id

        run_path = save_run(
            df=df,
            experiment_id=(f"benchmark_{config.id}"),
            experiment_name=("Authorization Conflict Benchmark v0.1"),
            model=config.model,
            temperature=0.0,
            notes=("Frozen 100-scenario, 11-condition benchmark."),
        )

        print(f"Saved: {run_path}")


if __name__ == "__main__":
    main()
