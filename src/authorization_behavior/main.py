# src/authorization_behavior/main.py

import pandas as pd

from authorization_behavior.analysis import (
    build_paired_results,
    print_experiment_summary,
)
from authorization_behavior.data import load_scenarios
from authorization_behavior.models import OllamaClient
from authorization_behavior.runner import run_experiment
from authorization_behavior.runs import save_run
from authorization_behavior.interventions import INTERVENTIONS


def main():
    scenarios = load_scenarios("data/scenarios.yaml")

    model = OllamaClient(
        model="qwen3:4b-instruct-2507-q4_K_M",
        temperature=0.0,
    )

    records = run_experiment(
        scenarios=scenarios,
        model=model,
        interventions=INTERVENTIONS,
    )

    df = pd.DataFrame(records)

    df = pd.DataFrame(records)

    # Save raw evidence
    save_run(
        df=df,
        experiment_id="e002",
        experiment_name="authority_conflict_100",
        model=model.model,
        temperature=model.temperature,
        prediction=None,
        notes=None,
    )

    main_df = df[df["variant"] == "original"]

    main_paired = build_paired_results(main_df)

    print_experiment_summary(
        main_df,
        main_paired,
    )



    # save_run(
    #     df=df,
    #     experiment_id="e001",
    #     experiment_name="generic_conflict",
    #     model=model.model,
    #     temperature=model.temperature,
    #     prediction=(
    #         "Clean authorization accuracy will remain near 100%. "
    #         "Generic conflicting context may cause some decisions "
    #         "to change, but I expect most cases to remain policy-compliant."
    #     ),
    #     notes=(
    #         "Initial 20 independent authorization scenarios. "
    #         "Conflict condition adds a generic colleague recommendation "
    #         "opposing the ground-truth decision."
    #     ),
    # )


if __name__ == "__main__":
    main()
