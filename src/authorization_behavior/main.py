# src/authorization_behavior/main.py

import pandas as pd
import argparse
from pathlib import Path

from authorization_behavior.analysis import (
    build_flip_signature,
    build_paired_results,
    build_scenario_similarity,
    print_experiment_summary,
)
from authorization_behavior.data import (
    filter_scenarios,
    load_scenarios,
    load_selection,
)

from authorization_behavior.interventions import INTERVENTIONS
from authorization_behavior.models import OllamaClient
from authorization_behavior.plotting import (
    plot_scenario_similarity,
)
from authorization_behavior.runner import run_experiment
from authorization_behavior.runs import save_run

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run authorization behavior experiments."
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("data/scenarios.yaml"),
        help=(
            "Path to the canonical scenario YAML file "
            "(default: data/scenarios.yaml)"
        ),
    )

    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help=(
            "Optional YAML file containing scenario IDs "
            "to select from the canonical scenario file."
        ),
    )

    return parser.parse_args()



def main():

    args = parse_args()

    scenarios = load_scenarios(
        args.scenarios
    )

    if args.selection is not None:
        scenario_ids = load_selection(
            args.selection
        )

        scenarios = filter_scenarios(
            scenarios,
            scenario_ids,
        )

    print(
        f"Loaded {len(scenarios)} scenarios "
        f"from {args.scenarios}"
    )

    if args.selection:
        print(
            f"Applied selection: "
            f"{args.selection}"
        )


    # scenarios = load_scenarios("data/scenarios.yaml")

    print(
        f"About to run {len(scenarios)} scenarios "
        f"x {len(INTERVENTIONS)} interventions "
        f"= {len(scenarios) * len(INTERVENTIONS)} runs"
    )

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

    signatures = build_flip_signature(df)

    similarity = build_scenario_similarity(signatures)

    plot_scenario_similarity(
        similarity,
        "results/figures/scenario_similarity_heatmap.png",
    )


if __name__ == "__main__":
    main()
