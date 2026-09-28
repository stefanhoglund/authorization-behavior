from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from authorization_behavior.interventions import (
    INTERVENTIONS,
)
from authorization_behavior.model_config import (
    create_model_client,
    load_model_configs,
)
from authorization_behavior.runner import (
    run_experiment,
)
from authorization_behavior.runs import save_run
from authorization_behavior.scenarios import (
    load_scenarios,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("data/scenarios.yaml"),
    )

    parser.add_argument(
        "--models",
        type=Path,
        default=Path("config/models.yaml"),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=("Run only one model ID from models.yaml"),
    )

    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help=("Run only models in this model group"),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Limit number of scenarios for smoke testing"),
    )

    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # -------------------------
    # Load scenarios
    # -------------------------

    scenarios = load_scenarios(args.scenarios)

    # Keep your existing selection
    # logic here if applicable.
    if args.selection is not None:
        scenarios = apply_selection(
            scenarios,
            args.selection,
        )

    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    # -------------------------
    # Load models
    # -------------------------

    model_configs = load_model_configs(args.models)

    if args.model is not None:
        model_configs = [config for config in model_configs if config.id == args.model]

    if args.group is not None:
        model_configs = [
            config for config in model_configs if config.group == args.group
        ]

    if not model_configs:
        raise ValueError("No models matched selection.")

    # -------------------------
    # Run summary
    # -------------------------

    n_scenarios = len(scenarios)
    n_conditions = len(INTERVENTIONS)
    n_models = len(model_configs)

    per_model = n_scenarios * n_conditions

    total = per_model * n_models

    print()
    print("Authorization Conflict Benchmark")
    print("-------------------------------")
    print(f"Scenarios:   {n_scenarios}")
    print(f"Conditions:  {n_conditions}")
    print(f"Models:      {n_models}")
    print(f"Per model:   {per_model}")
    print(f"Total calls: {total}")
    print()

    # -------------------------
    # Run each model
    # -------------------------

    for index, config in enumerate(
        model_configs,
        start=1,
    ):
        print("\n================================")
        print(f"Model {index}/{n_models}: {config.id}")
        print(f"Ollama model: {config.model}")
        print("================================")

        model = create_model_client(
            config=config,
            temperature=args.temperature,
        )

        records = run_experiment(
            scenarios=scenarios,
            interventions=INTERVENTIONS,
            model=model,
        )

        results = pd.DataFrame(records)

        run_path = save_run(
            df=results,
            experiment_id=(f"benchmark_{config.id}"),
            experiment_name=("Authorization Conflict Benchmark v0.1"),
            model=config.model,
            temperature=args.temperature,
            notes=("Frozen authorization conflict benchmark."),
        )

        print(f"\nSaved: {run_path}")


if __name__ == "__main__":
    main()

# # src/authorization_behavior/main.py

# import pandas as pd
# import argparse
# from pathlib import Path

# from authorization_behavior.analysis import (
#     build_flip_signature,
#     build_paired_results,
#     build_scenario_similarity,
#     print_experiment_summary,
# )
# from authorization_behavior.data import (
#     filter_scenarios,
#     load_scenarios,
#     load_selection,
# )

# from authorization_behavior.interventions import INTERVENTIONS
# from authorization_behavior.models import OllamaClient
# from authorization_behavior.plotting import (
#     plot_scenario_similarity,
# )
# from authorization_behavior.runner import run_experiment
# from authorization_behavior.runs import save_run

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Run authorization behavior experiments."
#     )

#     parser.add_argument(
#         "--scenarios",
#         type=Path,
#         default=Path("data/scenarios.yaml"),
#         help=(
#             "Path to the canonical scenario YAML file "
#             "(default: data/scenarios.yaml)"
#         ),
#     )

#     parser.add_argument(
#         "--selection",
#         type=Path,
#         default=None,
#         help=(
#             "Optional YAML file containing scenario IDs "
#             "to select from the canonical scenario file."
#         ),
#     )

#     return parser.parse_args()


# def main():

#     args = parse_args()

#     scenarios = load_scenarios(
#         args.scenarios
#     )

#     if args.selection is not None:
#         scenario_ids = load_selection(
#             args.selection
#         )

#         scenarios = filter_scenarios(
#             scenarios,
#             scenario_ids,
#         )

#     print(
#         f"Loaded {len(scenarios)} scenarios "
#         f"from {args.scenarios}"
#     )

#     if args.selection:
#         print(
#             f"Applied selection: "
#             f"{args.selection}"
#         )


#     # scenarios = load_scenarios("data/scenarios.yaml")

#     print(
#         f"About to run {len(scenarios)} scenarios "
#         f"x {len(INTERVENTIONS)} interventions "
#         f"= {len(scenarios) * len(INTERVENTIONS)} runs"
#     )

#     model = OllamaClient(
#         model="qwen3:4b-instruct-2507-q4_K_M",
#         temperature=0.0,
#     )


#     records = run_experiment(
#         scenarios=scenarios,
#         model=model,
#         interventions=INTERVENTIONS,
#     )

#     df = pd.DataFrame(records)

#     df = pd.DataFrame(records)

#     # Save raw evidence
#     save_run(
#         df=df,
#         experiment_id="e002",
#         experiment_name="authority_conflict_100",
#         model=model.model,
#         temperature=model.temperature,
#         prediction=None,
#         notes=None,
#     )

#     main_df = df[df["variant"] == "original"]

#     main_paired = build_paired_results(main_df)

#     print_experiment_summary(
#         main_df,
#         main_paired,
#     )

#     signatures = build_flip_signature(df)

#     similarity = build_scenario_similarity(signatures)

#     plot_scenario_similarity(
#         similarity,
#         "results/figures/scenario_similarity_heatmap.png",
#     )


# if __name__ == "__main__":
#     main()
