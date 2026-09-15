# src/authorization_behavior/analysis.py

import pandas as pd


# src/authorization_behavior/analysis.py

import pandas as pd


def build_paired_results(df: pd.DataFrame) -> pd.DataFrame:
    paired = df.pivot(
        index="scenario_id",
        columns="condition",
        values="decision",
    )

    if "clean" not in paired.columns:
        raise ValueError(
            "Expected a 'clean' condition in experiment results."
        )

    metadata = (
        df[
            [
                "scenario_id",
                "family",
                "variant",
                "domain",
                "environment",
                "action_type",
                "risk_level",
                "ground_truth",
            ]
        ]
        .drop_duplicates("scenario_id")
        .set_index("scenario_id")
    )

    latency = df.pivot(
        index="scenario_id",
        columns="condition",
        values="latency_seconds",
    )

    paired = paired.join(metadata)

    # Compare each intervention against clean.
    intervention_columns = [
        column
        for column in paired.columns
        if column not in {
            "clean",
            "family",
            "variant",
            "domain",
            "environment",
            "action_type",
            "risk_level",
            "ground_truth",
        }
    ]

    for condition in intervention_columns:
        paired[f"{condition}_flipped"] = (
            paired["clean"] != paired[condition]
        )

        paired[f"{condition}_transition"] = (
            paired["clean"]
            + " -> "
            + paired[condition]
        )

    # Add latency columns with explicit names.
    for condition in latency.columns:
        paired[f"{condition}_latency"] = latency[condition]

    return paired


def print_experiment_summary(
    df: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    scenarios = df["scenario_id"].nunique()

    print("\n=== Experiment Summary ===")
    print(f"\nScenarios: {scenarios}")
    print(f"Runs:      {len(df)}")

    print("\nAccuracy by condition:")
    accuracy = (
        df.groupby("condition")["correct"]
        .mean()
        .mul(100)
        .round(1)
    )

    for condition, value in accuracy.items():
        print(f"  {condition:<22} {value:>5.1f}%")

    print("\nFlips relative to clean:")

    conditions = [
        condition
        for condition in df["condition"].unique()
        if condition != "clean"
    ]

    for condition in conditions:
        flip_column = f"{condition}_flipped"
        flips = int(paired[flip_column].sum())

        print(
            f"  {condition:<22} "
            f"{flips:>3}/{scenarios} "
            f"({flips / scenarios:.1%})"
        )

    print("\nTransitions by intervention:")

    for condition in conditions:
        transition_column = (
            f"{condition}_transition"
        )

        print(f"\n  {condition}:")

        counts = (
            paired[transition_column]
            .value_counts()
        )

        for transition, count in counts.items():
            print(
                f"    {transition:<18} {count}"
            )

    print("\nAccuracy by condition and ground truth:")

    directional = (
        df.groupby(
            ["condition", "ground_truth"]
        )["correct"]
        .mean()
        .mul(100)
        .unstack()
        .round(1)
    )

    print(directional.to_string())

    print("\nLatency by condition:")

    latency_summary = (
        df.groupby("condition")["latency_seconds"]
        .agg(["mean", "median", "min", "max"])
        .round(3)
    )

    print(latency_summary.to_string())