# src/authorization_behavior/analysis.py

import pandas as pd


def build_paired_results(df: pd.DataFrame) -> pd.DataFrame:
    paired = df.pivot(
        index="scenario_id",
        columns="condition",
        values="decision",
    )

    paired["flipped"] = paired["clean"] != paired["conflict"]

    paired["transition"] = paired["clean"] + " -> " + paired["conflict"]

    metadata = (
        df[["scenario_id", "family", "variant", "ground_truth"]]
        .drop_duplicates("scenario_id")
        .set_index("scenario_id")
    )

    latency = df.pivot(
        index="scenario_id",
        columns="condition",
        values="latency_seconds",
    ).rename(
        columns={
            "clean": "clean_latency",
            "conflict": "conflict_latency",
        }
    )

    paired = paired.join(metadata)
    paired = paired.join(latency)

    return paired


def print_experiment_summary(
    df: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    total = len(paired)
    flips = int(paired["flipped"].sum())

    print("\n=== Experiment Summary ===")

    print(f"\nScenarios: {total}")
    print(f"Flips:     {flips}/{total} ({flips / total:.1%})")

    print("\nAccuracy by condition:")
    accuracy = df.groupby("condition")["correct"].mean().mul(100).round(1)

    for condition, value in accuracy.items():
        print(f"  {condition:<10} {value:>5.1f}%")

    print("\nTransitions:")
    transitions = paired["transition"].value_counts()

    for transition, count in transitions.items():
        print(f"  {transition:<18} {count}")

    print("\nAccuracy by ground truth:")
    directional = (
        df.groupby(["condition", "ground_truth"])["correct"]
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

    print("\nScenario-level results:")

    display = paired[
        [
            "clean",
            "conflict",
            "transition",
            "flipped",
            "clean_latency",
            "conflict_latency",
        ]
    ].copy()

    display["clean_latency"] = display["clean_latency"].round(3)

    display["conflict_latency"] = display["conflict_latency"].round(3)

    print(display.to_string())

    family_summary = paired.groupby("family").agg(
        variants=("flipped", "size"),
        flips=("flipped", "sum"),
    )

    family_summary["any_flip"] = family_summary["flips"] > 0

    print("\nFamily-level results:")
    print(family_summary)

    n_families = len(family_summary)
    families_with_flips = family_summary["any_flip"].sum()

    print(f"\nFamilies with flips: {families_with_flips}/{n_families}")
