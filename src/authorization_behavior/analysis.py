# src/authorization_behavior/analysis.py

import numpy as np
import pandas as pd

# src/authorization_behavior/analysis.py


def build_flip_signature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    decisions = df.pivot(
        index="scenario_id",
        columns="condition",
        values="decision",
    )

    if "clean" not in decisions.columns:
        raise ValueError("Expected a clean condition.")

    clean = decisions["clean"]

    return decisions.drop(columns="clean").ne(clean, axis=0).astype(int)


def build_scenario_similarity(
    signatures: pd.DataFrame,
) -> pd.DataFrame:
    values = signatures.to_numpy()
    n = len(signatures)

    similarity = np.zeros(
        (n, n),
        dtype=float,
    )

    for i in range(n):
        for j in range(n):
            similarity[i, j] = (values[i] == values[j]).mean()

    return pd.DataFrame(
        similarity,
        index=signatures.index,
        columns=signatures.index,
    )


def compare_conditions(
    df: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> dict:
    subset = df[df["condition"].isin([condition_a, condition_b])]

    wide = subset.pivot(
        index="scenario_id",
        columns="condition",
        values=[
            "decision",
            "correct",
            "ground_truth",
        ],
    )

    a_correct = wide["correct"][condition_a]
    b_correct = wide["correct"][condition_b]

    a_decision = wide["decision"][condition_a]
    b_decision = wide["decision"][condition_b]

    changed = a_decision != b_decision

    a_to_b_error = a_correct & ~b_correct

    a_to_b_repair = ~a_correct & b_correct

    both_correct = a_correct & b_correct

    both_wrong = ~a_correct & ~b_correct

    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "accuracy_a": a_correct.mean(),
        "accuracy_b": b_correct.mean(),
        "accuracy_delta": (b_correct.mean() - a_correct.mean()),
        "decision_changes": int(changed.sum()),
        "correct_to_incorrect": int(a_to_b_error.sum()),
        "incorrect_to_correct": int(a_to_b_repair.sum()),
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
    }


def build_paired_results(df: pd.DataFrame) -> pd.DataFrame:
    paired = df.pivot(
        index="scenario_id",
        columns="condition",
        values="decision",
    )

    if "clean" not in paired.columns:
        raise ValueError("Expected a 'clean' condition in experiment results.")

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
        if column
        not in {
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
        paired[f"{condition}_flipped"] = paired["clean"] != paired[condition]

        paired[f"{condition}_transition"] = paired["clean"] + " -> " + paired[condition]

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
    accuracy = df.groupby("condition")["correct"].mean().mul(100).round(1)

    for condition, value in accuracy.items():
        print(f"  {condition:<22} {value:>5.1f}%")

    print("\nFlips relative to clean:")

    conditions = [
        condition for condition in df["condition"].unique() if condition != "clean"
    ]

    for condition in conditions:
        flip_column = f"{condition}_flipped"
        flips = int(paired[flip_column].sum())

        print(f"  {condition:<22} {flips:>3}/{scenarios} ({flips / scenarios:.1%})")

    print("\nTransitions by intervention:")

    for condition in conditions:
        transition_column = f"{condition}_transition"

        print(f"\n  {condition}:")

        counts = paired[transition_column].value_counts()

        for transition, count in counts.items():
            print(f"    {transition:<18} {count}")

    print("\nAccuracy by condition and ground truth:")

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
