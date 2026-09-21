import pandas as pd

from authorization_behavior.analysis import (
    build_flip_overlap_crosstab,
    build_flip_overlap_details,
)


df = pd.read_parquet(
    "runs/e002_20260915T202315Z/results.parquet"
)


# --------------------------------------------------
# Aggregate overlap table
# --------------------------------------------------

overlap = build_flip_overlap_crosstab(
    df,
    "user_state_conflict",
    "emphatic_conflict",
)

print("\nUser-state vs emphatic flip overlap:")
print(overlap)


# --------------------------------------------------
# Per-scenario overlap details
# --------------------------------------------------

details = build_flip_overlap_details(
    df,
    "user_state_conflict",
    "emphatic_conflict",
)


# Keep only scenarios that flipped under BOTH
overlap_details = details[
    details["user_state_conflict_flip"]
    & details["emphatic_conflict_flip"]
].copy()


# --------------------------------------------------
# Transition labels
# --------------------------------------------------

overlap_details["user_state_transition"] = (
    overlap_details["clean"]
    + " -> "
    + overlap_details["user_state_conflict_decision"]
)

overlap_details["emphatic_transition"] = (
    overlap_details["clean"]
    + " -> "
    + overlap_details["emphatic_conflict_decision"]
)


# --------------------------------------------------
# Print overlapping scenarios
# --------------------------------------------------

print("\nScenarios that flipped under both:")

print(
    overlap_details[
        [
            "scenario_id",
            "clean",
            "user_state_transition",
            "emphatic_transition",
        ]
    ].to_string(index=False)
)


print(
    f"\nTotal overlapping scenarios: "
    f"{len(overlap_details)}"
)

details["susceptibility_group"] = "neither"

details.loc[
    details["user_state_conflict_flip"]
    & ~details["emphatic_conflict_flip"],
    "susceptibility_group",
] = "user_state_only"

details.loc[
    ~details["user_state_conflict_flip"]
    & details["emphatic_conflict_flip"],
    "susceptibility_group",
] = "emphatic_only"

details.loc[
    details["user_state_conflict_flip"]
    & details["emphatic_conflict_flip"],
    "susceptibility_group",
] = "both"


metadata = (
    df[
        [
            "scenario_id",
            "family",
            "domain",
            "environment",
            "action_type",
            "risk_level",
            "ground_truth",
        ]
    ]
    .drop_duplicates("scenario_id")
)

groups = details.merge(
    metadata,
    on="scenario_id",
    how="left",
)

print(
    groups["susceptibility_group"]
    .value_counts()
)

for column in [
    "ground_truth",
    "domain",
    "risk_level",
    "action_type",
]:
    print(f"\n=== {column} ===")

    print(
        pd.crosstab(
            groups["susceptibility_group"],
            groups[column],
        )
    )

for column in [
    "ground_truth",
    "domain",
    "risk_level",
    "action_type",
]:
    print(f"\n=== {column} proportions ===")

    print(
        pd.crosstab(
            groups["susceptibility_group"],
            groups[column],
            normalize="index",
        ).round(3)
    )

family_groups = (
    groups[
        [
            "family",
            "scenario_id",
            "ground_truth",
            "susceptibility_group",
        ]
    ]
    .sort_values(
        ["family", "ground_truth"]
    )
)
print()


print(
    family_groups.to_string(
        index=False
    )
)

both_group = groups[
    groups["susceptibility_group"] == "both"
]

print()

family_counts = (
    both_group
    .groupby("family")["scenario_id"]
    .count()
    .sort_values(ascending=False)
)

print(family_counts)


scenario_text = (
    df[
        [
            "scenario_id",
            "prompt",
            "condition",
        ]
    ]
    .query("condition == 'clean'")
    .drop(columns="condition")
)

both_cases = groups[
    groups["susceptibility_group"] == "both"
].merge(
    scenario_text,
    on="scenario_id",
    how="left",
)

for _, row in both_cases.iterrows():
    print("\n" + "=" * 80)
    print(row["scenario_id"])
    print(row["susceptibility_group"])
    print(row["prompt"])


def build_family_susceptibility(
    groups: pd.DataFrame,
    susceptible_groups: set[str],
) -> pd.DataFrame:
    data = groups.copy()

    data["susceptible"] = (
        data["susceptibility_group"]
        .isin(susceptible_groups)
    )

    paired = data.pivot(
        index="family",
        columns="ground_truth",
        values="susceptible",
    )

    return paired

user_state_pairs = build_family_susceptibility(
    groups,
    {
        "both",
        "user_state_only",
    },
)

print("\nUser-state susceptibility by matched family:")

print(
    pd.crosstab(
        user_state_pairs["ALLOW"],
        user_state_pairs["DENY"],
        rownames=["ALLOW susceptible"],
        colnames=["DENY susceptible"],
    )
)

from statsmodels.stats.contingency_tables import mcnemar


table = [
    [
        ((user_state_pairs["ALLOW"] == False)
         & (user_state_pairs["DENY"] == False)).sum(),

        ((user_state_pairs["ALLOW"] == False)
         & (user_state_pairs["DENY"] == True)).sum(),
    ],
    [
        ((user_state_pairs["ALLOW"] == True)
         & (user_state_pairs["DENY"] == False)).sum(),

        ((user_state_pairs["ALLOW"] == True)
         & (user_state_pairs["DENY"] == True)).sum(),
    ],
]

print(table)

result = mcnemar(
    table,
    exact=True,
)

print(
    f"McNemar exact p = {result.pvalue:.6f}"
)

emphatic_pairs = build_family_susceptibility(
    groups,
    {
        "both",
        "emphatic_only",
    },
)

print(
    pd.crosstab(
        emphatic_pairs["ALLOW"],
        emphatic_pairs["DENY"],
        rownames=["ALLOW susceptible"],
        colnames=["DENY susceptible"],
    )
)