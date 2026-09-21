import pandas as pd
import yaml

df = pd.read_parquet(
    "runs/e002_20260915T202315Z/results.parquet"
)

paired = (
    df[
        df["condition"].isin(
            ["clean", "user_state_conflict"]
        )
    ]
    .pivot(
        index="scenario_id",
        columns="condition",
        values="decision",
    )
)

flipped = paired[
    paired["clean"]
    != paired["user_state_conflict"]
].reset_index()

flipped["transition"] = (
    flipped["clean"]
    + " -> "
    + flipped["user_state_conflict"]
)

print(flipped)

selection = {
    "experiment_id": "e003_user_state_deep_dive",
    "source_run": "e002_20260915T202315Z",
    "selection_basis": (
        "Scenarios whose decision changed between clean "
        "and user_state_conflict."
    ),
    "susceptible": flipped["scenario_id"].tolist(),
}

with open(
    "data/selections/user_state_focus.yaml",
    "w",
) as f:
    yaml.safe_dump(
        selection,
        f,
        sort_keys=False,
    )