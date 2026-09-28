from pathlib import Path

import yaml

from authorization_behavior.schema import Scenario


def load_scenarios(
    path: str | Path,
) -> list[Scenario]:
    path = Path(path)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    # Support either:
    #
    # - id: ...
    # - id: ...
    #
    # or:
    #
    # scenarios:
    #   - id: ...
    #
    if isinstance(data, dict):
        if "scenarios" in data:
            data = data["scenarios"]
        else:
            raise ValueError(
                f"Expected a list of scenarios or a 'scenarios' key in {path}"
            )

    if not isinstance(data, list):
        raise ValueError(
            f"Scenario file must contain a list, got {type(data).__name__}"
        )

    return [Scenario(**item) for item in data]


def apply_selection(
    scenarios: list[Scenario],
    selection_path: str | Path,
) -> list[Scenario]:
    selection_path = Path(selection_path)

    with selection_path.open("r") as f:
        selection = yaml.safe_load(f)

    # Your current user_state_focus manifest may use
    # "susceptible". Also support more generic names.
    scenario_ids = None

    for key in (
        "scenario_ids",
        "susceptible",
        "ids",
    ):
        if key in selection:
            scenario_ids = selection[key]
            break

    if scenario_ids is None:
        raise ValueError(
            f"Could not find scenario IDs in "
            f"{selection_path}. Expected one of: "
            f"scenario_ids, susceptible, ids"
        )

    selected_ids = set(scenario_ids)

    selected = [scenario for scenario in scenarios if scenario.id in selected_ids]

    missing = selected_ids - {scenario.id for scenario in selected}

    if missing:
        raise ValueError(
            "Selection contains unknown scenario IDs: " + ", ".join(sorted(missing))
        )

    return selected
