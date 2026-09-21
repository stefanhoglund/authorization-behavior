from pathlib import Path

import yaml

from authorization_behavior.schema import Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    with open(path) as f:
        data = yaml.safe_load(f)

    return [Scenario(**item) for item in data]

def filter_scenarios(
    scenarios: list[Scenario],
    scenario_ids: set[str],
) -> list[Scenario]:
    selected = [
        scenario
        for scenario in scenarios
        if scenario.id in scenario_ids
    ]

    found_ids = {
        scenario.id
        for scenario in selected
    }

    missing_ids = (
        scenario_ids - found_ids
    )

    if missing_ids:
        raise ValueError(
            "Selection contains scenario IDs "
            f"not found in scenario file: "
            f"{sorted(missing_ids)}"
        )

    return selected

def load_selection(
    path: str | Path,
) -> set[str]:
    path = Path(path)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    scenario_ids = set()

    # Support your current deep-dive selection file.
    for key in [
        "susceptible",
        "controls",
    ]:
        scenario_ids.update(
            data.get(key, [])
        )

    return scenario_ids