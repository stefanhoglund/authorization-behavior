from pathlib import Path

import yaml

from authorization_behavior.schema import Scenario


def load_scenarios(path: str | Path) -> list[Scenario]:
    with open(path) as f:
        data = yaml.safe_load(f)

    return [Scenario(**item) for item in data]