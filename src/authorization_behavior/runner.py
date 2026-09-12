# src/authorization_behavior/runner.py

from authorization_behavior.models import ModelClient
from authorization_behavior.parsing import parse_decision
from authorization_behavior.prompts import (
    conflicting_context,
    render_prompt,
)
from authorization_behavior.schema import Scenario


def run_experiment(
    scenarios: list[Scenario],
    model: ModelClient,
) -> list[dict]:
    records = []

    for scenario in scenarios:
        conditions = {
            "clean": None,
            "conflict": conflicting_context(scenario.ground_truth),
        }

        for condition, additional_context in conditions.items():
            prompt = render_prompt(
                scenario,
                additional_context=additional_context,
            )

            response = model.generate(prompt)

            decision = parse_decision(response.text)

            records.append(
                {
                    "scenario_id": scenario.id,
                    "family": scenario.family,
                    "variant": scenario.variant,
                    "condition": condition,
                    "model": response.model,
                    "ground_truth": scenario.ground_truth,
                    "decision": decision.value,
                    "correct": (decision.value == scenario.ground_truth),
                    "prompt": prompt,
                    "raw_response": response.text,
                    "latency_seconds": (response.latency_seconds),
                }
            )

    return records
