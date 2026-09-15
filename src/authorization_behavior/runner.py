# src/authorization_behavior/runner.py

from __future__ import annotations

from tqdm import tqdm

from authorization_behavior.interventions import Intervention
from authorization_behavior.models import ModelClient
from authorization_behavior.parsing import parse_decision
from authorization_behavior.prompts import render_prompt
from authorization_behavior.schema import Scenario


def run_experiment(
    scenarios: list[Scenario],
    model: ModelClient,
    interventions: list[Intervention],
) -> list[dict]:
    total_runs = len(scenarios) * len(interventions)

    records: list[dict] = []
    correct_count = 0

    with tqdm(
        total=total_runs,
        desc="Running experiment",
        unit="run",
    ) as progress:

        for scenario in scenarios:
            for intervention in interventions:
                additional_context = intervention.render(
                    scenario.ground_truth
                )

                prompt = render_prompt(
                    scenario,
                    additional_context=additional_context,
                )

                response = model.generate(prompt)
                decision = parse_decision(response.text)

                correct = (
                    decision.value == scenario.ground_truth
                )

                if correct:
                    correct_count += 1

                records.append(
                    {
                        "scenario_id": scenario.id,
                        "family": scenario.family,
                        "variant": scenario.variant,
                        "domain": scenario.domain,
                        "environment": scenario.environment,
                        "action_type": scenario.action_type,
                        "risk_level": scenario.risk_level,
                        "condition": intervention.name,
                        "model": response.model,
                        "ground_truth": scenario.ground_truth,
                        "decision": decision.value,
                        "correct": correct,
                        "prompt": prompt,
                        "raw_response": response.text,
                        "latency_seconds": response.latency_seconds,
                    }
                )

                progress.update(1)

                progress.set_postfix(
                    scenario=scenario.id,
                    condition=intervention.name,
                    accuracy=f"{correct_count / len(records):.1%}",
                )

    return records