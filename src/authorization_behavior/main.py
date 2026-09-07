# src/authorization_behavior/main.py

import pandas as pd

from authorization_behavior.data import load_scenarios
from authorization_behavior.models import OllamaClient
from authorization_behavior.parsing import parse_decision
from authorization_behavior.prompts import render_prompt

def main():
    scenarios = load_scenarios("data/scenarios.yaml")

    model = OllamaClient(
        model="qwen3:4b-instruct-2507-q4_K_M",
        temperature=0.0,
    )

    records = []

    for scenario in scenarios:
        prompt = render_baseline_prompt(scenario)

        response = model.generate(prompt)
        decision = parse_decision(response.text)

        correct = decision.value == scenario.ground_truth

        records.append(
            {
                "scenario_id": scenario.id,
                "model": response.model,
                "ground_truth": scenario.ground_truth,
                "raw_response": response.text,
                "decision": decision.value,
                "correct": correct,
                "latency_seconds": response.latency_seconds,
            }
        )

        print(
            f"{scenario.id}: "
            f"{decision.value} "
            f"(expected {scenario.ground_truth}) "
            f"{'✓' if correct else '✗'}"
        )

    df = pd.DataFrame(records)

    print("\nResults")
    print(df)

    print("\nAccuracy")
    print(df["correct"].mean())

    print("\nMean latency:")
    print(df["latency_seconds"].mean())

    print("\nMedian latency:")
    print(df["latency_seconds"].median())


if __name__ == "__main__":
    main()