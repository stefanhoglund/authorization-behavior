# src/authorization_behavior/main.py

import pandas as pd

from authorization_behavior.data import load_scenarios
from authorization_behavior.models import OllamaClient
from authorization_behavior.parsing import parse_decision
from authorization_behavior.prompts import render_prompt, conflicting_context
from authorization_behavior.schema import Scenario

def get_scenario(
    scenarios: list[Scenario],
    scenario_id: str,
) -> Scenario:
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return scenario

    raise ValueError(
        f"Scenario not found: {scenario_id}"
    )

def print_experiment_summary(df: pd.DataFrame, paired: pd.DataFrame) -> None:
    total = len(paired)
    flips = int(paired["flipped"].sum())

    print("\n=== Experiment Summary ===")

    print(f"\nScenarios: {total}")
    print(f"Flips:     {flips}/{total} ({flips / total:.1%})")

    print("\nAccuracy by condition:")
    accuracy = (
        df.groupby("condition")["correct"]
        .mean()
        .mul(100)
        .round(1)
    )

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

    print("\nScenario-level results:")
    display = paired[
        ["clean", "conflict", "transition", "flipped"]
    ].copy()

    print(display.to_string())


def main():
    scenarios = load_scenarios("data/scenarios.yaml")

    model = OllamaClient(
        model="qwen3:4b-instruct-2507-q4_K_M",
        temperature=0.0,
    )

    records = []

    for scenario in scenarios:
        conditions = {
            "clean": None,
            "conflict": conflicting_context(scenario.ground_truth),
        }

        for condition, context in conditions.items():
            prompt = render_prompt(
                scenario,
                additional_context=context,
            )

            response = model.generate(prompt)
            decision = parse_decision(response.text)

            records.append(
                {
                    "scenario_id": scenario.id,
                    "condition": condition,
                    "ground_truth": scenario.ground_truth,
                    "decision": decision.value,
                    "correct": decision.value == scenario.ground_truth,
                    "raw_response": response.text,
                    "latency_seconds": response.latency_seconds,
                }
            )

    df = pd.DataFrame(records)

    paired = df.pivot(
        index="scenario_id",
        columns="condition",
        values="decision",
    )

    paired["flipped"] = (
        paired["clean"] != paired["conflict"]
    )

    print(paired)
    print("\nNumber of flips:", paired["flipped"].sum())
    print(
        df.groupby("condition")["correct"]
        .mean()
    )

    paired["transition"] = (
        paired["clean"] + " -> " + paired["conflict"]
    )

    print("\nTransitions:")
    print(paired["transition"].value_counts())

    results = (
        df.groupby(["condition", "ground_truth"])["correct"]
        .mean()
        .unstack()
    )

    print(results)


    df.groupby(["family", "condition"])["correct"].mean()


    # for scenario in scenarios:
    #     prompt = render_baseline_prompt(scenario)

    #     response = model.generate(prompt)
    #     decision = parse_decision(response.text)

    #     correct = decision.value == scenario.ground_truth

    #     records.append(
    #         {
    #             "scenario_id": scenario.id,
    #             "model": response.model,
    #             "ground_truth": scenario.ground_truth,
    #             "raw_response": response.text,
    #             "decision": decision.value,
    #             "correct": correct,
    #             "latency_seconds": response.latency_seconds,
    #         }
    #     )

    #     print(
    #         f"{scenario.id}: "
    #         f"{decision.value} "
    #         f"(expected {scenario.ground_truth}) "
    #         f"{'✓' if correct else '✗'}"
    #     )

    # df = pd.DataFrame(records)

    # print("\nResults")
    # print(df)

    # print("\nAccuracy")
    # print(df["correct"].mean())

    # print("\nMean latency:")
    # print(df["latency_seconds"].mean())

    # print("\nMedian latency:")
    # print(df["latency_seconds"].median())

    scenario_005 = get_scenario(
        scenarios,
        "baseline_005",
    )
    
    for i in range(5):
        response = model.generate(
            render_prompt(
                scenario_005,
                additional_context=conflicting_context(
                    scenario_005.ground_truth
                ),
            )
        )
        print(i, response.text)

if __name__ == "__main__":
    main()