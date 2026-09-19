from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_scenario_similarity(
    similarity: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(
        similarity.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    ax.set_title("Scenario Behavioral Similarity")

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Scenario")

    step = 5

    ticks = range(
        0,
        len(similarity),
        step,
    )

    labels = [similarity.index[i] for i in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=6,
    )

    ax.set_yticks(ticks)
    ax.set_yticklabels(
        labels,
        fontsize=6,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Behavioral similarity",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)
