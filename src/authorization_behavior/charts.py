from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


family = np.array([
    [29, 3],
    [16, 2],
])

fig, ax = plt.subplots(
    figsize=(6.2, 5.2),
    constrained_layout=True,
)

im = ax.imshow(
    family,
    cmap="Blues",
    aspect="equal",
)

ax.set_title(
    "User-State Susceptibility in Matched Policy Families",
    fontsize=14,
    pad=10,
)

ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["DENY robust", "DENY susceptible"],
    fontsize=11,
)

ax.set_yticks([0, 1])
ax.set_yticklabels(
    ["ALLOW robust", "ALLOW susceptible"],
    fontsize=11,
)

# Cell annotations with contrast-aware text color
threshold = family.max() / 2

for i in range(family.shape[0]):
    for j in range(family.shape[1]):
        value = family[i, j]
        text_color = "white" if value > threshold else "black"
        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=18,
            color=text_color,
        )

# Optional: remove spines for cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)

# Optional: add thin grid lines between cells
ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

output_path = Path("results/figures/matched_family_user_state_clean.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_path,
    dpi=220,
    bbox_inches="tight",
)

plt.show()

overlap = np.array([
    [75, 2],
    [12, 11],
])

fig, ax = plt.subplots(
    figsize=(6.2, 5.2),
    constrained_layout=True,
)

im = ax.imshow(
    overlap,
    cmap="Blues",
    aspect="equal",
)

ax.set_title(
    "User-State and Emphatic Susceptibility",
    fontsize=14,
    pad=10,
)

ax.set_xticks([0, 1])
ax.set_xticklabels(
    [
        "No emphatic flip",
        "Emphatic flip",
    ],
    fontsize=11,
)

ax.set_yticks([0, 1])
ax.set_yticklabels(
    [
        "No user-state flip",
        "User-state flip",
    ],
    fontsize=11,
)

threshold = overlap.max() / 2

for i in range(overlap.shape[0]):
    for j in range(overlap.shape[1]):
        value = overlap[i, j]

        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=18,
            color=(
                "white"
                if value > threshold
                else "black"
            ),
        )

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks(
    np.arange(-0.5, 2, 1),
    minor=True,
)
ax.set_yticks(
    np.arange(-0.5, 2, 1),
    minor=True,
)

ax.grid(
    which="minor",
    color="white",
    linewidth=2,
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

fig.savefig(
    "results/figures/user_state_emphatic_overlap_clean.png",
    dpi=220,
    bbox_inches="tight",
)

plt.show()