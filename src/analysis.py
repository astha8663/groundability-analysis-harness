import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from src.load import load_class_names

RESULTS_DIR = Path("results")
with open(
    RESULTS_DIR /
    "baseline_results.json",
    "r"
) as f:

    baseline_results = json.load(f)

with open(
    RESULTS_DIR /
    "groundability_scores.json",
    "r"
) as f:

    groundability_results = json.load(f)

groundability_scores = groundability_results
groundability_values = []
error_values = []
class_ids = []

for cls in groundability_scores:

    groundability = (
        groundability_scores[cls]["mean"]
    )
    error = (
        baseline_results[cls]["error"]
    )
    groundability_values.append(
        groundability
    )
    error_values.append(
        error
    )
    class_ids.append(
        int(cls)
    )

correlation, p_value = spearmanr(
    groundability_values,
    error_values
)
print(
    f"Spearman Correlation: "
    f"{correlation:.4f}"
)

print(
    f"Number of Classes: "
    f"{len(class_ids)}"
)

class_info = load_class_names()
class_names = []
for cls in class_ids:

    class_names.append(
        class_info[cls]["class_name"]
    )
plt.figure(
    figsize=(10, 8)
)
plt.scatter(
    groundability_values,
    error_values
)
for i in range(
    len(class_names)
):
    plt.annotate(
        class_names[i],
        (
            groundability_values[i],
            error_values[i]
        )
    )
plt.title(
    f"Groundability vs Error\n"
    f"Spearman = {correlation:.4f}"
)
plt.xlabel(
    "Groundability Score"
)

plt.ylabel(
    "Error Rate"
)
FIGURES_DIR = Path(
    "figures"
)

FIGURES_DIR.mkdir(
    exist_ok=True
)
plt.savefig(
    FIGURES_DIR /
    "groundability_vs_error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
