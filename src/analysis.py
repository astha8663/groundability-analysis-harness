import json
import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from src.load import load_class_names

RESULTS_DIR = Path("results")
def run_analysis(
    baseline_file,
    groundability_file,
    aggregation="mean"
):

    with open(
        baseline_file,
        "r"
    ) as f:

        baseline_results = json.load(f)

    with open(
        groundability_file,
        "r"
    ) as f:

        groundability_results = json.load(f)

    groundability_values = []
    error_values = []
    class_ids = []

    for cls in groundability_results:

        groundability = (
            groundability_results[cls][aggregation]
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
        f"groundability_vs_error_{aggregation}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "spearman": float(correlation),
        "p_value": float(p_value),
        "n_classes": len(class_ids)
    }
results_55_mean = run_analysis(
    "results/baseline_results_55_5.json",
    "results/groundability_scores_55_5.json",
    "mean"
)

results_55_max = run_analysis(
    "results/baseline_results_55_5.json",
    "results/groundability_scores_55_5.json",
    "max"
)

results_48_mean = run_analysis(
    "results/baseline_results_48_12.json",
    "results/groundability_scores_48_12.json",
    "mean"
)

results_48_max = run_analysis(
    "results/baseline_results_48_12.json",
    "results/groundability_scores_48_12.json",
    "max"
)

print("55/5 Mean:", results_55_mean)
print("55/5 Max:", results_55_max)
print("48/12 Mean:", results_48_mean)
print("48/12 Max:", results_48_max)


def compute_correlation(
    baseline_file,
    groundability_file,
    aggregation
):
    with open(
        baseline_file,
        "r"
    ) as f:
        baseline_results = json.load(f)
    with open(
        groundability_file,
        "r"
    ) as f:
        groundability_results = json.load(f)
    groundability_values = []
    error_values = []
    for cls in groundability_results:
        groundability = (
            groundability_results[cls][aggregation]
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
    correlation, _ = spearmanr(
        groundability_values,
        error_values
    )
    return (
        correlation,
        len(groundability_values)
    )
corr_55_mean, n_55 = compute_correlation(
    "results/baseline_results_55_5.json",
    "results/groundability_scores_55_5.json",
    "mean"
)
corr_55_max, _ = compute_correlation(
    "results/baseline_results_55_5.json",
    "results/groundability_scores_55_5.json",
    "max"
)
corr_48_mean, n_48 = compute_correlation(
    "results/baseline_results_48_12.json",
    "results/groundability_scores_48_12.json",
    "mean"
)
corr_48_max, _ = compute_correlation(
    "results/baseline_results_48_12.json",
    "results/groundability_scores_48_12.json",
    "max"
)   

rows = [
    {
        "split": "55/5",
        "seed": 42,
        "aggregation": "mean",
        "n_classes": n_55,
        "spearman": corr_55_mean
    },
    {
        "split": "55/5",
        "seed": 42,
        "aggregation": "max",
        "n_classes": n_55,
        "spearman": corr_55_max
    },
    {
        "split": "48/12",
        "seed": 42,
        "aggregation": "mean",
        "n_classes": n_48,
        "spearman": corr_48_mean
    },
    {
        "split": "48/12",
        "seed": 42,
        "aggregation": "max",
        "n_classes": n_48,
        "spearman": corr_48_max
    }
]

df = pd.DataFrame(rows)
df.to_csv(
    "results/phase3_results.csv",
    index=False
)
print(df)
