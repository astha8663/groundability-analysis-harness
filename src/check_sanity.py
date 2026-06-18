import json
from pathlib import Path

import numpy as np
import pandas as pd

def check_groundability_scores():
    files = [
        "groundability_scores_55_5.json",
        "groundability_scores_48_12.json"
    ]
    passed = True
    for filename in files:
        with open(
            Path("results") / filename,
            "r"
        ) as f:
            scores = json.load(f)

        for cls in scores:
            for aggregation in [
                "mean",
                "max"
            ]:
                value = (
                    scores[cls][aggregation]
                )
                if not np.isfinite(value):
                    print(
                        f"FAIL: {filename}, "
                        f"class {cls}, "
                        f"{aggregation} is not finite"
                    )
                    passed = False
                if (
                    value < -1
                    or
                    value > 1
                ):
                    print(
                        f"FAIL: {filename}, "
                        f"class {cls}, "
                        f"{aggregation} out of range"
                    )
                    passed = False
    if passed:
        print(
            "PASS: Groundability scores valid"
        )

    return passed

def check_baseline_accuracy():
    files = {
        "baseline_results_55_5.json": 5,
        "baseline_results_48_12.json": 12
    }
    passed = True
    for filename, num_unseen in files.items():
        with open(
            Path("results") / filename,
            "r"
        ) as f:
            results = json.load(f)
        accuracies = []
        for cls in results:
            accuracies.append(
                results[cls]["accuracy"]
            )
        overall_accuracy = np.mean(
            accuracies
        )
        chance = 1 / num_unseen
        if overall_accuracy <= chance:
            print(
                f"FAIL: {filename} "
                f"accuracy "
                f"{overall_accuracy:.4f} "
                f"is near chance "
                f"{chance:.4f}"
            )
            passed = False
    if passed:
        print(
            "PASS: Baseline accuracy sane"
        )
    return passed

def check_results_table():
    df = pd.read_csv(
        "results/phase3_results.csv"
    )
    passed = True
    expected_rows = 4
    if len(df) != expected_rows:
        print(
            f"FAIL: Expected "
            f"{expected_rows} rows "
            f"but found "
            f"{len(df)}"
        )
        passed = False
    if (
        df.isnull()
        .values
        .any()
    ):
        print(
            "FAIL: Results table "
            "contains missing values"
        )

        passed = False
    if passed:
        print(
            "PASS: Results table complete"
        )

    return passed

if __name__ == "__main__":
    check1 = (
        check_groundability_scores()
    )
    check2 = (
        check_baseline_accuracy()
    )
    check3 = (
        check_results_table()
    )
    if (
        check1
        and
        check2
        and
        check3
    ):
        print("All checks passed")
    else:
        print("Sanity check failed")