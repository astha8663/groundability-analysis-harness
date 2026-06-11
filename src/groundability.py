import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.load import (
    load_train_features,
    load_train_labels,
    load_test_features,
    load_test_labels,
    load_part_embeddings,
    load_split
)


def fit_part_probe(
    train_features,
    train_labels,
    part_embeddings
):
    targets = part_embeddings[train_labels]
    probe = Ridge(alpha=1.0)
    probe.fit(
        train_features,
        targets
    )
    return probe

def evaluate_probe(
    probe,
    features,
    labels,
    part_embeddings
):
    true_embeddings = part_embeddings[labels]
    predicted_embeddings = probe.predict(features)
    predicted_embeddings = (
        predicted_embeddings
        /
        np.linalg.norm(
            predicted_embeddings,
            axis=1,
            keepdims=True
        )
    )
    true_embeddings = (
        true_embeddings
        /
        np.linalg.norm(
            true_embeddings,
            axis=1,
            keepdims=True
        )
    )
    cosines = np.sum(
        predicted_embeddings * true_embeddings,
        axis=1
    )
    return np.mean(cosines)

def compute_groundability_scores(
    probes,
    parts,
    test_features,
    test_labels,
    unseen_classes
):
    groundability_scores = {}
    for class_id in unseen_classes:
        groundability_scores[int(class_id)] = {}
        mask = (test_labels == class_id)
        class_features = test_features[mask]
        if len(class_features) == 0:
            continue
        for part in parts:
            embeddings = load_part_embeddings(part)
            probe = probes[part]
            predicted_embeddings = probe.predict(
                class_features
            )
            true_embedding = embeddings[class_id]
            predicted_embeddings = (
                predicted_embeddings
                /
                np.linalg.norm(
                    predicted_embeddings,
                    axis=1,
                    keepdims=True
                )
            )
            true_embedding = (
                true_embedding
                /
                np.linalg.norm(true_embedding)
            )
            cosines = predicted_embeddings @ true_embedding
            mean_groundability = np.mean(cosines)
            groundability_scores[int(class_id)][part] = (
                float(mean_groundability)
            )
    return groundability_scores

def aggregate_groundability_scores(
    groundability_scores
):
    aggregated_scores = {}
    for class_id, part_scores in groundability_scores.items():
        scores = list(
            part_scores.values()
        )
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        aggregated_scores[class_id] = {
            "mean": float(mean_score),
            "max": float(max_score)
        }
    return aggregated_scores

if __name__ == "__main__":

    train_features = load_train_features()
    train_labels = load_train_labels()

    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=0.2,
        random_state=42
    )

    parts = [
        "head",
        "hand",
        "arm",
        "hip",
        "leg",
        "foot"
    ]
    probes = {}

    print("\n===== VALIDATION SCORES =====\n")

    for part in parts:
        embeddings = load_part_embeddings(part)
        probe = fit_part_probe(
            X_train,
            y_train,
            embeddings
        )
        probes[part] = probe
        score = evaluate_probe(
            probe,
            X_val,
            y_val,
            embeddings
        )

        print(
            f"{part.capitalize()} Probe Score: {score:.4f}"
        )

    test_features = load_test_features()
    test_labels = load_test_labels()

    seen_classes, unseen_classes = load_split(
        "rs55.npy",
        "ru5.npy"
    )

    groundability_scores = compute_groundability_scores(
        probes,
        parts,
        test_features,
        test_labels,
        unseen_classes
    )
    aggregated_scores = aggregate_groundability_scores(
        groundability_scores
    )
    import json
    with open(
        "results/groundability_scores.json",
        "w"
    ) as f:
        json.dump(
            aggregated_scores,
            f,
            indent=4
        )
    print("\n===== GROUNDABILITY SCORES =====\n")

    for class_id in groundability_scores:
        print(f"Class {class_id}")
        for part in parts:
            score = groundability_scores[class_id][part]

            print(
                f"   {part}: {score:.4f}"
            )

        print()
    
    print("\n===== AGGREGATED GROUNDABILITY SCORES =====\n")
    for class_id, scores in aggregated_scores.items():
        print(
            f"Class {class_id}"
        )
        print(
            f"   Mean Groundability: {scores['mean']:.4f}"
        )
        print(
            f"   Max Groundability : {scores['max']:.4f}"
        )

    print()