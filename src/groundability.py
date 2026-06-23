import argparse
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
    unseen_classes,
    seen_classes
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
            mean_text_embedding = np.mean(
                embeddings[seen_classes],
                axis=0
            )
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
            mean_text_embedding = (
                mean_text_embedding
                /
                np.linalg.norm(mean_text_embedding)
            )
            cosine_scores = np.dot(
                predicted_embeddings,
                true_embedding  
            )
            raw_cosine = float(
                np.mean(cosine_scores)
            )
            text_baseline = float(
                np.dot(
                    mean_text_embedding,
                    true_embedding
                )
            )

            skill_score = raw_cosine - text_baseline
            groundability_scores[int(class_id)][part] = {
                "raw_cosine": raw_cosine,
                "text_baseline": text_baseline,
                "skill": skill_score
            }
    return groundability_scores

def aggregate_groundability_scores(
    groundability_scores
):
    aggregated_scores = {}

    for class_id in groundability_scores:
        part_scores = []

        for part in groundability_scores[class_id]:
            value = groundability_scores[class_id][part]

            if isinstance(value, dict):
                score = value["skill"]

            else:
                score = value

            part_scores.append(score)

        aggregated_scores[int(class_id)] = {
            "mean": float(np.mean(part_scores)),
            "max": float(np.max(part_scores))
        }

    return aggregated_scores

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))

def mean_cosine_to_target(predictions, target):
    scores = []

    for pred in predictions:
        scores.append(
            cosine_similarity(pred, target)
        )

    return float(np.mean(scores))

def compute_mean_text_embedding(
    part_embeddings,
    seen_classes
):
    seen_text = part_embeddings[seen_classes]
    mean_text = np.mean(
        seen_text,
        axis=0
    )
    return mean_text

def compute_skill_score(
    predictions,
    class_text_embedding,
    mean_text_embedding
):
    pred_to_true = mean_cosine_to_target(
        predictions,
        class_text_embedding
    )

    baseline = cosine_similarity(
        mean_text_embedding,
        class_text_embedding
    )

    skill = pred_to_true - baseline

    return {
        "pred_to_true": float(pred_to_true),
        "text_baseline": float(baseline),
        "skill": float(skill)
    }



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seen",
        default="rs55.npy"
    )

    parser.add_argument(
        "--unseen",
        default="ru5.npy"
    )
    parser.add_argument(
        "--feature_dir",
        default="shift_ntu_60_5_r"
    )
    args = parser.parse_args()
    import src.load as load
    load.FEATURE_DIR = (
        load.BASE_DIR /
        "features" /
        args.feature_dir
    )
    seen_tag = args.seen.replace(".npy", "")
    unseen_tag = args.unseen.replace(".npy", "")
    split_tag = f"{seen_tag}_{unseen_tag}"

    print(
        f"Using feature directory: {load.FEATURE_DIR}"
    )
    
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
        args.seen,
        args.unseen
    )

    groundability_scores = compute_groundability_scores(
        probes,
        parts,
        test_features,
        test_labels,
        unseen_classes,
        seen_classes
    )
    aggregated_scores = aggregate_groundability_scores(
        groundability_scores
    )
    import json
    with open(
        f"results/groundability_scores_{split_tag}.json",
        "w"
    ) as f:
        json.dump(
            aggregated_scores,
            f,
            indent=4
        )
    print("\n===== GROUNDABILITY SCORES =====\n")
    for class_id, part_scores in groundability_scores.items():
        print(f"Class {class_id}")

        for part, score_info in part_scores.items():
            print(
                f"   {part}: "
                f"raw={score_info['raw_cosine']:.4f}, "
                f"baseline={score_info['text_baseline']:.4f}, "
                f"skill={score_info['skill']:.4f}"
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