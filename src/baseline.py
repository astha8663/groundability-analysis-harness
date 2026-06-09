import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from src.load import load_class_names


def fit_projection(seen_features, seen_labels, text_bank):
    targets = text_bank[seen_labels]
    model = Ridge(alpha=1.0)
    model.fit(seen_features, targets)
    return model

def evaluate_zero_shot(
    projection,
    test_features,
    test_labels,
    text_bank,
    unseen_classes
):
    projected = projection.predict(test_features)

    unseen_embeddings = text_bank[unseen_classes]

    projected = normalize(projected)
    unseen_embeddings = normalize(unseen_embeddings)

    similarities = cosine_similarity(
        projected,
        unseen_embeddings
    )

    pred_indices = np.argmax(similarities, axis=1)

    predictions = unseen_classes[pred_indices]

    overall_accuracy = accuracy_score(
        test_labels,
        predictions
    )

    per_class_accuracy = {}

    for cls in unseen_classes:

        mask = test_labels == cls

        cls_acc = accuracy_score(
            test_labels[mask],
            predictions[mask]
        )

        per_class_accuracy[int(cls)] = cls_acc

    cm = confusion_matrix(
        test_labels,
        predictions,
        labels=unseen_classes
    )

    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm,
        "predictions": predictions
    }

from src.load import (
    load_train_features,
    load_train_labels,
    load_test_features,
    load_test_labels,
    load_full_embeddings,
    load_split,
    load_class_names
)

if __name__ == "__main__":

    # Load data
    train_features = load_train_features()
    train_labels = load_train_labels()

    test_features = load_test_features()
    test_labels = load_test_labels()

    text_bank = load_full_embeddings()
    class_info = load_class_names()

    seen_classes, unseen_classes = load_split(
        "rs55.npy",
        "ru5.npy"
    )

    print("Training projection model...")

    # Train projection
    projection = fit_projection(
        train_features,
        train_labels,
        text_bank
    )

    print("Evaluating zero-shot performance...")

    # Evaluate
    results = evaluate_zero_shot(
        projection,
        test_features,
        test_labels,
        text_bank,
        unseen_classes
    )

    print("\nRESULTS")

    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")

    print("\nPer-Class Accuracy:")

    for cls, acc in results["per_class_accuracy"].items():
        class_name = class_info[cls]["class_name"]
        print(f"{class_name}: {acc:.4f}")

    print("\nConfusion Matrix Shape:")
    print(results["confusion_matrix"])