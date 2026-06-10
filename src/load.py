import json 
from pathlib import Path

import numpy as np

BASE_DIR = Path("data")

FEATURE_DIR = BASE_DIR / "features"
EMBEDDING_DIR = BASE_DIR / "embeddings"
SPLIT_DIR = BASE_DIR / "splits"

def load_train_features():
    path = FEATURE_DIR / "train.npy"
    features = np.load(path)
    return features

def load_train_labels():
    path = FEATURE_DIR / "train_label.npy"
    labels = np.load(path)
    return labels

def load_test_features():
    path = FEATURE_DIR / "ztest.npy"
    features = np.load(path)
    return features

def load_test_labels():
    path = FEATURE_DIR / "z_label.npy"
    labels = np.load(path)
    return labels

def load_split(seen_file, unseen_file):
    seen_path = SPLIT_DIR / seen_file
    unseen_path = SPLIT_DIR / unseen_file

    seen_classes = np.load(seen_path)
    unseen_classes = np.load(unseen_path)

    return seen_classes, unseen_classes

def load_full_embeddings():
    path = EMBEDDING_DIR / "lb_60.npy"
    embeddings = np.load(path)
    return embeddings

def load_part_embeddings():
    part_embeddings = {
        "head": np.load(EMBEDDING_DIR / "head_embeddings.npy"),
        "hand": np.load(EMBEDDING_DIR / "hand_embeddings.npy"),
        "arm": np.load(EMBEDDING_DIR / "arm_embeddings.npy"),
        "hip": np.load(EMBEDDING_DIR / "hip_embeddings.npy"),
        "leg": np.load(EMBEDDING_DIR / "leg_embeddings.npy"),
        "foot": np.load(EMBEDDING_DIR / "foot_embeddings.npy"),
    }
    return part_embeddings

def load_class_names():

    path = BASE_DIR / "descriptions.json"

    with open(path, "r") as f:
        descriptions = json.load(f)

    return descriptions["classes"]

def load_part_embeddings(part_name):

    path = EMBEDDING_DIR / f"{part_name}_embeddings.npy"

    return np.load(path)

def verify_alignment(
    full_embeddings,
    class_descriptions,
    train_labels,
    test_labels,
    unseen_classes,
):
    assert full_embeddings.shape[0] == 60, (
        f"Expected 60 embedding rows, got {full_embeddings.shape[0]}"
    )
    assert len(class_descriptions) == 60, (
        f"Expected 60 class descriptions, got {len(class_descriptions)}"
    )
    test_unique = set(np.unique(test_labels))
    unseen_set = set(unseen_classes)

    assert test_unique == unseen_set, (
        "Mismatch between unseen split and test labels"
    )
    train_unique = set(np.unique(train_labels))

    leakage = train_unique.intersection(unseen_set)

    assert len(leakage) == 0, (
        f"Unseen class leakage detected: {leakage}"
    )
    assert len(train_labels) > 0
    assert len(test_labels) > 0

    print("All alignment checks passed.")
    print("Unique test labels:")
    print(sorted(test_unique))

    print("Unseen split labels:")
    print(sorted(unseen_set))

if __name__ == "__main__":

    train_features = load_train_features()
    train_labels = load_train_labels()

    test_features = load_test_features()
    test_labels = load_test_labels()

    full_embeddings = load_full_embeddings()
    head = load_part_embeddings("head")

    print("Train Features:", train_features.shape)
    print("Train Labels:", train_labels.shape)

    print("Test Features:", test_features.shape)
    print("Test Labels:", test_labels.shape)

    print("Full Embeddings:", full_embeddings.shape)
    print(head.shape)

    class_descriptions = load_class_names()

    seen_classes, unseen_classes = load_split(
        "rs55.npy",
        "ru5.npy"
    )

    verify_alignment(
        full_embeddings,
        class_descriptions,
        train_labels,
        test_labels,
        unseen_classes
    )