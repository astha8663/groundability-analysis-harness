import argparse
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
    embeddings = (
        embeddings
        /
        np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )   
    )
    return embeddings

def load_all_part_embeddings():
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
    embeddings = np.load(path)
    assert embeddings.shape == (60, 1024), (
        f"{part_name} embeddings have shape "
        f"{embeddings.shape}, expected (60,1024)"
    )
    return embeddings

def verify_alignment(
    text_bank,
    class_descriptions,
    train_features,
    train_labels,
    test_features,
    test_labels,
    unseen_classes
):
    
    assert len(train_features) == len(train_labels), (
        "Train features and labels length mismatch"
    )
    assert len(test_features) == len(test_labels), (
        "Test features and labels length mismatch"
    )
    assert text_bank.shape[0] == 60, (
        f"Expected 60 text bank rows, got {text_bank.shape[0]}"
    )
    assert not np.isnan(train_features).any(), (
    "NaN found in train features"
    )
    assert not np.isnan(test_features).any(), (
        "NaN found in test features"
    )
    assert not np.isinf(train_features).any(), (
        "Inf found in train features"
    )
    assert not np.isinf(test_features).any(), (
        "Inf found in test features"
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seen",
        default="rs55.npy"
    )

    parser.add_argument(
        "--unseen",
        default="ru5.npy"
    )
    args = parser.parse_args()

    seen_classes, unseen_classes = load_split(
        args.seen,
        args.unseen
    )

    verify_alignment(
        full_embeddings,
        class_descriptions,
        train_features,
        train_labels,
        test_features,
        test_labels,
        unseen_classes
    )