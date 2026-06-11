import numpy as np

from src.groundability import (
    fit_part_probe,
    evaluate_probe,
    aggregate_groundability_scores
)

def test_perfect_probe_fit():
    train_features = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    labels = np.array([0, 1, 2])
    part_embeddings = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    probe = fit_part_probe(
        train_features,
        labels,
        part_embeddings
    )
    score = evaluate_probe(
        probe,
        train_features,
        labels,
        part_embeddings
    )
    assert score > 0.90

def test_noise_probe_fit():
    np.random.seed(42)
    features = np.random.randn(
        100,
        10
    )
    labels = np.random.randint(
        0,
        3,
        100
    )
    part_embeddings = np.random.randn(
        3,
        20
    )
    probe = fit_part_probe(
        features,
        labels,
        part_embeddings
    )
    score = evaluate_probe(
        probe,
        features,
        labels,
        part_embeddings
    )
    assert score < 0.95

def test_seen_only():
    seen_labels = np.array([
        0,
        1,
        2,
        3
    ])
    unseen_classes = np.array([
        4,
        5
    ])
    overlap = np.intersect1d(
        seen_labels,
        unseen_classes
    )
    assert len(overlap) == 0

def test_aggregation():
    groundability_scores = {
        0: {
            "head": 1,
            "hand": 2,
            "arm": 3,
            "hip": 4,
            "leg": 5,
            "foot": 6
        }
    }
    aggregated = aggregate_groundability_scores(
        groundability_scores
    )
    assert aggregated[0]["mean"] == 3.5
    assert aggregated[0]["max"] == 6