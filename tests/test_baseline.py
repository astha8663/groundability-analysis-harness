import numpy as np
import pytest

from src.baseline import evaluate_zero_shot
from src.load import verify_alignment

def test_identity_projection():

    class IdentityProjection:

        def predict(self, x):
            return x

    test_features = np.array([
        [1, 0],
        [0, 1]
    ])

    text_bank = np.array([
        [1, 0],
        [0, 1]
    ])

    test_labels = np.array([0, 1])

    unseen_classes = np.array([0, 1])

    results = evaluate_zero_shot(
        IdentityProjection(),
        test_features,
        test_labels,
        text_bank,
        unseen_classes
    )

    assert results["overall_accuracy"] == 1.0

def test_known_answer():

    class IdentityProjection:

        def predict(self, x):
            return x

    test_features = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    text_bank = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    test_labels = np.array([0, 1, 2])

    unseen_classes = np.array([0, 1, 2])

    results = evaluate_zero_shot(
        IdentityProjection(),
        test_features,
        test_labels,
        text_bank,
        unseen_classes
    )

    expected_predictions = np.array([0, 1, 2])

    assert np.array_equal(
        results["predictions"],
        expected_predictions
    )

    assert results["overall_accuracy"] == 1.0

def test_alignment_failure():

    text_bank = np.zeros((60, 1024))

    class_descriptions = [{} for _ in range(60)]

    train_labels = np.array([0, 1, 2])

    test_labels = np.array([3, 4])

    unseen_classes = np.array([10, 11])

    with pytest.raises(AssertionError):

        verify_alignment(
            text_bank,
            class_descriptions,
            train_labels,
            test_labels,
            unseen_classes
        )

def test_determinism():

    class IdentityProjection:

        def predict(self, x):
            return x

    test_features = np.array([
        [1, 0],
        [0, 1]
    ])

    text_bank = np.array([
        [1, 0],
        [0, 1]
    ])

    test_labels = np.array([0, 1])

    unseen_classes = np.array([0, 1])

    results1 = evaluate_zero_shot(
        IdentityProjection(),
        test_features,
        test_labels,
        text_bank,
        unseen_classes
    )

    results2 = evaluate_zero_shot(
        IdentityProjection(),
        test_features,
        test_labels,
        text_bank,
        unseen_classes
    )

    assert (
        results1["overall_accuracy"]
        ==
        results2["overall_accuracy"]
    )

def test_confusion_matrix_shape():

    class IdentityProjection:

        def predict(self, x):
            return x

    test_features = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    text_bank = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    test_labels = np.array([0, 1, 2])

    unseen_classes = np.array([0, 1, 2])

    results = evaluate_zero_shot(
        IdentityProjection(),
        test_features,
        test_labels,
        text_bank,
        unseen_classes
    )

    cm = results["confusion_matrix"]

    assert cm.shape == (3, 3)

def test_prediction_length():

    class IdentityProjection:

        def predict(self, x):
            return x

    test_features = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5]
    ])

    text_bank = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    test_labels = np.array([0, 1, 2, 2])

    unseen_classes = np.array([0, 1, 2])

    results = evaluate_zero_shot(
        IdentityProjection(),
        test_features,
        test_labels,
        text_bank,
        unseen_classes
    )

    predictions = results["predictions"]

    assert len(predictions) == len(test_labels)