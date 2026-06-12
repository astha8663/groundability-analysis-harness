import numpy as np
import pytest

from src.load import verify_alignment


def test_alignment_failure():

    text_bank = np.zeros((60, 1024))

    class_descriptions = [{} for _ in range(60)]
    train_features = np.zeros((3, 256))
    test_features = np.zeros((2, 256))

    train_labels = np.array([0, 1, 2])

    test_labels = np.array([10, 11])

    unseen_classes = np.array([20, 21])

    with pytest.raises(AssertionError):

        verify_alignment(
            text_bank,
            class_descriptions,
            train_features,
            train_labels,
            test_features,
            test_labels,
            unseen_classes
        )

def test_text_bank_size():

    text_bank = np.zeros((59, 1024))

    class_descriptions = [{} for _ in range(60)]
    train_features = np.zeros((3, 256))
    test_features = np.zeros((2, 256))

    train_labels = np.array([0])

    test_labels = np.array([1])

    unseen_classes = np.array([1])

    with pytest.raises(AssertionError):

        verify_alignment(
            text_bank,
            class_descriptions,
            train_features,
            train_labels,
            test_features,
            test_labels,
            unseen_classes
        )

def test_class_description_count():

    text_bank = np.zeros((60, 1024))

    class_descriptions = [{} for _ in range(59)]
    train_features = np.zeros((3, 256))
    test_features = np.zeros((2, 256))

    train_labels = np.array([0])

    test_labels = np.array([1])

    unseen_classes = np.array([1])

    with pytest.raises(AssertionError):

        verify_alignment(
            text_bank,
            class_descriptions,
            train_features,
            train_labels,
            test_features,
            test_labels,
            unseen_classes  
        )