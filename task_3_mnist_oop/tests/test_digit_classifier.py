import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from digit_classifier import DigitClassifier
from lenet5 import LeNet5


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_image():
    """A valid (28, 28, 1) numpy array."""
    return np.random.randint(0, 255, (28, 28, 1), dtype=np.uint8)


@pytest.fixture
def cnn_model():
    """Untrained LeNet5 — enough to test interface, not accuracy."""
    return LeNet5(num_classes=10)


@pytest.fixture
def rf_model():
    """RF fitted on minimal dummy data — enough to call predict."""
    model = RandomForestClassifier(n_estimators=2)
    model.fit(np.zeros((10, 784)), list(range(10)))
    return model


# ── Return type and range ─────────────────────────────────────────────────────

def test_cnn_returns_int(valid_image, cnn_model):
    clf = DigitClassifier("cnn", model=cnn_model)
    result = clf.predict(valid_image)
    assert isinstance(result, int)


def test_cnn_result_in_range(valid_image, cnn_model):
    clf = DigitClassifier("cnn", model=cnn_model)
    assert 0 <= clf.predict(valid_image) <= 9


def test_rf_returns_int(valid_image, rf_model):
    clf = DigitClassifier("rf", model=rf_model)
    result = clf.predict(valid_image)
    assert isinstance(result, int)


def test_rf_result_in_range(valid_image, rf_model):
    clf = DigitClassifier("rf", model=rf_model)
    assert 0 <= clf.predict(valid_image) <= 9


def test_random_returns_int(valid_image):
    clf = DigitClassifier("rand")
    result = clf.predict(valid_image)
    assert isinstance(result, int)


def test_random_result_in_range(valid_image):
    clf = DigitClassifier("rand")
    for _ in range(20):  # run several times given it's random
        assert 0 <= clf.predict(valid_image) <= 9


# ── Error handling ────────────────────────────────────────────────────────────

def test_invalid_algorithm_raises_value_error():
    with pytest.raises(ValueError):
        DigitClassifier("invalid_algo")


def test_invalid_image_shape_raises_value_error(cnn_model):
    clf = DigitClassifier("cnn", model=cnn_model)
    bad_image = np.zeros((28, 28))  # missing channel dim
    with pytest.raises(ValueError):
        clf.predict(bad_image)


def test_invalid_image_type_raises_value_error(cnn_model):
    clf = DigitClassifier("cnn", model=cnn_model)
    with pytest.raises(ValueError):
        clf.predict([[1, 2], [3, 4]])  # list, not ndarray


def test_fit_raises_not_implemented():
    clf = DigitClassifier("rand")
    with pytest.raises(NotImplementedError):
        clf.fit()


# ── Consistent interface across all algorithms ────────────────────────────────

def test_all_algorithms_same_input_output(valid_image, cnn_model, rf_model):
    """All three algorithms must accept the same input and return the same type."""
    classifiers = [
        DigitClassifier("cnn", model=cnn_model),
        DigitClassifier("rf", model=rf_model),
        DigitClassifier("rand"),
    ]
    for clf in classifiers:
        result = clf.predict(valid_image)
        assert isinstance(result, int), f"{clf.classifier.__class__.__name__} did not return int"
        assert 0 <= result <= 9, f"{clf.classifier.__class__.__name__} returned out-of-range value {result}"