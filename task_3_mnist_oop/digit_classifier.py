import numpy as np
from typing import Dict
from interfaces import DigitClassificationInterface
from cnn_classifier import CNNClassifier
from random_forest_classifier import RandomForestClassifier
from random_classifier import RandomModelClassifier


class DigitClassifier:

    classifiers: Dict[str, DigitClassificationInterface] = {
        "cnn": CNNClassifier,
        "rf": RandomForestClassifier,
        "rand": RandomModelClassifier,
    }

    def __init__(self, label_of_algorithm="cnn", model=None):
        algorithm_to_use = self.classifiers.get(label_of_algorithm)
        if algorithm_to_use is None:
            raise ValueError("Unsupported algorithm")

        self.classifier: DigitClassificationInterface = algorithm_to_use(model)

    def predict(self, image: np.array):
        """This method takes 28x28x1 image and passes it directly to algorithm of interest"""
        if not isinstance(image, np.ndarray) or image.shape != (28, 28, 1):
            raise ValueError(
                f"Expected numpy array of shape (28, 28, 1), got {type(image)} {getattr(image, 'shape', '')}"
            )
        predicted_digit = self.classifier.predict(image)
        return predicted_digit

    def fit(self):
        raise NotImplementedError(
            "Training is not implemented in DigitClassifier. "
            "Train your model externally and inject it via the `model` parameter."
        )