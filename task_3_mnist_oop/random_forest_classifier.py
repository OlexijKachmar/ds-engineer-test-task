import numpy as np
import random
from interfaces import DigitClassificationInterface


class RandomForestClassifier(DigitClassificationInterface):

    def __init__(self, model):
        self.model = model

    def predict(self, image: np.ndarray) -> int:
        """
        Args:
            image: numpy array of shape (28, 28, 1)
 
        Returns:
            Predicted digit as int in [0, 9]
        """
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected image shape (28, 28, 1), got {image.shape}")
        flat = image.reshape(1, 784)
        return int(self.model.predict(flat)[0])
