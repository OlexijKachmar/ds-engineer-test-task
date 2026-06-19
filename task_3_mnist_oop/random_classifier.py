import numpy as np
import random
from interfaces import DigitClassificationInterface


class RandomModelClassifier(DigitClassificationInterface):

    def __init__(self, model=None):
        pass

    def predict(self, image: np.ndarray) -> int:
        """
        Takes the 10x10 center crop of the image (unused) and returns a random digit.
 
        Args:
            image: numpy array of shape (28, 28, 1)
 
        Returns:
            Random digit as int in [0, 9]
        """
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected image shape (28, 28, 1), got {image.shape}")
        center_crop = image[9:19, 9:19]  
        return int(random.randint(0, 9))
