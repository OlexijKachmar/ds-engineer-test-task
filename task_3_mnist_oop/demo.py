"""
demo.py — runs all three classifiers on a sample image.

Usage:
    python demo.py

Note: Uses a synthetic random image for portability.
      To use a real MNIST image, replace load_sample_image() with:

        from torchvision import datasets
        dataset = datasets.MNIST(root="./data", train=True, download=True)
        pil_img, label = dataset[0]
        image = np.array(pil_img).reshape(28, 28, 1)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier 

from lenet5 import LeNet5
from digit_classifier import DigitClassifier


def load_sample_image():
    """Returns a synthetic (28, 28, 1) uint8 image for demonstration."""
    rng = np.random.default_rng(seed=42)
    image = rng.integers(0, 255, (28, 28, 1), dtype=np.uint8)
    return image


def main():
    image = load_sample_image()
    print(f"Input image shape: {image.shape}\n")

    # CNN — untrained, demonstrates interface only
    print("── CNN (untrained LeNet5) ──")
    lenet = LeNet5(num_classes=10)
    clf = DigitClassifier("cnn", model=lenet)
    print(f"  Prediction  : {clf.predict(image)}")
    print("  (untrained — demonstrates interface only)\n")

    # Random Forest — minimally fitted on dummy data
    print("── Random Forest (dummy-fitted) ──")
    rf_model = RandomForestClassifier(n_estimators=10)
    rf_model.fit(np.zeros((10, 784)), list(range(10)))
    clf = DigitClassifier("rf", model=rf_model)
    print(f"  Prediction  : {clf.predict(image)}")
    print("  (dummy-fitted — demonstrates interface only)\n")

    # Random model
    print("── Random Model ──")
    clf = DigitClassifier("rand")
    print(f"  Prediction  : {clf.predict(image)}")
    print("  (returns a random digit by design)\n")

    print("All three classifiers ran on the same (28, 28, 1) input.")


if __name__ == "__main__":
    main()