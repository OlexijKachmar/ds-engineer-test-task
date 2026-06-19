"""
demo.py — runs all three classifiers on real MNIST test images.

Usage:
    python demo.py

This demo proves the saved-and-reloaded training loop actually works end to
end: each trained model is loaded from its checkpoint file (not re-trained in
this process) and injected into DigitClassifier, then used to predict real
MNIST test images, which are compared against their true labels.

If a checkpoint is missing, the corresponding classifier falls back to an
untrained/dummy-fitted model so the demo still runs and the interface can
still be demonstrated — but this is clearly flagged in the output.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torchvision import datasets

from digit_classifier import DigitClassifier
from lenet5 import LeNet5

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
RF_CHECKPOINT = ARTIFACTS_DIR / "random_forest.joblib"
CNN_CHECKPOINT = ARTIFACTS_DIR / "lenet5.pt"

NUM_SAMPLES = 5


def load_mnist_samples(num_samples: int, data_root: str = "task_3_mnist_oop/data"):
    """Load a handful of real MNIST test images as (28, 28, 1) numpy arrays,
    paired with their true labels."""
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True)

    images, labels = [], []
    for i in range(num_samples):
        pil_img, label = test_dataset[i]
        image = np.array(pil_img, dtype=np.float32).reshape(28, 28, 1)
        images.append(image)
        labels.append(label)
    return images, labels


def load_rf_classifier() -> DigitClassifier:
    if RF_CHECKPOINT.exists():
        artifact = joblib.load(RF_CHECKPOINT)
        rf_model = artifact["model"]
        print(f"  Loaded trained checkpoint: {RF_CHECKPOINT}")
        print(f"  (reported test accuracy at training time: {artifact.get('test_accuracy', 'n/a')})")
    else:
        from sklearn.ensemble import RandomForestClassifier

        print(f"  No checkpoint found at {RF_CHECKPOINT} — falling back to a dummy-fitted model.")
        print("  Run `python training/train_rf.py` first for a real, trained model.")
        rf_model = RandomForestClassifier(n_estimators=10)
        rf_model.fit(np.zeros((10, 784)), list(range(10)))

    return DigitClassifier("rf", model=rf_model)


def load_cnn_classifier() -> DigitClassifier:
    lenet = LeNet5(num_classes=10)

    if CNN_CHECKPOINT.exists():
        lenet.load_state_dict(torch.load(CNN_CHECKPOINT, map_location="cpu"))
        lenet.eval()
        print(f"  Loaded trained checkpoint: {CNN_CHECKPOINT}")
    else:
        lenet.eval()
        print(f"  No checkpoint found at {CNN_CHECKPOINT} — using an untrained LeNet5.")
        print("  Run `python training/train_cnn.py` first for a real, trained model.")

    return DigitClassifier("cnn", model=lenet)


def run_classifier(name: str, clf: DigitClassifier, images, labels) -> None:
    correct = 0
    for image, true_label in zip(images, labels):
        prediction = clf.predict(image)
        is_correct = prediction == true_label
        correct += int(is_correct)
        mark = "✓" if is_correct else "✗"
        print(f"  true={true_label}  predicted={prediction}  {mark}")
    print(f"  {correct}/{len(images)} correct on this sample\n")


def main():
    images, labels = load_mnist_samples(NUM_SAMPLES)
    print(f"Loaded {len(images)} real MNIST test images, shape {images[0].shape}\n")

    print("── CNN (LeNet5) ──")
    cnn_clf = load_cnn_classifier()
    run_classifier("cnn", cnn_clf, images, labels)

    print("── Random Forest ──")
    rf_clf = load_rf_classifier()
    run_classifier("rf", rf_clf, images, labels)

    print("── Random Model ──")
    rand_clf = DigitClassifier("rand")
    run_classifier("rand", rand_clf, images, labels)
    print("  (returns a random digit by design — accuracy is not meaningful)\n")

    print("All three classifiers ran on the same (28, 28, 1) real MNIST inputs.")


if __name__ == "__main__":
    main()