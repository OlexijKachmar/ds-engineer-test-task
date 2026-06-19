"""
train_rf.py — train a RandomForestClassifier on MNIST and save the checkpoint.

Usage:
    python training/train_rf.py \
        --train-size 15000 \
        --n-estimators 100 \
        --model-path artifacts/random_forest.joblib

Loads MNIST via torchvision (downloads to --data-root on first run), flattens
each 28x28 image to a length-784 vector, trains a RandomForestClassifier on a
stratified subsample of the training set, evaluates on the full MNIST test
set, and saves the fitted model with joblib.
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torchvision import datasets


def dataset_to_numpy(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert a torchvision MNIST dataset into flattened (N, 784) features
    and (N,) integer labels."""
    images = np.array([np.array(img) for img, _ in dataset], dtype=np.float32)
    labels = np.array([label for _, label in dataset], dtype=np.int64)
    images_flat = images.reshape(len(images), -1)
    return images_flat, labels


def load_mnist(data_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading MNIST from/to '{data_root}' ...")
    try:
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True)
    except RuntimeError as exc:
        # torchvision's default mirrors (yann.lecun.com / ossci-datasets S3) are
        # occasionally unreachable from restricted network environments. Fall
        # back to a GitHub-hosted mirror of the same files (checksums are
        # still verified against torchvision's known MD5s).
        print(f"  Default MNIST mirrors unreachable ({exc}); retrying with a GitHub mirror...")
        import torchvision.datasets.mnist as mnist_module

        mnist_module.MNIST.mirrors = [
            "https://raw.githubusercontent.com/fgnt/mnist/master/"
        ]
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    print(f"  Full train set: {X_train.shape}, full test set: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def train(
    data_root: str,
    train_size: int,
    n_estimators: int,
    model_path: str,
    random_state: int = 42,
) -> None:
    X_train_full, y_train_full, X_test, y_test = load_mnist(data_root)

    # Stratified subsample keeps class balance across digits 0-9.
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train_full,
        y_train_full,
        train_size=train_size,
        random_state=random_state,
        stratify=y_train_full,
    )
    print(f"Training on a stratified subsample of {len(X_train_sub)} images.")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train_sub, y_train_sub)
    print(f"Training completed in {time.time() - start:.1f}s.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy on full MNIST test set ({len(X_test)} images): {accuracy:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    artifact = {
        "model": model,
        "input_shape": (28, 28, 1),
        "train_size": train_size,
        "test_accuracy": accuracy,
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"\nModel saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RandomForestClassifier on MNIST and save the checkpoint."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="task_3_mnist_oop/data",
        help="Directory to download/load MNIST from (torchvision format).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=15000,
        help="Number of training images to use (stratified subsample).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="task_3_mnist_oop/artifacts/random_forest.joblib",
        help="Path where the trained model artifact will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train(
        data_root=args.data_root,
        train_size=args.train_size,
        n_estimators=args.n_estimators,
        model_path=args.model_path,
    )