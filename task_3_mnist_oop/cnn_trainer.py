"""
train_cnn.py — train LeNet5 on MNIST and save the checkpoint.

Usage:
    python training/train_cnn.py \
        --epochs 2 \
        --batch-size 128 \
        --model-path artifacts/lenet5.pt

Loads MNIST via torchvision (downloads to --data-root on first run), trains
LeNet5 for a small number of epochs (MNIST converges fast, so 1-2 epochs is
enough for a meaningful, non-trivial checkpoint), evaluates accuracy on the
full MNIST test set after every epoch, and saves the final state_dict.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# lenet5.py lives in the parent package directory (task_3_mnist_oop/), not in
# training/, so make sure it's importable regardless of the working
# directory this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lenet5 import LeNet5

RESIZE = transforms.Resize((32, 32))  # LeNet5 expects 32x32, MNIST is 28x28


def load_mnist_loaders(data_root: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    print(f"Loading MNIST from/to '{data_root}' ...")
    transform = transforms.Compose([transforms.ToTensor()])

    try:
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
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
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"  Train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = RESIZE(images).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = RESIZE(images).to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(
    data_root: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_path: str,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = load_mnist_loaders(data_root, batch_size)

    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_test_acc = 0.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        final_test_acc = test_acc
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f} — "
            f"test_acc: {test_acc:.4f} — {elapsed:.1f}s"
        )

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nFinal test accuracy: {final_test_acc:.4f}")
    print(f"Model state_dict saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LeNet5 on MNIST and save the checkpoint."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="task_3_mnist_oop/data",
        help="Directory to download/load MNIST from (torchvision format).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="task_3_mnist_oop/artifacts/lenet5.pt",
        help="Path where the trained model state_dict will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
    )