# Task 3 — MNIST Digit Classifier (OOP)

A unified digit classification interface that supports three interchangeable algorithms behind a single `DigitClassifier` entry point.

## Structure

```
task_3/
├── interfaces.py               # DigitClassificationInterface (ABC)
├── lenet5.py                   # LeNet5 CNN architecture
├── cnn_classifier.py           # CNNClassifier
├── random_forest_classifier.py # RandomForestClassifier
├── random_classifier.py        # RandomModelClassifier
├── digit_classifier.py         # DigitClassifier (dispatcher)
├── demo.py                     # Runnable demo
└── tests/
    └── test_digit_classifier.py
```

## Design

### Public contract

Every algorithm accepts the same input and returns the same output type:

| | |
|---|---|
| **Input** | `np.ndarray` of shape `(28, 28, 1)`, dtype `uint8` or `float32` |
| **Output** | Single Python `int` in range `[0, 9]` |

### Internal preprocessing per algorithm

Each classifier handles its own preprocessing internally — the caller always passes a raw `(28, 28, 1)` array:

- **CNN** — permutes to `(1, 28, 28)` tensor, resizes to `(32, 32)` for LeNet5, runs forward pass
- **RF** — flattens to a `(1, 784)` array before calling `model.predict`
- **Random** — takes the `10×10` center crop (`image[9:19, 9:19]`), returns a random digit

### Model injection

Training is intentionally outside `DigitClassifier` (`fit()` raises `NotImplementedError`). Pre-trained models are injected via the constructor:

```python
# CNN
lenet = LeNet5(num_classes=10)
lenet.load_state_dict(torch.load("lenet5.pth"))
clf = DigitClassifier("cnn", model=lenet)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
clf = DigitClassifier("rf", model=rf)

# Random (no model needed)
clf = DigitClassifier("rand")
```

This keeps model lifecycle management (training, serialization, loading) separate from prediction logic.

### Adding a new algorithm

1. Create a class that inherits from `DigitClassificationInterface`
2. Implement `predict(self, image: np.ndarray) -> int`
3. Register it in `DigitClassifier._classifiers`

## Usage

```python
import numpy as np
from digit_classifier import DigitClassifier
from lenet5 import LeNet5

image = np.random.randint(0, 255, (28, 28, 1), dtype=np.uint8)

clf = DigitClassifier("cnn", model=LeNet5(num_classes=10))
digit = clf.predict(image)  # int in [0, 9]
```

Supported algorithm names: `"cnn"`, `"rf"`, `"rand"`.

## Running

```bash
# Demo
python demo.py

# Tests
python -m pytest tests/ -v
```

## Dependencies

```
torch
torchvision
scikit-learn
numpy
pytest
```

## Notes on model quality

The demo and tests use an untrained CNN and a dummy-fitted RF. They exist only to verify that the interface works correctly — not to produce accurate predictions. For meaningful accuracy, train both models on the full MNIST dataset and inject the saved weights.
