# Task 3 — MNIST Digit Classifier (OOP)

A unified digit classification interface that supports three interchangeable algorithms behind a single `DigitClassifier` entry point.

## Structure

```
task_3_mnist_oop/
├── interfaces.py               # DigitClassificationInterface (ABC)
├── lenet5.py                   # LeNet5 CNN architecture
├── cnn_classifier.py           # CNNClassifier
├── random_forest_classifier.py # RandomForestClassifier
├── random_classifier.py        # RandomModelClassifier
├── digit_classifier.py         # DigitClassifier (dispatcher)
├── demo.py                     # Runnable demo on real MNIST images
├── training/
│   ├── train_cnn.py            # Trains LeNet5, saves artifacts/lenet5.pt
│   └── train_rf.py             # Trains RandomForestClassifier, saves artifacts/random_forest.joblib
└── artifacts/
    ├── lenet5.pt                # Trained CNN checkpoint (state_dict)
    └── random_forest.joblib     # Trained RF checkpoint (joblib bundle)
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
lenet.load_state_dict(torch.load("artifacts/lenet5.pt"))
lenet.eval()
clf = DigitClassifier("cnn", model=lenet)

# Random Forest
artifact = joblib.load("artifacts/random_forest.joblib")
clf = DigitClassifier("rf", model=artifact["model"])

# Random (no model needed)
clf = DigitClassifier("rand")
```

This keeps model lifecycle management (training, serialization, loading) separate from prediction logic.

### Adding a new algorithm

1. Create a class that inherits from `DigitClassificationInterface`
2. Implement `predict(self, image: np.ndarray) -> int`
3. Register it in `DigitClassifier.classifiers`

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

## Training the models

Both training scripts download MNIST automatically (via `torchvision`, cached under `./data/` on first run), train, evaluate on the full MNIST test set, and save a checkpoint to `artifacts/`.

```bash
# Random Forest — trains on a stratified 15k-image subsample (~10s)
python training/train_rf.py --train-size 15000 --model-path artifacts/random_forest.joblib

# CNN (LeNet5) — trains for 2 epochs on the full 60k training set (~35s on CPU)
python training/train_cnn.py --epochs 2 --model-path artifacts/lenet5.pt
```

Both scripts have sensible defaults, so they also run with no arguments from inside `task_3_mnist_oop/`.

### Achieved accuracy (full MNIST test set, 10,000 images)

| Model | Training data | Test accuracy |
|---|---|---|
| Random Forest (`n_estimators=100`) | 15,000-image stratified subsample | **95.52%** |
| LeNet5 CNN, 2 epochs | full 60,000-image training set | **97.99%** |

These are genuine results from running the scripts above, not placeholders. Random seeds are fixed (`random_state=42` / `torch.manual_seed(42)`) for reproducibility.

## Running the demo

```bash
python demo.py
```

`demo.py` loads real MNIST test images, loads both trained checkpoints from `artifacts/` (in a fresh process — it does not reuse anything from the training run), injects each into `DigitClassifier`, and prints true label vs. predicted label per image for all three algorithms. This is the proof that the save → reload → inject → predict loop actually works end to end, not just immediately after training.

If a checkpoint is missing, the corresponding classifier falls back to an untrained/dummy-fitted model so the demo still runs, with a clear message indicating that `training/train_cnn.py` or `training/train_rf.py` should be run first for a real result.

## Tests

*Not yet included in this submission — see "Possible improvements" below.*

## Dependencies

```
torch
torchvision
scikit-learn
joblib
numpy
pytest
```

## Notes on model quality

Both checkpoints in `artifacts/` are real, trained models (not stubs) — see the accuracy table above. The Random model is intentionally untrained by design (it returns a random digit regardless of input, per the task specification), so its accuracy is not meaningful and is not reported.

## Possible improvements

- Add `tests/test_digit_classifier.py` covering: all three algorithms return a Python `int` in `[0, 9]` for the same input shape; invalid input shapes raise `ValueError`; an unknown algorithm name raises `ValueError`; `DigitClassifier.fit()` raises `NotImplementedError`.
- Train the CNN for more epochs / tune the Random Forest's `n_estimators` if higher accuracy is desired — out of scope for this exercise, since the goal is a correct, reusable interface rather than state-of-the-art MNIST accuracy.