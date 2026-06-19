import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_artifact(model_path: str) -> dict:
    """Load the trained model bundle saved by train.py."""
    artifact = joblib.load(model_path)

    required_keys = ["model", "feature_columns"]
    missing_keys = [key for key in required_keys if key not in artifact]
    if missing_keys:
        raise ValueError(
            f"Model artifact at '{model_path}' is missing expected keys: {missing_keys}"
        )

    return artifact


def load_test_data(path: str, feature_columns: list[str]) -> pd.DataFrame:
    """Load the test set and validate it contains the required feature columns."""
    data = pd.read_csv(path)

    missing_columns = [col for col in feature_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Test data at '{path}' is missing required columns: {missing_columns}"
        )

    # Select and reorder columns to exactly match the order used during training.
    X = data[feature_columns]

    if X.isna().any().any():
        raise ValueError("Test features contain missing values.")

    return X


def predict(test_path: str, model_path: str, output_path: str) -> None:
    """Load the trained model, generate predictions, and save them to a CSV file."""
    artifact = load_artifact(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X_test = load_test_data(test_path, feature_columns)

    predictions = model.predict(X_test)

    if len(predictions) != len(X_test):
        raise RuntimeError(
            f"Prediction count ({len(predictions)}) does not match "
            f"input row count ({len(X_test)})."
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("Predictions contain NaN or infinite values.")

    output = pd.DataFrame({"prediction": predictions})

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)

    print(f"Generated {len(output)} predictions.")
    print(f"Predictions saved to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions on the hidden test set using a trained model."
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default="task_2/hidden_test.csv",
        help="Path to hidden_test.csv.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="task_2/artifacts/model.joblib",
        help="Path to the trained model artifact saved by train.py.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="task_2/predictions/predictions.csv",
        help="Path where the predictions CSV will be written.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    predict(
        test_path=args.test_path,
        model_path=args.model_path,
        output_path=args.output_path,
    )