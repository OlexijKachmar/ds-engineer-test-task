import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


TARGET_COLUMN = "target"
FEATURE_COLUMNS = ["6", "7"]


def build_model() -> Pipeline:
    """Create the final regression pipeline."""
    return Pipeline(
        steps=[
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
            ),
            (
                "linear_regression",
                LinearRegression(),
            ),
        ]
    )


def load_training_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the training dataset."""
    data = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    if X.isna().any().any():
        raise ValueError("Training features contain missing values.")

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    return X, y


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Evaluate the model using cross-validated RMSE."""
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )

    rmse_scores = -scores

    print(f"CV RMSE scores: {rmse_scores}")
    print(f"Mean CV RMSE: {rmse_scores.mean():.10f}")
    print(f"CV RMSE standard deviation: {rmse_scores.std():.10f}")

    return float(rmse_scores.mean())


def train(
    train_path: str,
    model_path: str,
) -> None:
    """Evaluate, train and save the final model."""
    X, y = load_training_data(train_path)

    model = build_model()

    evaluate_model(model, X, y)

    # Fit again using all available training rows.
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, output_path)

    print(f"Model saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the tabular regression model."
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default="task_2/train.csv",
        help="Path to train.csv.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="task_2/artifacts/model.joblib",
        help="Path where the trained model will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    train(
        train_path=args.train_path,
        model_path=args.model_path,
    )