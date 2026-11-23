"""Train the attrition logistic regression model and save it as a pickle."""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (
    FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
    configure_logging,
    load_config,
)

logger = logging.getLogger(__name__)


def build_estimator() -> LogisticRegression:
    """Return the starter logistic regression estimator and its required settings."""
    return LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )


def train_model() -> LogisticRegression:
    """Fit the attrition model on finaldata.csv and save trainedmodel.pkl."""
    config = load_config()
    dataset_path = Path(config["output_folder_path"]) / "finaldata.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Ingested dataset not found at {dataset_path}. Run ingestion.py first."
        )

    dataframe = pd.read_csv(dataset_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Ingested dataset is missing columns: " + ", ".join(missing_columns)
        )

    usable = dataframe.loc[:, FEATURES + [TARGET]].dropna()
    if len(usable) < 2:
        raise ValueError(f"Need at least 2 complete records to train, found {len(usable)}.")
    if usable[TARGET].nunique(dropna=True) < 2:
        raise ValueError("Training data must contain both exited classes (0 and 1).")

    # scikit-learn 1.8 still accepts the starter arguments and warns that penalty is deprecated.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*penalty.*")
        warnings.filterwarnings("ignore", message=".*l1_ratio.*")
        model = build_estimator()
        model.fit(usable.loc[:, FEATURES], usable[TARGET])

    model_dir = Path(config["output_model_path"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trainedmodel.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    logger.info(
        "Trained logistic regression on %s rows and saved %s",
        len(usable),
        model_path,
    )
    return model


def main() -> None:
    """Train from the repository root."""
    configure_logging()
    train_model()


if __name__ == "__main__":
    main()
