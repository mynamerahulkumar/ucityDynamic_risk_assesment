"""Plot a confusion matrix for the deployed model on the configured test data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

from common import TARGET, configure_logging, load_config
from diagnostics import model_predictions

logger = logging.getLogger(__name__)

CLASS_LABELS = ["0 (stayed)", "1 (exited)"]


def score_model(output_filename: str = "confusionmatrix.png") -> Path:
    """Save a confusion matrix for the deployed model. The filename is not overwritten by default."""
    filename = Path(output_filename).name
    if filename in {"", ".", ".."}:
        raise ValueError("A confusion-matrix file name is required")

    config = load_config()
    test_path = Path(config["test_data_path"]) / "testdata.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    dataframe = pd.read_csv(test_path)
    if TARGET not in dataframe.columns:
        raise ValueError(f"Test dataset is missing the {TARGET} column")

    predictions = model_predictions(dataframe)
    actual = [int(value) for value in dataframe[TARGET].tolist()]
    if len(predictions) != len(actual):
        raise RuntimeError("Predictions and labels have different lengths")

    matrix = metrics.confusion_matrix(actual, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    try:
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], CLASS_LABELS)
        axis.set_yticks([0, 1], CLASS_LABELS)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_title("Confusion matrix for the deployed attrition model")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="black",
                )
        figure.colorbar(image, ax=axis, label="Count")
        figure.tight_layout()
        output_dir = Path(config["output_model_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / filename
        figure.savefig(figure_path, dpi=120, bbox_inches="tight", facecolor="white")
        logger.info("Saved confusion matrix to %s", figure_path)
    finally:
        plt.close(figure)
    return figure_path


def main(argv: list[str] | None = None) -> None:
    """Write confusionmatrix.png, or the filename given with --output."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Save the deployed model's confusion matrix.")
    parser.add_argument("--output", default="confusionmatrix.png")
    args = parser.parse_args(argv)
    score_model(args.output)


if __name__ == "__main__":
    main()
