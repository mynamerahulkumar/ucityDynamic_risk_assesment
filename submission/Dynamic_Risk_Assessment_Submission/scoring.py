"""Score a fitted attrition model with F1 and optionally save that score."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sklearn import metrics

from common import FEATURES, TARGET, configure_logging, load_config, load_pickle

logger = logging.getLogger(__name__)


def score_model(model, dataframe: pd.DataFrame, score_path: str | os.PathLike | None = None) -> float:
    """Return the model's F1 score. Write latestscore.txt only when score_path is set."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if dataframe.empty:
        raise ValueError("Cannot score an empty dataset")

    missing = [column for column in FEATURES + [TARGET] if column not in dataframe.columns]
    if missing:
        raise ValueError("Scoring dataset is missing columns: " + ", ".join(missing))

    usable = dataframe.loc[:, FEATURES + [TARGET]].dropna()
    if usable.empty:
        raise ValueError("Scoring dataset has no complete feature and target rows")

    predictions = model.predict(usable.loc[:, FEATURES])
    f1_score = float(metrics.f1_score(usable[TARGET], predictions))
    if score_path is not None:
        destination = Path(score_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{f1_score}\n", encoding="utf-8")
        logger.info("Wrote F1 score %s to %s", f1_score, destination)
    else:
        logger.info("Calculated F1 score %s without writing a score file", f1_score)
    return f1_score


def score_candidate_model() -> float:
    """Score the candidate model on testdata.csv and save output_model_path/latestscore.txt."""
    config = load_config()
    model_dir = Path(config["output_model_path"])
    model = load_pickle(model_dir / "trainedmodel.pkl")
    test_data = pd.read_csv(Path(config["test_data_path"]) / "testdata.csv")
    return score_model(model, test_data, score_path=model_dir / "latestscore.txt")


def main() -> None:
    """Score the candidate model from the repository root."""
    configure_logging()
    score_candidate_model()


if __name__ == "__main__":
    main()
