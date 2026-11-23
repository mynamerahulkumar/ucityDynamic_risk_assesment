"""Shared configuration, model loading, and score-file helpers."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

FEATURES = [
    "lastmonth_activity",
    "lastyear_activity",
    "number_of_employees",
]
TARGET = "exited"
REQUIRED_COLUMNS = ["corporation", *FEATURES, TARGET]
CONFIG_KEYS = [
    "input_folder_path",
    "output_folder_path",
    "test_data_path",
    "output_model_path",
    "prod_deployment_path",
]


def repo_root() -> Path:
    """Return the directory that contains the project scripts."""
    return Path(__file__).resolve().parent


def configure_logging() -> None:
    """Send process logs to stderr once the script starts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_config(config_path: str | Path | None = None) -> dict:
    """Read config.json from the working directory unless a path is supplied."""
    path = Path(config_path or "config.json")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"config.json is missing keys: {', '.join(missing)}")
    return config


def load_pickle(path: str | Path):
    """Load a pickle file or raise a clear error when it is missing."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def read_score(path: str | Path) -> float:
    """Read a single F1 score from a text file."""
    score_path = Path(path)
    if not score_path.is_file():
        raise FileNotFoundError(f"Score file not found: {score_path}")
    text = score_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Score file is empty: {score_path}")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Score file does not contain a number: {score_path}") from exc
