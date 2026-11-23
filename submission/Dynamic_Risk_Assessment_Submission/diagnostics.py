"""Predictions, data statistics, timing, and dependency diagnostics."""

from __future__ import annotations

import importlib.metadata
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from common import FEATURES, configure_logging, load_config, load_pickle, repo_root

logger = logging.getLogger(__name__)


def _finaldata_path() -> Path:
    config = load_config()
    return Path(config["output_folder_path"]) / "finaldata.csv"


def _read_finaldata() -> pd.DataFrame:
    dataset_path = _finaldata_path()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Ingested dataset not found at {dataset_path}. Run ingestion.py first."
        )
    return pd.read_csv(dataset_path)


def _json_number(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def model_predictions(dataframe: pd.DataFrame) -> list[int]:
    """Predict exited labels with the deployed model, one prediction per input row."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if dataframe.empty:
        return []

    missing = [column for column in FEATURES if column not in dataframe.columns]
    if missing:
        raise ValueError("Prediction dataset is missing columns: " + ", ".join(missing))
    features = dataframe.loc[:, FEATURES]
    if features.isna().any().any():
        raise ValueError("Feature columns contain missing values, so every row cannot be scored.")

    config = load_config()
    model = load_pickle(Path(config["prod_deployment_path"]) / "trainedmodel.pkl")
    predictions = model.predict(features)
    result = [int(value) for value in predictions]
    if len(result) != len(dataframe):
        raise RuntimeError("Prediction count does not match the input row count")
    return result


def dataframe_summary() -> list[dict]:
    """Return mean, median, and sample standard deviation for each numeric column."""
    dataframe = _read_finaldata()
    numeric = dataframe.select_dtypes(include="number")
    summary = []
    for column in numeric.columns:
        series = numeric[column]
        summary.append(
            {
                "column": str(column),
                "mean": None if series.empty else _json_number(series.mean()),
                "median": None if series.empty else _json_number(series.median()),
                "std": None if series.empty else _json_number(series.std()),
            }
        )
    return summary


def missing_data() -> list[dict]:
    """Return missing-value counts and percentages for every column, in column order."""
    dataframe = _read_finaldata()
    total = int(len(dataframe))
    results = []
    for column in dataframe.columns:
        missing = int(dataframe[column].isna().sum())
        percentage = 0.0 if total == 0 else (missing / total) * 100.0
        results.append(
            {
                "column": str(column),
                "missing_count": missing,
                "total_count": total,
                "percentage": percentage,
            }
        )
    return results


def execution_time() -> list[float]:
    """Time ingestion.py and training.py separately. Do not deploy the new candidate."""
    timings: list[float] = []
    for script_name in ("ingestion.py", "training.py"):
        script = repo_root() / script_name
        logger.info("Timing %s", script)
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"{script_name} exited with status {completed.returncode}.\n{detail}")
        timings.append(elapsed)
        logger.info("%s finished in %.6f seconds", script_name, elapsed)
    return timings


def parse_requirements(path: str | Path) -> list[str]:
    """Return package names from a requirements file, ignoring comments and versions."""
    names = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return names


def parse_latest_version(output: str) -> str | None:
    """Extract the LATEST version from `pip index versions` output."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("LATEST:"):
            version = stripped.split(":", 1)[1].strip()
            if version:
                return version
    return None


def installed_version(package_name: str) -> str:
    """Return the installed distribution version, or unavailable when it is missing."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        logger.warning("Package %s is not installed", package_name)
        return "unavailable"


def latest_version(package_name: str, timeout: float = 20) -> str:
    """Look up the newest published version. Network failure becomes 'unavailable'."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", package_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Could not look up the latest version of %s: %s", package_name, exc)
        return "unavailable"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        logger.warning("pip index versions failed for %s: %s", package_name, detail)
        return "unavailable"
    parsed = parse_latest_version(f"{completed.stdout}\n{completed.stderr}")
    if parsed is None:
        logger.warning("pip did not report a latest version for %s", package_name)
        return "unavailable"
    return parsed


def outdated_packages_list() -> list[dict]:
    """Return installed and latest versions for every requirements.txt package."""
    requirements_path = repo_root() / "requirements.txt"
    if not requirements_path.is_file():
        raise FileNotFoundError(f"requirements.txt not found at {requirements_path}")
    names = parse_requirements(requirements_path)
    latest: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(latest_version, name): name for name in names}
        for future, name in futures.items():
            try:
                latest[name] = future.result()
            except Exception as exc:
                logger.warning("Dependency lookup failed for %s: %s", name, exc)
                latest[name] = "unavailable"
    return [
        {
            "package_name": name,
            "installed_version": installed_version(name),
            "latest_version": latest[name],
        }
        for name in names
    ]


def main() -> None:
    """Run every diagnostic from the repository root and print the results."""
    configure_logging()
    config = load_config()
    test_data = pd.read_csv(Path(config["test_data_path"]) / "testdata.csv")
    print("predictions:", model_predictions(test_data))
    print("summary:", dataframe_summary())
    print("missing:", missing_data())
    print("timings:", execution_time())
    print("dependencies:", outdated_packages_list())


if __name__ == "__main__":
    main()
