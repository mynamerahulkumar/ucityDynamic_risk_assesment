"""Call the running Flask API over HTTP and save a combined report."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import requests

from common import configure_logging, load_config

logger = logging.getLogger(__name__)


def _decode(name: str, response: requests.Response):
    if response.status_code != 200:
        raise RuntimeError(f"{name} returned HTTP {response.status_code}: {response.text}")
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"{name} did not return JSON: {response.text}") from exc


def format_report(responses: dict) -> str:
    """Build a readable report containing every value the rubric asks for."""
    diagnostics = responses["diagnostics"]
    lines = [
        "Predictions",
        json.dumps(responses["predictions"], indent=2),
        "",
        "F1 score",
        json.dumps(responses["f1_score"], indent=2),
        "",
        "Summary statistics",
        json.dumps(responses["summary_statistics"], indent=2),
        "",
        "Ingestion execution time (seconds)",
        str(diagnostics.get("ingestion_time")),
        "",
        "Training execution time (seconds)",
        str(diagnostics.get("training_time")),
        "",
        "Missing data",
        json.dumps(diagnostics.get("missing_data"), indent=2),
        "",
        "Dependency versions",
        "package_name\tinstalled_version\tlatest_version",
    ]
    for row in diagnostics.get("dependencies", []):
        lines.append(
            f"{row.get('package_name')}\t{row.get('installed_version')}\t{row.get('latest_version')}"
        )
    return "\n".join(lines) + "\n"


def call_apis(base_url: str = "http://127.0.0.1:8000", output_filename: str = "apireturns.txt") -> dict:
    """Call all four endpoints and write the combined responses into the model directory."""
    filename = Path(output_filename).name
    if filename in {"", ".", ".."}:
        raise ValueError("An API report file name is required")

    root = base_url.rstrip("/")
    config = load_config()
    dataset_path = str(Path(config["test_data_path"]) / "testdata.csv")
    try:
        prediction = requests.post(
            f"{root}/prediction",
            json={"dataset_path": dataset_path},
            timeout=60,
        )
        scoring_response = requests.get(f"{root}/scoring", timeout=60)
        summary_response = requests.get(f"{root}/summarystats", timeout=60)
        diagnostics_response = requests.get(f"{root}/diagnostics", timeout=600)
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Could not connect to the API at {root}. Start it with `python app.py` and try again."
        ) from exc
    except requests.Timeout as exc:
        raise TimeoutError(f"The API at {root} did not respond before the timeout.") from exc

    responses = {
        "predictions": _decode("prediction", prediction),
        "f1_score": _decode("scoring", scoring_response),
        "summary_statistics": _decode("summarystats", summary_response),
        "diagnostics": _decode("diagnostics", diagnostics_response),
    }
    output_dir = Path(config["output_model_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    destination.write_text(format_report(responses), encoding="utf-8")
    logger.info("Wrote API responses to %s", destination)
    return responses


def main(argv: list[str] | None = None) -> None:
    """Call the API and save apireturns.txt, or the filename given with --output."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Call the risk-assessment API and save the responses.")
    parser.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", default="apireturns.txt")
    args = parser.parse_args(argv)
    call_apis(args.url, args.output)


if __name__ == "__main__":
    main()
