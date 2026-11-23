"""Flask API for predictions, scoring, summary statistics, and diagnostics."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from common import configure_logging, load_config, load_pickle
from diagnostics import dataframe_summary, execution_time, missing_data, model_predictions, outdated_packages_list
from scoring import score_model

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"


def _json_error(message: str, status: int):
    return jsonify({"error": message}), status


def _preflight():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    return None


def _dataset_path_from_request() -> str | None:
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        dataset_path = payload.get("dataset_path")
    elif isinstance(payload, str):
        dataset_path = payload
    else:
        dataset_path = None
    if not dataset_path and request.form:
        dataset_path = request.form.get("dataset_path")
    if not dataset_path and request.data and not request.is_json:
        dataset_path = request.data.decode("utf-8").strip()
    if dataset_path is None:
        return None
    dataset_path = str(dataset_path).strip()
    return dataset_path or None


@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    """Return deployed-model predictions for the CSV path supplied by the caller."""
    preflight = _preflight()
    if preflight is not None:
        return preflight

    dataset_path = _dataset_path_from_request()
    if not dataset_path:
        return _json_error("dataset_path is required", 400)

    path = Path(dataset_path)
    if not path.exists():
        return _json_error(f"Dataset not found: {dataset_path}", 404)
    if not path.is_file():
        return _json_error(f"Dataset path is not a file: {dataset_path}", 400)

    try:
        dataframe = pd.read_csv(path)
        predictions = model_predictions(dataframe)
    except (ValueError, pd.errors.ParserError) as exc:
        logger.exception("Prediction request was rejected")
        return _json_error(str(exc), 400)
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:
        logger.exception("Prediction failed")
        return _json_error(str(exc), 500)
    return jsonify({"predictions": predictions}), 200


@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring_endpoint():
    """Score the deployed model on testdata.csv without writing a score file."""
    preflight = _preflight()
    if preflight is not None:
        return preflight

    try:
        config = load_config()
        model = load_pickle(Path(config["prod_deployment_path"]) / "trainedmodel.pkl")
        test_path = Path(config["test_data_path"]) / "testdata.csv"
        if not test_path.is_file():
            return _json_error(f"Test dataset not found: {test_path}", 404)
        f1_score = score_model(model, pd.read_csv(test_path))
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        logger.exception("Scoring failed")
        return _json_error(str(exc), 500)
    return jsonify({"f1_score": f1_score}), 200


@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats_endpoint():
    """Return numeric summary statistics for the ingested dataset."""
    preflight = _preflight()
    if preflight is not None:
        return preflight
    try:
        summary = dataframe_summary()
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:
        logger.exception("Summary statistics failed")
        return _json_error(str(exc), 500)
    return jsonify(summary), 200


@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics_endpoint():
    """Return timings, missing-data percentages, and dependency versions."""
    preflight = _preflight()
    if preflight is not None:
        return preflight
    try:
        timings = execution_time()
        missing = missing_data()
        dependencies = outdated_packages_list()
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:
        logger.exception("Diagnostics failed")
        return _json_error(str(exc), 500)
    return jsonify(
        {
            "ingestion_time": timings[0],
            "training_time": timings[1],
            "missing_data": missing,
            "dependencies": dependencies,
        }
    ), 200


def main() -> None:
    """Run the development API. API_PORT overrides the default port 8000."""
    configure_logging()
    port = int(os.environ.get("API_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
