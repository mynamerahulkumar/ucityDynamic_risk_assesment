"""Ingest new files, check the deployed model for drift, and report on the current model."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

import deployment
import ingestion
import scoring
import training
from common import configure_logging, load_config, load_pickle, read_score, repo_root

logger = logging.getLogger(__name__)


def deployed_filenames(production_dir: str | Path) -> set[str]:
    """Read the deployed ingestion record as a set of file names."""
    record = Path(production_dir) / "ingestedfiles.txt"
    if not record.is_file():
        raise FileNotFoundError(
            f"Deployed ingestion record not found: {record}. Deploy a model before running fullprocess.py."
        )
    return {line.strip() for line in record.read_text(encoding="utf-8").splitlines() if line.strip()}


def find_new_files() -> list[str]:
    """Return CSV names in the input directory that are not in the deployed record."""
    config = load_config()
    current = [path.name for path in ingestion.list_csv_files(config["input_folder_path"])]
    known = deployed_filenames(config["prod_deployment_path"])
    return [name for name in current if name not in known]


def _port_from_url(url: str) -> str:
    parsed = requests.utils.urlparse(url)
    if parsed.port is None:
        return "8000"
    return str(parsed.port)


def api_is_reachable(base_url: str) -> bool:
    """Return whether any HTTP response is available at the scoring endpoint."""
    try:
        requests.get(f"{base_url.rstrip('/')}/scoring", timeout=5)
    except requests.RequestException:
        return False
    return True


def _tail(path: Path, limit: int = 40) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def start_api(base_url: str) -> tuple[subprocess.Popen, object, Path]:
    """Start app.py on the URL port and wait until it answers."""
    env = os.environ.copy()
    env["API_PORT"] = _port_from_url(base_url)
    log_path = Path(tempfile.gettempdir()) / f"dynamic_risk_api_{os.getpid()}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(repo_root() / "app.py")],
        cwd=Path.cwd(),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if api_is_reachable(base_url):
            logger.info("Started the API at %s", base_url)
            return process, log_handle, log_path
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"API exited before it was ready.\n{_tail(log_path)}")
        time.sleep(0.25)
    process.terminate()
    log_handle.close()
    raise RuntimeError(f"API did not become ready at {base_url}.\n{_tail(log_path)}")


def stop_process(process: subprocess.Popen | None, log_handle=None) -> None:
    """Stop an API process that this workflow started."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if log_handle is not None and not log_handle.closed:
        log_handle.close()


def run_second_reports(base_url: str) -> None:
    """Run reporting.py and apicalls.py for the second-run artifact names."""
    reporting = subprocess.run(
        [sys.executable, str(repo_root() / "reporting.py"), "--output", "confusionmatrix2.png"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    if reporting.returncode != 0:
        detail = (reporting.stderr or reporting.stdout).strip()
        raise RuntimeError(f"reporting.py failed with status {reporting.returncode}.\n{detail}")

    started = None
    log_handle = None
    if not api_is_reachable(base_url):
        started, log_handle, _log_path = start_api(base_url)
    try:
        api_call = subprocess.run(
            [
                sys.executable,
                str(repo_root() / "apicalls.py"),
                "--output",
                "apireturns2.txt",
                "--url",
                base_url,
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        if api_call.returncode != 0:
            detail = (api_call.stderr or api_call.stdout).strip()
            raise RuntimeError(f"apicalls.py failed with status {api_call.returncode}.\n{detail}")
    finally:
        stop_process(started, log_handle)


def run_full_process(run_reports: bool = True, base_url: str | None = None) -> str:
    """Detect new data, check for drift, and write second-run reports when data is new."""
    url = base_url or os.environ.get("API_URL", "http://127.0.0.1:8000")
    new_files = find_new_files()
    if not new_files:
        message = (
            "No new data found. Stopping before drift checks, retraining, deployment, and reporting."
        )
        logger.info(message)
        print(message)
        return "no_new_data"

    logger.info("New files detected: %s", ", ".join(new_files))
    print("New files detected: " + ", ".join(new_files))
    ingestion.merge_multiple_dataframe()

    config = load_config()
    production_dir = Path(config["prod_deployment_path"])
    model = load_pickle(production_dir / "trainedmodel.pkl")
    baseline = read_score(production_dir / "latestscore.txt")
    ingested = pd.read_csv(Path(config["output_folder_path"]) / "finaldata.csv")
    new_score = scoring.score_model(model, ingested)
    print(f"new_data_f1_score={new_score}")
    print(f"production_baseline_f1_score={baseline}")
    logger.info("Drift check new F1=%s baseline F1=%s", new_score, baseline)

    if new_score < baseline:
        logger.info("Drift detected. Training, scoring, and deploying a replacement model.")
        print("Drift detected. Retraining and redeploying.")
        training.train_model()
        scoring.score_candidate_model()
        deployment.store_model_into_pickle()
        status = "drift"
    else:
        logger.info("No drift detected. Keeping the deployed model and baseline.")
        print("No drift detected. Keeping the deployed model and baseline.")
        status = "no_drift"

    if run_reports:
        logger.info("Writing second-run reports with the currently deployed model.")
        run_second_reports(url)
    return status


def main() -> None:
    """Run the automation workflow from the repository root."""
    configure_logging()
    run_full_process()


if __name__ == "__main__":
    main()
