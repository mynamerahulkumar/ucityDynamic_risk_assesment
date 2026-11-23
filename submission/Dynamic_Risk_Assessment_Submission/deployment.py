"""Copy the trained model, its F1 score, and the ingestion record into production."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from common import configure_logging, load_config

logger = logging.getLogger(__name__)


def store_model_into_pickle(model=None) -> list[Path]:
    """Copy the current candidate artifacts into prod_deployment_path.

    The model argument is accepted so the starter signature still works.
    Deployment does not train, score, or reserialize a model. It copies the
    pickle and score that training and scoring already saved, plus ingestedfiles.txt.
    """
    if model is not None:
        logger.info("Ignoring the in-memory model. Deployment copies the saved artifacts.")

    config = load_config()
    model_dir = Path(config["output_model_path"])
    output_dir = Path(config["output_folder_path"])
    production_dir = Path(config["prod_deployment_path"])
    production_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        model_dir / "trainedmodel.pkl": production_dir / "trainedmodel.pkl",
        model_dir / "latestscore.txt": production_dir / "latestscore.txt",
        output_dir / "ingestedfiles.txt": production_dir / "ingestedfiles.txt",
    }
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot deploy because these files are missing: " + ", ".join(missing)
        )

    copied: list[Path] = []
    for source, destination in sources.items():
        shutil.copy2(source, destination)
        copied.append(destination)
        logger.info("Copied %s to %s", source, destination)
    return copied


def main() -> None:
    """Deploy the current candidate artifacts from the repository root."""
    configure_logging()
    store_model_into_pickle()


if __name__ == "__main__":
    main()
