"""Discover source CSV files, drop duplicate rows, and record what was ingested."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common import configure_logging, load_config

logger = logging.getLogger(__name__)


def list_csv_files(input_folder: str | Path) -> list[Path]:
    """Return sorted CSV files, skipping hidden, generated, and non-CSV files."""
    folder = Path(input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory not found: {folder}")

    selected: list[Path] = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            logger.info("Skipping hidden file %s", path.name)
            continue
        if path.name == "finaldata.csv":
            logger.info("Skipping generated file %s", path.name)
            continue
        if path.suffix.lower() != ".csv":
            logger.info("Skipping non-CSV file %s", path.name)
            continue
        selected.append(path)
    return selected


def merge_multiple_dataframe() -> pd.DataFrame:
    """Combine every source CSV, remove duplicates, and write the ingestion record."""
    config = load_config()
    input_folder = Path(config["input_folder_path"])
    output_folder = Path(config["output_folder_path"])
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_files = list_csv_files(input_folder)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        logger.info("Reading %s", csv_path)
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception:
            logger.exception("Failed to read %s", csv_path)
            raise

    combined = pd.concat(frames, ignore_index=True)
    deduplicated = combined.drop_duplicates().reset_index(drop=True)
    logger.info(
        "Combined %s files into %s rows and removed %s duplicates",
        len(csv_files),
        len(deduplicated),
        len(combined) - len(deduplicated),
    )

    final_path = output_folder / "finaldata.csv"
    record_path = output_folder / "ingestedfiles.txt"
    deduplicated.to_csv(final_path, index=False)
    record_path.write_text(
        "".join(f"{path.name}\n" for path in csv_files),
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", final_path, record_path)
    return deduplicated


def main() -> None:
    """Run ingestion from the repository root."""
    configure_logging()
    merge_multiple_dataframe()


if __name__ == "__main__":
    main()
