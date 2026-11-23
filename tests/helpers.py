"""Temporary project workspaces for tests. These never use the real production directory."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deployment
import ingestion
import scoring
import training

DEFAULT_CONFIG = {
    "input_folder_path": "practicedata",
    "output_folder_path": "ingesteddata",
    "test_data_path": "testdata",
    "output_model_path": "practicemodels",
    "prod_deployment_path": "production_deployment",
}


class Workspace:
    """A temporary working directory with its own config and data copies."""

    def __init__(self):
        self.temp = None
        self.root = None
        self.previous = None

    def __enter__(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = Path.cwd()
        os.chdir(self.root)
        return self

    def close(self) -> None:
        if self.previous is not None:
            os.chdir(self.previous)
            self.previous = None
        if self.temp is not None:
            self.temp.cleanup()
            self.temp = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def write_config(self, **overrides) -> dict:
        config = dict(DEFAULT_CONFIG)
        config.update(overrides)
        (self.root / "config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
        return config


def use_practice_project(workspace: Workspace, **config_overrides) -> None:
    """Ingest, train, score, and deploy the supplied practice data inside a workspace."""
    shutil.copytree(REPO_ROOT / "practicedata", workspace.root / "practicedata")
    shutil.copytree(REPO_ROOT / "testdata", workspace.root / "testdata")
    workspace.write_config(**config_overrides)
    ingestion.merge_multiple_dataframe()
    training.train_model()
    scoring.score_candidate_model()
    deployment.store_model_into_pickle()
