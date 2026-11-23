"""New-data detection and the no-drift and drift automation branches."""

import contextlib
import hashlib
import io
import shutil
import socket
import unittest
from unittest import mock

import fullprocess
from common import read_score
from helpers import REPO_ROOT, Workspace, use_practice_project


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _printed_score(output: str, label: str) -> float:
    prefix = label + "="
    for line in output.splitlines():
        if line.startswith(prefix):
            return float(line.split("=", 1)[1])
    raise AssertionError(f"{label} was not printed:\n{output}")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FullProcessTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace().__enter__()
        self.addCleanup(self.workspace.close)
        use_practice_project(self.workspace)
        self.production = self.workspace.root / "production_deployment"
        self.baseline_hash = _sha256(self.production / "trainedmodel.pkl")
        self.baseline_score = (self.production / "latestscore.txt").read_bytes()
        self.baseline_files = (self.production / "ingestedfiles.txt").read_text(encoding="utf-8")

    def _prepare_new_data(self, fixture: str) -> None:
        shutil.copytree(REPO_ROOT / "validation_data", self.workspace.root / "validation_data")
        self.workspace.write_config(
            input_folder_path=f"validation_data/{fixture}",
            output_model_path="models",
        )
        model_dir = self.workspace.root / "models"
        model_dir.mkdir()
        (model_dir / "confusionmatrix.png").write_bytes(b"first-matrix")
        (model_dir / "apireturns.txt").write_text("first-api\n", encoding="utf-8")

    def _production_snapshot(self):
        return (
            _sha256(self.production / "trainedmodel.pkl"),
            (self.production / "latestscore.txt").read_bytes(),
            (self.production / "ingestedfiles.txt").read_text(encoding="utf-8"),
        )

    def test_no_new_data_stops_before_drift_and_reports(self):
        before = self._production_snapshot()
        ingested = (self.workspace.root / "ingesteddata" / "finaldata.csv").read_bytes()

        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = fullprocess.run_full_process(run_reports=False)

        self.assertEqual(status, "no_new_data")
        self.assertIn("No new data found", output.getvalue())
        self.assertEqual(self._production_snapshot(), before)
        self.assertEqual((self.workspace.root / "ingesteddata" / "finaldata.csv").read_bytes(), ingested)
        self.assertFalse((self.workspace.root / "models" / "confusionmatrix2.png").exists())
        self.assertFalse((self.workspace.root / "practicemodels" / "apireturns2.txt").exists())

    def test_equal_score_keeps_the_deployed_model(self):
        extra = self.workspace.root / "practicedata" / "extra.csv"
        shutil.copy(self.workspace.root / "practicedata" / "dataset1.csv", extra)
        before = self._production_snapshot()
        baseline = read_score(self.production / "latestscore.txt")

        with mock.patch("fullprocess.scoring.score_model", return_value=baseline):
            status = fullprocess.run_full_process(run_reports=False)

        self.assertEqual(status, "no_drift")
        self.assertEqual(self._production_snapshot(), before)
        self.assertEqual(fullprocess.find_new_files(), ["extra.csv"])
        self.assertIn("extra.csv", (self.workspace.root / "ingesteddata" / "ingestedfiles.txt").read_text())

    def test_no_drift_fixture_keeps_production_and_writes_second_reports(self):
        self._prepare_new_data("no_drift")
        port = _free_port()
        before = self._production_snapshot()

        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = fullprocess.run_full_process(
                run_reports=True,
                base_url=f"http://127.0.0.1:{port}",
            )

        new_score = _printed_score(output.getvalue(), "new_data_f1_score")
        baseline = _printed_score(output.getvalue(), "production_baseline_f1_score")
        self.assertGreaterEqual(new_score, baseline)
        self.assertAlmostEqual(new_score, 1.0, places=4)
        self.assertAlmostEqual(baseline, 0.5714, places=4)
        self.assertEqual(status, "no_drift")
        self.assertEqual(self._production_snapshot(), before)
        model_dir = self.workspace.root / "models"
        self.assertEqual((model_dir / "confusionmatrix.png").read_bytes(), b"first-matrix")
        self.assertEqual((model_dir / "apireturns.txt").read_text(encoding="utf-8"), "first-api\n")
        self.assertTrue((model_dir / "confusionmatrix2.png").read_bytes().startswith(b"\x89PNG"))
        report = (model_dir / "apireturns2.txt").read_text(encoding="utf-8")
        self.assertIn("Predictions", report)
        self.assertIn("F1 score", report)
        self.assertIn("Summary statistics", report)
        self.assertIn("Ingestion execution time", report)
        self.assertIn("Training execution time", report)
        self.assertIn("Missing data", report)
        self.assertIn("installed_version", report)
        self.assertIn("latest_version", report)

    def test_drift_fixture_retrains_and_deploys_without_touching_first_reports(self):
        self._prepare_new_data("drift")

        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = fullprocess.run_full_process(run_reports=False)

        new_score = _printed_score(output.getvalue(), "new_data_f1_score")
        baseline = _printed_score(output.getvalue(), "production_baseline_f1_score")
        self.assertLess(new_score, baseline)
        self.assertAlmostEqual(new_score, 0.0, places=4)
        self.assertEqual(status, "drift")
        self.assertNotEqual(_sha256(self.production / "trainedmodel.pkl"), self.baseline_hash)
        self.assertNotEqual((self.production / "latestscore.txt").read_bytes(), self.baseline_score)
        self.assertIn("drift.csv", (self.production / "ingestedfiles.txt").read_text(encoding="utf-8"))
        self.assertEqual(
            (self.production / "latestscore.txt").read_bytes(),
            (self.workspace.root / "models" / "latestscore.txt").read_bytes(),
        )
        self.assertEqual(
            (self.production / "trainedmodel.pkl").read_bytes(),
            (self.workspace.root / "models" / "trainedmodel.pkl").read_bytes(),
        )
        model_dir = self.workspace.root / "models"
        self.assertEqual((model_dir / "confusionmatrix.png").read_bytes(), b"first-matrix")
        self.assertEqual((model_dir / "apireturns.txt").read_text(encoding="utf-8"), "first-api\n")
        self.assertFalse((model_dir / "confusionmatrix2.png").exists())
        self.assertFalse((model_dir / "apireturns2.txt").exists())


if __name__ == "__main__":
    unittest.main()
