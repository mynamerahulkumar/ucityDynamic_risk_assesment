"""Confusion matrix, Flask endpoints, and the HTTP API client."""

import hashlib
import unittest

import pandas as pd

import reporting
import wsgi
from app import app
from common import read_score
from helpers import Workspace, use_practice_project


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReportingAndApiTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace().__enter__()
        self.addCleanup(self.workspace.close)
        use_practice_project(self.workspace)
        self.client = app.test_client()

    def test_confusion_matrix_names_and_first_run_preservation(self):
        first = reporting.score_model("confusionmatrix.png")
        self.assertEqual(first.name, "confusionmatrix.png")
        self.assertTrue(first.read_bytes().startswith(b"\x89PNG"))
        original = first.read_bytes()

        second = reporting.score_model("confusionmatrix2.png")

        self.assertEqual(second.name, "confusionmatrix2.png")
        self.assertTrue(second.is_file())
        self.assertEqual(first.read_bytes(), original)

    def test_prediction_scoring_and_summary_endpoints(self):
        missing = self.client.post("/prediction", json={})
        self.assertEqual(missing.status_code, 400)

        absent = self.client.post("/prediction", json={"dataset_path": "testdata/missing.csv"})
        self.assertEqual(absent.status_code, 404)

        dataset = self.workspace.root / "testdata" / "testdata.csv"
        predicted = self.client.post("/prediction", json={"dataset_path": str(dataset)})
        self.assertEqual(predicted.status_code, 200)
        predictions = predicted.get_json()["predictions"]
        self.assertEqual(len(predictions), len(pd.read_csv(dataset)))

        score_path = self.workspace.root / "production_deployment" / "latestscore.txt"
        before = score_path.read_bytes()
        scored = self.client.get("/scoring")
        self.assertEqual(scored.status_code, 200)
        self.assertEqual(scored.get_json()["f1_score"], read_score(score_path))
        self.assertEqual(score_path.read_bytes(), before)
        candidate_score = self.workspace.root / "practicemodels" / "latestscore.txt"
        self.assertEqual(candidate_score.read_bytes(), before)

        summary = self.client.get("/summarystats")
        self.assertEqual(summary.status_code, 200)
        self.assertIsInstance(summary.get_json(), list)
        self.assertTrue(summary.get_json())

        options = self.client.open("/scoring", method="OPTIONS")
        self.assertEqual(options.status_code, 200)

    def test_diagnostics_endpoint_and_wsgi_app(self):
        production_model = self.workspace.root / "production_deployment" / "trainedmodel.pkl"
        before = _sha256(production_model)

        response = self.client.get("/diagnostics")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload["ingestion_time"], float)
        self.assertIsInstance(payload["training_time"], float)
        self.assertTrue(payload["missing_data"])
        self.assertTrue(payload["dependencies"])
        self.assertIn("package_name", payload["dependencies"][0])
        self.assertEqual(_sha256(production_model), before)
        self.assertIs(wsgi.app, app)


if __name__ == "__main__":
    unittest.main()
