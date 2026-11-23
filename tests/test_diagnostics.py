"""Diagnostics for predictions, statistics, timing, and dependencies."""

import hashlib
import subprocess
import unittest
from unittest import mock

import pandas as pd

import diagnostics
from common import repo_root
from helpers import Workspace, use_practice_project


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace().__enter__()
        self.addCleanup(self.workspace.close)

    def test_predictions_match_row_count_and_statistics_cover_numeric_columns(self):
        use_practice_project(self.workspace)
        dataframe = pd.read_csv(self.workspace.root / "testdata" / "testdata.csv")

        predictions = diagnostics.model_predictions(dataframe)

        self.assertEqual(len(predictions), len(dataframe))
        self.assertTrue(all(prediction in (0, 1) for prediction in predictions))

        summary = diagnostics.dataframe_summary()
        self.assertIsInstance(summary, list)
        columns = [item["column"] for item in summary]
        self.assertNotIn("corporation", columns)
        self.assertIn("lastmonth_activity", columns)
        for item in summary:
            self.assertEqual(set(item), {"column", "mean", "median", "std"})
            self.assertIsInstance(item["mean"], float)

        missing = diagnostics.missing_data()
        self.assertEqual([item["column"] for item in missing], list(dataframe.columns))
        for item in missing:
            self.assertEqual(item["missing_count"], 0)
            self.assertEqual(item["total_count"], 26)
            self.assertEqual(item["percentage"], 0.0)

    def test_missing_percentages_handle_zero_rows(self):
        self.workspace.write_config()
        ingested = self.workspace.root / "ingesteddata"
        ingested.mkdir()
        (ingested / "finaldata.csv").write_text(
            "corporation,lastmonth_activity,lastyear_activity,number_of_employees,exited\n",
            encoding="utf-8",
        )

        missing = diagnostics.missing_data()

        self.assertEqual(len(missing), 5)
        self.assertTrue(all(item["total_count"] == 0 and item["percentage"] == 0.0 for item in missing))

    def test_timing_runs_both_scripts_without_replacing_production(self):
        use_practice_project(self.workspace)
        production_model = self.workspace.root / "production_deployment" / "trainedmodel.pkl"
        production_score = self.workspace.root / "production_deployment" / "latestscore.txt"
        before = (_sha256(production_model), production_score.read_bytes())

        timings = diagnostics.execution_time()

        self.assertEqual(len(timings), 2)
        self.assertTrue(all(timing >= 0 for timing in timings))
        self.assertEqual((_sha256(production_model), production_score.read_bytes()), before)

    def test_dependency_table_lists_requirements_and_survives_lookup_failure(self):
        with mock.patch("diagnostics.latest_version", return_value="9.9.9"):
            rows = diagnostics.outdated_packages_list()

        names = [row["package_name"] for row in rows]
        self.assertEqual(names, diagnostics.parse_requirements(repo_root() / "requirements.txt"))
        self.assertIn("pandas", names)
        self.assertIn("scikit-learn", names)
        self.assertIn("Flask", names)
        for row in rows:
            self.assertEqual(
                set(row),
                {"package_name", "installed_version", "latest_version"},
            )
        pandas_row = next(row for row in rows if row["package_name"] == "pandas")
        self.assertEqual(pandas_row["installed_version"], "2.3.3")
        self.assertEqual(pandas_row["latest_version"], "9.9.9")

        self.assertEqual(
            diagnostics.parse_latest_version("pandas (1.2.3)\n  LATEST:    1.2.3\n"),
            "1.2.3",
        )
        self.assertIsNone(diagnostics.parse_latest_version("no version here"))

        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="offline")
        with mock.patch("diagnostics.subprocess.run", return_value=failed):
            self.assertEqual(diagnostics.latest_version("pandas"), "unavailable")
        with mock.patch(
            "diagnostics.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=1),
        ):
            self.assertEqual(diagnostics.latest_version("pandas"), "unavailable")


if __name__ == "__main__":
    unittest.main()
