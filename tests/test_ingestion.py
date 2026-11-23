"""Ingestion discovery, concatenation, and duplicate removal."""

import shutil
import unittest
from pathlib import Path

import pandas as pd

import ingestion
from helpers import REPO_ROOT, Workspace


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace().__enter__()
        self.addCleanup(self.workspace.close)

    def test_discovers_csvs_removes_duplicates_and_ignores_unrelated_files(self):
        source = self.workspace.root / "input"
        source.mkdir()
        (source / "first.csv").write_text(
            "corporation,lastmonth_activity,lastyear_activity,number_of_employees,exited\n"
            "alpha,1,2,3,0\n"
            "beta,4,5,6,1\n",
            encoding="utf-8",
        )
        (source / "second.csv").write_text(
            "corporation,lastmonth_activity,lastyear_activity,number_of_employees,exited\n"
            "beta,4,5,6,1\n"
            "gamma,7,8,9,0\n",
            encoding="utf-8",
        )
        (source / "third.csv").write_text(
            "corporation,lastmonth_activity,lastyear_activity,number_of_employees,exited\n"
            "delta,10,11,12,1\n",
            encoding="utf-8",
        )
        (source / "notes.txt").write_text("not a dataset\n", encoding="utf-8")
        (source / ".hidden.csv").write_text(
            "corporation,lastmonth_activity,lastyear_activity,number_of_employees,exited\n"
            "hidden,9,9,9,1\n",
            encoding="utf-8",
        )
        (source / "finaldata.csv").write_text(
            "corporation,lastmonth_activity,lastyear_activity,number_of_employees,exited\n"
            "generated,1,1,1,1\n",
            encoding="utf-8",
        )
        self.workspace.write_config(input_folder_path="input", output_folder_path="output")

        result = ingestion.merge_multiple_dataframe()

        self.assertEqual(len(result), 4)
        self.assertNotIn("hidden", result["corporation"].tolist())
        self.assertNotIn("generated", result["corporation"].tolist())
        saved = pd.read_csv(self.workspace.root / "output" / "finaldata.csv")
        self.assertEqual(len(saved), 4)
        record = (self.workspace.root / "output" / "ingestedfiles.txt").read_text(encoding="utf-8")
        self.assertEqual(record.splitlines(), ["first.csv", "second.csv", "third.csv"])

    def test_empty_input_directory_raises_after_creating_output(self):
        (self.workspace.root / "input").mkdir()
        self.workspace.write_config(input_folder_path="input", output_folder_path="output")

        with self.assertRaises(FileNotFoundError):
            ingestion.merge_multiple_dataframe()

        self.assertTrue((self.workspace.root / "output").is_dir())

    def test_practice_files_deduplicate_to_26_rows(self):
        shutil.copytree(REPO_ROOT / "practicedata", self.workspace.root / "practicedata")
        self.workspace.write_config()

        result = ingestion.merge_multiple_dataframe()

        self.assertEqual(len(result), 26)
        record = (self.workspace.root / "ingesteddata" / "ingestedfiles.txt").read_text(encoding="utf-8")
        self.assertEqual(record.splitlines(), ["dataset1.csv", "dataset2.csv"])


if __name__ == "__main__":
    unittest.main()
