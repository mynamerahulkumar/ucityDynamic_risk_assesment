"""Training, F1 scoring, and deployment artifact copying."""

import pickle
import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

import deployment
import scoring
import training
from common import FEATURES, TARGET
from helpers import Workspace, use_practice_project


class TrainingScoringDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace().__enter__()
        self.addCleanup(self.workspace.close)

    def _write_training_csv(self, rows: str) -> None:
        ingested = self.workspace.root / "ingesteddata"
        ingested.mkdir()
        (ingested / "finaldata.csv").write_text(
            "corporation,lastmonth_activity,lastyear_activity,number_of_employees,exited\n" + rows,
            encoding="utf-8",
        )
        self.workspace.write_config()

    def test_training_saves_logistic_regression_with_three_features(self):
        use_practice_project(self.workspace)

        model_path = self.workspace.root / "practicemodels" / "trainedmodel.pkl"
        with model_path.open("rb") as handle:
            model = pickle.load(handle)

        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(list(model.feature_names_in_), FEATURES)
        self.assertNotIn("corporation", model.feature_names_in_)
        self.assertEqual(model.random_state, 0)
        self.assertEqual(model.solver, "liblinear")

    def test_training_rejects_one_class_and_missing_columns(self):
        self._write_training_csv("alpha,1,2,3,0\nbeta,2,3,4,0\n")
        with self.assertRaises(ValueError):
            training.train_model()

        ingested = self.workspace.root / "ingesteddata" / "finaldata.csv"
        ingested.write_text("corporation,exited\nalpha,1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            training.train_model()

    def test_score_is_written_only_when_a_path_is_provided(self):
        self._write_training_csv("alpha,0,1,2,0\nalpha2,0,1,3,0\nbeta,50,60,2,1\nbeta2,55,70,1,1\n")
        model = training.train_model()
        dataframe = pd.read_csv(self.workspace.root / "ingesteddata" / "finaldata.csv")
        score_path = self.workspace.root / "practicemodels" / "latestscore.txt"

        returned = scoring.score_model(model, dataframe)
        self.assertFalse(score_path.exists())
        self.assertGreaterEqual(returned, 0.0)
        self.assertLessEqual(returned, 1.0)

        written = scoring.score_model(model, dataframe, score_path=score_path)
        self.assertEqual(written, returned)
        self.assertEqual(float(score_path.read_text(encoding="utf-8").strip()), written)
        self.assertIn(TARGET, dataframe.columns)

    def test_candidate_score_uses_test_data_file(self):
        use_practice_project(self.workspace)
        score_path = self.workspace.root / "practicemodels" / "latestscore.txt"
        self.assertTrue(score_path.is_file())
        self.assertGreater(float(score_path.read_text(encoding="utf-8").strip()), 0)

    def test_deployment_copies_three_artifacts_and_rejects_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            self.workspace.write_config()
            deployment.store_model_into_pickle()

        use_practice_project(self.workspace)
        production = self.workspace.root / "production_deployment"
        for name in ("trainedmodel.pkl", "latestscore.txt", "ingestedfiles.txt"):
            source_dir = "ingesteddata" if name == "ingestedfiles.txt" else "practicemodels"
            self.assertEqual(
                (production / name).read_bytes(),
                (self.workspace.root / source_dir / name).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
