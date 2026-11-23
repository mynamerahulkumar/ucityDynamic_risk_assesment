# Initial project audit

This audit records the repository as it existed before implementation on 23 November 2026. It is the baseline assessment required before coding. The completed-run evidence is in `FINAL_RUBRIC_VERIFICATION.md`.

## Existing files and their purposes

| Path | Purpose |
| --- | --- |
| `ingestion.py` | Starter for combining source CSVs and recording ingested filenames. |
| `training.py` | Starter for logistic-regression training and `trainedmodel.pkl`. |
| `scoring.py` | Starter for F1 scoring. The required `score_model(model, dataframe, score_path=None)` signature is present and empty. |
| `deployment.py` | Starter for copying the model, score, and ingestion record into production. |
| `diagnostics.py` | Starter for predictions, summary statistics, missing data, timing, and dependency checks. |
| `reporting.py` | Starter for the deployed-model confusion matrix. |
| `app.py` | Starter Flask app with `/prediction`, `/scoring`, `/summarystats`, and `/diagnostics`. |
| `wsgi.py` | Exposes the Flask `app` object. This file is already usable. |
| `apicalls.py` | Starter API client. The file does not parse. |
| `fullprocess.py` | Comments describing the new-data and drift workflow. No executable logic. |
| `config.json` | Relative directory settings. Still on the practice paths. |
| `requirements.txt` | Pinned pandas, numpy, scikit-learn, Flask, requests, matplotlib, and related packages. |
| `README.md` | Course instructions for the unfinished starter. |
| `practicedata/` | `dataset1.csv` (17 rows) and `dataset2.csv` (19 rows). |
| `sourcedata/` | `dataset3.csv` and `dataset4.csv` (26 rows combined, already unique). |
| `testdata/testdata.csv` | 5 labeled evaluation rows. |
| `validation_data/drift/` and `validation_data/no_drift/` | Eight-row fixtures with opposite labels. Must stay unchanged. |
| `ingesteddata/`, `practicemodels/`, `models/`, `production_deployment/` | Empty output directories kept by `.gitkeep`. |
| `.gitignore` | Ignores virtual environments, caches, and `*.zip`. |

## Existing implementation status

The Python pipeline is a starter. `wsgi.py` imports `app`. Every other script stops at a docstring, a comment, or an empty function. No model, score, confusion matrix, API return file, or cron entry exists.

`config.json` currently contains:

```json
{
    "input_folder_path": "practicedata",
    "output_folder_path": "ingesteddata",
    "test_data_path": "testdata",
    "output_model_path": "practicemodels",
    "prod_deployment_path": "production_deployment"
}
```

Step 5 and the submission need `input_folder_path` set to `sourcedata` and `output_model_path` set to `models` after the automation scenarios are finished.

## Missing functions

- `ingestion.merge_multiple_dataframe` has no body.
- `training.train_model` does not load data, fit a model, or save a pickle. It also has no `__main__` block, so `python training.py` does nothing.
- `scoring.score_model` returns `None`. `scoring.py` and `deployment.py` also have no `__main__` block.
- `diagnostics.model_predictions`, `dataframe_summary`, `missing_data`, `execution_time`, and `outdated_packages_list` are empty.
- `reporting.score_model` does not build or save a figure.
- All four Flask view functions return `None`.
- `apicalls.py` does not call the API or write `apireturns.txt`.
- `fullprocess.py` does not compare files, score drift, retrain, deploy, or launch the second-run reports.
- `cronjob.txt`, `tests/`, and `FINAL_RUBRIC_VERIFICATION.md` are absent.

## Incomplete functions

The training stub constructs a `LogisticRegression` with the required settings and then discards it:

```python
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   n_jobs=None, penalty='l2', random_state=0,
                   solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
```

Those arguments must be kept. On scikit-learn 1.8 they still fit, with a deprecation warning for `penalty`.

`deployment.store_model_into_pickle(model)` is the starter name, but the comments require copying files rather than pickling the argument. The parameter stays optional so existing call style still works.

## Incorrect implementations

- `apicalls.py` assigns comments to `response1` through `responses`. That is a `SyntaxError`, so the module cannot be imported.
- Several scripts import Flask, session, and split helpers they do not use. Those imports do not implement the required behavior.
- Import-time `config.json` reads would freeze paths for the whole process. Functions need to read the config when they run so tests and Step 5 see the current file.
- `diagnostics.py` imports `timeit` but never times `ingestion.py` and `training.py` as separate subprocesses.

## Existing dependencies

`requirements.txt` pins Flask 3.1.3, pandas 2.3.3, numpy 2.3.5, scikit-learn 1.8.0, matplotlib 3.10.8, requests 2.34.2, seaborn 0.13.2, joblib 1.5.3, and their supporting packages. The local interpreter is Python 3.14.6 at `/opt/homebrew/bin/python3.14`. No virtual environment was present at audit time. The pins should not be changed unless a pin cannot run.

## Dataset structure

Every supplied CSV has the same columns:

`corporation`, `lastmonth_activity`, `lastyear_activity`, `number_of_employees`, `exited`

- Model features are the three numeric activity and employee columns.
- `exited` is the target. `1` means the client left.
- `corporation` is an identifier and must not be a model input.
- Practice files contain 36 rows and 26 rows after duplicate removal. None of the supplied files contain missing values.
- Source files contain 26 unique rows.
- Test data contains 5 rows.
- Each validation fixture contains 8 rows, both classes, and the same feature values with opposite labels.

## Configuration problems

- Final automation and the submission artifacts belong under `models/`, while the starter writes to `practicemodels/`.
- The final input directory is `sourcedata`, not `practicedata`.
- Paths are relative and only work when commands run from the repository root.
- Output directories exist but contain no artifacts, so training, scoring, deployment, and the API would fail until ingestion and training have run.
- `.gitignore` ignores every `*.zip`. The submission archive can still be created on disk; git will not track it.

## Potential runtime errors

- Importing `apicalls.py` raises `SyntaxError`.
- Running the other scripts before implementation succeeds without producing artifacts, or returns `None` from the API.
- `python app.py` would start Flask and then fail inside the empty views once a client called them.
- Training cannot accept NaN features. The supplied files have none; later data still needs an explicit complete-row check.
- A one-class dataset would make logistic regression fail or produce a meaningless F1 score. Training should reject that case.
- Diagnostics timing must not call `deployment.py`, or it would replace the production model.
- `/scoring` and the drift check must not pass `score_path`, or they would overwrite `latestscore.txt`.
- `fullprocess.py` must stop when every current CSV name is already in the deployed ingestion record. Generating second-run reports in that case would violate the rubric.

## Rubric requirements not yet satisfied

- Step 1: automatic CSV discovery, concatenation, duplicate removal, `finaldata.csv`, and `ingestedfiles.txt`.
- Step 2: logistic regression training, pickle persistence, F1 scoring, `latestscore.txt`, and deployment of all three production files.
- Step 3: deployed-model predictions, mean/median/standard deviation, missing-data percentages, separate ingestion and training timings, and dependency versions. Diagnostics must leave the production model in place.
- Step 4: confusion matrix, four HTTP 200 endpoints, an HTTP API client, and `apireturns.txt`.
- Step 5: new-data detection, no-new-data early exit, F1 drift comparison, conditional retraining and redeployment, `confusionmatrix2.png`, `apireturns2.txt`, preservation of the first-run files, and `cronjob.txt`.

## Recommended implementation sequence

1. Ingestion, then training, scoring, and deployment.
2. Diagnostics, with production files left untouched.
3. Reporting, Flask endpoints, WSGI, and the HTTP client.
4. `fullprocess.py`, then `cronjob.txt` and README run instructions.
5. Isolated tests for ingestion, model artifacts, diagnostics, the API, and both drift branches.
6. Practice-model scenario runs, then a restored `sourcedata` / `models` pipeline and the submission ZIP.

The validation fixtures stay read-only. Scenario runs use temporary directories so one branch cannot replace another branch's deployed model.
