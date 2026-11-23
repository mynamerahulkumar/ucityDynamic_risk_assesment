# A Dynamic Risk Assessment System

A completed workflow for Udacity's **ML Model Scoring and Monitoring** course. It trains a logistic regression model that predicts whether a corporate client will leave (`exited`), deploys that model, and checks later data for drift.

The model uses `lastmonth_activity`, `lastyear_activity`, and `number_of_employees`. `corporation` is not a model input.

## Set up

Run commands from the repository root. The scripts read `config.json` from the current directory.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

This environment was prepared with Python 3.14.6 and the pinned versions in `requirements.txt`.

## Configuration

`config.json` controls every input and output directory:

| Setting | Final value |
| --- | --- |
| `input_folder_path` | `sourcedata` |
| `output_folder_path` | `ingesteddata` |
| `test_data_path` | `testdata` |
| `output_model_path` | `models` |
| `prod_deployment_path` | `production_deployment` |

The original practice step uses `practicedata` and `practicemodels`. Automation scenarios can point `input_folder_path` at `validation_data/no_drift` or `validation_data/drift`. Leave those fixture files unchanged.

## Run the pipeline

```bash
python ingestion.py
python training.py
python scoring.py
python deployment.py
```

That writes:

- `ingesteddata/finaldata.csv` and `ingesteddata/ingestedfiles.txt`
- `models/trainedmodel.pkl` and `models/latestscore.txt`
- copies of the model, score, and ingestion record in `production_deployment/`

Start the API, then build the first reports in a second terminal:

```bash
python app.py
python apicalls.py
python reporting.py
```

The API listens on port **8000**. `apicalls.py` calls the running server with HTTP requests. It does not import the Flask view functions. The first reports are `models/apireturns.txt` and `models/confusionmatrix.png`.

| Method | Endpoint | Response |
| --- | --- | --- |
| POST | `/prediction` | Predictions for JSON `{"dataset_path": "testdata/testdata.csv"}`. |
| GET | `/scoring` | F1 score of the deployed model on `testdata/testdata.csv`. This does not rewrite `latestscore.txt`. |
| GET | `/summarystats` | Mean, median, and standard deviation for each numeric ingested column. |
| GET | `/diagnostics` | Ingestion time, training time, missing-data percentages, and dependency versions. |

`/diagnostics` runs `ingestion.py` and `training.py` to measure them. That refreshes the candidate files under `output_model_path`. It does not call `deployment.py`, so the production model and baseline stay in place.

## Process automation

`python fullprocess.py` compares CSV names in `input_folder_path` with `production_deployment/ingestedfiles.txt`.

- No new file: it prints that there is no new data and stops. It does not check drift, retrain, redeploy, or write second-run reports.
- New files: it ingests them and scores the **deployed** model on the new `finaldata.csv` without saving that score.
- A lower F1 than `production_deployment/latestscore.txt` retrains on the ingested data, scores the replacement on `testdata/testdata.csv`, saves `models/latestscore.txt`, and then deploys the model, score, and ingestion record together.
- An equal or higher F1 leaves the deployed model and baseline unchanged.
- After either new-data result, it writes `models/confusionmatrix2.png` and `models/apireturns2.txt` from the model that is deployed at that moment. `confusionmatrix.png` and `apireturns.txt` are left as they are.

If port 8000 is already serving this API, the second-run client uses it. Otherwise `fullprocess.py` starts `app.py` for the API calls and stops only the server it started.

`validation_data/no_drift/no_drift.csv` and `validation_data/drift/drift.csv` are synthetic control-flow fixtures. With the practice-data model, the no-drift fixture scores at the top of the F1 range and the drift fixture scores at the bottom. The initial test-data baseline is about 0.5714. Those values are calculated during the run. Do not edit `latestscore.txt` to force a branch.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use temporary directories. They do not write the repository's `production_deployment` directory.

## Cron

`cronjob.txt` runs `fullprocess.py` every 10 minutes with this machine's project path and virtual-environment Python:

```bash
crontab cronjob.txt
crontab -l
```

Remove it later with `crontab -r` or by editing `crontab -e`. On macOS, cron needs permission to read the project directory. This repository does not install the job into the user crontab automatically, and a cron run was not claimed for this environment.

`POST /prediction` example:

```bash
curl -s -X POST http://127.0.0.1:8000/prediction \
  -H 'Content-Type: application/json' \
  -d '{"dataset_path": "testdata/testdata.csv"}'
```

## Data

The CSV files are fabricated corporate records. `practicedata` is the original training sample. `sourcedata` is the later source drop used for the final configuration. `testdata` is the fixed evaluation set used for `latestscore.txt`, the scoring endpoint, and the confusion matrix.

## License

See [LICENSE](LICENSE).
