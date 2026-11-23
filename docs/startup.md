# Run the project locally

Run every command from the repository root. The scripts read `config.json` from the current directory and use relative paths.

This guide follows the five Udacity steps: ingestion, training/scoring/deployment, diagnostics, reporting and the API, then process automation.

## 1. Create the environment

Python 3.14 is the interpreter this repository was run with. The packages are pinned in `requirements.txt`.

```bash
cd /Users/rahulkumar/Documents/GitHub/ucityDynamic_risk_assesment
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

After activation, `python` is the virtual environment. The checked-in `cronjob.txt` calls `.venv/bin/python` by its absolute path, so the environment needs to stay at `.venv`.

Confirm the configuration before the first run:

```bash
python -c "import json; print(json.load(open('config.json')))"
```

The submission configuration is:

| Key | Directory | Role |
| --- | --- | --- |
| `input_folder_path` | `sourcedata` | CSV files ingested for the final model |
| `output_folder_path` | `ingesteddata` | `finaldata.csv` and `ingestedfiles.txt` |
| `test_data_path` | `testdata` | Fixed evaluation file `testdata.csv` |
| `output_model_path` | `models` | Candidate model, score, and reports |
| `prod_deployment_path` | `production_deployment` | Deployed model, score, and ingestion record |

`practicedata` and `practicemodels` are the original practice locations. The drift fixtures live in `validation_data` and are used only in the isolated scenario section below.

## 2. Ingest, train, score, and deploy

These four commands are Steps 1 and 2. Run them in this order:

```bash
python ingestion.py
python training.py
python scoring.py
python deployment.py
```

What each command must leave behind:

| Command | Rubric result |
| --- | --- |
| `ingestion.py` | Discovers every CSV under `sourcedata` (`dataset3.csv`, `dataset4.csv`), concatenates them, drops duplicate rows, and writes `ingesteddata/finaldata.csv` plus `ingesteddata/ingestedfiles.txt` with one filename per line. |
| `training.py` | Fits logistic regression on `lastmonth_activity`, `lastyear_activity`, and `number_of_employees`. Target is `exited`. Saves `models/trainedmodel.pkl`. |
| `scoring.py` | Scores that model on `testdata/testdata.csv` with F1 and writes `models/latestscore.txt`. |
| `deployment.py` | Copies `trainedmodel.pkl`, `latestscore.txt`, and `ingestedfiles.txt` into `production_deployment/`. It does not train a new model. |

On the supplied source files this produces 26 ingested rows and a test F1 of about `0.3333`. That number is calculated. Do not type it into `latestscore.txt`.

Quick check:

```bash
cat ingesteddata/ingestedfiles.txt
cat models/latestscore.txt
cat production_deployment/latestscore.txt
ls production_deployment
```

The two score files and the two ingestion records must match. `production_deployment` must contain `trainedmodel.pkl`, `latestscore.txt`, and `ingestedfiles.txt`.

## 3. Start the API and make the first reports

Step 4 needs a running server. Use two terminals, both with the virtual environment activated and both in the repository root.

Terminal 1:

```bash
source .venv/bin/activate
python app.py
```

The app listens on port **8000**. Leave this terminal open. `API_PORT` can change the port for a test run; the normal local run uses 8000.

Terminal 2:

```bash
source .venv/bin/activate
python apicalls.py
python reporting.py
```

`apicalls.py` sends HTTP requests to `http://127.0.0.1:8000`. It does not import the Flask view functions. It writes `models/apireturns.txt`. `reporting.py` writes `models/confusionmatrix.png` for the **deployed** model on `testdata/testdata.csv`.

Call the four endpoints yourself:

```bash
curl -s -X POST http://127.0.0.1:8000/prediction \
  -H 'Content-Type: application/json' \
  -d '{"dataset_path": "testdata/testdata.csv"}'

curl -s http://127.0.0.1:8000/scoring
curl -s http://127.0.0.1:8000/summarystats
curl -s http://127.0.0.1:8000/diagnostics
```

Expected HTTP behavior:

| Call | Success |
| --- | --- |
| `POST /prediction` | 200 and one prediction per test row. A missing file returns 404. A request without `dataset_path` returns 400. |
| `GET /scoring` | 200 and the deployed model's F1 on `testdata/testdata.csv`. |
| `GET /summarystats` | 200 and a list of mean, median, and standard deviation for each numeric ingested column. |
| `GET /diagnostics` | 200 with ingestion time, training time, missing-data percentages, and installed plus latest versions for every package in `requirements.txt`. |

`/scoring` must not change `production_deployment/latestscore.txt`. Compare the file before and after the call. `/diagnostics` runs `ingestion.py` and `training.py` to time them. That can refresh the candidate files in `models/`. It does not call `deployment.py`, so the production model and baseline stay in place.

`/diagnostics` is the slow endpoint because it retrains and asks the package index for latest versions. `apicalls.py` allows several minutes for that call.

Stop the server with Ctrl+C in terminal 1 when the reports exist:

```text
models/apireturns.txt
models/confusionmatrix.png
models/trainedmodel.pkl
models/latestscore.txt
```

## 4. Run the automation on the real repository

With the source files already listed in `production_deployment/ingestedfiles.txt`, the live workflow has nothing new to ingest:

```bash
python fullprocess.py
```

The required result is a stop before drift checking, retraining, deployment, and second-run reports:

```text
No new data found. Stopping before drift checks, retraining, deployment, and reporting.
```

Leave `dataset3.csv` and `dataset4.csv` as the deployed record. Do not delete that record just to force the new-data branch on the submission copy.

`cronjob.txt` is the scheduler entry for this machine:

```text
*/10 * * * * cd /Users/rahulkumar/Documents/GitHub/ucityDynamic_risk_assesment && /Users/rahulkumar/Documents/GitHub/ucityDynamic_risk_assesment/.venv/bin/python fullprocess.py
```

Install it only if you want the job on this user account:

```bash
crontab cronjob.txt
crontab -l
```

On macOS, cron needs permission to read the project directory. Remove the job later with `crontab -e`. Installing it is optional for a local walkthrough. The rubric file is the one-line `cronjob.txt` included in the submission ZIP.

## 5. Prove the three automation branches in a copy

The no-drift and drift fixtures are judged against a model trained on **all 26 deduplicated practice rows**, scored on the unchanged `testdata/testdata.csv`. That practice baseline is about `0.5714`. The no-drift fixture then scores about `1.0`. The drift fixture scores about `0.0` for that same deployed model. The drift fixture's `0.0` is the old model's score on the new file, not the replacement model's test score.

Run these scenarios in a temporary copy so one branch cannot replace another branch's model, and so the sourcedata deployment in this repository stays intact.

```bash
COPY="$(mktemp -d)/risk-copy"
mkdir -p "$COPY"
cp ingestion.py training.py scoring.py deployment.py diagnostics.py reporting.py \
   app.py wsgi.py apicalls.py fullprocess.py common.py requirements.txt "$COPY/"
cp -R practicedata testdata validation_data "$COPY/"
cd "$COPY"
```

Practice deployment inside the copy:

```bash
python - << 'PY'
import json
from pathlib import Path
Path("config.json").write_text(json.dumps({
    "input_folder_path": "practicedata",
    "output_folder_path": "ingesteddata",
    "test_data_path": "testdata",
    "output_model_path": "practicemodels",
    "prod_deployment_path": "production_deployment",
}, indent=4))
PY
python ingestion.py
python training.py
python scoring.py
python deployment.py
cat production_deployment/latestscore.txt
```

Keep that score. The later comparisons use whatever value was just written.

### Scenario A — no new data

Leave `input_folder_path` as `practicedata` and run:

```bash
python fullprocess.py
```

Expected: the process stops. `production_deployment/trainedmodel.pkl` and `latestscore.txt` stay the same. `confusionmatrix2.png` and `apireturns2.txt` are not created.

### Scenario B — new data, no drift

Point the copy at the no-drift fixture and at `models` for later reports. Keep the practice model that is already in `production_deployment`.

```bash
python - << 'PY'
import json
from pathlib import Path
config = json.loads(Path("config.json").read_text())
config["input_folder_path"] = "validation_data/no_drift"
config["output_model_path"] = "models"
Path("config.json").write_text(json.dumps(config, indent=4))
Path("models").mkdir(exist_ok=True)
PY
python fullprocess.py
```

Expected log lines include a new file `no_drift.csv`, a new-data F1 at or above the practice baseline (about `1.0`), and the message that the deployed model and baseline stay in place. The copy then writes:

```text
models/confusionmatrix2.png
models/apireturns2.txt
```

If port 8000 is already serving this project, the script uses it. Otherwise it starts `app.py` for the API report and stops only the server it started. The API must be able to read this copy's `config.json`, so start it from the copy if you start it yourself.

### Scenario C — new data, drift

Start again from a fresh practice deployment. Do not continue in the directory left by Scenario B, because that directory has already ingested `no_drift.csv`.

Repeat the copy and practice-deployment commands, then:

```bash
python - << 'PY'
import json
from pathlib import Path
config = json.loads(Path("config.json").read_text())
config["input_folder_path"] = "validation_data/drift"
config["output_model_path"] = "models"
Path("config.json").write_text(json.dumps(config, indent=4))
Path("models").mkdir(exist_ok=True)
PY
python fullprocess.py
```

Expected: `drift.csv` is new, the deployed model's F1 on that file is lower than the practice baseline (about `0.0`), and the script retrains, scores the replacement on `testdata/testdata.csv`, writes `models/latestscore.txt`, and only then copies the model, that score, and the new `ingestedfiles.txt` into `production_deployment`. Reports `confusionmatrix2.png` and `apireturns2.txt` are written after deployment.

The repository's own `production_deployment` is unchanged by these copies. Return to the original directory afterward. Its `config.json` stays on `sourcedata` and `models`.

## 6. Run the test suite

From the original repository root:

```bash
cd /Users/rahulkumar/Documents/GitHub/ucityDynamic_risk_assesment
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The tests build temporary workspaces. They cover ingestion, training, F1 scoring, deployment, diagnostics, the four endpoints, baseline preservation, the no-new-data stop, the no-drift branch, and the drift branch. A passing run is the local regression check for the rubric.

## 7. Files a complete local run produces

After Steps 1–4 on `sourcedata`, plus the second-run pair saved from a new-data scenario:

```text
ingesteddata/finaldata.csv
ingesteddata/ingestedfiles.txt
models/trainedmodel.pkl
models/latestscore.txt
models/confusionmatrix.png
models/apireturns.txt
models/confusionmatrix2.png
models/apireturns2.txt
production_deployment/trainedmodel.pkl
production_deployment/latestscore.txt
production_deployment/ingestedfiles.txt
cronjob.txt
```

The submission archive that packages those outputs is `submission/Dynamic_Risk_Assessment_Submission.zip`.
