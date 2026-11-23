# ROLE: Senior Python ML Engineer, MLOps Engineer, and Udacity Project Rubric Specialist

You are responsible for completing my existing Udacity project, **A Dynamic Risk Assessment System**, end to end.

## 1. Project repository

GitHub repository:

https://github.com/mynamerahulkumar/ucityDynamic_risk_assesment

Project title: A Dynamic Risk Assessment System  
Course: Udacity ML Model Scoring and Monitoring

You must work directly inside the currently opened repository. Do not create a separate project or replace the existing repository structure.

The objective is to implement every requirement in the project instructions and satisfy every applicable rubric point.

**Important:** Do not simply generate code that looks correct. Implement the project, execute it, test all functions, validate the API, verify the automation scenarios, and correct all failures.

---

## 2. Mandatory instructions before coding

First, inspect the complete repository.

Read and understand:

- README.md
- config.json
- requirements.txt
- All Python scripts
- All supplied CSV datasets
- Existing directory structure
- .gitignore
- Existing project validation fixtures

Inspect these existing files individually:

1. ingestion.py
2. training.py
3. scoring.py
4. deployment.py
5. diagnostics.py
6. reporting.py
7. app.py
8. wsgi.py
9. apicalls.py
10. fullprocess.py

Inspect the actual implementation in every file before modifying it.

Do not assume that a function or script is missing until you inspect it.

Create an initial project assessment in `PROJECT_AUDIT.md` containing:

- Existing files and their purposes
- Existing implementation status
- Missing functions
- Incomplete functions
- Incorrect implementations
- Existing dependencies
- Dataset structure
- Configuration problems
- Potential runtime errors
- Rubric requirements not yet satisfied
- Recommended implementation sequence

Use the Udacity project requirements described below as the acceptance criteria.

Do not stop after writing the audit. Continue implementing the complete project.

---

# 3. Existing repository structure

Preserve and use the existing directories:

```text
ucityDynamic_risk_assesment/
│
├── ingestion.py
├── training.py
├── scoring.py
├── deployment.py
├── diagnostics.py
├── reporting.py
├── app.py
├── wsgi.py
├── apicalls.py
├── fullprocess.py
│
├── config.json
├── requirements.txt
├── README.md
│
├── practicedata/
│   ├── dataset1.csv
│   └── dataset2.csv
│
├── sourcedata/
│   ├── dataset3.csv
│   └── dataset4.csv
│
├── testdata/
│   └── testdata.csv
│
├── ingesteddata/
├── practicemodels/
├── models/
├── production_deployment/
│
└── validation_data/
    ├── drift/
    └── no_drift/
```

Do not delete or overwrite the supplied datasets.

The validation fixtures must remain separate from the normal ingestion directories.

Preserve the existing `.gitkeep` files where appropriate.

---

# 4. Project business objective

Build a dynamic machine learning system that predicts whether a corporate client is likely to exit a contract.

The business scenario has approximately 10,000 corporate clients. The company needs an attrition-risk model to identify clients who may leave so that managers can prioritize customer outreach.

The target column is:

- `exited`: 1 means the client exited.
- `exited`: 0 means the client did not exit.

The input columns include:

- corporation
- lastmonth_activity
- lastyear_activity
- number_of_employees
- exited

Use the three numerical model features specified in the starter project:

```python
FEATURES = [
    "lastmonth_activity",
    "lastyear_activity",
    "number_of_employees"
]
```

Use `exited` as the target.

Do not use `corporation` as a model input.

Preserve the Logistic Regression approach in the existing starter `training.py`. Do not unnecessarily replace the requested algorithm or change its expected settings.

---

# 5. Implementation requirements — Step 1: Data ingestion

File: `ingestion.py`

Implement a complete ingestion pipeline.

Required functionality:

1. Read configuration from config.json.
2. Read the configured input directory.
3. Discover all CSV files automatically.
4. Do not hardcode dataset filenames.
5. Support a variable number of CSV files.
6. Load each CSV using pandas.
7. Combine all datasets into one DataFrame.
8. Remove duplicate records.
9. Save the resulting DataFrame as:

```text
ingesteddata/finaldata.csv
```

10. Record every ingested filename in:

```text
ingesteddata/ingestedfiles.txt
```

Store one filename per line.

11. Create destination directories when necessary.
12. Handle empty input directories appropriately.
13. Exclude generated files, hidden files, and non-CSV files.
14. Use robust error handling.
15. Provide useful logging.
16. Ensure repeated execution does not accidentally ingest the generated output file as input.

Expected behavior:

When the input directory contains two datasets, ingestion must automatically find both, concatenate their data, remove duplicates, and save the final dataset.

When a third CSV file is added, the next execution must discover it without changing the Python code.

Do not hardcode the number of files.

### Step 1 acceptance tests

- Add a temporary CSV file to the configured input directory.
- Run ingestion.py.
- Verify that the new file appears in ingestedfiles.txt.
- Verify that finaldata.csv includes its records.
- Verify duplicate records are removed.
- Verify that unrelated files are ignored.

---

# 6. Implementation requirements — Step 2: Training, scoring, and deployment

## A. training.py

Implement model training.

Requirements:

1. Read config.json.
2. Load finaldata.csv from output_folder_path.
3. Validate that the required columns exist.
4. Select the three specified numerical features.
5. Use `exited` as the target.
6. Train the Logistic Regression model using the existing starter approach and settings.
7. Validate that the dataset has enough usable records and both target classes.
8. Persist the trained model using pickle or joblib consistently with the starter project.
9. Save the model at:

```text
practicemodels/trainedmodel.pkl
```

or the directory specified by `output_model_path`.

10. Do not use corporation as an input feature.
11. Do not silently replace the required algorithm with another algorithm.
12. Provide clear logging and useful errors.

The code must be reusable when output_model_path changes from practicemodels to models.

## B. scoring.py

Implement model scoring.

Required function:

```python
def score_model(model, dataframe, score_path=None):
    pass
```

The function must:

1. Accept a trained model.
2. Accept a pandas DataFrame.
3. Separate the model features from the target.
4. Generate predictions.
5. Calculate the F1 score.
6. Return the F1 score.
7. Write the score to score_path only when score_path is provided.

For the Step 2 scoring process:

- Load the trained model from output_model_path.
- Load testdata.csv from test_data_path.
- Calculate F1.
- Save the result to:

```text
output_model_path/latestscore.txt
```

Use the optional score_path feature to make the function reusable.

**Important:** When an API or drift-check process calls:

```python
score_model(model, dataframe)
```

it must not modify the existing latestscore.txt.

This prevents accidental overwriting of the production baseline.

## C. deployment.py

Implement model deployment.

Copy exactly the required production artifacts from their respective source directories:

```text
output_model_path/trainedmodel.pkl
output_model_path/latestscore.txt
output_folder_path/ingestedfiles.txt
```

Deploy them into:

```text
production_deployment/
```

Expected deployment files:

```text
production_deployment/
    trainedmodel.pkl
    latestscore.txt
    ingestedfiles.txt
```

Requirements:

- Create destination directories if necessary.
- Copy existing files rather than retraining a model.
- Do not modify the model or score during deployment.
- Handle missing files with meaningful errors.
- Ensure all three files are deployed.

### Step 2 acceptance tests

Run:

```bash
python ingestion.py
python training.py
python scoring.py
python deployment.py
```

Verify that all required output artifacts exist and can be loaded.

---

# 7. Implementation requirements — Step 3: Diagnostics

File: `diagnostics.py`

Implement all required diagnostic functions.

## A. Model predictions

Implement:

```python
def model_predictions(dataframe):
    pass
```

Requirements:

- Load the deployed model from production_deployment.
- Accept a DataFrame as input.
- Select the exact three model features.
- Generate predictions.
- Return predictions as a Python list.
- Preserve row order.
- Return one prediction for every row.
- Do not retrain or redeploy a model.

## B. Summary statistics

Implement a function that:

- Reads finaldata.csv from output_folder_path.
- Calculates mean.
- Calculates median.
- Calculates standard deviation.
- Handles all numeric columns.
- Returns the results in a Python list as expected by the project instructions.

Exclude non-numeric columns from these calculations.

Return structured, readable results.

## C. Missing-data diagnostics

Implement a function that:

- Loads finaldata.csv.
- Counts missing values in every column.
- Calculates the missing-data percentage for every column.
- Returns results as a Python list.
- Handles datasets with zero rows safely.
- Includes the column name, missing count, total count, and percentage in its returned data.

## D. Timing diagnostics

Implement a function that measures the execution time of:

1. ingestion.py
2. training.py

Use Python's time module and subprocess.

Return both timings as a Python list, in seconds.

Important:

- Measure ingestion and training independently.
- Run the actual project scripts.
- Use the correct current Python interpreter.
- Capture subprocess errors.
- Do not replace the deployed production model.
- Do not call deployment.py from diagnostics.
- Keep timing execution isolated from production deployment.

The training process may generate a new candidate artifact in the configured model output directory. It must not automatically deploy that artifact.

## E. Dependency diagnostics

Implement a function that checks:

- All packages used by the project.
- Installed package versions.
- Latest available package versions.

Use requirements.txt and installed package metadata.

Where practical, obtain latest versions using pip's supported package information commands.

Return a structured table/list with:

```text
package_name
installed_version
latest_version
```

Requirements:

- Do not automatically install or upgrade packages.
- Handle unavailable package metadata.
- Handle network restrictions gracefully.
- Do not crash the API if package-index access fails.
- Report the diagnostic limitation clearly.

### Step 3 acceptance tests

Verify each function independently.

Test:

- Deployed model predictions.
- Summary statistics.
- Missing-data percentages.
- Execution timing.
- Dependency version checks.

Ensure predictions have the same length as the supplied DataFrame.

Ensure diagnostics does not change the currently deployed production model.

---

# 8. Implementation requirements — Step 4: Reporting

File: `reporting.py`

Generate a confusion matrix for the deployed model.

Requirements:

1. Load testdata.csv from the configured test_data_path.
2. Load deployed model through the diagnostics prediction function.
3. Generate predictions.
4. Compare predictions with actual `exited` labels.
5. Generate a confusion matrix using scikit-learn.
6. Plot using matplotlib.
7. Save the figure to:

```text
output_model_path/confusionmatrix.png
```

Requirements:

- Add meaningful axis labels.
- Add a title.
- Include class labels.
- Ensure the figure is readable.
- Close the figure after saving.
- Do not retrain or replace the deployed model.
- Reuse the configured model output path.

When running the reporting process for the second time in Step 5, preserve the original report and save:

```text
confusionmatrix2.png
```

Do not overwrite the first report.

---

# 9. Implementation requirements — Step 4: Flask API

Files:

- app.py
- wsgi.py

Implement all four required endpoints.

Run the Flask application on port 8000 as expected by the starter project.

## A. POST /prediction

Input:

A dataset file location supplied in the request.

Behavior:

1. Validate the request.
2. Load the specified CSV.
3. Pass the DataFrame into diagnostics.model_predictions().
4. Return model predictions as JSON.
5. Handle invalid file paths and missing files.
6. Return appropriate error responses.

## B. GET /scoring

Behavior:

1. Load deployed trainedmodel.pkl from production_deployment.
2. Load testdata.csv from test_data_path.
3. Call score_model(model, dataframe).
4. Return the F1 score.

**Critical requirement:**

Do not pass a score_path.

The API scoring endpoint must not overwrite production_deployment/latestscore.txt or the original model baseline.

## C. GET /summarystats

Behavior:

- Call the summary statistics function.
- Return its result as JSON.
- Handle missing files or unavailable data correctly.

## D. GET /diagnostics

Behavior:

Return:

- Ingestion timing.
- Training timing.
- Missing-data diagnostics.
- Dependency diagnostics.

Use the functions implemented in diagnostics.py.

## API quality requirements

- Every successful endpoint must return HTTP 200.
- Return proper JSON responses.
- Validate request payloads.
- Use appropriate error status codes for failures.
- Avoid duplicating ML logic inside the API.
- Reuse the project functions.
- Keep the WSGI application correctly exposed through wsgi.py.
- Do not automatically retrain the model through API requests.

### API acceptance tests

Start:

```bash
python app.py
```

Test all four endpoints.

Verify:

- `/prediction`
- `/scoring`
- `/summarystats`
- `/diagnostics`

Confirm successful responses return HTTP 200.

Confirm the deployed latestscore.txt remains unchanged after `/scoring`.

---

# 10. Implementation requirements — Step 4: API client

File: `apicalls.py`

Implement a client that calls the actual running Flask server through HTTP requests.

Do not bypass the API by directly importing endpoint functions.

Required actions:

1. Send a prediction request using testdata/testdata.csv.
2. Call the scoring endpoint.
3. Call the summary statistics endpoint.
4. Call the diagnostics endpoint.
5. Validate HTTP responses.
6. Extract response content.
7. Combine all results into one readable output.
8. Save the result to:

```text
output_model_path/apireturns.txt
```

The output must include:

- Predictions.
- F1 score.
- Summary statistics.
- Ingestion execution time.
- Training execution time.
- Missing-data information.
- Installed package versions.
- Latest package versions.

Use requests with suitable timeouts.

Handle API connection errors clearly.

When Step 5 runs again, preserve the original result and create:

```text
apireturns2.txt
```

---

# 11. Implementation requirements — Step 5: Full process automation

File: `fullprocess.py`

This is one of the most important parts of the project.

Implement the complete dynamic data-ingestion, model-drift, retraining, redeployment, and reporting workflow.

Before beginning Step 5, ensure config.json contains:

```json
{
    "input_folder_path": "sourcedata",
    "output_folder_path": "ingesteddata",
    "test_data_path": "testdata",
    "output_model_path": "models",
    "prod_deployment_path": "production_deployment"
}
```

Use existing configuration conventions. Do not hardcode absolute paths.

The automation must follow this exact decision flow.

## Step 1: Detect new input data

- Read production_deployment/ingestedfiles.txt.
- Discover all CSV files in the configured input directory.
- Compare current filenames with deployed ingestion filenames.
- Identify new CSV files.

Do not ingest the validation_data directory by default.

## Step 2: Stop if no new data exists

If there are no new files:

- Print/log that there is no new data.
- Stop the processing workflow.
- Do not perform drift checks.
- Do not retrain.
- Do not redeploy.
- Do not generate second-run reports.

This is a mandatory requirement.

## Step 3: Ingest new data

When new CSV files exist:

- Run ingestion.py.
- Rebuild finaldata.csv from the configured input dataset.
- Update ingestedfiles.txt.
- Continue to drift evaluation.

Do not lose existing ingested records where the configured ingestion strategy requires rebuilding from all source CSVs.

## Step 4: Evaluate production model on new data

Load:

```text
production_deployment/trainedmodel.pkl
production_deployment/latestscore.txt
```

Read the newly ingested finaldata.csv.

Calculate the deployed model's F1 score on the newly ingested data.

Use scoring.py's reusable score_model(model, dataframe) function.

Do not save this drift score over the production baseline.

Compare:

```text
new_data_f1_score
production_baseline_f1_score
```

Drift decision:

```python
drift_detected = new_data_f1_score < production_baseline_f1_score
```

A lower F1 score means drift for this project's specified workflow.

An equal or higher F1 score means no drift.

## Step 5: No-drift branch

When:

```python
new_data_f1_score >= production_baseline_f1_score
```

Then:

- Do not retrain.
- Do not redeploy.
- Do not replace trainedmodel.pkl.
- Do not replace production_deployment/latestscore.txt.
- Keep the deployed baseline unchanged.
- Continue to the reporting and API steps.

## Step 6: Drift branch

When:

```python
new_data_f1_score < production_baseline_f1_score
```

Then:

1. Run training.py.
2. Train a new Logistic Regression model using the latest ingested data.
3. Save the candidate model under the configured models directory.
4. Score the replacement model on the configured test_data_path/testdata.csv.
5. Save the replacement model's F1 score to models/latestscore.txt.
6. Run deployment.py.
7. Deploy the replacement model, latest test-data F1 score, and updated ingestedfiles.txt.

The replacement model and its score must be deployed together.

Do not deploy before successfully training and scoring the replacement model.

Do not replace the deployed model if retraining or scoring fails.

## Step 7: Generate second-run reports

After processing new data, run both:

```text
apicalls.py
reporting.py
```

This must happen in both cases:

- New data with no drift.
- New data with drift.

When drift occurs, complete deployment before generating reports.

Use the currently deployed model.

Use the configured testdata.csv for scoring and confusion-matrix reporting, according to the project instructions.

Save the second set of reports as:

```text
models/apireturns2.txt
models/confusionmatrix2.png
```

Preserve the original:

```text
models/apireturns.txt
models/confusionmatrix.png
```

For repeatability, ensure the workflow can distinguish first-run and second-run artifact names without destructive overwriting.

Important:

When there is no drift, identical predictions, F1 scores, and confusion matrices are valid. Do not force retraining just to make results different.

---

# 12. Validation fixtures: test both automation branches

The repository already includes:

```text
validation_data/
    drift/
        drift.csv
    no_drift/
        no_drift.csv
```

Read and understand these fixtures.

Use them to validate the automation logic.

Do not modify the fixtures.

## Scenario A: No new data

Configure input_folder_path to the already-deployed input directory.

Expected:

- No new data is detected.
- No ingestion runs.
- No drift check runs.
- No retraining.
- No redeployment.
- No second-run reports.

## Scenario B: New data, no drift

Use:

```text
validation_data/no_drift
```

Expected:

- New data is detected.
- Data ingestion runs.
- Drift score is calculated.
- Score is equal to or greater than baseline.
- No retraining occurs.
- No redeployment occurs.
- Production model remains unchanged.
- Production baseline remains unchanged.
- Both second-run reports are generated.

## Scenario C: New data, drift

Use:

```text
validation_data/drift
```

Expected:

- New data is detected.
- Data ingestion runs.
- Deployed model is evaluated.
- New-data F1 is lower than baseline.
- Retraining occurs.
- Replacement model is scored on testdata.csv.
- Replacement model is deployed.
- New baseline is deployed with the replacement model.
- Both second-run reports are generated.

The README describes the initial fixture scores:

- No-drift fixture: approximately 1.0.
- Drift fixture: approximately 0.0.
- Initial test-data baseline: approximately 0.5714.

These are expected for the specified initial Logistic Regression model and fixture setup. Do not hardcode these scores. Calculate them from the actual data and model.

**Important:** Run each scenario using an independent copy or carefully resettable artifact state. Do not allow one scenario's model or ingestion record to contaminate another scenario's test.

Restore the final project configuration to:

```text
input_folder_path = sourcedata
output_model_path = models
```

---

# 13. Cron job automation

The project requires a cron job that runs fullprocess.py every 10 minutes.

Create a file:

```text
cronjob.txt
```

It must contain the actual cron command for the current project environment.

Expected format:

```bash
*/10 * * * * cd /absolute/path/to/ucityDynamic_risk_assesment && /absolute/path/to/python fullprocess.py
```

Do not leave placeholder paths in the final file.

Determine the real project path and Python executable for the environment in which the cron command is configured.

Document the commands required to activate/start cron where applicable.

Do not assume that a cron service is available on Windows or every development container. Provide environment-appropriate instructions.

The cron file must be included in the final submission ZIP.

---

# 14. Requirements and environment

Inspect requirements.txt before changing dependencies.

Use the current working Python environment.

Do not unnecessarily upgrade or downgrade all libraries.

Ensure the required packages are available, including the project's dependencies for:

- pandas
- numpy
- scikit-learn
- Flask
- requests
- matplotlib
- joblib or pickle support

Use the dependency versions already specified in the repository where possible.

Check compatibility before changing any pinned versions.

Do not introduce unnecessary dependencies.

Provide a reliable setup process in README.md.

---

# 15. Code quality and engineering requirements

All implementation must follow these practices:

1. Use Python functions and modular architecture.
2. Avoid duplicated code.
3. Reuse existing starter functions where appropriate.
4. Use pathlib or reliable path handling.
5. Use configuration-driven input and output directories.
6. Use type hints where useful.
7. Add meaningful docstrings.
8. Add logging for important workflow events.
9. Add proper error handling.
10. Do not silently swallow exceptions.
11. Avoid hardcoded filenames for source data.
12. Do not hardcode model output directories.
13. Validate required CSV columns.
14. Handle missing values safely.
15. Preserve row alignment between features and labels.
16. Keep production and candidate artifacts separate.
17. Ensure all Python scripts can be executed from the repository root.
18. Avoid unnecessary modifications to supplied data.
19. Make scripts reusable and maintainable.
20. Keep the original project scope and required Logistic Regression implementation.

Add automated tests where helpful, but do not alter the required submission behavior to make tests pass.

---

# 16. Project test suite

Create a suitable `tests/` directory and implement tests for:

- Data ingestion.
- CSV discovery.
- Duplicate removal.
- Ingested filename recording.
- Model training.
- Model serialization/loading.
- F1 scoring.
- Optional score file writing.
- Deployment artifact copying.
- Model prediction.
- Summary statistics.
- Missing-data percentages.
- Timing diagnostics.
- Dependency checks.
- Confusion matrix generation.
- All four Flask API endpoints.
- API scoring baseline preservation.
- New-data detection.
- No-new-data early exit.
- No-drift branch.
- Drift and retraining branch.
- Preservation of first-run reports.
- Second-run artifact naming.

Tests must use temporary directories or isolated test fixtures wherever possible.

Do not run destructive tests directly against the final production_deployment directory.

Use independent test environments and copy/reset test artifacts before each scenario.

---

# 17. Final end-to-end execution

Once all implementation is complete, run the project in this order:

## Initial training and deployment

```bash
python ingestion.py
python training.py
python scoring.py
python deployment.py
```

## Start API

```bash
python app.py
```

Use another terminal:

```bash
python apicalls.py
python reporting.py
```

Verify the first report pair:

```text
models/apireturns.txt
models/confusionmatrix.png
```

## Automation validation

Run isolated tests for:

1. No new data.
2. New data without drift.
3. New data with drift.

For each scenario, verify the exact expected branch behavior.

After validation, restore the final config and final artifacts for the actual project submission.

## Final execution check

Run:

```bash
python fullprocess.py
```

Confirm that the actual final workflow behaves correctly with the supplied `sourcedata` files and current deployed ingestion record.

If the repository already records the same source filenames in production, the workflow should stop at the no-new-data condition. Do not change the deployed record merely to force a new-data branch.

---

# 18. Submission deliverables

Prepare a clean final submission ZIP.

Include the required project scripts:

```text
ingestion.py
training.py
scoring.py
deployment.py
diagnostics.py
reporting.py
app.py
wsgi.py
apicalls.py
fullprocess.py
```

Include:

```text
config.json
requirements.txt
README.md
cronjob.txt
PROJECT_AUDIT.md
```

Include generated outputs:

```text
models/
    trainedmodel.pkl
    latestscore.txt
    confusionmatrix.png
    confusionmatrix2.png
    apireturns.txt
    apireturns2.txt

ingesteddata/
    finaldata.csv
    ingestedfiles.txt

production_deployment/
    trainedmodel.pkl
    latestscore.txt
    ingestedfiles.txt
```

Include any necessary project test files.

Include the supplied datasets only as needed by the assignment submission structure.

Do not include:

- .git directory
- .venv
- __pycache__
- Temporary files
- Unnecessary cache files
- Unrelated validation output artifacts
- API logs containing unnecessary system information

Do not include secrets or credentials.

Create:

```text
submission/Dynamic_Risk_Assessment_Submission.zip
```

The ZIP must have a clean folder structure and all required artifacts.

---

# 19. Final rubric verification report

Create:

```text
FINAL_RUBRIC_VERIFICATION.md
```

Use a checklist with these columns:

| Rubric requirement | Implementation file | Test performed | Result | Evidence |
|---|---|---|---|---|

Cover all requirements from all five project steps.

At minimum verify:

### Step 1
- Automatic CSV discovery.
- No hardcoded source filenames.
- Concatenation of datasets.
- Duplicate removal.
- finaldata.csv generation.
- ingestedfiles.txt generation.

### Step 2
- Logistic Regression model training.
- Correct feature selection.
- Model serialization.
- F1 score calculation.
- latestscore.txt generation.
- Deployment of all three required artifacts.

### Step 3
- Model predictions.
- Mean, median, standard deviation.
- Missing-data counts.
- Missing-data percentages.
- Separate ingestion timing.
- Separate training timing.
- Dependency diagnostics.

### Step 4
- Confusion matrix.
- Prediction API.
- Scoring API.
- Summary statistics API.
- Diagnostics API.
- API client uses HTTP requests.
- apireturns.txt output.

### Step 5
- New-data detection.
- No-new-data early exit.
- Drift comparison.
- No-drift branch.
- Drift and retraining branch.
- Replacement model scoring.
- Replacement deployment.
- Second-run API report.
- Second-run confusion matrix.
- First-run reports preserved.
- Ten-minute cron job.
- cronjob.txt included.

Only mark a requirement PASS when actual execution or an appropriate test provides evidence.

If a requirement cannot be tested because the environment does not support it, mark it BLOCKED and explain why. Do not falsely mark it as passed.

---

# 20. Important execution rules for Cursor

Work independently and complete the implementation end to end.

Do not stop after:

- Reading the README.
- Creating the audit.
- Writing one or two scripts.
- Generating code without running it.
- Passing syntax checks only.

Do not ask me for confirmation between project steps.

Make reasonable decisions based on the repository's existing code, the project requirements, and the provided datasets.

Do not change the required project architecture or Logistic Regression algorithm just to simplify implementation.

Do not hide failing tests.

Do not fake API responses.

Do not fabricate successful cron execution when cron is unavailable.

Do not overwrite the deployed production model during diagnostics or API scoring.

Do not change the production baseline merely to force a drift scenario.

If a test fails, investigate the root cause, fix it, and run the affected test again.

Keep a clear record of the commands executed and their actual results.

## Final response required from Cursor

At completion, provide:

1. Repository audit summary.
2. Files modified and created.
3. Implemented features by project step.
4. Tests executed.
5. Test results.
6. API endpoint test results.
7. Drift scenario test results.
8. Cron configuration status.
9. Final ZIP location.
10. Remaining blockers, if any.
11. Rubric verification summary.

**Final objective: Deliver a fully implemented, runnable, tested, and submission-ready Udacity Dynamic Risk Assessment System that satisfies the project instructions and rubric—not merely a collection of generated Python files.**