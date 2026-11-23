Use this project rubric to understand and assess the project criteria.

Data Ingestion
Criteria	Submission Requirements
Student will be able to update the training data for ML models in preparation for retraining.

ingestion.py performs the ingestion process.
The script automatically discovers and reads every CSV file in the data directory specified by input_folder_path in config.json; input filenames are not hard-coded.
The script combines the input files into one pandas DataFrame and removes duplicate rows.
The script writes the deduplicated DataFrame to finaldata.csv in the directory specified by output_folder_path in config.json.
The script writes the names of all ingested CSV files to ingestedfiles.txt in the directory specified by output_folder_path.
Training, Scoring, and Deployment
Criteria	Submission Requirements
Student will be able to write a script that trains an ML model and writes the model to storage.

Students should write training.py to accomplish this. The model should be saved in the pickle format.

Student will be able to write a script that performs scoring of ML models

Students should write a scoring script in the scoring.py starter file. Scoring should be performed using the F1 score.

Student will be able to write ML model scores to persistent storage in files or databases.

The scoring.py script should write the F1 score to a .txt file called latestscore.txt.

Student will be able to regularly re-deploy models

The deployment.py script should copy the trained model, the F1 score, and the ingested file record to a production deployment directory.

Diagnostics
Criteria	Submission Requirements
Student will be able to write a script that automatically measures the execution time of the data ingestion and model training processes.

diagnostics.py measures the execution time of ingestion.py and training.py in seconds.
The timing function returns or reports both measurements so that ingestion and training performance can be evaluated separately.
Running the diagnostics does not retrain, redeploy, or otherwise replace the deployed production model.
diagnostics.py also provides a prediction function that uses the deployed model and a supplied dataset, as required by the API and reporting steps.
The script calculates the mean, median, and standard deviation for each numeric column in the ingested dataset and returns the results as a Python list.
Student will be able to check data integrity and stability.

The diagnostics.py script should perform this step.
Data integrity should be checked by measuring the percentage of NA values in each of the numeric dataset’s columns.
Student will be able to check for dependency changes in scripts.

The diagnostics.py script should perform this step.
All modules in requirements.txt need to have their latest versions and currently installed versions checked.
Reporting
Criteria	Submission Requirements
Student will be able to create APIs that provide automated access to ML model predictions, scores, and diagnostics.

The app.py script provides all of the following endpoints:

a scoring endpoint that returns an F1 score using the test data in the directory specified by test_data_path in config.json;
a summary-statistics endpoint that returns summary statistics for the ingested data in the directory specified by output_folder_path;
a diagnostics endpoint that returns execution timing, dependency information, and missing-data percentages for the ingested data;
a prediction endpoint that accepts an input dataset and returns predictions from the deployed model in the directory specified by prod_deployment_path.
The reporting.py script generates a confusion matrix for the deployed model using the configured test data and saves it as confusionmatrix.png in the configured model-output directory.

Student will be able to call ML scoring APIs and use their returned data in reports.

apicalls.py sends HTTP requests to the endpoints implemented in app.py.
The script obtains model predictions, the F1 score, summary statistics, execution timing, dependency information, and missing-data results from the API endpoints.
The calls are made to the running API and do not bypass the API by importing and directly calling the endpoint functions.
The script combines the endpoint responses and writes them to apireturns.txt in the project output location.
apireturns.txt is included in the final submission ZIP.
Process Automation
Criteria	Submission Requirements
Student will be able to determine whether a deployed ML model needs to be updated based on newly available data and model performance.

fullprocess.py compares files in input_folder_path with the deployed ingestedfiles.txt; if none are new, it stops before drift checking or reporting.
With new data, it ingests the data and compares the deployed model's F1 score on that data with the deployed latestscore.txt.
A lower F1 triggers retraining and redeployment. An equal or higher F1 keeps the deployed model and baseline unchanged and proceeds to reporting.
Student will be able to automate conditional model retraining and redeployment and schedule the complete process to run regularly.

Retrain only after new data and drift are detected. Evaluate the replacement model on test_data_path/testdata.csv and save its F1 as output_model_path/latestscore.txt before deployment.
deployment.py copies the replacement model and score from output_model_path, plus the current ingestedfiles.txt, to prod_deployment_path.
After processing new data, run reporting and API calls on the current deployed model in both drift and no-drift cases; complete any redeployment first. Use configured test data for the confusion matrix and API score. Save confusionmatrix2.png and apireturns2.txt, preserving the first-run files.
Schedule fullprocess.py every 10 minutes with cron and include the one-line entry as cronjob.txt in the submission ZIP.
Suggestions to Make Your Project Stand Out
Generate pdf reports that contain one or more plots, plus model diagnostics and summary statistics.
Store and analyze data related to time trends, including changes in latency and missing values percentages.
Store datasets and records in SQL databases instead of .csv and .txt files.




