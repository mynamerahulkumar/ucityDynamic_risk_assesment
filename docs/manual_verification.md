# Manual verification

The ingestion, training, scoring, deployment, API, unit tests, three automation copies, final `fullprocess.py` check, and submission zip were run from this repository. Nothing else in that pipeline needs to be started by hand.

## Cron tick

The scheduler entry is already installed. `crontab -l` shows this single line, which is also the whole of `cronjob.txt`:

```text
*/10 * * * * cd /Users/rahulkumar/Documents/GitHub/ucityDynamic_risk_assesment && /Users/rahulkumar/Documents/GitHub/ucityDynamic_risk_assesment/.venv/bin/python fullprocess.py
```

Do not run `crontab cronjob.txt` again while that line is already the crontab. Replacing the crontab is unnecessary.

This session ran that Python command directly. With `dataset3.csv` and `dataset4.csv` already deployed, it printed:

```text
No new data found. Stopping before drift checks, retraining, deployment, and reporting.
```

That direct run is not a cron tick. A real tick of this same job also writes no files, because there is no new data, so file timestamps will not show that cron fired.

To confirm the scheduler itself:

1. Wait for the next 10-minute boundary (`:00`, `:10`, `:20`, `:30`, `:40`, or `:50`).
2. Check cron mail or Console for `fullprocess.py`.
3. If nothing runs, give cron permission to read the project directory. On macOS that is System Settings, Privacy & Security, Full Disk Access, for `/usr/sbin/cron`.

Leave `config.json` on `sourcedata` and `models`. Do not point the live project at `validation_data` to force a new-data run.
