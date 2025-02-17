import mlflow
import pandas as pd

# Set the tracking URI if needed
mlflow.set_tracking_uri('file:/mlruns/732989260917745662')  # Adjust the path as needed

# Get all runs for a specific experiment
experiment_name = 'Musharib'
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Convert runs to a DataFrame
runs_df = runs[['run_id', 'start_time', 'end_time', 'status', 'params', 'metrics']]
print(runs_df)

# Optionally, export the DataFrame to CSV
runs_df.to_csv('mlflow_report.csv', index=False)
