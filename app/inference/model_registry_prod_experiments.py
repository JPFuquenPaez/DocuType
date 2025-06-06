"""This module handles the registration and promotion of machine learning models into production.

It contains functions to promote model versions to production using MLflow aliases and
manage the model registry for production experiments.
"""

import logging
import os
import sys
from pathlib import Path

import mlflow

os.makedirs("model_deployment_logs", exist_ok=True)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "model_deployment_logs/model_deployment.log"
        ),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to console as well
    ],
)


def display_artifacts(client, run_id):
    """Display all artifacts in a run."""
    try:
        artifacts = client.list_artifacts(run_id)
        print("\nAvailable artifacts:")
        for artifact in artifacts:
            print(f"{artifact.path} {'(dir)' if artifact.is_dir else '(file)'}")
            if artifact.is_dir:
                nested_artifacts = client.list_artifacts(run_id, artifact.path)
                for nested in nested_artifacts:
                    print(f" - {nested.path}")
        return artifacts
    except Exception as e:
        raise Exception(f"Error listing artifacts for run {run_id}: {str(e)}")


def select_model_paths(artifacts):
    """Select only the model artifact directory."""
    dirs = [art.path for art in artifacts if art.is_dir]

    if not dirs:
        raise Exception("No directory artifacts found.")

    # Select only the model directory
    selected_paths = []
    for path in dirs:
        if "model" in path:
            selected_paths.append(path)
            break  # Ensure only one model path is selected

    if not selected_paths:
        raise Exception("Required artifact 'model' not found.")

    return selected_paths


def get_model_uri(
    tracking_uri,
    experiment_name=None,
    run_id=None,
    metric_name="validation_accuracy",
):
    """Retrieve the URI(s) of the best model run from MLflow experiments.

    This function connects to an MLflow tracking server and locates the best run(s) based on a specified metric
    (default: "validation_accuracy") across one or more experiments. It can also retrieve model URIs from a specific run ID.

    Args:
        tracking_uri (str): The URI of the MLflow tracking server.
        experiment_name (str or list of str, optional): Name or list of names of experiments to search.
            If None, searches across all available experiments.
        run_id (str, optional): If provided, retrieves model URIs from this specific run.
        metric_name (str, optional): The metric to use for selecting the best run. Defaults to "validation_accuracy".

    Returns:
        tuple:
            - model_uris (list of str): List of MLflow model URIs for the selected run.
            - run_id (str): The run ID of the selected/best run.
            - experiment_id (str): The experiment ID of the selected/best run.

    Raises:
        Exception: If no valid experiments or successful runs are found.

    Notes:
        - If no runs contain the specified metric, the latest finished run is used as a fallback.
        - Prints informative messages about the selection process and any issues encountered.
    """
    """Get model URIs from the best run across ALL experiments or specified experiment(s)."""

    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using tracking URI: {tracking_uri}")

    client = mlflow.tracking.MlflowClient()

    if run_id:
        print(f"Loading model from run ID: {run_id}")
        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id
        artifacts = display_artifacts(client, run_id)
        model_paths = select_model_paths(artifacts)
        model_uris = [
            f"runs:/{run_id}/{model_path}" for model_path in model_paths
        ]
        return model_uris, run_id, experiment_id

    # Handle experiment selection
    if experiment_name:
        experiment_names = (
            [experiment_name]
            if isinstance(experiment_name, str)
            else experiment_name
        )
    else:
        print("Searching across ALL available experiments")
        experiment_names = [exp.name for exp in mlflow.search_experiments()]

    # Get valid experiment IDs
    experiment_ids = []
    for name in experiment_names:
        exp = mlflow.get_experiment_by_name(name)
        if not exp:
            print(f"Experiment '{name}' not found. Skipping...")
            continue
        experiment_ids.append(exp.experiment_id)

    if not experiment_ids:
        available_exps = [e.name for e in mlflow.search_experiments()]
        raise Exception(
            f"No valid experiments found. Available experiments: {available_exps}"
        )

    # Search runs globally with metric filtering
    print(
        f"Searching best run across {len(experiment_ids)} experiment(s) using metric '{metric_name}'..."
    )

    # Get all finished runs
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string="status = 'FINISHED'",
        output_format="pandas",
    )

    if runs_df.empty:
        raise Exception("No successful runs found in specified experiments")

    # Handle metric presence
    metric_col = f"metrics.{metric_name}"

    if metric_col in runs_df.columns:
        metric_runs = runs_df[runs_df[metric_col].notnull()]
        if not metric_runs.empty:
            best_run = metric_runs.loc[metric_runs[metric_col].idxmax()]
            run_id = best_run.run_id
            experiment_id = best_run.experiment_id
            metric_value = best_run[metric_col]
            print(
                f"Found best run: ID={run_id}, {metric_name}={metric_value:.4f}"
            )
        else:
            print(
                f"No runs with metric '{metric_name}' found. Using latest run as fallback"
            )
            run_id = runs_df.iloc[0].run_id
            experiment_id = runs_df.iloc[0].experiment_id
    else:
        print(
            f"Metric '{metric_name}' not found in any runs. Using latest run as fallback"
        )
        run_id = runs_df.iloc[0].run_id
        experiment_id = runs_df.iloc[0].experiment_id

    exp_name = mlflow.get_experiment(experiment_id).name
    print(f"Selected run: ID={run_id}, Experiment={exp_name}")

    artifacts = display_artifacts(client, run_id)
    model_paths = select_model_paths(artifacts)
    model_uris = [f"runs:/{run_id}/{model_path}" for model_path in model_paths]
    return model_uris, run_id, experiment_id


def register_model(model_uris, model_name, tags=None):
    """Register a single model and set its tags."""
    print(f"\nRegistering model from: {model_uris}")
    print(f"Model name: {model_name}")

    client = mlflow.tracking.MlflowClient()
    model_details_list = []

    for model_uri in model_uris:
        try:
            model_details = mlflow.register_model(model_uri, model_name)
            print(f"Model registered with version: {model_details.version}")

            if tags:
                for key, value in tags.items():
                    client.set_registered_model_tag(model_name, key, value)
                print("Tags set successfully")

            model_details_list.append(model_details)
            break  # Ensure only one model is registered

        except Exception as e:
            print(f"Failed to register model: {str(e)}")
            raise

    return model_details_list


def promote_to_production(model_name, version_numbers):
    """Promote the latest model version to production using MLflow aliases."""
    client = mlflow.tracking.MlflowClient()
    try:
        latest_version = max(version_numbers)  # Get the latest version number

        try:
            current_prod = client.get_model_version_by_alias(model_name, "prod")
            print(
                f"Existing production model: {current_prod.name} v{current_prod.version}"
            )
            print(f"Replacing with new version: v{latest_version}")
        except Exception as e:
            print(f"No existing production model found: {str(e)}")

        client.set_registered_model_alias(model_name, "prod", latest_version)
        print(f"\nModel version {latest_version} assigned 'prod' alias")

        alias_info = client.get_model_version_by_alias(model_name, "prod")
        print(
            f"Production model confirmed: {alias_info.name} v{alias_info.version}"
        )
    except Exception as e:
        print(f"\nFailed to promote model to production: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Configuration
        TRACKING_URI = f"file:{Path('./app/training/mlruns').absolute()}"
        EXPERIMENT_NAME = (
            None  # None = search ALL experiments, or specify name/list
        )
        RUN_ID = None  # Specify a run ID if you want to select a specific model
        MODEL_NAME = "Global_Best_Model"  # Fixed name for global best model from all experiments
        TARGET_METRIC = "validation_accuracy"  # Change if needed

        # Get best model URIs globally or from a specific run
        model_uris, RUN_ID, exp_id = get_model_uri(
            TRACKING_URI, EXPERIMENT_NAME, RUN_ID, metric_name=TARGET_METRIC
        )

        # tags
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(RUN_ID)
        tags = run.data.tags

        # Register models
        model_details_list = register_model(model_uris, MODEL_NAME, tags)

        # Push to prod
        version_numbers = [
            model_details.version for model_details in model_details_list
        ]
        promote_to_production(MODEL_NAME, version_numbers)

        # Getting experiment name for display
        exp = mlflow.get_experiment(exp_id)

        print("\nDeployment Summary:")
        print(f"Model Name: {MODEL_NAME}")
        print(f"Production Versions: {version_numbers}")
        print(f"Source Experiment: {exp.name} ({exp_id})")
        print(f"Run ID: {RUN_ID}")
        print(f"Tracking URI: {TRACKING_URI}")
        print(f"Model URIs: {model_uris}")

        logging.info("\nDeployment Summary:")
        logging.info(f"Model Name: {MODEL_NAME}")
        logging.info(f"Production Versions: {version_numbers}")
        logging.info(f"Source Experiment: {exp.name} ({exp_id})")
        logging.info(f"Run ID: {RUN_ID}")
        logging.info(f"Tracking URI: {TRACKING_URI}")
        logging.info(f"Model URIs: {model_uris}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
