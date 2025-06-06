"""This module provides detailed information and utilities for managing production models.

It includes functions and classes to interact with the model registry, retrieve model details,
and perform operations related to production models just in case we need to check the model details in prod.
"""

import logging
import os
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pathlib import Path

os.makedirs("model_deployment_logs", exist_ok=True)

# config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "model_deployment_logs/production_model_details.log"
        ),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to console as well
    ],
)


def list_registered_models():
    """Lists and prints the names of all registered models in the MLflow registry."""
    client = MlflowClient()
    print("Registered Models:")
    for rm in client.search_registered_models():
        print(f"Model Name: {rm.name}")


def get_production_model_details(model_name):
    """Retrieve details of the model version in prod.

    Args:
        model_name (str): Name of the registered model.

    Returns:
        dict or None: Dictionary containing model details (name, version, run_id, source, registration time, metrics, params, tags),
                      or None if the model or production version is not found.
    """
    # Set the tracking URI
    tracking_uri = f"file:{Path('./app/training/mlruns').absolute()}"
    mlflow.set_tracking_uri(tracking_uri)

    # Verifying uri
    print("Tracking URI:", mlflow.get_tracking_uri())

    client = MlflowClient()

    try:
        # All registered models to verify
        list_registered_models()

        # Check if the model exists
        model = client.get_registered_model(model_name)
        print(f"Model found: {model.name}")

        # Getting 'prod' model
        prod_model = client.get_model_version_by_alias(model_name, "prod")
        print(f"Production Model Version: {prod_model.version}")

        # Get run details to access metrics
        run = client.get_run(prod_model.run_id)

        # Getting whatever detaimls
        details = {
            "model_name": prod_model.name,
            "version": prod_model.version,
            "run_id": prod_model.run_id,
            "source": prod_model.source,
            "registered_at": pd.to_datetime(
                prod_model.creation_timestamp, unit="ms"
            ),
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }

        return details

    except mlflow.exceptions.MlflowException as e:
        print(f"Error retrieving production model: {str(e)}")
        return None


def print_model_details(details):
    """Logs details of the model in production, including its name, version, run ID, registration time, and metrics.

    Args:
        details (dict): Dictionary containing model details and metrics.
    """
    if not details:
        logging.warning("No production model found in prod")
        return

    logging.info("###### Production Model Details ######")
    logging.info("#" * 40)
    logging.info(f"Model Name: {details['model_name']}")
    logging.info(f"Version: {details['version']}")
    logging.info(f"Run ID: {details['run_id']}")
    logging.info(f"Registered At: {details['registered_at']}")

    metrics = details["metrics"]
    accuracy_key = (
        "validation_accuracy" if "validation_accuracy" in metrics else "accuracy"
    )
    accuracy_value = metrics.get(accuracy_key, "Not found")
    logging.info(f"{accuracy_key.replace('_', ' ').title()}: {accuracy_value}")

    # Log other metrics if needed
    logging.info("\nMetrics:")
    for metric, value in metrics.items():
        logging.info(f"- {metric}: {value}")


if __name__ == "__main__":
    model_name = "Global_Best_Model"
    prod_details = get_production_model_details(model_name)
    print_model_details(prod_details)
