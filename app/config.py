"""This file contains the configuration settings JUST for the FastAPI APP.

uses Pydantic for settings management and allows for environment variable overrides.
"""

from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings configuration for the Image Classification API.

    This class defines application-wide settings and configuration options, including:
    - MLflow tracking and model artifact paths.
    - Application metadata such as title, description, and version.
    - Debug mode toggle for development.
    - Pydantic settings for environment variable management.

    Attributes:
        MLFLOW_TRACKING_URI (str): URI for MLflow tracking server or local directory.
        MODEL_NAME (str): Name of the registered model in MLflow.
        ONNX_MODEL_ARTIFACT_NAME (str): Filename for the exported ONNX model artifact.
        TRANSFORM_ARTIFACT_PATH (str): Path to the serialized preprocessing transforms.
        MODEL_ALIAS (str): Alias for the model version ('prod').
        APP_TITLE (str): Title..
        APP_DESCRIPTION (str): Description..
        APP_VERSION (str): Version..
        DEBUG (bool): Flag to enable or disable debug mode.
        model_config (ClassVar[SettingsConfigDict]): Pydantic settings for environment variable loading and validation.
    """

    # MLflow and Model Config
    MLFLOW_TRACKING_URI: str = "file:///mlruns"
    MODEL_NAME: str = "Global_Best_Model"
    ONNX_MODEL_ARTIFACT_NAME: str = "model.onnx"
    TRANSFORM_ARTIFACT_PATH: str = "preprocessing/data_transforms.pkl"
    MODEL_ALIAS: str = "prod"

    # APP Metadata
    APP_TITLE: str = "Image Classification API"
    APP_DESCRIPTION: str = (
        "FastAPI service for handwritten vs. typewritten image classification!"
    )
    APP_VERSION: str = "1.0.0"

    # Application Behavior
    DEBUG: bool = False  # Set to True in .env or env var for development

    # Pydantic Settings Configuration
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",  # Load environment variables from a .env file
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env or environment
        case_sensitive=False,
    )


# importable instance of the settings
settings = Settings()
