"""Module for performing inference using ONNX models with MLflow integration.

This module provides the `ModelPredictorONNX` class, which is designed to load ONNX models
and associated preprocessing transforms from MLflow, and perform inference on input images.
It supports operations such as model verification, loading, and prediction with ONNX Runtime
"""

import io
import pickle
import tempfile
import typing as tp

import mlflow
import numpy as np
import onnxruntime
from mlflow.tracking import MlflowClient
from PIL import Image


class ModelPredictorONNX:
    """ModelPredictorONNX instance with model and artifact.

    Args:
        model_name (str): Name of the model.
        onnx_model_artifact_name (str): Name of the ONNX model artifact.
        transform_artifact_path (str): Path to the transform artifact.
        alias (str, optional): Alias for the model version. Defaults to "prod".
        tracking_uri (str, optional): MLflow tracking URI. If not provided, uses the default tracking URI.
    """

    def __init__(
        self,
        model_name,
        onnx_model_artifact_name,
        transform_artifact_path,
        alias="prod",
        tracking_uri=None,
    ):
        """Initialize the ModelPredictorONNX instance with model and artifact information."""
        self.model_name = model_name
        self.onnx_model_artifact_name = onnx_model_artifact_name
        self.transform_artifact_path = transform_artifact_path
        self.alias = alias
        self.tracking_uri = tracking_uri or self._get_default_tracking_uri()

        self.ort_session = None
        self.transform = None
        self.model_version = None
        self.model_info = None
        self.input_name = None
        self.output_name = None
        print(
            f"ModelPredictorONNX initialized for model: {model_name}, alias: {alias}"
        )
        print(f"Tracking URI: {self.tracking_uri}")

    def _get_default_tracking_uri(self):
        """Get default MLflow tracking URI if not provided."""
        return mlflow.get_tracking_uri() or "file:///mlruns"

    def _softmax_numpy(self, x, axis=1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _verify_model_exists(self):
        """Check if the registered model exists in MLflow."""
        client = MlflowClient(tracking_uri=self.tracking_uri)
        try:
            client.get_registered_model(self.model_name)
            return True
        except Exception:
            print(
                f"Registered model '{self.model_name}' not found. Available models:"
            )
            for model in client.search_registered_models():
                print(f"- {model.name}")
            return False

    def _get_model_version_by_alias(self):
        """Get production model version using alias."""
        client = MlflowClient(tracking_uri=self.tracking_uri)
        try:
            model_version = client.get_model_version_by_alias(
                self.model_name, self.alias
            )
            print(
                f"Found model version {model_version.version} with alias '{self.alias}'"
            )
            return model_version
        except Exception as e:
            print(f"!!! Error retrieving model by alias: {str(e)}")
            print(f"Model versions for '{self.model_name}':")
            for mv in client.search_model_versions(f"name='{self.model_name}'"):
                aliases = ", ".join(mv.aliases) if mv.aliases else "None"
                print(
                    f"- Version {mv.version}: Aliases={aliases}, Run ID={mv.run_id}"
                )
            raise ValueError(
                f"No '{self.alias}' alias found for model '{self.model_name}'"
            )

    def load_model(self):
        """Loads the production ONNX model and its associated preprocessing transforms from MLflow.

        This method performs the following steps:
            1. Checks if the ONNX model session and transforms are already loaded; if so, it returns early.
            2. Sets the MLflow tracking URI.
            3. Verifies that the specified model exists in the MLflow model registry.
            4. Retrieves the production model version and associated run ID using the provided alias.
            5. Downloads the ONNX model artifact and initializes an ONNX Runtime inference session.
            6. Downloads and loads the preprocessing transforms artifact.
            7. Updates internal state with model version, input/output names, and loaded transforms.

        Raises:
            ValueError: If the model does not exist in the registry or if the model version has no associated run ID.
            RuntimeError: If downloading or loading the ONNX model or transforms fails.
            Exception: For any other unexpected errors during the loading process.
        """
        if self.ort_session and self.transform:
            print("Model and transforms already loaded.")
            return

        print("Attempting to load model and transforms...")

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Verify model exists
        if not self._verify_model_exists():
            raise ValueError(f"Model '{self.model_name}' not found in registry")

        try:
            # Get production model version
            model_version_obj = self._get_model_version_by_alias()
            self.model_version = model_version_obj.version
            self.model_info = f"Alias: {self.alias}"
            run_id = model_version_obj.run_id

            if not run_id:
                raise ValueError(
                    f"Model version {self.model_version} has no associated run ID"
                )

            print(f"Loading artifacts from run ID: {run_id}")

            with tempfile.TemporaryDirectory() as tmpdir:
                # Download ONNX model
                try:
                    local_onnx_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=self.onnx_model_artifact_name,
                        dst_path=tmpdir,
                    )
                    print(f"ONNX model downloaded to: {local_onnx_path}")
                    self.ort_session = onnxruntime.InferenceSession(
                        local_onnx_path
                    )
                    self.input_name = self.ort_session.get_inputs()[0].name
                    self.output_name = self.ort_session.get_outputs()[0].name
                    print(
                        f"ONNX model loaded. Input: {self.input_name}, Output: {self.output_name}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load ONNX model: {str(e)}"
                    ) from e

                # Download and load transforms
                try:
                    local_transform_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=self.transform_artifact_path,
                        dst_path=tmpdir,
                    )
                    print(
                        f"Transforms artifact downloaded to: {local_transform_path}"
                    )
                    with open(local_transform_path, "rb") as f:
                        self.transform = pickle.load(f)
                    print("Data transformations loaded successfully")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load transforms: {str(e)}"
                    ) from e

            print(
                f" Model loaded successfully (Version: {self.model_version}, {self.model_info})"
            )

        except Exception as e:
            self.ort_session = None
            self.transform = None
            print(f" Failed to load model: {str(e)}")
            raise

    def predict_from_bytes(
        self, image_bytes: bytes, image_name: tp.Optional[str]
    ) -> dict:
        """Run Trasnform/preprocessing/inference on an image provided as bytes using the loaded transform and corresponding ONNX model.

        Args:
            image_bytes (bytes): The image data in bytes.
            image_name (str, optional): An identifier for the image.

        Returns:
            dict: Prediction results with class, confidence, and model info.
        """
        if image_name is None:
            image_name = "processed_from_bytes.unknown"

        if not self.ort_session or not self.transform:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Loading and preprocessing/transform image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_tensor = self.transform(image)
            input_numpy = (
                image_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)
            )

            # Inference with ONNX Runtime
            ort_outputs = self.ort_session.run(
                [self.output_name], {self.input_name: input_numpy}
            )
            output_data = ort_outputs[0]

            # Process results
            probabilities = self._softmax_numpy(output_data, axis=1)
            confidence = float(np.max(probabilities, axis=1)[0]) # Finding best/highest proba
            predicted_idx = int(np.argmax(probabilities, axis=1)[0]) # Getting index of that proba

            # Class mapping
            class_mapping = {0: "handwritten", 1: "printed"}
            class_name = class_mapping.get(predicted_idx, "unknown")

            return {
                "image_name": image_name,
                "predicted_class": class_name,
                "confidence": confidence,
                "model_version": str(self.model_version),
                "model_info": self.model_info,
            }

        except Exception as e:
            print(f" Prediction error for '{image_name}': {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}") from e
