"""Tests for model inference using ONNX models."""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app.inference.inference_onnx as inference_onnx


@pytest.fixture
def predictor(monkeypatch):
    """Fixture that provides a mocked ModelPredictorONNX instance for testing.

    This fixture sets up a ModelPredictorONNX object with mocked methods and attributes,
    allowing tests to run without requiring actual model files or external dependencies.

    Args:
    monkeypatch: pytest fixture for patching.

    Returns:
    ModelPredictorONNX: A mocked predictor instance.
    """
    pred = inference_onnx.ModelPredictorONNX(
        model_name="test_model",
        onnx_model_artifact_name="model.onnx",
        transform_artifact_path="transform.pkl",
        alias="prod",
        tracking_uri="file:///tmp/mlruns",
    )
    pred.ort_session = MagicMock()
    pred.transform = MagicMock()
    pred.model_version = "1"
    pred.model_info = "Alias: prod"
    pred.input_name = "input"
    pred.output_name = "output"
    return pred


def test_softmax_numpy():
    """Test the softmax_numpy method to ensure it computes softmax correctly."""
    pred = inference_onnx.ModelPredictorONNX(
        model_name="test_model",
        onnx_model_artifact_name="model.onnx",
        transform_artifact_path="transform.pkl",
    )
    arr = np.array([[1.0, 2.0, 3.0]])
    softmax = pred._softmax_numpy(arr, axis=1)
    assert np.allclose(np.sum(softmax, axis=1), 1.0)


def test_verify_model_exists_found(monkeypatch):
    """Test that the model exists in the registry."""
    pred = inference_onnx.ModelPredictorONNX(
        model_name="test_model",
        onnx_model_artifact_name="model.onnx",
        transform_artifact_path="transform.pkl",
    )
    mock_client = MagicMock()
    mock_client.get_registered_model.return_value = True
    monkeypatch.setattr(
        inference_onnx, "MlflowClient", lambda tracking_uri=None: mock_client
    )
    assert pred._verify_model_exists() is True


def test_verify_model_exists_not_found(monkeypatch):
    """Test that the model does not exist in the registry."""
    pred = inference_onnx.ModelPredictorONNX(
        model_name="test_model",
        onnx_model_artifact_name="model.onnx",
        transform_artifact_path="transform.pkl",
    )
    mock_client = MagicMock()
    mock_client.get_registered_model.side_effect = Exception("not found")
    mock_client.search_registered_models.return_value = [
        types.SimpleNamespace(name="foo")
    ]
    monkeypatch.setattr(
        inference_onnx, "MlflowClient", lambda tracking_uri=None: mock_client
    )
    assert pred._verify_model_exists() is False


def test_get_model_version_by_alias_success(monkeypatch):
    """Test getting model version by alias."""
    pred = inference_onnx.ModelPredictorONNX(
        model_name="test_model",
        onnx_model_artifact_name="model.onnx",
        transform_artifact_path="transform.pkl",
    )
    mock_client = MagicMock()
    mv = types.SimpleNamespace(version="1", run_id="abc", aliases=["prod"])
    mock_client.get_model_version_by_alias.return_value = mv
    monkeypatch.setattr(
        inference_onnx, "MlflowClient", lambda tracking_uri=None: mock_client
    )
    result = pred._get_model_version_by_alias()
    assert result.version == "1"
    assert result.run_id == "abc"


def test_predict_from_bytes_happy_path(predictor):
    """Test the predict_from_bytes method with a valid image input."""
    # Patch ort_session.run to return logits for two classes
    predictor.ort_session.run.return_value = [np.array([[0.1, 0.9]])]
    predictor.transform.return_value = types.SimpleNamespace(
        unsqueeze=lambda dim: types.SimpleNamespace(
            cpu=lambda: types.SimpleNamespace(
                numpy=lambda: np.ones((3, 32, 32), dtype=np.float32)
            )
        )
    )
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG header
    with patch("PIL.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_open.return_value = mock_img
        mock_img.convert.return_value = mock_img
        result = predictor.predict_from_bytes(image_bytes, image_name="img.png")
    assert result["predicted_class"] in {"handwritten", "printed"}
    assert 0.0 <= result["confidence"] <= 1.0
