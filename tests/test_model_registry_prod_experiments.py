"""Test cases for model registry production experiments module."""

import types
from unittest.mock import MagicMock

import pytest

import app.inference.model_registry_prod_experiments as model_registry


def test_select_model_paths_success():
    """Test selecting model paths from artifacts."""
    artifacts = [
        types.SimpleNamespace(path="model", is_dir=True),
        types.SimpleNamespace(path="other", is_dir=True),
    ]
    result = model_registry.select_model_paths(artifacts)
    assert "model" in result[0]


def test_register_model(monkeypatch):
    """Test registering a model with MLflow."""
    mock_mlflow = MagicMock()
    mock_mlflow.register_model.return_value = types.SimpleNamespace(version="1")
    monkeypatch.setattr(
        model_registry,
        "mlflow",
        MagicMock(
            register_model=mock_mlflow.register_model,
            tracking=MagicMock(MlflowClient=lambda: MagicMock()),
        ),
    )
    model_uris = ["runs:/abc/model"]
    model_name = "foo"
    tags = {"a": "b"}
    result = model_registry.register_model(model_uris, model_name, tags)
    assert result[0].version == "1"


def test_promote_to_production(monkeypatch):
    """Test promoting a model to prod (error path)."""
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("not found")
    monkeypatch.setattr(
        model_registry,
        "mlflow",
        MagicMock(tracking=MagicMock(MlflowClient=lambda: mock_client)),
    )
    with pytest.raises(Exception, match="not found"):
        model_registry.promote_to_production("foo", [1, 2, 3])
