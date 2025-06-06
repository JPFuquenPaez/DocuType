"""Tests for model production details module."""

import types
from unittest.mock import MagicMock

import app.inference.model_prod_details as model_prod_details


def test_list_registered_models(monkeypatch):
    """Test listing registered models."""
    mock_client = MagicMock()
    mock_client.search_registered_models.return_value = [
        types.SimpleNamespace(name="foo")
    ]
    monkeypatch.setattr(model_prod_details, "MlflowClient", lambda: mock_client)
    model_prod_details.list_registered_models()


def test_get_production_model_details_success(monkeypatch):
    """Test getting details of a production model."""
    mock_client = MagicMock()
    mock_model = types.SimpleNamespace(name="foo")
    mock_prod_model = types.SimpleNamespace(
        version="1",
        run_id="abc",
        name="foo",
        source="source",
        creation_timestamp=1234567890,
    )
    mock_run = types.SimpleNamespace(
        data=types.SimpleNamespace(
            metrics={"validation_accuracy": 0.99},
            params={"p": "v"},
            tags={"t": "v"},
        )
    )
    mock_client.get_registered_model.return_value = mock_model
    mock_client.get_model_version_by_alias.return_value = mock_prod_model
    mock_client.get_run.return_value = mock_run
    monkeypatch.setattr(model_prod_details, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(model_prod_details, "pd", __import__("pandas"))
    details = model_prod_details.get_production_model_details("foo")
    assert details["model_name"] == "foo"
    assert details["version"] == "1"
    assert details["metrics"]["validation_accuracy"] == 0.99


def test_print_model_details_none(caplog):
    """Test printing model details when no production model is found."""
    model_prod_details.print_model_details(None)
    assert "No production model found" in caplog.text
