"""Implement tests for webservice functions. Tests files are not documented by choice."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.exceptions import BaseAPIException
from app.webservice import app


class MockUploadFile:
    """Mocking an upload file."""

    def __init__(self, content, content_type, filename):
        """Init the mock."""
        self.content = content
        self.content_type = content_type
        self.filename = filename


class _PredictInput(BaseModel):
    """Mocking an input."""

    action: str


class _PredictOutput(BaseModel):
    """Mocking an output."""

    status: str


async def _predict(item: _PredictInput) -> _PredictOutput:
    """Simulates a prediction operation for testing purposes.

    Args:
        item (_PredictInput): The input object containing the action to simulate.

    Returns:
        _PredictOutput: The output object with a status of "Success" if the action is "SuccessfulResponse".

    Raises:
        BaseAPIException: If the action is "BaseAPIException", this exception is raised for testing.
        Exception: If the action is unrecognized, a generic exception is raised for testing.
    """
    if item.action == "SuccessfulResponse":
        return _PredictOutput(status="Success")
    elif item.action == "BaseAPIException":
        raise BaseAPIException("This is a BaseAPIException raised for testing.")
    else:
        raise Exception("This is an unknown exception raised for testing.")


@pytest.fixture(name="client")
def app_client_fixture():
    """Use the real FastAPI app for testing."""
    client = TestClient(app)
    return client


def test_base_routes_of_create_app(client):
    """Test the health and root endpoints."""
    response = client.get("/health")
    assert response.status_code == 200
    # Accept both loaded and not loaded states
    data = response.json()
    assert "status" in data
    assert data["status"] in {"ok", "error"}
    # If not loaded, check for expected fields
    if data["status"] == "error":
        assert data["model_status"] == "not_loaded_or_error"
        assert data["model_info"] == "Predictor not initialized"


def test_base_exceptions_str_representation():
    """Test the string representation of BaseAPIException."""
    exc = BaseAPIException("Test error")
    assert (
        str(exc)
        == "[status_code=500][title=internal server error][details=Test error]"
    )


def test_base_api_exception_response():
    """Test the response() method of BaseAPIException."""
    exc = BaseAPIException("Test error response")
    response = exc.response()

    assert response.status_code == 500
    assert json.loads(response.body.decode()) == {
        "details": "Test error response",
        "status_code": 500,
        "title": "internal server error",
    }


def test_health_endpoint_when_model_not_loaded(client):
    """Health endpoint returns error when model is not loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["model_status"] == "not_loaded_or_error"
    assert data["model_info"] == "Predictor not initialized"


def test_predict_endpoint_model_not_loaded(client):
    """Predict endpoint returns 503 if model is not loaded."""
    response = client.post(
        "/predict/",
        files={"file": ("test.png", b"fakeimagebytes", "image/png")},
    )
    assert response.status_code == 503
    assert "Model service is not available" in response.text
