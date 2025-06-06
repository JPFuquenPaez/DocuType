"""Base API."""

import logging
import os
import sys  # Keep sys for StreamHandler(sys.stdout)
import typing as tp

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.inference.inference_onnx import ModelPredictorONNX

PREDICTIONS_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "webservice_predictions"
)
LOG_FILE_NAME = "predictions.log"
os.makedirs(PREDICTIONS_LOG_DIR, exist_ok=True)
log_file_path = os.path.join(PREDICTIONS_LOG_DIR, LOG_FILE_NAME)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Added %(name)s
    handlers=[
        logging.FileHandler(log_file_path),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to console as well
    ],
)
logger = logging.getLogger("webserviceapp")  # Use a named logger for this module


# --- Pydantic Models for API ---
class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field("ok", examples=["ok"])
    model_status: str = Field(
        ..., examples=["loaded"]
    )  # Or Try? Field(..., example="loaded")
    model_version: tp.Optional[str] = Field(None, examples=["1"])
    model_info: tp.Optional[str] = Field(None, examples=["Alias: prod"])


class PredictionResponse(BaseModel):
    """Represents the response returned after making a prediction on an image.

    Attributes:
        image_name (str): Name of the input image.
        predicted_class (str): Predicted class label for the image.
        confidence (float): Confidence score of the prediction.
        model_version (Optional[str]): Version of the model used for prediction.
        model_info (Optional[str]): Additional information about the model.
    """

    image_name: str = Field(..., examples=["test_image.png"])
    predicted_class: str = Field(..., examples=["printed"])
    confidence: float = Field(..., examples=[0.9876])
    model_version: tp.Optional[str] = Field(None, examples=["1"])
    model_info: tp.Optional[str] = Field(None, examples=["Alias: prod"])


class ErrorDetail(BaseModel):
    """Represents the details of an error response."""

    message: str = Field(..., examples=["An unexpected error occurred."])
    type: tp.Optional[str] = Field(None, examples=["PredictionError"])


# --- Custom Exceptions ---
class BaseAPIException(HTTPException):
    """Construct base class for all custom API exceptions."""

    def __init__(
        self, status_code: int, detail: str, exc_type: str = "APIException"
    ):
        """Base class init."""
        super().__init__(status_code=status_code, detail=detail)
        self.exc_type = exc_type


class ModelLoadException(BaseAPIException):
    """Construct the ModelLoadError exception with a given detail message."""

    def __init__(self, detail: str):
        """ModelLoadException init."""
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            exc_type="ModelLoadError",
        )


class PredictionException(BaseAPIException):
    """Construct the PredictionException with a specific detail message.

    Args:
        detail (str): A message describing the details of the prediction error.
    """

    def __init__(self, detail: str):
        """PredictionException init."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            exc_type="PredictionError",
        )


# --- Global Predictor Instance ---
image_classifier: tp.Optional[ModelPredictorONNX] = None


# --- FastAPI Event Handlers ---
async def startup_predictor_event() -> None:
    """Asynchronously initializes the global image classifier by loading the ONNX model and associated transforms at FastAPI application startup.

    Logs the process and handles exceptions by setting the classifier to None and logging a critical error.
    """
    global image_classifier
    logger.info("FastAPI application startup: Initializing ModelPredictorONNX...")
    image_classifier = ModelPredictorONNX(
        model_name=settings.MODEL_NAME,
        onnx_model_artifact_name=settings.ONNX_MODEL_ARTIFACT_NAME,
        transform_artifact_path=settings.TRANSFORM_ARTIFACT_PATH,
        alias=settings.MODEL_ALIAS,
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    try:
        logger.info("Loading MLflow model and associated transforms...")
        image_classifier.load_model()
        logger.info(
            "Model and transforms loaded successfully. Model Version: %s, Info: %s",
            getattr(image_classifier, "model_version", "N/A"),
            getattr(image_classifier, "model_info", "N/A"),
        )
    except Exception as e:
        image_classifier = None
        logger.critical(
            f"CRITICAL STARTUP ERROR: Failed to load model: {e}", exc_info=True
        )


# --- Base API Route Functions ---
async def root_redirect(req: Request) -> RedirectResponse:
    """Redirects the root URL to the API documentation page.

    Args:
        req (Request): The incoming HTTP request object.

    Returns:
        RedirectResponse: A response that redirects the client to the '/docs' endpoint,
        preserving any root path specified in the request's scope.
    """
    root_path = req.scope.get("root_path", "").rstrip("/")
    return RedirectResponse(root_path + "/docs")


async def health_check() -> HealthCheckResponse:
    """Performs a health check on the image classifier model.

    Returns:
        HealthCheckResponse: An object containing the status of the service, model loading status,
        model version, and additional model information.
    """
    global image_classifier
    if (
        image_classifier
        and image_classifier.ort_session
        and image_classifier.transform
    ):
        return HealthCheckResponse(
            status="ok",
            model_status="loaded",
            model_version=image_classifier.model_version,
            model_info=image_classifier.model_info,
        )

    model_version = (
        getattr(image_classifier, "model_version", None)
        if image_classifier
        else None
    )
    model_info_str = (
        f"Alias: {getattr(image_classifier, 'alias', settings.MODEL_ALIAS)}"  # Use configured alias as fallback
        if image_classifier and hasattr(image_classifier, "alias")
        else "Predictor not initialized"
    )
    logger.warning(
        "Health check: Model is not loaded or an error occurred. Status: error, Model Status: not_loaded_or_error, Version: %s, Info: %s",
        model_version,
        model_info_str,
    )
    return HealthCheckResponse(
        status="error",
        model_status="not_loaded_or_error",
        model_version=model_version,
        model_info=model_info_str,
    )


def add_base_routes(app_instance: FastAPI) -> None:
    """Adds base routes to the given FastAPI application instance, including a health check endpoint and a root redirect.

    Args:
        app_instance (FastAPI): The FastAPI application to which the routes will be added.
    """
    app_instance.add_api_route(
        "/health",
        health_check,
        response_model=HealthCheckResponse,
        tags=["Utility"],
    )
    app_instance.add_api_route("/", root_redirect, include_in_schema=False)


# --- FastAPI App Factory ---
def create_application(
    debug: bool = bool(
        os.getenv("DEBUG", False)
    ),  # Consider moving DEBUG to central config too
    title: str = settings.APP_TITLE,
    description: str = settings.APP_DESCRIPTION,
    version: str = settings.APP_VERSION,
    **kwargs: tp.Any,
) -> FastAPI:
    """Creates and configures a FastAPI application instance.

    Args:
        debug (bool, optional): Whether to run the application in debug mode. Defaults to value from environment variable 'DEBUG'.
        title (str, optional): The title of the application. Defaults to settings.APP_TITLE.
        description (str, optional): The description of the application. Defaults to settings.APP_DESCRIPTION.
        version (str, optional): The version of the application. Defaults to settings.APP_VERSION.
        **kwargs: Additional keyword arguments passed to the FastAPI constructor.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    # Reconfigure root logger's level if debug status changes, mainly affects console output
    # Note: FileHandler will still log at INFO unless its level is also explicitly set.
    # For simplicity here, basicConfig sets the shared level.
    # If you need different levels for file vs console, you'd configure handlers individually.
    logging.getLogger().setLevel(log_level)  # Sets root logger level

    logger.info(
        "Creating FastAPI application. Debug: %s, Title: %s, Version: %s, Log Level: %s",
        settings.DEBUG,
        settings.APP_TITLE,
        settings.APP_VERSION,
        logging.getLevelName(log_level),
    )

    new_app = FastAPI(
        debug=settings.DEBUG,
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        **kwargs,
    )
    add_base_routes(new_app)
    new_app.add_event_handler("startup", startup_predictor_event)
    logger.info("FastAPI application created and configured.")
    return new_app


app = create_application()


# --- API Endpoints ---
predict_extra_responses: tp.Dict[int | str, tp.Dict[str, tp.Any]] = {
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": ErrorDetail,
        "description": "Prediction error (e.g., invalid image data).",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorDetail,
        "description": "Model not loaded or unavailable.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorDetail,
        "description": "An unexpected internal server error.",
    },
}


@app.post(
    "/predict/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Classification"],
    summary="Classify an uploaded image as handwritten or printed.",
    responses=predict_extra_responses,
)
async def predict_image_endpoint(
    file: UploadFile = File(
        ..., description="Image file to classify (e.g., PNG, JPG)."
    ),
) -> PredictionResponse:
    """Endpoint to classify an uploaded image as handwritten or printed.

    Receives an image file via POST request, processes it using the global image_classifier,
    and returns the predicted class, confidence, model version, and model info.

    Args:
        file (UploadFile): Image file to classify (e.g., PNG, JPG).

    Returns:
        PredictionResponse: The prediction result containing the predicted class, confidence,
        model version, and model info.

    Raises:
        ModelLoadException: If the model service is not available.
        PredictionException: If no image data is received or a ValueError occurs during prediction.
        HTTPException: For FastAPI or custom HTTP exceptions.
        BaseAPIException: For any other unexpected internal server errors.

    Logs:
        - Logs the receipt of a prediction request.
        - Logs errors if the model is not available.
        - Logs warnings if no image data is received or a ValueError occurs.
        - Logs successful inference with prediction details.
        - Logs unexpected errors during prediction.
    """
    global image_classifier
    # This log indicates the start of a prediction attempt
    logger.info(f"Prediction request received for image: '{file.filename}'")

    if (
        not image_classifier
        or not image_classifier.ort_session
        or not image_classifier.transform
    ):
        logger.error(
            "Prediction endpoint called for '%s' but model predictor is not available.",
            file.filename,
        )
        raise ModelLoadException(detail="Model service is not available !!!")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            logger.warning(
                f"No image data received in uploaded file: '{file.filename}'"
            )
            raise PredictionException(
                detail="No image data received in uploaded file."
            )
        # This log can be useful for debugging image processing issues
        logger.debug(
            f"Image '{file.filename}' read successfully. Size: {len(image_bytes)} bytes. Starting inference."
        )

        # This is where the actual inference happens
        prediction_result = image_classifier.predict_from_bytes(
            image_bytes, image_name=file.filename
        )

        # THIS IS THE KEY LOG FOR EACH SUCCESSFUL PREDICTION:
        # It's a distinct log entry (due to timestamp and content) for each inference,
        # and includes the image name and the prediction outputs.
        logger.info(
            f"Inference successful for '{file.filename}': "
            f"Predicted Class='{prediction_result['predicted_class']}', "
            f"Confidence={prediction_result['confidence']:.4f}, "
            f"Model Version='{prediction_result['model_version']}', "
            f"Model Info='{prediction_result['model_info']}'"
        )

        return PredictionResponse(**prediction_result)

    except ValueError as ve:  # Typically for issues during model's preprocessing or prediction logic
        logger.warning(
            f"Prediction ValueError for '{file.filename}': {ve}", exc_info=False
        )
        raise PredictionException(detail=str(ve))
    except HTTPException:  # Re-raise FastAPI/custom HTTP exceptions
        raise
    except Exception as e:  # Catch any other unexpected errors during prediction
        logger.error(
            f"Unexpected error during prediction for '{file.filename}': {e}",
            exc_info=True,
        )
        raise BaseAPIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred: {str(e)}",
            exc_type="InternalServerError",
        )
