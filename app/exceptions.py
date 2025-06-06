"""Module for handling custom exceptions."""

from fastapi import status
from starlette.responses import JSONResponse

from app.schemas import BaseAPIExceptionModel


class BaseAPIException(Exception):
    """Base error for custom API exceptions."""

    pydantic_model = BaseAPIExceptionModel
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, details: str) -> None:
        """Init the base Exception.

        Args:
            details: details about the exception in a string format
        """
        super().__init__(details)
        self.content = self.pydantic_model(details=details)

    def __str__(self) -> str:
        """Str repr of the exception."""
        return (
            f"[status_code={self.content.status_code}]"
            f"[title={self.content.title}][details={self.content.details}]"
        )

    def response(self) -> JSONResponse:
        """Return response based on pydantic model."""
        return JSONResponse(
            content=self.content.model_dump(),
            status_code=self.content.status_code,
        )

    @classmethod
    def response_model(
        cls,
    ):
        """Return response model to show on swagger."""
        return {cls.status_code: {"model": cls.pydantic_model}}
