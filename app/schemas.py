"""Module that contains the schemas."""

import typing as tp

from pydantic import BaseModel  # pylint: disable=no-name-in-module


class BaseAPIExceptionModel(BaseModel):
    """Schema for a base api exception."""

    details: str
    status_code: tp.Literal[500] = 500
    title: tp.Literal["internal server error"] = "internal server error"
