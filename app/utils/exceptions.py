from typing import Optional, Any
from fastapi import HTTPException, status


class AppException(HTTPException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = ""

    def __init__(self) -> None:
        super().__init__(status_code=self.status_code, detail=self.detail)


class MLRequestNotFoundException(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Request not found"


class MLModelNotFoundException(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Model not found"


class MLModelLoadException(AppException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Failed to load ML model"


class MLInferenceException(AppException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Inference error"


class MLInvalidDataException(AppException):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    detail = "Invalid input data"

    def __init__(self, errors: Optional[list[Any]] = None) -> None:
        super().__init__()
        if errors:
            self.detail = {"message": self.detail, "errors": errors}


class MQServiceException(AppException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Message queue unavailable"

    def __init__(self, request_id: Optional[Any] = None, original_exception: Optional[Exception] = None) -> None:
        self.request_id = request_id
        self.original_exception = original_exception
        super().__init__()



