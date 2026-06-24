import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.utils.exceptions import AppException, MQServiceException

logger = logging.getLogger(__name__)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Обработчик кастомных исключений приложения."""
    if isinstance(exc, MQServiceException):
        msg = f"Ошибка RabbitMQ"
        if exc.request_id:
            msg += f" (задача {exc.request_id})"
        if exc.original_exception:
            msg += f": {exc.original_exception}"
        logger.error(msg)
    else:
        logger.warning(f"Ошибка приложения: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Обработчик ошибок валидации (Pydantic)."""
    errors = exc.errors()
    simplified_errors = [
        {"field": ".".join(map(str, error["loc"][1:])), "message": error["msg"]}
        for error in errors
    ]
    logger.warning(f"Ошибка валидации: {simplified_errors}")
    return JSONResponse(
        status_code=422,
        content={"status": "validation_error", "errors": simplified_errors}
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Обработчик ошибок базы данных."""
    logger.error(f"Database error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "db_error", "message": "Ошибка при работе с базой данных"}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Обработчик всех остальных непредвиденных ошибок."""
    logger.critical(f"НЕПРЕДВИДЕННАЯ ОШИБКА: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "critical_error", "message": "Произошла внутренняя ошибка сервера"}
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Регистрация всех обработчиков исключений."""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
