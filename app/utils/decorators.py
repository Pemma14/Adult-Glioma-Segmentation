import functools
import logging
from typing import Any, Callable, TypeVar, cast
import inspect

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

def transactional(func: F) -> F:
    """
    Декоратор для автоматического управления транзакциями SQLAlchemy.
    Работает как с синхронными, так и с асинхронными методами классов,
    у которых есть атрибут self.session.
    """

    @functools.wraps(func)
    async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = await func(self, *args, **kwargs)
            if hasattr(self, "session") and self.session:
                self.session.commit()
            return result
        except Exception as e:
            if hasattr(self, "session") and self.session:
                self.session.rollback()
            logger.error(f"Ошибка в транзакции (async {func.__name__}): {e}")
            raise

    @functools.wraps(func)
    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = func(self, *args, **kwargs)
            if hasattr(self, "session") and self.session:
                self.session.commit()
            return result
        except Exception as e:
            if hasattr(self, "session") and self.session:
                self.session.rollback()
            logger.error(f"Ошибка в транзакции (sync {func.__name__}): {e}")
            raise

    if inspect.iscoroutinefunction(func):
        return cast(F, async_wrapper)
    return cast(F, sync_wrapper)
