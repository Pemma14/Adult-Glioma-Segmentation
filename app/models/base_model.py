from sqlalchemy.orm import DeclarativeBase, mapped_column
from typing import Annotated

# Базовый класс для моделей
class Base(DeclarativeBase):
    def __repr__(self):
        cols = [f"{col}={getattr(self, col)}" for col in self.__table__.columns.keys()]
        return f"<{self.__class__.__name__}({', '.join(cols)})>"


# Аннотации для часто используемых типов колонок
int_pk = Annotated[int, mapped_column(primary_key=True)]
str_uniq = Annotated[str, mapped_column(unique=True, nullable=False)]
str_null_true = Annotated[str, mapped_column(nullable=True)]
