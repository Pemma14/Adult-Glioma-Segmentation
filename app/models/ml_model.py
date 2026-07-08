from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, TYPE_CHECKING
from app.models.base_model import Base, int_pk, str_uniq

if TYPE_CHECKING:
    from app.models import MLRequest


class MLModel(Base):
    __tablename__ = "ml_model"

    id: Mapped[int_pk]
    name: Mapped[str_uniq]
    code_name: Mapped[str_uniq]
    description: Mapped[str] = mapped_column(nullable=True)
    version: Mapped[str] = mapped_column(default="1.0.0")
    is_active: Mapped[bool] = mapped_column(default=True)

    ml_requests: Mapped[List["MLRequest"]] = relationship(
        back_populates="ml_model",
        cascade="all, delete-orphan"
    )
