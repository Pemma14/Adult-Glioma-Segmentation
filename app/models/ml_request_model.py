from datetime import datetime, timezone
from enum import Enum
from typing import Optional, TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base_model import Base, int_pk

if TYPE_CHECKING:
    from app.models import MLModel


class MLRequestStatus(str, Enum):
    success = "success"
    fail = "fail"
    pending = "pending"


class InferenceMode(str, Enum):
    ensemble = "ensemble"


class MLRequest(Base):
    __tablename__ = "ml_request"

    id: Mapped[int_pk]
    model_id: Mapped[int] = mapped_column(ForeignKey("ml_model.id"), nullable=False, index=True)
    input_data: Mapped[Any] = mapped_column(JSON, nullable=False)
    prediction: Mapped[Any] = mapped_column(JSON, nullable=True)
    errors: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[MLRequestStatus] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        server_default=text('now()'),
        nullable=False,
        index=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    message: Mapped[Optional[str]] = mapped_column(nullable=True)

    inference_mode: Mapped[Optional[InferenceMode]] = mapped_column(nullable=True)
    input_file_path: Mapped[Optional[str]] = mapped_column(nullable=True)
    output_dir: Mapped[Optional[str]] = mapped_column(nullable=True)
    prediction_path: Mapped[Optional[str]] = mapped_column(nullable=True)
    uncertainty_path: Mapped[Optional[str]] = mapped_column(nullable=True)
    report_path: Mapped[Optional[str]] = mapped_column(nullable=True)
    visualization_paths: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    ml_model: Mapped["MLModel"] = relationship(back_populates="ml_requests")