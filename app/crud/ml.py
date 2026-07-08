import logging
from typing import List, Optional, Any
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload
from app.models import MLModel, MLRequest, MLRequestStatus

logger = logging.getLogger(__name__)


def get_active_model(session: Session) -> Optional[MLModel]:
    model = session.execute(select(MLModel).where(MLModel.is_active == True)).scalars().first()
    logger.info("Active model: %s", model.code_name if model else None)
    return model


def create_request_record(
    session: Session,
    model_id: int,
    input_data: Any,
    status: MLRequestStatus = MLRequestStatus.pending,
) -> MLRequest:
    new_request = MLRequest(
        model_id=model_id,
        input_data=input_data,
        status=status,
    )
    session.add(new_request)
    session.flush()
    logger.info("Created request %s", new_request.id)
    return new_request


def update_request(session: Session, request_id: int, **kwargs: Any) -> Optional[MLRequest]:
    db_request = session.get(MLRequest, request_id)
    if db_request:
        for key, value in kwargs.items():
            if hasattr(db_request, key):
                setattr(db_request, key, value)
        session.flush()
        logger.info("Updated request %s: %s", request_id, kwargs)
    return db_request


def get_history(session: Session) -> List[MLRequest]:
    query = (
        select(MLRequest)
        .options(joinedload(MLRequest.ml_model))
        .order_by(MLRequest.created_at.desc())
    )
    return list(session.execute(query).scalars().all())


def get_request_by_id(session: Session, request_id: int) -> Optional[MLRequest]:
    return session.execute(
        select(MLRequest).options(joinedload(MLRequest.ml_model)).where(MLRequest.id == request_id)
    ).scalar_one_or_none()
