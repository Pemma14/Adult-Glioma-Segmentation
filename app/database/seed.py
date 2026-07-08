import logging
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.config import settings
from app.models import MLModel

logger = logging.getLogger(__name__)


def seed_db(session: Session):
    existing = session.execute(
        select(MLModel).where(MLModel.code_name == "glioma_segmentation")
    ).scalars().first()

    if not existing:
        model = MLModel(
            name="Adult Glioma Segmentation",
            code_name="glioma_segmentation",
            description="3D ensemble segmentation of adult-type diffuse gliomas (WT/TC/ET)",
            version=settings.glioma.MODEL_VERSION,
        )
        session.add(model)
        logger.info("ML model seeded")

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Seed failed: {e}")
        raise
