import os
import logging
from typing import Generator
from sqlalchemy import create_engine, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import Base, MLModel
from app.database.seed import seed_db

logger = logging.getLogger(__name__)

engine_kwargs = settings.db.get_engine_kwargs()
logger.info(f"Connecting to DB (host: {settings.db.HOST})")

engine: Engine = create_engine(**engine_kwargs)
session_maker = sessionmaker(engine, expire_on_commit=False)


def get_database_engine() -> Engine:
    return engine


def get_session() -> Generator[Session, None, None]:
    with session_maker() as session:
        yield session


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: logger.info(f"Retrying DB init... (attempt {retry_state.attempt_number})")
)
def init_db(drop_all: bool = False) -> None:
    if os.getenv("TESTING"):
        logger.info("TESTING mode: skipping db init")
        return
    try:
        if drop_all:
            Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        logger.info("Database tables initialized")

        with session_maker() as session:
            model_count = session.execute(select(func.count()).select_from(MLModel.__table__)).scalar()
            if model_count == 0:
                seed_db(session)
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        raise
