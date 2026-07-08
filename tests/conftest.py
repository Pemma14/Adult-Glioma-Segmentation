import os

os.environ["GLIOMA__MODEL_VERSION"] = "v1.0.0"
os.environ["AUTH__API_KEY"] = "test-api-key"
os.environ["TESTING"] = "1"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database.database as db_module
from app.models import Base, MLModel

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(engine, expire_on_commit=False)

db_module.engine = engine
db_module.session_maker = TestSession

from app.database.database import get_session
from app.main import app
from fastapi.testclient import TestClient
import pytest


@pytest.fixture(scope="function")
def session():
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="function")
def active_model(session):
    model = MLModel(
        name="Test Model",
        code_name="glioma_segmentation",
        version="1.0.0",
        is_active=True,
    )
    session.add(model)
    session.flush()
    return model


@pytest.fixture(scope="function")
def mock_mq_service():
    from unittest.mock import AsyncMock, MagicMock
    mock_mq = MagicMock()
    mock_mq.send_task = AsyncMock(return_value=None)
    return mock_mq


@pytest.fixture(scope="function")
def client(session, active_model, mock_mq_service):
    def override_get_session():
        yield session

    app.dependency_overrides[get_session] = override_get_session
    app.state.mq_service = mock_mq_service

    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
