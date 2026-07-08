import logging

import aio_pika
import asyncio
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import settings
from app.database.database import init_db
from app.services.mq_publisher import MLTaskPublisher
from app.services.mq_consumer import ResultsConsumer
from aio_pika.pool import Pool
from app.routes.glioma_router import router as glioma_router
from app.routes.home_router import router as home_router
from app.utils import setup_logging, setup_exception_handlers

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    import os
    is_test = os.getenv("TESTING")

    if not is_test:
        logger.info("Initializing database...")
        try:
            init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

        logger.info("Connecting to RabbitMQ...")
        try:
            async def get_connection():
                return await aio_pika.connect_robust(
                    settings.mq.amqp_url,
                    timeout=settings.mq.TIMEOUT
                )

            connection_pool = Pool(get_connection, max_size=2)
            application.state.mq_service = MLTaskPublisher(connection_pool)

            application.state.results_consumer = ResultsConsumer()
            application.state.results_consumer_task = asyncio.create_task(
                application.state.results_consumer.run()
            )
            logger.info("RabbitMQ services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RabbitMQ: {e}")
            application.state.mq_service = None

    yield

    if not is_test:
        logger.info("Shutting down...")
        if hasattr(application.state, "results_consumer") and application.state.results_consumer:
            await application.state.results_consumer.stop()
        if application.state.mq_service:
            await application.state.mq_service.close()
            await application.state.mq_service.connection_pool.close()
        logger.info("Shutdown complete")


def create_application() -> FastAPI:
    application = FastAPI(
        title="Glioma Segmentation Service",
        description="On-premise API for automated 3D segmentation of adult-type diffuse gliomas",
        version="1.0.0",
        docs_url="/api/docs" if settings.app.DEBUG else None,
        lifespan=lifespan,
    )

    setup_logging()
    setup_exception_handlers(application)

    settings.glioma.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/outputs",
        StaticFiles(directory=settings.glioma.OUTPUT_DIR, check_dir=False),
        name="outputs",
    )

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/viewer",
        StaticFiles(directory=str(static_dir), html=True),
        name="static",
    )

    application.include_router(home_router, tags=["Home"])
    application.include_router(glioma_router, prefix="/api/v1/segmentation", tags=["Segmentation"])

    return application


app = create_application()


if __name__ == '__main__':
    uvicorn.run(
        'app.main:app',
        host=settings.app.HOST,
        port=settings.app.PORT,
        reload=settings.app.DEBUG,
        log_level="debug" if settings.app.DEBUG else "info",
    )
