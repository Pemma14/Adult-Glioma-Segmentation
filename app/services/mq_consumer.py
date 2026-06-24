import asyncio
import json
import logging
from typing import Optional

import aio_pika
from sqlalchemy import select

from app.config import settings
from app.database.database import session_maker
from app.models import MLRequest, MLRequestStatus
from app.schemas.ml_task_schemas import MLResult
from app.services.glioma_service import GliomaRequestService

logger = logging.getLogger(__name__)


class ResultsConsumer:
    """
    Потребитель результатов из RabbitMQ. Получает сообщения с результатами ML,
    сохраняет их через MLRequestService.process_task_result.
    Работает как фоновая задача FastAPI-приложения.
    """

    def __init__(self, amqp_url: Optional[str] = None) -> None:
        self.amqp_url = amqp_url or settings.mq.amqp_url
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.RobustChannel] = None
        self.queue: Optional[aio_pika.abc.AbstractRobustQueue] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        retry_interval = 5
        while not self._stop_event.is_set():
            try:
                logger.info("[ResultsConsumer] Подключение к RabbitMQ...")
                self.connection = await aio_pika.connect_robust(self.amqp_url)
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=10)

                # Объявляем обменник и очередь результатов
                exchange = await self.channel.declare_exchange(
                    settings.mq.RESULTS_EXCHANGE_NAME,
                    type=aio_pika.ExchangeType.DIRECT,
                    durable=True,
                )
                self.queue = await self.channel.declare_queue(
                    settings.mq.RESULTS_QUEUE_NAME,
                    durable=True,
                )
                await self.queue.bind(exchange, routing_key=settings.mq.RESULTS_ROUTING_KEY)

                await self.queue.consume(self._on_message)
                logger.info(
                    f"[ResultsConsumer] Запущен. Очередь: {settings.mq.RESULTS_QUEUE_NAME}, обменник: {settings.mq.RESULTS_EXCHANGE_NAME}"
                )
                break
            except Exception as e:
                logger.error(f"[ResultsConsumer] Ошибка при старте: {e}. Повтор через {retry_interval} сек...")
                await asyncio.sleep(retry_interval)

    async def _on_message(self, message: aio_pika.IncomingMessage) -> None:
        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                result = MLResult(**payload)
                logger.info(f"[ResultsConsumer] Получен результат для task_id={result.task_id}")

                with session_maker() as session:
                    request_id = result.request_id

                    existing = session.execute(
                        select(MLRequest).where(MLRequest.id == request_id)
                    ).scalar_one_or_none()

                    if not existing:
                        logger.error(f"[ResultsConsumer] Запрос {request_id} не найден в базе")
                        return

                    if existing.status != MLRequestStatus.pending:
                        logger.warning(f"[ResultsConsumer] Запрос {request_id} уже обработан (статус: {existing.status})")
                        return

                    service = GliomaRequestService(session)
                    await service.process_result(result)
                    session.commit()

            except Exception as e:
                logger.error(f"[ResultsConsumer] Ошибка обработки сообщения: {e}")
                raise

    async def stop(self) -> None:
        logger.info("[ResultsConsumer] Остановка...")
        self._stop_event.set()
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
        except Exception as e:
            logger.warning(f"[ResultsConsumer] Ошибка при остановке: {e}")

    async def run(self) -> None:
        await self.start()
        await self._stop_event.wait()
