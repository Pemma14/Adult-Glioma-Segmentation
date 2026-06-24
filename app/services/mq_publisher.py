import logging
import aio_pika
import asyncio
import uuid
from time import time
from typing import Optional, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import Request
from aio_pika.pool import Pool
from app.config import settings
from app.schemas.ml_task_schemas import MLTask
from app.utils import MQServiceException

logger = logging.getLogger(__name__)

class MLTaskPublisher:
    def __init__(self, connection_pool: Pool[aio_pika.RobustConnection]) -> None:
        self.connection_pool = connection_pool
        self.channel_pool: Pool[aio_pika.RobustChannel] = Pool(
            self._get_channel,
            max_size=10
        )
        self._infrastructure_ready: bool = False

    async def _get_channel(self) -> aio_pika.RobustChannel:
        async with self.connection_pool.acquire() as connection:
            return await connection.channel(publisher_confirms=True)

    async def ensure_infrastructure(self) -> None:
        if self._infrastructure_ready:
            return

        async with self.channel_pool.acquire() as channel:
            # Объявляем обменник
            exchange = await channel.declare_exchange(
                settings.mq.EXCHANGE_NAME,
                type=aio_pika.ExchangeType.DIRECT,
                durable=True
            )
            # Объявляем очередь
            queue = await channel.declare_queue(
                settings.mq.QUEUE_NAME,
                durable=True
            )
            # Привязываем очередь к обменнику
            await queue.bind(exchange, routing_key=settings.mq.QUEUE_NAME)

            self._infrastructure_ready = True
            logger.info("RabbitMQ infrastructure (Exchange/Queue) ensured")

    @retry(
        stop=stop_after_attempt(settings.mq.RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=settings.mq.RETRY_MULTIPLIER,
            min=settings.mq.RETRY_MIN,
            max=settings.mq.RETRY_MAX
        ),
        reraise=True
    )
    async def _publish_with_retry(self, message: aio_pika.Message) -> None:
        async with self.channel_pool.acquire() as channel:
            exchange = await channel.get_exchange(settings.mq.EXCHANGE_NAME)
            await exchange.publish(
                message,
                routing_key=settings.mq.QUEUE_NAME,
                mandatory=True
            )

    async def send_task(self, task: MLTask) -> None:
        try:
            await self.ensure_infrastructure()

            message = aio_pika.Message(
                body=task.model_dump_json().encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
                message_id=task.task_id,
                app_id="glioma-service",
                timestamp=task.timestamp,
            )

            await self._publish_with_retry(message)
            logger.info(f"Задача {task.task_id} успешно отправлена в RabbitMQ")

        except aio_pika.exceptions.AMQPError as e:
            raise MQServiceException(request_id=task.task_id, original_exception=e)
        except Exception as e:
            raise MQServiceException(request_id=task.task_id, original_exception=e)

    async def close(self) -> None:
        await self.channel_pool.close()

class RPCPublisher:
    def __init__(self, connection_pool: Pool[aio_pika.RobustConnection]) -> None:
        self.connection_pool = connection_pool
        self.futures: Dict[str, Tuple[asyncio.Future, float]] = {}
        self._callback_queue: Optional[aio_pika.abc.AbstractRobustQueue] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._loop = asyncio.get_event_loop()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def _get_channel(self) -> aio_pika.RobustChannel:
        async with self.connection_pool.acquire() as connection:
            return await connection.channel()

    async def ensure_ready(self) -> None:
        if self._channel and not self._channel.is_closed:
            return

        async with self.connection_pool.acquire() as connection:
            self._channel = await connection.channel()
            self._callback_queue = await self._channel.declare_queue(
                exclusive=True,
                auto_delete=True
            )
            await self._callback_queue.consume(self.on_response, no_ack=True)
            logger.info(f"RPC Client ready, callback queue: {self._callback_queue.name}")

        # Запускаем задачу очистки устаревших фьючерсов
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_futures())

    async def _cleanup_stale_futures(self) -> None:
        """Удаляет фьючерсы старше 5 минут каждую минуту."""
        while True:
            try:
                await asyncio.sleep(60)
                now = time()
                stale_ids = [
                    cid for cid, (_, ts) in self.futures.items()
                    if now - ts > 300
                ]
                for cid in stale_ids:
                    logger.warning(f"Удаление устаревшей RPC future: {cid}")
                    self.futures.pop(cid, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в задаче очистки RPC фьючерсов: {e}")
                await asyncio.sleep(10)

    async def on_response(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        """Обработчик ответов из callback очереди."""
        if message.correlation_id is None:
            logger.warning(f"Получено сообщение без correlation_id: {message!r}")
            return

        item = self.futures.pop(message.correlation_id, None)
        if item:
            future, _ = item
            if not future.done():
                future.set_result(message.body)

    async def call(self, payload: bytes, routing_key: str, timeout: float = 10.0) -> bytes:
        """Выполняет RPC запрос."""
        await self.ensure_ready()

        correlation_id = str(uuid.uuid4())
        future = self._loop.create_future()
        self.futures[correlation_id] = (future, time())

        try:
            await self._channel.default_exchange.publish(
                aio_pika.Message(
                    payload,
                    content_type="application/json",
                    correlation_id=correlation_id,
                    reply_to=self._callback_queue.name,
                ),
                routing_key=routing_key,
            )

            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise MQServiceException(original_exception=e)
        finally:
            self.futures.pop(correlation_id, None)

    async def close(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._channel:
            await self._channel.close()

def get_mq_service(request: Request) -> MLTaskPublisher:
    """Зависимость для получения сервиса RabbitMQ из состояния приложения."""
    mq_service = request.app.state.mq_service
    if mq_service is None:
        raise MQServiceException()
    return mq_service

def get_rpc_client(request: Request) -> RPCPublisher:
    """Зависимость для получения RPC клиента из состояния приложения."""
    rpc_client = request.app.state.rpc_client
    if rpc_client is None:
        raise MQServiceException()
    return rpc_client
