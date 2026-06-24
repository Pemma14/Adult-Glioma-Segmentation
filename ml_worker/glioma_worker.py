"""RabbitMQ worker for glioma segmentation inference."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import aio_pika

# Ensure project root is on sys.path for src.glioma imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from app.config import settings
from app.schemas.ml_task_schemas import MLResult, MLTask
from src.glioma.inference import predict

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _build_report(result: dict, version: str) -> dict:
    """Build a JSON-serializable clinical report from inference result."""
    return {
        "case_id": result.get("case_id"),
        "model_version": version,
        "inference_mode": result.get("inference_mode", "ensemble"),
        "volumes_ml": result.get("volumes_ml"),
        "prediction_path": result.get("prediction_path"),
        "uncertainty_path": result.get("uncertainty_path"),
        "region_prediction_path": result.get("region_prediction_path"),
        "visualization_paths": result.get("visualization_paths", []),
    }


def _run_inference(task: MLTask) -> MLResult:
    """Run the segmentation inference and return an MLResult."""
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Running ensemble inference for request %s on %s",
        task.request_id,
        device,
    )

    try:
        result = predict(
            image_path=task.input_file_path,
            output_dir=output_dir,
            device=device,
            folds=None,
            save_uncertainty=task.save_uncertainty,
            save_regions=True,
            save_visualization=True,
            n_slices=3,
        )
    except Exception as exc:
        logger.exception("Inference failed for request %s", task.request_id)
        return MLResult(
            task_id=task.task_id,
            request_id=task.request_id,
            status="fail",
            error=str(exc),
        )

    # Save JSON report
    report = _build_report(result, settings.glioma.MODEL_VERSION)
    report_path = output_dir / f"{result['case_id']}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result["report_path"] = str(report_path)
    result["inference_mode"] = task.inference_mode

    return MLResult(
        task_id=task.task_id,
        request_id=task.request_id,
        status="success",
        case_id=result.get("case_id"),
        prediction_path=result.get("prediction_path"),
        uncertainty_path=result.get("uncertainty_path"),
        report_path=str(report_path),
        visualization_paths=result.get("visualization_paths"),
        volumes_ml=result.get("volumes_ml"),
    )


async def process_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    async with message.process():
        try:
            payload = json.loads(message.body.decode())
            task = MLTask(**payload)
            logger.info("Received task %s for request %s", task.task_id, task.request_id)

            result = _run_inference(task)
            await publish_result(result)
        except Exception as exc:
            logger.exception("Failed to process message: %s", exc)


async def publish_result(result: MLResult) -> None:
    amqp_url = settings.mq.amqp_url
    connection = await aio_pika.connect_robust(amqp_url)
    try:
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            settings.mq.RESULTS_EXCHANGE_NAME,
            type=aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        message = aio_pika.Message(
            body=result.model_dump_json().encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
            message_id=result.task_id,
        )
        await exchange.publish(
            message,
            routing_key=settings.mq.RESULTS_ROUTING_KEY,
        )
        logger.info("Published result for request %s (status=%s)", result.request_id, result.status)
    finally:
        await connection.close()


async def main() -> None:
    configure_logging()
    logger.info("Starting glioma segmentation worker")
    logger.info("Model version: %s", settings.glioma.MODEL_VERSION)

    connection = await aio_pika.connect_robust(settings.mq.amqp_url)
    try:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        exchange = await channel.declare_exchange(
            settings.mq.EXCHANGE_NAME,
            type=aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(
            settings.mq.QUEUE_NAME,
            durable=True,
        )
        await queue.bind(exchange, routing_key=settings.mq.QUEUE_NAME)

        logger.info("Worker waiting for tasks on queue: %s", settings.mq.QUEUE_NAME)
        await queue.consume(process_message)

        # Keep running
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
