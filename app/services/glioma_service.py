from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.crud import ml as ml_crud
from app.models import MLModel, MLRequest, MLRequestStatus
from app.schemas.glioma_schemas import (
    SGliomaResultResponse,
    SGliomaStatusResponse,
    VolumesML,
)
from app.schemas.ml_task_schemas import MLResult
from app.utils import (
    MLModelNotFoundException,
    MLRequestNotFoundException,
    MLInvalidDataException,
)

logger = logging.getLogger(__name__)


class GliomaRequestService:
    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def _ensure_dirs() -> None:
        settings.glioma.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.glioma.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_file(file: UploadFile) -> None:
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if filename.endswith(".nii.gz"):
            ext = ".nii.gz"
        if ext not in settings.glioma.ALLOWED_EXTENSIONS:
            raise MLInvalidDataException(
                errors=[{"field": "file", "message": f"Only {settings.glioma.ALLOWED_EXTENSIONS} files are supported"}]
            )

    @staticmethod
    def _save_upload_file(file: UploadFile, request_id: int) -> Path:
        GliomaRequestService._ensure_dirs()
        filename = file.filename or "unknown.nii.gz"
        safe_name = Path(filename).name
        request_dir = settings.glioma.UPLOAD_DIR / f"request_{request_id}"
        request_dir.mkdir(parents=True, exist_ok=True)
        dest_path = request_dir / safe_name
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved upload for request %s to %s", request_id, dest_path)
        return dest_path

    def _get_active_glioma_model(self) -> MLModel:
        model = ml_crud.get_active_model(self.session)
        if not model:
            raise MLModelNotFoundException
        return model

    async def create_segmentation_request(
        self,
        file: UploadFile,
        mq_service: Any,
        save_uncertainty: bool = True,
    ) -> MLRequest:
        self._validate_file(file)
        model = self._get_active_glioma_model()

        db_request = ml_crud.create_request_record(
            session=self.session,
            model_id=model.id,
            input_data={"filename": file.filename, "inference_mode": "ensemble"},
            status=MLRequestStatus.pending,
        )

        input_path = self._save_upload_file(file, db_request.id)
        output_dir = settings.glioma.OUTPUT_DIR / f"request_{db_request.id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        db_request.input_file_path = str(input_path)
        db_request.output_dir = str(output_dir)
        db_request.inference_mode = "ensemble"
        db_request.message = "Request accepted and queued for inference"

        from app.schemas.ml_task_schemas import MLTask
        task = MLTask(
            request_id=db_request.id,
            input_file_path=str(input_path),
            output_dir=str(output_dir),
            inference_mode="ensemble",
            save_uncertainty=save_uncertainty,
            model=model.code_name,
        )
        await mq_service.send_task(task)

        self.session.commit()
        logger.info("Published segmentation task for request %s", db_request.id)

        return db_request

    def get_status(self, request_id: int) -> SGliomaStatusResponse:
        db_request = self._get_request(request_id)
        return SGliomaStatusResponse(
            request_id=db_request.id,
            status=db_request.status,
            message=db_request.message,
            inference_mode=db_request.inference_mode,
            created_at=db_request.created_at,
            completed_at=db_request.completed_at,
        )

    def _get_request(self, request_id: int) -> MLRequest:
        db_request = ml_crud.get_request_by_id(self.session, request_id)
        if not db_request:
            raise MLRequestNotFoundException
        return db_request

    def _file_url(self, file_path: Optional[str]) -> Optional[str]:
        if not file_path:
            return None
        relative = Path(file_path).relative_to(settings.glioma.OUTPUT_DIR)
        return f"/outputs/{'/'.join(quote(part) for part in relative.parts)}"

    def get_result(self, request_id: int) -> SGliomaResultResponse:
        db_request = self._get_request(request_id)
        volumes = None
        if db_request.prediction and isinstance(db_request.prediction, dict):
            raw_volumes = db_request.prediction.get("volumes_ml")
            if raw_volumes:
                volumes = VolumesML(**raw_volumes)

        return SGliomaResultResponse(
            request_id=db_request.id,
            status=db_request.status,
            case_id=db_request.prediction.get("case_id") if isinstance(db_request.prediction, dict) else None,
            inference_mode=db_request.inference_mode,
            volumes_ml=volumes,
            prediction_url=self._file_url(db_request.prediction_path),
            uncertainty_url=self._file_url(db_request.uncertainty_path),
            report_url=self._file_url(db_request.report_path),
            visualization_urls=[self._file_url(p) for p in (db_request.visualization_paths or [])],
            created_at=db_request.created_at,
            completed_at=db_request.completed_at,
            errors=db_request.errors,
        )

    async def process_result(self, result: MLResult) -> None:
        request_id = result.request_id
        db_request = ml_crud.get_request_by_id(self.session, request_id)
        if not db_request:
            logger.error("Request %s not found for result processing", request_id)
            raise MLRequestNotFoundException

        if db_request.status != MLRequestStatus.pending:
            logger.warning("Request %s already processed with status %s", request_id, db_request.status)
            return

        if result.status == "success":
            db_request.status = MLRequestStatus.success
            db_request.prediction = {
                "case_id": result.case_id,
                "volumes_ml": result.volumes_ml,
            }
            db_request.prediction_path = result.prediction_path
            db_request.uncertainty_path = result.uncertainty_path
            db_request.report_path = result.report_path
            db_request.visualization_paths = result.visualization_paths
            db_request.message = "Segmentation completed successfully"
        else:
            db_request.status = MLRequestStatus.fail
            db_request.errors = [{"error": result.error}] if result.error else None
            db_request.message = result.error or "Inference failed"

        db_request.completed_at = datetime.now(timezone.utc)
        logger.info("Processed result for request %s with status %s", request_id, db_request.status)

    def get_history(self) -> list[MLRequest]:
        return ml_crud.get_history(self.session)
