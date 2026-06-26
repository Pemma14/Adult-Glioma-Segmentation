"""Lightweight Colab-ready FastAPI app for synchronous GPU inference.

This module runs the glioma segmentation service as a single process without
Postgres, RabbitMQ, or background workers. It is intended for Google Colab GPU
demos and local development only.
"""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import uvicorn
import asyncio

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas.glioma_schemas import SGliomaResultResponse, SGliomaStatusResponse, VolumesML
from src.glioma.inference import predict
from src.glioma.model import load_ensemble_for_inference
from src.glioma.settings import load_model_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_ID = 1
_colab_state: dict[str, Any] = {
    "status": "pending",
    "message": "No request yet",
    "result": None,
    "input_url": None,
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the ensemble once at startup so inference is fast."""
    config = load_model_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading ensemble on %s", device)
    models = load_ensemble_for_inference(device, config)
    application.state.config = config
    application.state.device = device
    application.state.models = models
    logger.info("Ready for inference")
    yield
    # Cleanup GPU memory when the app shuts down.
    del application.state.models
    if device.type == "cuda":
        torch.cuda.empty_cache()


def create_application() -> FastAPI:
    application = FastAPI(
        title="Glioma Segmentation — Colab Demo",
        description="Single-process GPU demo for adult glioma segmentation.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Allow the UI served from any origin (useful when exposing via ngrok).
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.mount(
        "/outputs",
        StaticFiles(directory=str(OUTPUT_DIR), check_dir=False),
        name="outputs",
    )

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/viewer",
        StaticFiles(directory=str(static_dir), html=True),
        name="static",
    )

    @application.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    def _run_inference(input_path: Path, request_dir: Path) -> None:
        """Run inference synchronously in a background thread."""
        _colab_state["message"] = "Running inference..."
        try:
            result = predict(
                image_path=input_path,
                output_dir=request_dir,
                device=application.state.device,
                config=application.state.config,
                models=application.state.models,
                save_uncertainty=True,
                save_visualization=True,
                n_slices=3,
            )
            _colab_state["result"] = result
            _colab_state["status"] = "success"
            _colab_state["message"] = "Segmentation completed successfully"
            logger.info("Inference completed for colab request")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed")
            _colab_state["status"] = "fail"
            _colab_state["message"] = str(exc)

    @application.post("/api/v1/segmentation/upload")
    async def upload_scan(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        global _colab_state  # noqa: PLW0603
        _colab_state = {
            "status": "pending",
            "message": "Uploading...",
            "result": None,
            "input_url": None,
        }

        request_dir = OUTPUT_DIR / "colab_request"
        request_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file.filename or "input.nii.gz").name
        input_path = UPLOAD_DIR / f"request_{REQUEST_ID}" / safe_name
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Copy input into the output directory so the viewer can serve it.
        output_input_path = request_dir / safe_name
        shutil.copy2(str(input_path), str(output_input_path))
        _colab_state["input_url"] = f"/outputs/colab_request/{safe_name}"

        _colab_state["message"] = "Running inference..."
        background_tasks.add_task(_run_inference, input_path, request_dir)

        return {
            "request_id": REQUEST_ID,
            "status": _colab_state["status"],
            "message": _colab_state["message"],
        }

    @application.get("/api/v1/segmentation/status/{request_id}")
    async def get_status(request_id: int) -> SGliomaStatusResponse:
        return SGliomaStatusResponse(
            request_id=REQUEST_ID,
            status=_colab_state["status"],
            message=_colab_state["message"],
            created_at=datetime.now(timezone.utc),
        )

    def _relative_url(file_path: str | Path | None) -> str | None:
        if not file_path:
            return None
        rel = Path(file_path).relative_to(OUTPUT_DIR)
        return f"/outputs/{rel.as_posix()}"

    @application.get("/api/v1/segmentation/result/{request_id}")
    async def get_result(request_id: int) -> SGliomaResultResponse:
        result = _colab_state.get("result") or {}
        request_dir = OUTPUT_DIR / "colab_request"

        volumes = None
        raw_volumes = result.get("volumes_ml")
        if raw_volumes:
            volumes = VolumesML(**raw_volumes)

        # Generate a JSON report if it does not exist yet.
        report_path = request_dir / "report.json"
        if not report_path.exists():
            report = {
                "case_id": result.get("case_id"),
                "volumes_ml": raw_volumes,
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        visualization_urls = [
            _relative_url(p) for p in result.get("visualization_paths", []) if p
        ]

        return SGliomaResultResponse(
            request_id=REQUEST_ID,
            status=_colab_state["status"],
            case_id=result.get("case_id"),
            volumes_ml=volumes,
            prediction_url=_relative_url(result.get("prediction_path")),
            uncertainty_url=_relative_url(result.get("uncertainty_path")),
            report_url=_relative_url(report_path),
            input_url=_colab_state.get("input_url"),
            visualization_urls=visualization_urls,
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc)
            if _colab_state["status"] == "success"
            else None,
            errors=None,
        )

    @application.get("/api/v1/segmentation/download/{request_id}/{file_type}")
    async def download_file(request_id: int, file_type: str) -> FileResponse:
        if _colab_state["status"] != "success":
            raise HTTPException(
                status_code=400, detail="Request is not completed yet"
            )

        result = _colab_state["result"] or {}
        request_dir = OUTPUT_DIR / "colab_request"

        path_map: dict[str, str | None] = {
            "mask": result.get("prediction_path"),
            "uncertainty": result.get("uncertainty_path"),
            "report": str(request_dir / "report.json"),
        }

        if file_type.startswith("vis-"):
            idx = int(file_type.replace("vis-", ""))
            vis_paths = result.get("visualization_paths", []) or []
            if idx < 0 or idx >= len(vis_paths):
                raise HTTPException(status_code=404, detail="Visualization not found")
            path_map[file_type] = vis_paths[idx]

        file_path = path_map.get(file_type)
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=file_path,
            filename=Path(file_path).name,
            media_type="application/octet-stream",
        )

    return application


app = create_application()

if __name__ == "__main__":
    import os

    port = int(os.getenv("COLAB_APP_PORT", "8500"))
    uvicorn.run("app.main_colab:app", host="0.0.0.0", port=port)
