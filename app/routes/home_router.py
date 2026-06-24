from fastapi import APIRouter
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def home_page() -> Dict[str, Any]:
    return {
        "name": "Glioma Segmentation Service",
        "version": "1.0.0",
        "description": "On-premise automated 3D segmentation of adult-type diffuse gliomas",
        "endpoints": {
            "upload": "POST /api/v1/segmentation/upload",
            "from_dicom": "POST /api/v1/segmentation/from-dicom",
            "status": "GET /api/v1/segmentation/status/{id}",
            "result": "GET /api/v1/segmentation/result/{id}",
            "history": "GET /api/v1/segmentation/history",
            "download": "GET /api/v1/segmentation/download/{id}/{type}",
            "rtstruct": "GET /api/v1/segmentation/{id}/rtstruct",
        }
    }


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    return {"status": "ok"}
