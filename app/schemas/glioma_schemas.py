from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.ml_request_model import InferenceMode, MLRequestStatus


class InferenceModeInput(str, Enum):
    ensemble = "ensemble"


class SGliomaUploadResponse(BaseModel):
    request_id: int = Field(..., description="ID of the created segmentation request")
    status: MLRequestStatus = Field(..., description="Current request status")
    message: str = Field(..., description="Human-readable status message")


class SGliomaStatusResponse(BaseModel):
    request_id: int = Field(..., description="ID of the segmentation request")
    status: MLRequestStatus = Field(..., description="Current request status")
    message: Optional[str] = Field(None, description="Status or error message")
    inference_mode: Optional[InferenceMode] = Field(None, description="Inference mode")
    created_at: datetime = Field(..., description="Request creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class VolumesML(BaseModel):
    wt: float = Field(..., description="Whole tumor volume in milliliters")
    tc: float = Field(..., description="Tumor core volume in milliliters")
    et: float = Field(..., description="Enhancing tumor volume in milliliters")


class SGliomaResultResponse(BaseModel):
    request_id: int = Field(..., description="ID of the segmentation request")
    status: MLRequestStatus = Field(..., description="Request status")
    case_id: Optional[str] = Field(None, description="Patient/case identifier")
    inference_mode: Optional[InferenceMode] = Field(None, description="Inference mode")
    volumes_ml: Optional[VolumesML] = Field(None, description="Computed region volumes")
    prediction_url: Optional[str] = Field(None, description="URL to download the segmentation mask")
    uncertainty_url: Optional[str] = Field(None, description="URL to download the uncertainty map")
    report_url: Optional[str] = Field(None, description="URL to download the JSON report")
    input_url: Optional[str] = Field(None, description="URL to download the original input scan")
    visualization_urls: Optional[List[str]] = Field(None, description="URLs of PNG visualizations")
    created_at: datetime = Field(..., description="Request creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details if failed")
