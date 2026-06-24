from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class MLTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: int = Field(..., description="ID of the MLRequest in the API database")
    input_file_path: str = Field(..., description="Path to the uploaded NIfTI file")
    output_dir: str = Field(..., description="Directory where results should be saved")
    inference_mode: str = Field(default="ensemble", description="ensemble (5-fold)")
    save_uncertainty: bool = Field(default=True, description="Generate uncertainty map")
    model: str = Field(default="glioma_segmentation")
    timestamp: datetime = Field(default_factory=datetime.now)


class MLResult(BaseModel):
    task_id: str
    request_id: int
    status: str
    case_id: Optional[str] = None
    prediction_path: Optional[str] = None
    uncertainty_path: Optional[str] = None
    report_path: Optional[str] = None
    visualization_paths: Optional[List[str]] = None
    volumes_ml: Optional[Dict[str, float]] = None
    error: Optional[str] = None
