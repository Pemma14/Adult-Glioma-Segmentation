import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.api_key import verify_api_key
from app.config import settings
from app.models.ml_request_model import MLRequestStatus
from app.schemas.glioma_schemas import (
    SGliomaResultResponse,
    SGliomaStatusResponse,
    SGliomaUploadResponse,
)
from app.services.glioma_service import GliomaRequestService
from app.services.mq_publisher import MLTaskPublisher, get_mq_service
from app.database.database import get_session

router = APIRouter()

logger = logging.getLogger(__name__)


def get_glioma_request_service(session=Depends(get_session)):
    return GliomaRequestService(session)


@router.post(
    "/upload",
    response_model=SGliomaUploadResponse,
    summary="Upload a NIfTI scan for segmentation",
    description="Uploads a .nii/.nii.gz file and enqueues it for glioma segmentation.",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_scan(
    file: UploadFile = File(..., description="NIfTI scan (.nii or .nii.gz)"),
    save_uncertainty: bool = Form(True, description="Generate uncertainty map"),
    mq_service: MLTaskPublisher = Depends(get_mq_service),
    service: GliomaRequestService = Depends(get_glioma_request_service),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    db_request = await service.create_segmentation_request(
        file=file,
        mq_service=mq_service,
        save_uncertainty=save_uncertainty,
    )
    return {
        "request_id": db_request.id,
        "status": db_request.status,
        "message": db_request.message,
    }


@router.post(
    "/from-dicom",
    response_model=SGliomaUploadResponse,
    summary="Upload a DICOM series (zip) for segmentation",
    description="Accepts a zip archive of a DICOM series, converts to NIfTI, and enqueues for segmentation.",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_dicom(
    file: UploadFile = File(..., description="ZIP archive containing DICOM series"),
    save_uncertainty: bool = Form(True, description="Generate uncertainty map"),
    mq_service: MLTaskPublisher = Depends(get_mq_service),
    service: GliomaRequestService = Depends(get_glioma_request_service),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    from app.dicom.reader import extract_zip_and_find_modalities, combine_modalities_to_multichannel, dicom_series_to_nifti_file

    temp_dir = settings.dicom.TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)

    zip_path = temp_dir / (file.filename or "dicom.zip")
    with zip_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        modalities = extract_zip_and_find_modalities(zip_path, temp_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not modalities:
        raise HTTPException(status_code=400, detail="No DICOM series found in the archive")

    nifti_path = temp_dir / "converted.nii.gz"

    if len(modalities) == 1 and not any(m in modalities for m in ["T1", "T1ce", "T2", "FLAIR"]):
        single_dir = next(iter(modalities.values()))
        dicom_series_to_nifti_file(single_dir, nifti_path)
    else:
        volumes = {}
        for mod in ["T1", "T1ce", "T2", "FLAIR"]:
            if mod in modalities:
                mod_nifti = temp_dir / f"{mod}.nii.gz"
                dicom_series_to_nifti_file(modalities[mod], mod_nifti)
                import nibabel as nib
                volumes[mod] = nib.load(str(mod_nifti)).get_fdata()
        if volumes:
            combine_modalities_to_multichannel(volumes, nifti_path)

    db_request = await service.create_segmentation_request(
        file=UploadFile(filename=nifti_path.name, file=nifti_path.open("rb")),
        mq_service=mq_service,
        save_uncertainty=save_uncertainty,
    )

    ref_dir = Path(db_request.input_file_path).parent
    ref_dir.mkdir(parents=True, exist_ok=True)
    for mod_dir in modalities.values():
        dcm_files = sorted(Path(mod_dir).rglob("*.dcm"))
        if dcm_files:
            import shutil as sh
            sh.copy2(str(dcm_files[0]), str(ref_dir / "reference.dcm"))
            break

    shutil.rmtree(temp_dir)
    return {
        "request_id": db_request.id,
        "status": db_request.status,
        "message": db_request.message,
    }


@router.get(
    "/status/{request_id}",
    response_model=SGliomaStatusResponse,
    summary="Get segmentation request status",
)
async def get_status(
    request_id: int,
    service: GliomaRequestService = Depends(get_glioma_request_service),
    _: bool = Depends(verify_api_key),
) -> SGliomaStatusResponse:
    return service.get_status(request_id)


@router.get(
    "/result/{request_id}",
    response_model=SGliomaResultResponse,
    summary="Get segmentation result",
)
async def get_result(
    request_id: int,
    service: GliomaRequestService = Depends(get_glioma_request_service),
    _: bool = Depends(verify_api_key),
) -> SGliomaResultResponse:
    return service.get_result(request_id)


@router.get(
    "/history",
    summary="Segmentation request history",
)
async def get_history(
    service: GliomaRequestService = Depends(get_glioma_request_service),
    _: bool = Depends(verify_api_key),
) -> List[Dict[str, Any]]:
    requests = service.get_history()
    return [
        {
            "id": r.id,
            "status": r.status,
            "case_id": r.prediction.get("case_id") if isinstance(r.prediction, dict) else None,
            "volumes_ml": r.prediction.get("volumes_ml") if isinstance(r.prediction, dict) else None,
            "created_at": r.created_at.isoformat(),
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in requests
    ]


@router.get(
    "/download/{request_id}/{file_type}",
    summary="Download a segmentation output file",
)
async def download_file(
    request_id: int,
    file_type: str,
    service: GliomaRequestService = Depends(get_glioma_request_service),
    _: bool = Depends(verify_api_key),
):
    db_request = service._get_request(request_id)
    if db_request.status != MLRequestStatus.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request is not completed yet",
        )

    path_map = {
        "mask": db_request.prediction_path,
        "uncertainty": db_request.uncertainty_path,
        "report": db_request.report_path,
    }

    if file_type in path_map:
        file_path = path_map[file_type]
    elif file_type.startswith("vis-"):
        idx = int(file_type.replace("vis-", ""))
        vis_paths = db_request.visualization_paths or []
        if idx < 0 or idx >= len(vis_paths):
            raise HTTPException(status_code=404, detail="Visualization not found")
        file_path = vis_paths[idx]
    else:
        raise HTTPException(status_code=404, detail="Unknown file type")

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        media_type="application/octet-stream",
    )


@router.get(
    "/{request_id}/rtstruct",
    summary="Download segmentation as DICOM RTSTRUCT",
)
async def download_rtstruct(
    request_id: int,
    service: GliomaRequestService = Depends(get_glioma_request_service),
    _: bool = Depends(verify_api_key),
):
    db_request = service._get_request(request_id)
    if db_request.status != MLRequestStatus.success:
        raise HTTPException(status_code=400, detail="Request is not completed yet")
    if not db_request.prediction_path:
        raise HTTPException(status_code=404, detail="Prediction mask not found")

    from app.dicom.writer import prediction_to_rtstruct

    ref_dir = Path(db_request.input_file_path).parent
    if not any(ref_dir.rglob("*.dcm")):
        raise HTTPException(status_code=400, detail="Reference DICOM series not found for RTSTRUCT generation")

    output_path = Path(db_request.output_dir) / "rtstruct.dcm"
    result = prediction_to_rtstruct(
        prediction_path=Path(db_request.prediction_path),
        reference_dicom_dir=ref_dir,
        output_path=output_path,
    )

    return FileResponse(
        path=result,
        filename=f"segmentation_{request_id}_rtstruct.dcm",
        media_type="application/dicom",
    )
