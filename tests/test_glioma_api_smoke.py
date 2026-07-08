import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ["GLIOMA__MODEL_VERSION"] = "v1.0.0"
os.environ["AUTH__API_KEY"] = "test-api-key"

import io
import zipfile
import json
import numpy as np
import nibabel as nib
from datetime import datetime, timezone
from pathlib import Path

API_KEY = "test-api-key"
HEADERS = {"X-API-Key": API_KEY}


def _make_nifti_bytes(shape=(32, 32, 32), is_pred=False):
    import tempfile
    if is_pred:
        data = np.zeros(shape, dtype=np.uint8)
        data[10:20, 10:20, 10:20] = 1
        data[12:18, 12:18, 12:18] = 2
        data[14:16, 14:16, 14:16] = 3
    else:
        data = np.random.randn(*shape).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        tmp = f.name
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), tmp)
    with open(tmp, "rb") as f:
        buf = io.BytesIO(f.read())
    os.unlink(tmp)
    buf.seek(0)
    return buf


class TestAuth:
    def test_upload_without_api_key(self, client):
        resp = client.post("/api/v1/segmentation/upload")
        assert resp.status_code == 422

    def test_upload_with_wrong_key(self, client):
        resp = client.post(
            "/api/v1/segmentation/upload",
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 401


class TestUpload:
    def test_upload_nifti_and_status(self, client):
        buf = _make_nifti_bytes()
        resp = client.post(
            "/api/v1/segmentation/upload",
            files={"file": ("scan.nii.gz", buf, "application/gzip")},
            headers=HEADERS,
        )
        assert resp.status_code == 202, resp.text
        data = resp.json()
        assert data["status"] == "pending"
        assert data["request_id"] > 0

        status = client.get(
            f"/api/v1/segmentation/status/{data['request_id']}",
            headers=HEADERS,
        )
        assert status.status_code == 200
        assert status.json()["request_id"] == data["request_id"]

    def test_upload_invalid_extension(self, client):
        buf = io.BytesIO(b"not a nifti")
        resp = client.post(
            "/api/v1/segmentation/upload",
            files={"file": ("scan.txt", buf, "text/plain")},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_upload_from_dicom_bad_zip(self, client):
        buf = io.BytesIO(b"not a zip file")
        resp = client.post(
            "/api/v1/segmentation/from-dicom",
            files={"file": ("dicom.zip", buf, "application/zip")},
            headers=HEADERS,
        )
        assert resp.status_code == 400

    def test_upload_from_dicom_empty_zip(self, client):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("empty.txt", "not a dicom")
        zip_buf.seek(0)
        resp = client.post(
            "/api/v1/segmentation/from-dicom",
            files={"file": ("dicom.zip", zip_buf, "application/zip")},
            headers=HEADERS,
        )
        assert resp.status_code in (400, 202)


class TestStatus:
    def test_status_not_found(self, client):
        resp = client.get("/api/v1/segmentation/status/99999", headers=HEADERS)
        assert resp.status_code == 404

    def test_result_not_found(self, client):
        resp = client.get("/api/v1/segmentation/result/99999", headers=HEADERS)
        assert resp.status_code == 404

    def test_download_not_found(self, client):
        resp = client.get("/api/v1/segmentation/download/99999/mask", headers=HEADERS)
        assert resp.status_code == 404


class TestFullFlow:
    def test_full_flow_with_mocked_result(self, client, session):
        buf = _make_nifti_bytes()
        resp = client.post(
            "/api/v1/segmentation/upload",
            files={"file": ("scan.nii.gz", buf, "application/gzip")},
            headers=HEADERS,
        )
        assert resp.status_code == 202
        request_id = resp.json()["request_id"]

        from app.models import MLRequest, MLRequestStatus
        db_req = session.get(MLRequest, request_id)
        assert db_req is not None

        output_dir = Path(db_req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_buf = _make_nifti_bytes(is_pred=True)
        pred_path = output_dir / "pred_mask.nii.gz"
        pred_path.write_bytes(pred_buf.read())

        db_req.status = MLRequestStatus.success
        db_req.prediction_path = str(pred_path)
        db_req.prediction = {
            "case_id": "test_case",
            "volumes_ml": {"wt": 45.2, "tc": 23.8, "et": 12.1},
        }
        db_req.completed_at = datetime.now(timezone.utc)
        session.flush()

        status = client.get(f"/api/v1/segmentation/status/{request_id}", headers=HEADERS)
        assert status.status_code == 200
        assert status.json()["status"] == MLRequestStatus.success.value

        result = client.get(f"/api/v1/segmentation/result/{request_id}", headers=HEADERS)
        assert result.status_code == 200
        rdata = result.json()
        assert rdata["volumes_ml"]["wt"] == 45.2
        assert rdata["volumes_ml"]["tc"] == 23.8
        assert rdata["volumes_ml"]["et"] == 12.1
        assert rdata["case_id"] == "test_case"
        assert rdata["prediction_url"] is not None

        dl = client.get(f"/api/v1/segmentation/download/{request_id}/mask", headers=HEADERS)
        assert dl.status_code == 200

        history = client.get("/api/v1/segmentation/history", headers=HEADERS)
        assert history.status_code == 200
        assert any(r["id"] == request_id for r in history.json())

    def test_rtstruct_without_reference(self, client, session):
        from app.models import MLRequest, MLRequestStatus, MLModel
        from app.crud import ml as ml_crud

        model = ml_crud.get_active_model(session)
        assert model is not None

        req = MLRequest(
            model_id=model.id,
            input_data={"test": True},
            status=MLRequestStatus.pending,
        )
        session.add(req)
        session.flush()
        req_id = req.id

        output_dir = Path("/tmp/test_glioma_rtstruct") / str(req_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / "pred.nii.gz"
        pred_path.write_bytes(_make_nifti_bytes(is_pred=True).read())

        req.status = MLRequestStatus.success
        req.prediction_path = str(pred_path)
        req.input_file_path = str(pred_path)
        req.output_dir = str(output_dir)
        req.completed_at = datetime.now(timezone.utc)
        session.flush()

        resp = client.get(f"/api/v1/segmentation/{req_id}/rtstruct", headers=HEADERS)
        assert resp.status_code == 400, resp.text
