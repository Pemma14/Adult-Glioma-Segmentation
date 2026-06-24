from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROI_NAMES = {
    1: "Whole Tumor (WT)",
    2: "Tumor Core (TC)",
    3: "Enhancing Tumor (ET)",
}

ROI_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}


def _load_reference_dataset(dicom_dir: Path):
    import pydicom
    dicom_files = sorted(dicom_dir.rglob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    ds = pydicom.dcmread(str(dicom_files[0]))
    return ds, dicom_files


def _get_image_orientation(dicom_files: List[Path]):
    import pydicom
    ds = pydicom.dcmread(str(dicom_files[0]))
    iop = ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    ipp = ds.get("ImagePositionPatient", [0, 0, 0])
    spacing = ds.get("PixelSpacing", [1.0, 1.0])
    return iop, ipp, spacing


def _contour_from_mask_slice(
    mask_slice: np.ndarray,
    iop: List[float],
    ipp: List[float],
    spacing: Tuple[float, float],
    slice_z: float,
) -> List[float]:
    from skimage import measure
    contours = measure.find_contours(mask_slice, level=0.5)
    all_points = []
    for contour in contours:
        for point in contour:
            y, x = point
            px = ipp[0] + x * spacing[0] * iop[0] + y * spacing[1] * iop[1]
            py = ipp[1] + x * spacing[0] * iop[3] + y * spacing[1] * iop[4]
            pz = float(ipp[2]) + slice_z
            all_points.extend([round(px, 2), round(py, 2), round(pz, 2)])
    return all_points


def _contour_from_mask_slice_simple(
    mask_slice: np.ndarray,
    slice_idx: int,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
) -> List[float]:
    from skimage import measure
    contours = measure.find_contours(mask_slice, level=0.5)
    all_points = []
    for contour in contours:
        for point in contour:
            y, x = point
            px = origin[0] + x * spacing[0]
            py = origin[1] + y * spacing[1]
            pz = origin[2] + slice_idx * spacing[2]
            all_points.extend([round(px, 2), round(py, 2), round(pz, 2)])
    return all_points


def _get_contour_data(
    roi_mask: np.ndarray,
    reference_nifti: Path,
) -> Tuple[List[List[float]], List[int]]:
    import nibabel as nib
    nii = nib.load(str(reference_nifti))
    affine = nii.affine
    spacing = tuple(abs(affine[i, i]) for i in range(3))
    origin = tuple(float(affine[i, 3]) for i in range(3))

    contour_data = []
    num_contours = []

    for slice_idx in range(roi_mask.shape[-1]):
        mask_slice = roi_mask[..., slice_idx]
        if mask_slice.sum() == 0:
            continue

        points = _contour_from_mask_slice_simple(
            mask_slice, slice_idx, spacing, origin
        )
        if points:
            contour_data.append(points)
            num_contours.append(len(points) // 3)

    return contour_data, num_contours


def prediction_to_rtstruct(
    prediction_path: Path,
    reference_dicom_dir: Path,
    output_path: Path,
    patient_info: Optional[Dict[str, str]] = None,
) -> Path:
    import nibabel as nib
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import generate_uid

    logger.info("Generating RTSTRUCT from %s", prediction_path)

    prediction_nii = nib.load(str(prediction_path))
    prediction_data = prediction_nii.get_fdata()

    ref_ds, ref_files = _load_reference_dataset(reference_dicom_dir)
    patient_name = patient_info.get("name", "") if patient_info else ""
    patient_id = patient_info.get("id", "") if patient_info else ""
    study_uid = str(ref_ds.get("StudyInstanceUID", generate_uid()))

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"

    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 500
    ds.Modality = "RTSTRUCT"
    ds.StructureSetLabel = "AI Segmentation"
    ds.StructureSetName = "Glioma AI Segmentation"
    ds.StructureSetDate = ref_ds.get("StudyDate", "")
    ds.StructureSetTime = ref_ds.get("StudyTime", "")

    ds.ReferencedFrameOfReferenceSequence = Dataset()
    frame_of_ref = Dataset()
    frame_of_ref.FrameOfReferenceUID = ref_ds.get(
        "FrameOfReferenceUID", generate_uid()
    )
    ds.ReferencedFrameOfReferenceSequence = [frame_of_ref]

    structure_set_roi_sequence = []
    roi_contour_sequence = []
    rt_roi_observations_sequence = []

    for roi_number in [1, 2, 3]:
        mask = (prediction_data == roi_number).astype(np.uint8)
        if mask.sum() == 0:
            continue

        roi_name = ROI_NAMES[roi_number]
        color = ROI_COLORS[roi_number]

        contour_data, num_contours = _get_contour_data(mask, prediction_path)

        if not contour_data:
            logger.warning("No contours found for %s", roi_name)
            continue

        roi_obs = Dataset()
        roi_obs.ObservationNumber = roi_number
        roi_obs.ReferencedROINumber = roi_number
        roi_obs.ROIObservationLabel = roi_name
        roi_obs.RTROIInterpretedType = "ORGAN"
        roi_obs.ROIInterpreter = patient_name
        rt_roi_observations_sequence.append(roi_obs)

        roi = Dataset()
        roi.ROINumber = roi_number
        roi.ReferencedFrameOfReferenceUID = frame_of_ref.FrameOfReferenceUID
        roi.ROIName = roi_name
        roi.ROIDisplayColor = list(color)
        structure_set_roi_sequence.append(roi)

        contour_seq = []
        for i, (points, n_pts) in enumerate(zip(contour_data, num_contours)):
            c = Dataset()
            c.ContourImageSequence = [Dataset()]
            c.ContourImageSequence[0].ReferencedSOPClassUID = ref_ds.SOPClassUID
            c.ContourImageSequence[0].ReferencedSOPInstanceUID = (
                ref_ds.SOPInstanceUID
            )
            c.ContourGeometricType = "CLOSED_PLANAR"
            c.NumberOfContourPoints = n_pts
            c.ContourData = points
            contour_seq.append(c)

        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = list(color)
        roi_contour.ContourSequence = contour_seq
        roi_contour.ReferencedROINumber = roi_number
        roi_contour_sequence.append(roi_contour)

    ds.StructureSetROISequence = structure_set_roi_sequence
    ds.ROIContourSequence = roi_contour_sequence
    ds.RTROIObservationsSequence = rt_roi_observations_sequence

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = ref_ds.SeriesInstanceUID
    ref_series.SeriesNumber = ref_ds.SeriesNumber
    ref_series.ContourImageSequence = []
    for f in ref_files:
        ref_sop = Dataset()
        ref_sop.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ref_sop.ReferencedSOPInstanceUID = pydicom.dcmread(
            str(f), stop_before_pixels=True
        ).SOPInstanceUID
        ref_series.ContourImageSequence.append(ref_sop)
    ds.ReferencedStructureSetSequence = [ref_series]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))
    logger.info("Saved RTSTRUCT to %s", output_path)
    return output_path



