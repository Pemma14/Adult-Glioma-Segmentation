from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]


def _find_series_by_description(dicom_dir: Path, description_keyword: str) -> Optional[Path]:
    import SimpleITK as sitk
    series_reader = sitk.ImageSeriesReader()
    series_ids = series_reader.GetGDCMSeriesIDs(str(dicom_dir))
    keyword_upper = description_keyword.upper()

    for series_id in series_ids:
        file_names = series_reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        if not file_names:
            continue
        header = sitk.ReadImage(file_names[0])
        desc = header.GetMetaData("0008|103e").upper() if header.HasMetaDataKey("0008|103e") else ""
        words = desc.replace(",", " ").replace("_", " ").replace("-", " ").split()
        if any(keyword_upper == w or keyword_upper == w.rstrip("CE") for w in words):
            return Path(file_names[0]).parent
    return None


def _find_series_by_modality(dicom_dir: Path, modality: str) -> Optional[List[Path]]:
    import SimpleITK as sitk
    series_reader = sitk.ImageSeriesReader()
    series_ids = series_reader.GetGDCMSeriesIDs(str(dicom_dir))

    for series_id in series_ids:
        file_names = series_reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        if not file_names:
            continue
        header = sitk.ReadImage(file_names[0])
        mod = header.GetMetaData("0008|0060") if header.HasMetaDataKey("0008|0060") else ""
        if mod.upper() == modality.upper():
            return [Path(f) for f in file_names]
    return None


def read_dicom_series_as_nifti(dicom_dir: Path) -> np.ndarray:
    import SimpleITK as sitk
    vol = sitk.ReadImage(str(dicom_dir))
    return sitk.GetArrayFromImage(vol)


def dicom_series_to_nifti_file(dicom_dir: Path, output_path: Path) -> Path:
    import SimpleITK as sitk
    vol = sitk.ReadImage(str(dicom_dir))
    sitk.WriteImage(vol, str(output_path))
    logger.info("Converted DICOM series to %s", output_path)
    return output_path


def extract_zip_and_find_modalities(zip_path: Path, temp_dir: Path) -> dict:
    extract_dir = temp_dir / zip_path.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(extract_dir))
    except zipfile.BadZipFile:
        raise ValueError("File is not a valid ZIP archive")

    modalities: dict[str, Path] = {}
    for mod in MODALITIES:
        series_dir = _find_series_by_description(extract_dir, mod)
        if series_dir:
            modalities[mod] = series_dir
            logger.info("Found %s series in %s", mod, series_dir)

    if not modalities:
        series_ids = _find_series_by_modality(extract_dir, "MR")
        if series_ids:
            modalities["MR"] = extract_dir
            logger.info("Found MR series (modality-based fallback)")

    return modalities


def combine_modalities_to_multichannel(
    modality_volumes: dict[str, np.ndarray],
    output_path: Path,
    target_shape: Optional[tuple] = None,
) -> Path:
    import SimpleITK as sitk

    ordered = [modality_volumes[mod] for mod in MODALITIES if mod in modality_volumes]
    if not ordered:
        raise ValueError("No modality volumes to combine")

    stack = np.stack(ordered, axis=-1)
    if target_shape is not None:
        import torch
        from monai.transforms import ResizeWithPadOrCrop
        t = torch.from_numpy(stack).permute(3, 0, 1, 2).unsqueeze(0)
        t = ResizeWithPadOrCrop(target_shape, mode="constant")(t)
        stack = t.squeeze(0).permute(1, 2, 3, 0).numpy()

    img = sitk.GetImageFromArray(stack, isVector=True)
    sitk.WriteImage(img, str(output_path))
    logger.info("Saved multichannel NIfTI to %s (shape %s)", output_path, stack.shape)
    return output_path
