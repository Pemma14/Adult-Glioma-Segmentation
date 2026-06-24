"""Unified entrypoint for inference result analysis.

Provides subcommands for worst-case analysis, per-case reports,
uncertainty/error correlation, visualization of problematic cases,
and selection of anomalous cases.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Project root is the directory that contains src/pyproject.toml
EXPERIMENTS_DIR = Path(__file__).resolve()
while EXPERIMENTS_DIR != EXPERIMENTS_DIR.parent and not (EXPERIMENTS_DIR / "pyproject.toml").exists():
    EXPERIMENTS_DIR = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.stats import pointbiserialr  # noqa: E402

from src.scripts.utils.analysis import (  # noqa: E402
    load_metrics,
    load_nifti,
    multiclass_to_region,
    resolve_path,
    select_cases,
)

logger = logging.getLogger(__name__)

FP_COLOR = np.array([0.2, 0.4, 1.0, 1.0])  # blue
FN_COLOR = np.array([1.0, 0.9, 0.2, 1.0])  # yellow
CMAP_FP = ListedColormap([FP_COLOR])
CMAP_FN = ListedColormap([FN_COLOR])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Worst cases
# --------------------------------------------------------------------------- #
def run_worst_cases(metrics_source: str | Path, output_dir: Path) -> None:
    """Analyze inference metrics and produce a list of worst-performing cases."""
    df = load_metrics(metrics_source)
    output_dir = _ensure_output_dir(output_dir)

    worst_dice = df.nsmallest(20, "mean_dice").copy()
    worst_hd95 = df.nlargest(20, "mean_hd95").copy()

    worst_wt_dice = df.nsmallest(10, "dice_wt").copy()
    worst_tc_dice = df.nsmallest(10, "dice_tc").copy()
    worst_et_dice = df.nsmallest(10, "dice_et").copy()

    worst_wt_hd95 = df.nlargest(10, "hd95_wt").copy()
    worst_tc_hd95 = df.nlargest(10, "hd95_tc").copy()
    worst_et_hd95 = df.nlargest(10, "hd95_et").copy()

    dice_min, dice_max = df["mean_dice"].min(), df["mean_dice"].max()
    hd95_min, hd95_max = df["mean_hd95"].min(), df["mean_hd95"].max()

    df["dice_norm"] = 1 - (df["mean_dice"] - dice_min) / (dice_max - dice_min)
    df["hd95_norm"] = (df["mean_hd95"] - hd95_min) / (hd95_max - hd95_min)
    df["badness_score"] = 0.5 * df["dice_norm"] + 0.5 * df["hd95_norm"]

    worst_overall = df.nlargest(20, "badness_score").copy()

    worst_overall[
        [
            "case_id",
            "mean_dice",
            "dice_wt",
            "dice_tc",
            "dice_et",
            "mean_hd95",
            "hd95_wt",
            "hd95_tc",
            "hd95_et",
            "badness_score",
        ]
    ].to_csv(output_dir / "worst_cases_overall.csv", index=False)

    md = ["# Worst-performing inference cases\n"]
    md.append(
        f"Dataset: {len(df)} cases from ensemble inference.\n"
        f"Mean Dice range: `{dice_min:.4f} – {dice_max:.4f}`, "
        f"Mean HD95 range: `{hd95_min:.2f} – {hd95_max:.2f}`.\n"
    )

    md.append("## Overall worst cases (combined low Dice + high HD95)\n")
    md.append(
        _df_to_md(
            worst_overall[
                [
                    "case_id",
                    "mean_dice",
                    "mean_hd95",
                    "dice_wt",
                    "dice_tc",
                    "dice_et",
                    "hd95_wt",
                    "hd95_tc",
                    "hd95_et",
                    "badness_score",
                ]
            ]
        )
    )

    md.append("## Lowest mean Dice (top 20)\n")
    md.append(
        _df_to_md(
            worst_dice[
                [
                    "case_id",
                    "mean_dice",
                    "dice_wt",
                    "dice_tc",
                    "dice_et",
                    "mean_hd95",
                    "hd95_wt",
                    "hd95_tc",
                    "hd95_et",
                ]
            ]
        )
    )

    md.append("## Highest mean HD95 (top 20)\n")
    md.append(
        _df_to_md(
            worst_hd95[
                [
                    "case_id",
                    "mean_hd95",
                    "hd95_wt",
                    "hd95_tc",
                    "hd95_et",
                    "mean_dice",
                    "dice_wt",
                    "dice_tc",
                    "dice_et",
                ]
            ]
        )
    )

    md.append("## Worst per-region Dice\n")
    md.append("### Lowest WT Dice (top 10)\n")
    md.append(_df_to_md(worst_wt_dice[["case_id", "dice_wt", "mean_dice", "mean_hd95"]]))
    md.append("### Lowest TC Dice (top 10)\n")
    md.append(_df_to_md(worst_tc_dice[["case_id", "dice_tc", "mean_dice", "mean_hd95"]]))
    md.append("### Lowest ET Dice (top 10)\n")
    md.append(_df_to_md(worst_et_dice[["case_id", "dice_et", "mean_dice", "mean_hd95"]]))

    md.append("## Worst per-region HD95\n")
    md.append("### Highest WT HD95 (top 10)\n")
    md.append(_df_to_md(worst_wt_hd95[["case_id", "hd95_wt", "mean_dice", "mean_hd95"]]))
    md.append("### Highest TC HD95 (top 10)\n")
    md.append(_df_to_md(worst_tc_hd95[["case_id", "hd95_tc", "mean_dice", "mean_hd95"]]))
    md.append("### Highest ET HD95 (top 10)\n")
    md.append(_df_to_md(worst_et_hd95[["case_id", "hd95_et", "mean_dice", "mean_hd95"]]))

    (output_dir / "worst_cases_analysis.md").write_text("\n".join(md), encoding="utf-8")

    logger.info("Saved worst-cases report to %s", output_dir)


# --------------------------------------------------------------------------- #
# Per-case report
# --------------------------------------------------------------------------- #
def _classify_case(row: pd.Series) -> str:
    wt_bad = row["hd95_wt"] > 20
    tc_bad = row["hd95_tc"] > 10
    et_bad = row["hd95_et"] > 10
    dice_low = row["mean_dice"] < 0.75

    if dice_low and wt_bad and tc_bad and et_bad:
        return "global_failure"
    if (
        row["hd95_wt"] > 30
        and row["hd95_tc"] < 3
        and row["hd95_et"] < 3
        and row["mean_dice"] > 0.82
    ):
        return "wt_only_boundary"
    if row["mean_dice"] < 0.75:
        return "low_dice"
    return "mixed"


def _case_observation(row: pd.Series) -> str:
    parts = []
    pattern = row["pattern"]

    if pattern == "wt_only_boundary":
        parts.append(
            "Хорошее совпадение по Dice, но огромный HD95 в WT. "
            "Ошибка сосредоточена на границе WT (периферийный отёк). TC/ET почти идеальны."
        )
    elif pattern == "global_failure":
        parts.append(
            "Полный провал: низкий Dice и высокий HD95 по всем регионам. "
            "Модель существенно недосегментирует опухоль."
        )
    elif pattern == "low_dice":
        parts.append(
            "Сильно занижен Dice, особенно в одном или нескольких регионах. "
            "Сегментация фрагментирована или сильно меньше GT."
        )
    else:
        parts.append(
            "Смешанный паттерн: умеренно низкий Dice и высокий HD95 по нескольким регионам."
        )

    if row["wt_ratio"] < 0.6:
        parts.append(
            f"WT предсказано в {row['wt_ratio']:.2f} раза меньше GT "
            f"({row['wt_pred_ml']:.1f} vs {row['wt_gt_ml']:.1f} мл)."
        )
    elif row["wt_ratio"] > 1.4:
        parts.append(
            f"WT предсказано в {row['wt_ratio']:.2f} раза больше GT "
            f"({row['wt_pred_ml']:.1f} vs {row['wt_gt_ml']:.1f} мл)."
        )

    if row["tc_ratio"] < 0.5:
        parts.append(f"TC сильно недосегментирован ({row['tc_pred_ml']:.1f} vs {row['tc_gt_ml']:.1f} мл).")
    if row["et_ratio"] < 0.5:
        parts.append(f"ET сильно недосегментирован ({row['et_pred_ml']:.1f} vs {row['et_gt_ml']:.1f} мл).")

    return " ".join(parts)


def run_per_case_report(
    metrics_source: str | Path,
    output_dir: Path,
    data_dir: Path,
    predictions_dir: Path,
) -> None:
    """Generate a per-case analysis report with volumes and error patterns."""
    metrics = load_metrics(metrics_source)
    output_dir = _ensure_output_dir(output_dir)

    voxel_vol_ml = 1.0 / 1000.0
    rows = []

    for _, row in metrics.iterrows():
        case_id = row["case_id"]
        label_path = data_dir / "labelsTr" / f"{case_id}.nii.gz"
        pred_path = predictions_dir / f"{case_id}_pred_mask.nii.gz"

        label = load_nifti(label_path, dtype=np.uint8)
        pred = load_nifti(pred_path, dtype=np.uint8)

        volumes = {}
        for region in ["wt", "tc", "et"]:
            gt_region = multiclass_to_region(label, region)
            pred_region = multiclass_to_region(pred, region)
            volumes[f"{region}_gt_ml"] = gt_region.sum() * voxel_vol_ml
            volumes[f"{region}_pred_ml"] = pred_region.sum() * voxel_vol_ml
            volumes[f"{region}_diff_ml"] = (pred_region.sum() - gt_region.sum()) * voxel_vol_ml
            volumes[f"{region}_ratio"] = (
                pred_region.sum() / gt_region.sum() if gt_region.sum() > 0 else float("inf")
            )

        rec = {**row.to_dict(), **volumes}
        rec["pattern"] = _classify_case(pd.Series(rec))
        rec["observation"] = _case_observation(pd.Series(rec))
        rows.append(rec)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(output_dir / "per_case_report.csv", index=False)

    md = ["# Отчёт по кейсам\n"]
    md.append(f"Всего проанализировано кейсов: {len(report_df)}\n")
    md.append("## Распределение паттернов\n")
    for pattern, count in report_df["pattern"].value_counts().items():
        md.append(f"- **{pattern}**: {count}")
    md.append("")

    md.append("## Сводная таблица\n")
    summary_cols = [
        "case_id",
        "pattern",
        "mean_dice",
        "mean_hd95",
        "dice_wt",
        "dice_tc",
        "dice_et",
        "hd95_wt",
        "hd95_tc",
        "hd95_et",
        "wt_gt_ml",
        "wt_pred_ml",
        "tc_gt_ml",
        "tc_pred_ml",
        "et_gt_ml",
        "et_pred_ml",
    ]
    md.append(report_df[summary_cols].to_markdown(index=False, floatfmt=".3f"))
    md.append("")

    md.append("## Детальный анализ по кейсам\n")
    for _, row in report_df.sort_values("mean_hd95", ascending=False).iterrows():
        md.append(f"### {row['case_id']} ({row['pattern']})\n")
        md.append(f"- **mean_dice**: {row['mean_dice']:.4f}")
        md.append(f"- **mean_hd95**: {row['mean_hd95']:.2f} мм")
        md.append(
            f"- **Dice**: WT={row['dice_wt']:.3f}, TC={row['dice_tc']:.3f}, ET={row['dice_et']:.3f}"
        )
        md.append(
            f"- **HD95**: WT={row['hd95_wt']:.2f}, TC={row['hd95_tc']:.2f}, ET={row['hd95_et']:.2f} мм"
        )
        md.append(
            f"- **Объёмы (GT → Pred)**: WT={row['wt_gt_ml']:.1f} → {row['wt_pred_ml']:.1f} мл, "
            f"TC={row['tc_gt_ml']:.1f} → {row['tc_pred_ml']:.1f} мл, "
            f"ET={row['et_gt_ml']:.1f} → {row['et_pred_ml']:.1f} мл"
        )
        md.append(f"- **Наблюдение**: {row['observation']}")
        md.append("")

    md.append("## Рекомендации\n")
    md.append("1. **wt_only_boundary**: фокус на постпроцессинге/взвешивании границы WT.")
    md.append(
        "2. **global_failure/low_dice**: проверить разметку и качество входных сканов; возможно, редкие анатомические варианты."
    )
    md.append("3. **mixed**: комбинированный подход — улучшить и сегментацию ядра, и границу WT.")

    (output_dir / "per_case_report.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Saved per-case report to %s", output_dir)


# --------------------------------------------------------------------------- #
# Uncertainty / error correlation
# --------------------------------------------------------------------------- #
def _analyze_uncertainty_case(
    case_id: str,
    data_dir: Path,
    predictions_dir: Path,
    uncertainty_dir: Path,
    region: str = "wt",
) -> dict:
    label = load_nifti(data_dir / "labelsTr" / f"{case_id}.nii.gz", dtype=np.uint8)
    pred = load_nifti(predictions_dir / f"{case_id}_pred_mask.nii.gz", dtype=np.uint8)
    uncertainty = load_nifti(uncertainty_dir / f"{case_id}_uncertainty.nii.gz", dtype=np.float32)

    label_mask = multiclass_to_region(label, region)
    pred_mask = multiclass_to_region(pred, region)

    tp = label_mask & pred_mask
    fp = ~label_mask & pred_mask
    fn = label_mask & ~pred_mask
    tn = ~label_mask & ~pred_mask

    region_names = ["TP", "FP", "FN", "TN"]
    regions = [tp, fp, fn, tn]
    mean_unc = {
        f"unc_{r.lower()}": float(uncertainty[m].mean()) if m.any() else 0.0
        for r, m in zip(region_names, regions)
    }
    std_unc = {
        f"unc_std_{r.lower()}": float(uncertainty[m].std()) if m.any() else 0.0
        for r, m in zip(region_names, regions)
    }
    voxels = {f"voxels_{r.lower()}": int(m.sum()) for r, m in zip(region_names, regions)}

    error_mask = fp | fn
    if error_mask.any():
        pb_corr, pb_p = pointbiserialr(error_mask.ravel(), uncertainty.ravel())
    else:
        pb_corr, pb_p = np.nan, np.nan

    if fn.any():
        fn_corr, fn_p = pointbiserialr(fn.ravel(), uncertainty.ravel())
    else:
        fn_corr, fn_p = np.nan, np.nan

    if fp.any():
        fp_corr, fp_p = pointbiserialr(fp.ravel(), uncertainty.ravel())
    else:
        fp_corr, fp_p = np.nan, np.nan

    return {
        "case_id": case_id,
        "region": region,
        **mean_unc,
        **std_unc,
        **voxels,
        "error_corr": float(pb_corr) if not np.isnan(pb_corr) else None,
        "error_pvalue": float(pb_p) if not np.isnan(pb_p) else None,
        "fn_corr": float(fn_corr) if not np.isnan(fn_corr) else None,
        "fn_pvalue": float(fn_p) if not np.isnan(fn_p) else None,
        "fp_corr": float(fp_corr) if not np.isnan(fp_corr) else None,
        "fp_pvalue": float(fp_p) if not np.isnan(fp_p) else None,
    }


def _plot_uncertainty_by_region(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    regions = ["TP", "FP", "FN", "TN"]
    colors = ["green", "blue", "orange", "gray"]

    for ax, region, color in zip(axes.flat, regions, colors):
        col = f"unc_{region.lower()}"
        ax.bar(df["case_id"], df[col], color=color, alpha=0.7)
        ax.set_ylabel("Mean uncertainty")
        ax.set_title(f"Mean uncertainty in {region} voxels")
        ax.set_ylim(0, max(df[col].max() * 1.1, 0.001))
        ax.tick_params(axis="x", rotation=90)

    plt.suptitle("Mean uncertainty by error region")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["error_corr"], width, label="Error (FP+FN)", color="red", alpha=0.7)
    ax.bar(x, df["fn_corr"], width, label="FN only", color="orange", alpha=0.7)
    ax.bar(x + width, df["fp_corr"], width, label="FP only", color="blue", alpha=0.7)

    ax.set_ylabel("Point-biserial correlation")
    ax.set_title("Voxel-wise correlation: uncertainty vs errors")
    ax.set_xticks(x)
    ax.set_xticklabels(df["case_id"], rotation=90)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_uncertainty_correlation(
    metrics_source: str | Path,
    output_dir: Path,
    data_dir: Path,
    predictions_dir: Path,
    uncertainty_dir: Path,
    region: str = "wt",
) -> None:
    """Analyze voxel-wise correlation between uncertainty and segmentation errors."""
    metrics = load_metrics(metrics_source)
    output_dir = _ensure_output_dir(output_dir)

    rows = []
    for case_id in metrics["case_id"]:
        rows.append(
            _analyze_uncertainty_case(
                case_id,
                data_dir=data_dir,
                predictions_dir=predictions_dir,
                uncertainty_dir=uncertainty_dir,
                region=region,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "uncertainty_error_correlation.csv", index=False)

    _plot_uncertainty_by_region(df, output_dir / "uncertainty_by_region.png")
    _plot_correlation(df, output_dir / "uncertainty_error_correlation.png")

    md = ["# Корреляция неопределённости и ошибок\n"]
    md.append("## Сводная таблица\n")
    md.append(df.to_markdown(index=False, floatfmt=".4f"))
    md.append("\n## Интерпретация\n")
    md.append("- **error_corr** — корреляция неопределённости с любыми ошибками (FP+FN).")
    md.append("- **fn_corr** — корреляция с FN (пропущенные области GT).")
    md.append("- **fp_corr** — корреляция с FP (лишние области предсказания).")
    md.append("- Положительная корреляция означает: чем выше неопределённость, тем вероятнее ошибка.")
    md.append("\n## Ключевые выводы\n")

    mean_error_corr = df["error_corr"].mean()
    mean_fn_corr = df["fn_corr"].mean()
    mean_fp_corr = df["fp_corr"].mean()
    md.append(f"- Средняя корреляция неопределённости с ошибкой: **{mean_error_corr:.4f}**")
    md.append(f"- Средняя корреляция с FN: **{mean_fn_corr:.4f}**")
    md.append(f"- Средняя корреляция с FP: **{mean_fp_corr:.4f}**")

    top_fn = df.nlargest(3, "fn_corr")["case_id"].tolist()
    top_error = df.nlargest(3, "error_corr")["case_id"].tolist()
    md.append(f"- Топ-3 по корреляции с FN: {', '.join(top_fn)}")
    md.append(f"- Топ-3 по корреляции с общей ошибкой: {', '.join(top_error)}")

    (output_dir / "uncertainty_error_correlation.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Saved uncertainty correlation analysis to %s", output_dir)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _normalize_slice(slice_: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros_like(slice_, dtype=np.float32)
    return np.clip((slice_.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _select_slices(label: np.ndarray, pred: np.ndarray, n_slices: int = 3) -> list[int]:
    foreground = label | pred
    n_z = label.shape[2]
    z_counts = foreground.sum(axis=(0, 1))
    z_indices = np.where(z_counts > 0)[0]
    if len(z_indices) == 0:
        center = n_z // 2
        slice_indices = list(range(center - n_slices // 2, center + n_slices // 2 + 1))
    else:
        heaviest = int(np.argmax(z_counts))
        half = n_slices // 2
        slice_indices = list(range(heaviest - half, heaviest + half + 1))

    slice_indices = [max(0, min(s, n_z - 1)) for s in slice_indices]
    return sorted(set(slice_indices))


def _plot_case_overlay(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    case_id: str,
    region: str,
    slice_idx: int,
    metrics: dict[str, float],
    out_path: Path,
    modality: int = 0,
) -> None:
    if image.ndim == 4:
        img_slice = _normalize_slice(image[:, :, slice_idx, modality])
    else:
        img_slice = _normalize_slice(image[:, :, slice_idx])
    label_slice = label[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx]
    unc_slice = uncertainty[:, :, slice_idx]

    fp = pred_slice & ~label_slice
    fn = label_slice & ~pred_slice

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Image\n{case_id} ({region.upper()})")
    axes[0].axis("off")

    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.5 * label_slice)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(pred_slice, cmap="Greens", alpha=0.5 * pred_slice)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(img_slice, cmap="gray")
    axes[3].imshow(fp, cmap=CMAP_FP, alpha=0.6 * fp)
    axes[3].imshow(fn, cmap=CMAP_FN, alpha=0.6 * fn)
    axes[3].set_title("Errors: FP=blue, FN=yellow")
    axes[3].axis("off")

    im = axes[4].imshow(unc_slice, cmap="hot")
    axes[4].set_title("Uncertainty (TTA std)")
    axes[4].axis("off")
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    axes[5].imshow(img_slice, cmap="gray")
    axes[5].imshow(pred_slice, cmap="Greens", alpha=0.3 * pred_slice)
    im2 = axes[5].imshow(unc_slice, cmap="hot", alpha=0.5)
    axes[5].set_title("Prediction + Uncertainty")
    axes[5].axis("off")
    plt.colorbar(im2, ax=axes[5], fraction=0.046, pad=0.04)

    title = (
        f"{case_id} | {region.upper()} | slice {slice_idx} | "
        f"mean_dice={metrics.get('mean_dice', -1):.3f} | "
        f"mean_hd95={metrics.get('mean_hd95', -1):.2f}"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _visualize_case(
    case_row: pd.Series,
    predictions_dir: Path,
    uncertainty_dir: Path,
    output_dir: Path,
    regions: list[str],
    n_slices: int,
    modality: int = 0,
) -> None:
    case_id = str(case_row["case_id"])
    image_path = resolve_path(case_row["image_path"])
    label_path = resolve_path(case_row["label_path"])
    pred_path = predictions_dir / f"{case_id}_pred_mask.nii.gz"
    uncertainty_path = uncertainty_dir / f"{case_id}_uncertainty.nii.gz"

    for p, name in [
        (image_path, "image"),
        (label_path, "label"),
        (pred_path, "prediction"),
        (uncertainty_path, "uncertainty map"),
    ]:
        if not p.exists():
            logger.warning("%s not found for %s: %s", name.capitalize(), case_id, p)
            return

    logger.info("Visualizing %s", case_id)

    image = load_nifti(image_path)
    label_multiclass = load_nifti(label_path, dtype=np.uint8)
    pred_multiclass = load_nifti(pred_path, dtype=np.uint8)
    uncertainty = load_nifti(uncertainty_path, dtype=np.float32)

    metrics = {
        "mean_dice": case_row.get("mean_dice", -1),
        "mean_hd95": case_row.get("mean_hd95", -1),
    }

    for region in regions:
        label_mask = multiclass_to_region(label_multiclass, region)
        pred_mask = multiclass_to_region(pred_multiclass, region)
        slice_indices = _select_slices(label_mask, pred_mask, n_slices=n_slices)

        for slice_idx in slice_indices:
            out_path = output_dir / region / f"{case_id}_slice{slice_idx:03d}.png"
            _plot_case_overlay(
                image=image,
                label=label_mask,
                pred=pred_mask,
                uncertainty=uncertainty,
                case_id=case_id,
                region=region,
                slice_idx=slice_idx,
                metrics=metrics,
                out_path=out_path,
                modality=modality,
            )


def run_visualize(
    metrics_source: str | Path,
    output_dir: Path,
    predictions_dir: Path,
    uncertainty_dir: Path,
    max_cases: int,
    hd95_threshold: float,
    dice_threshold: float,
    regions: list[str],
    n_slices: int,
    modality: int,
    mode: str = "or",
) -> None:
    """Generate PNG overlays for problematic inference cases with uncertainty maps."""
    df = load_metrics(metrics_source)
    required = {"case_id", "image_path", "label_path", "mean_dice", "mean_hd95"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {missing}")

    problematic = select_cases(
        df,
        hd95_threshold=hd95_threshold,
        dice_threshold=dice_threshold,
        max_cases=max_cases,
        sort_by="hd95",
        mode=mode,
    )
    logger.info(
        "Selected %d problematic cases (hd95 > %.2f %s dice < %.2f)",
        len(problematic),
        hd95_threshold,
        "OR" if mode == "or" else "AND",
        dice_threshold,
    )

    predictions_dir = resolve_path(predictions_dir)
    uncertainty_dir = resolve_path(uncertainty_dir)
    output_dir = _ensure_output_dir(output_dir)

    for _, row in problematic.iterrows():
        _visualize_case(
            case_row=row,
            predictions_dir=predictions_dir,
            uncertainty_dir=uncertainty_dir,
            output_dir=output_dir,
            regions=regions,
            n_slices=n_slices,
            modality=modality,
        )

    logger.info("Saved visualizations to %s", output_dir)


# --------------------------------------------------------------------------- #
# Select anomalous cases
# --------------------------------------------------------------------------- #
def run_select_anomalous(
    metrics_source: str | Path,
    output: Path | None,
    hd95_threshold: float,
    dice_threshold: float,
    max_cases: int | None,
    sort_by: str,
    mode: str,
) -> None:
    """Select case IDs with high HD95 and/or low Dice."""
    df = load_metrics(metrics_source)
    selected = select_cases(
        df,
        hd95_threshold=hd95_threshold,
        dice_threshold=dice_threshold,
        max_cases=max_cases,
        sort_by=sort_by,
        mode=mode,
    )

    case_ids = selected["case_id"].tolist()
    output_lines = "\n".join(case_ids) + "\n"

    if output:
        out_path = resolve_path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_lines)
        logger.info("Saved %d case IDs to %s", len(case_ids), out_path)
    else:
        sys.stdout.write(output_lines)

    summary_cols = ["case_id", "mean_dice", "mean_hd95"]
    logger.info(
        "Selected anomalous cases (criteria: hd95 > %.2f %s dice < %.2f):\n%s",
        hd95_threshold,
        "OR" if mode == "or" else "AND",
        dice_threshold,
        selected[summary_cols].to_string(index=False),
    )


# --------------------------------------------------------------------------- #
# Run all analyses
# --------------------------------------------------------------------------- #
def run_all(args: argparse.Namespace) -> None:
    """Run worst-cases, per-case-report and uncertainty-correlation analyses."""
    logger.info("Running full analysis pipeline")

    run_worst_cases(args.metrics, args.output_dir)
    run_per_case_report(
        args.metrics,
        output_dir=args.output_dir / "per_case_report",
        data_dir=args.data_dir,
        predictions_dir=args.predictions_dir,
    )
    run_uncertainty_correlation(
        args.metrics,
        output_dir=args.output_dir / "uncertainty_analysis",
        data_dir=args.data_dir,
        predictions_dir=args.predictions_dir,
        uncertainty_dir=args.uncertainty_dir,
        region=args.region,
    )

    if args.run_visualize:
        run_visualize(
            args.metrics,
            output_dir=args.output_dir / "visualizations",
            predictions_dir=args.predictions_dir,
            uncertainty_dir=args.uncertainty_dir,
            max_cases=args.max_cases,
            hd95_threshold=args.hd95_threshold,
            dice_threshold=args.dice_threshold,
            regions=args.regions,
            n_slices=args.n_slices,
            modality=args.modality,
            mode=args.mode,
        )

    logger.info("Full analysis pipeline complete. Results in %s", args.output_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_metrics_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--metrics",
        type=str,
        required=required,
        help="Path to inference_metrics.csv or ClearML task ID",
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data_dir",
        type=str,
        default="src/data/dev/UPENN-GBM",
        help="Directory containing labelsTr/ folder",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="src/results/inference/predictions",
        help="Directory with *_pred_mask.nii.gz predictions",
    )
    parser.add_argument(
        "--uncertainty_dir",
        type=str,
        default="src/results/inference/uncertainty",
        help="Directory with *_uncertainty.nii.gz maps",
    )


def _add_output_arg(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default,
        help="Directory to save outputs",
    )


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max_cases",
        type=int,
        default=20,
        help="Maximum number of problematic cases",
    )
    parser.add_argument(
        "--hd95_threshold",
        type=float,
        default=10.0,
        help="HD95 threshold for case selection",
    )
    parser.add_argument(
        "--dice_threshold",
        type=float,
        default=0.75,
        help="Dice threshold for case selection",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="or",
        choices=["or", "and"],
        help="Combine HD95 and Dice thresholds with OR or AND",
    )


def _add_visualize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["wt", "tc", "et"],
        choices=["wt", "tc", "et"],
        help="Regions to visualize",
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=3,
        help="Number of axial slices per case/region",
    )
    parser.add_argument(
        "--modality",
        type=int,
        default=0,
        help="Image modality channel (0=FLAIR, 1=T1, 2=T1ce, 3=T2)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified inference result analysis tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # worst-cases
    p_worst = subparsers.add_parser("worst-cases", help="Worst-performing cases report")
    _add_metrics_arg(p_worst)
    _add_output_arg(p_worst, "src/results/analysis/worst_cases")

    # per-case-report
    p_report = subparsers.add_parser("per-case-report", help="Per-case volume and pattern report")
    _add_metrics_arg(p_report)
    _add_output_arg(p_report, "src/results/analysis/per_case_report")
    _add_data_args(p_report)

    # uncertainty-correlation
    p_unc = subparsers.add_parser(
        "uncertainty-correlation",
        help="Correlation between uncertainty and segmentation errors",
    )
    _add_metrics_arg(p_unc)
    _add_output_arg(p_unc, "src/results/analysis/uncertainty_analysis")
    _add_data_args(p_unc)
    p_unc.add_argument(
        "--region",
        type=str,
        default="wt",
        choices=["wt", "tc", "et"],
        help="Region for uncertainty analysis",
    )

    # visualize
    p_vis = subparsers.add_parser(
        "visualize",
        help="Generate PNG overlays for problematic cases",
    )
    _add_metrics_arg(p_vis)
    _add_output_arg(p_vis, "src/results/analysis/visualizations")
    _add_data_args(p_vis)
    _add_filter_args(p_vis)
    _add_visualize_args(p_vis)

    # select-anomalous
    p_sel = subparsers.add_parser(
        "select-anomalous",
        help="Select anomalous case IDs by thresholds",
    )
    _add_metrics_arg(p_sel)
    p_sel.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file to save selected case_ids (one per line)",
    )
    _add_filter_args(p_sel)
    p_sel.add_argument(
        "--sort_by",
        type=str,
        default="hd95",
        choices=["hd95", "dice"],
        help="Sort by worst HD95 or worst Dice",
    )

    # all
    p_all = subparsers.add_parser("all", help="Run worst-cases, per-case-report and uncertainty-correlation")
    _add_metrics_arg(p_all)
    _add_output_arg(p_all, "src/results/analysis")
    _add_data_args(p_all)
    p_all.add_argument(
        "--region",
        type=str,
        default="wt",
        choices=["wt", "tc", "et"],
        help="Region for uncertainty analysis",
    )
    p_all.add_argument(
        "--run_visualize",
        action="store_true",
        help="Also generate visualizations (slower)",
    )
    _add_filter_args(p_all)
    _add_visualize_args(p_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    command = args.command
    if command == "worst-cases":
        run_worst_cases(args.metrics, resolve_path(args.output_dir))
    elif command == "per-case-report":
        run_per_case_report(
            args.metrics,
            output_dir=resolve_path(args.output_dir),
            data_dir=resolve_path(args.data_dir),
            predictions_dir=resolve_path(args.predictions_dir),
        )
    elif command == "uncertainty-correlation":
        run_uncertainty_correlation(
            args.metrics,
            output_dir=resolve_path(args.output_dir),
            data_dir=resolve_path(args.data_dir),
            predictions_dir=resolve_path(args.predictions_dir),
            uncertainty_dir=resolve_path(args.uncertainty_dir),
            region=args.region,
        )
    elif command == "visualize":
        run_visualize(
            args.metrics,
            output_dir=resolve_path(args.output_dir),
            predictions_dir=resolve_path(args.predictions_dir),
            uncertainty_dir=resolve_path(args.uncertainty_dir),
            max_cases=args.max_cases,
            hd95_threshold=args.hd95_threshold,
            dice_threshold=args.dice_threshold,
            regions=args.regions,
            n_slices=args.n_slices,
            modality=args.modality,
            mode=args.mode,
        )
    elif command == "select-anomalous":
        output = resolve_path(args.output) if args.output else None
        run_select_anomalous(
            args.metrics,
            output=output,
            hd95_threshold=args.hd95_threshold,
            dice_threshold=args.dice_threshold,
            max_cases=args.max_cases,
            sort_by=args.sort_by,
            mode=args.mode,
        )
    elif command == "all":
        args.output_dir = resolve_path(args.output_dir)
        args.data_dir = resolve_path(args.data_dir)
        args.predictions_dir = resolve_path(args.predictions_dir)
        args.uncertainty_dir = resolve_path(args.uncertainty_dir)
        run_all(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
