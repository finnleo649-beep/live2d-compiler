"""阶段 2：图片分析，产出 image_report（对齐 v0.3 文档 4.1）。"""
from __future__ import annotations

import numpy as np

from schemas.pipeline_report import ImageReport
from pipeline.imaging import foreground_mask, laplacian_variance


def _background_complexity(arr: np.ndarray, bg_kind: str) -> str:
    """透明背景无背景复杂度；不透明背景按四角 patch 颜色离散度估计。"""
    if bg_kind == "transparent":
        return "n/a"

    h, w = arr.shape[:2]
    p = min(24, h // 4, w // 4)
    p = max(p, 1)
    patches = [
        arr[:p, :p, :3],
        arr[:p, w - p:, :3],
        arr[h - p:, :p, :3],
        arr[h - p:, w - p:, :3],
    ]
    # 量化到 32 级再数唯一色，作为背景复杂度估计。
    q = 32
    uniq = set()
    for patch in patches:
        flat = (patch.astype(np.float64) // (256.0 / q)).astype(int)
        for px in flat.reshape(-1, 3):
            uniq.add(tuple(px))
    n = len(uniq)
    if n <= 4:
        return "low"
    if n <= 16:
        return "medium"
    return "high"


def analyze_image(arr: np.ndarray, info: dict) -> ImageReport:
    """分析 RGBA 数组，返回 ImageReport。"""
    h, w = arr.shape[:2]
    alpha = arr[..., 3]
    has_alpha = bool((alpha < 255).any())

    mask, bg_kind = foreground_mask(arr)
    subject_ratio = float(mask.mean())

    report = ImageReport(
        format=info.get("ext", "").lstrip("."),
        width=w,
        height=h,
        color_mode="RGBA",
        has_alpha=has_alpha,
        is_transparent_bg=(bg_kind == "transparent"),
        subject_ratio=round(subject_ratio, 4),
        background_complexity=_background_complexity(arr, bg_kind),
        blur_score=round(laplacian_variance(arr), 2),
        warnings=[],
        allowed=True,
    )

    if subject_ratio < 0.02:
        report.warnings.append("block:subject_ratio too small (%.4f)" % subject_ratio)
        report.allowed = False
    elif subject_ratio < 0.10:
        report.warnings.append("warn:subject occupies a very small area")
    if subject_ratio > 0.99:
        report.warnings.append("warn:no transparent/背景 region detected")
    if report.blur_score < 5.0:
        report.warnings.append("warn:image may be blurry (laplacian var=%.1f)" % report.blur_score)

    return report
