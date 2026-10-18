"""阶段 5：输出对齐校验（对齐 v0.3 文档 4.2 / 7.1 的分层与交付门禁）。"""
from __future__ import annotations

import os

import numpy as np

from schemas.layer_manifest import LayerManifest
from schemas.pipeline_report import AlignmentReport
from pipeline.imaging import composite_to_uint8, foreground_mask


def validate_output(arr: np.ndarray, manifest: LayerManifest,
                    layer_arrays: list, output_dir: str | None = None) -> AlignmentReport:
    """合成所有图层并与原图对比，产出 AlignmentReport。"""
    report = AlignmentReport()

    # 画布尺寸一致
    h, w = arr.shape[:2]
    report.canvas_ok = (manifest.canvas_width == w and manifest.canvas_height == h)
    report.layer_count = len(layer_arrays)

    # 图层非空
    nonempty = all(int(a[..., 3].sum()) > 0 for a in layer_arrays)
    report.layers_nonempty = nonempty
    if not nonempty:
        report.messages.append("warn:some layers are empty")

    # 文件齐全检查
    if output_dir:
        missing = [
            e.file for e in manifest.layers
            if not os.path.exists(os.path.join(output_dir, e.file))
        ]
        if missing:
            report.messages.append("error:missing layer files: %s" % missing)

    if not layer_arrays:
        report.messages.append("error:no layers produced")
        return report

    composite = composite_to_uint8(layer_arrays)
    mask, _ = foreground_mask(arr)

    # 前景区域逐像素对比（RGB + alpha）
    fg = mask
    if int(fg.sum()) == 0:
        report.messages.append("error:no foreground to validate")
        return report

    diff = np.abs(composite.astype(np.float64) - arr.astype(np.float64))[fg]
    report.composite_mae = round(float(diff.mean()), 4)

    mse = float((diff ** 2).mean())
    if mse <= 1e-12:
        report.composite_psnr = None  # 无损
        report.messages.append("info:composite is lossless in foreground")
    else:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
        report.composite_psnr = round(psnr, 2)

    # 覆盖：前景像素中，合成图 alpha>0 的比例
    comp_alpha = composite[..., 3] > 0
    report.coverage = round(float(comp_alpha[fg].mean()), 4)
    if report.coverage < 0.999:
        report.messages.append(
            "warn:coverage=%.4f (some foreground pixels were dropped)" % report.coverage
        )

    return report
