"""阶段 3：语义分层（路 B：纯算法颜色聚类）。

诚实边界：本原型不做语义识别，只做「背景分离 + 前景颜色聚类」，
每个颜色簇产出为一个 `unknown_*` 图层。这正是 v0.3 文档 4.2 允许的
「未识别图层保留为 unknown_*」策略——画师拿到的是 0→60 分的初稿底板。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from schemas.layer_manifest import LABEL_VERSION, LayerEntry, LayerManifest
from pipeline.imaging import foreground_mask

MIN_AREA_DEFAULT = 64


def segment(arr: np.ndarray, K: int = 8, output_dir: str | None = None,
            min_area: int = MIN_AREA_DEFAULT):
    """对 RGBA 数组做颜色聚类分层。

    返回 (LayerManifest, list[np.ndarray])；若 output_dir 给定，同时写出 PNG。
    """
    h, w = arr.shape[:2]
    mask, bg_kind = foreground_mask(arr)

    # 对 RGB 做调色板量化（MEDIANCUT），每个实际出现的调色板色作为一个候选图层。
    rgb = Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")
    q = rgb.quantize(colors=K, method=Image.Quantize.MEDIANCUT)
    qarr = np.asarray(q, dtype=np.uint8)

    raw = []
    for idx in np.unique(qarr):
        m = (qarr == idx) & mask
        area = int(m.sum())
        if area < min_area:
            continue
        layer_arr = np.zeros_like(arr)
        layer_arr[m] = arr[m]
        ys, xs = np.nonzero(m)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        mean = tuple(float(v) for v in arr[m][:, :3].mean(axis=0))
        raw.append({"area": area, "arr": layer_arr, "bbox": bbox, "mean": mean})

    raw.sort(key=lambda d: -d["area"])

    layers = []
    layer_arrays = []
    for i, d in enumerate(raw):
        layer_id = "unknown_%03d" % i
        file_name = layer_id + ".png"
        if output_dir:
            Image.fromarray(d["arr"], mode="RGBA").save(
                os.path.join(output_dir, file_name)
            )
        layers.append(LayerEntry(
            layer_id=layer_id,
            label=layer_id,
            label_version=LABEL_VERSION,
            file=file_name,
            area_pixels=d["area"],
            bbox=d["bbox"],
            mean_color=d["mean"],
            source="quantize_cluster",
        ))
        layer_arrays.append(d["arr"])

    manifest = LayerManifest(
        version=LABEL_VERSION,
        canvas_width=w,
        canvas_height=h,
        background_kind=bg_kind,
        layers=layers,
    )
    return manifest, layer_arrays
