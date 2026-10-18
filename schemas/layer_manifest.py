"""图层清单契约（对齐 v0.3 文档 4.2 / 6.3）。

关键铁律：未识别图层保留为 `unknown_*`，绝不静默丢弃图层。
每个图层都要能追溯到来源（本原型统一为颜色聚类 `quantize_cluster`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


LABEL_VERSION = "0.1.0"


@dataclass
class LayerEntry:
    """单个图层条目。"""
    layer_id: str                 # e.g. "unknown_000"
    label: str                    # 与 layer_id 一致；后续可映射到标准标签
    label_version: str = LABEL_VERSION
    file: str = ""                # 相对 manifest 的输出文件名
    area_pixels: int = 0
    bbox: tuple = (0, 0, 0, 0)    # (x0, y0, x1, y1) 闭区间
    mean_color: tuple = (0, 0, 0)  # (r, g, b) 平均色，浮点
    source: str = "quantize_cluster"
    confidence: Optional[float] = None  # 语义置信度；纯算法阶段为 None


@dataclass
class LayerManifest:
    """图层清单：画布信息 + 全部图层。"""
    version: str = LABEL_VERSION
    canvas_width: int = 0
    canvas_height: int = 0
    background_kind: str = "unknown"  # transparent | opaque
    layers: list = field(default_factory=list)  # list[LayerEntry]
    postprocess_notes: list = field(default_factory=list)  # 后处理修正记录
