"""流水线报告契约：图片检测报告 + 对齐报告 + 总报告。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .job import JobStatus


@dataclass
class ImageReport:
    """输入图片检测报告（对齐 v0.3 文档 4.1）。"""
    format: str = ""
    width: int = 0
    height: int = 0
    color_mode: str = ""
    has_alpha: bool = False
    is_transparent_bg: bool = False
    subject_ratio: float = 0.0          # 前景（人物）像素占比 0~1
    background_complexity: str = "unknown"  # low | medium | high | n/a
    blur_score: float = 0.0             # 拉普拉斯方差，越大越清晰
    warnings: list = field(default_factory=list)
    allowed: bool = True                # 是否允许继续

    @property
    def blocked_reasons(self) -> list:
        return [w for w in self.warnings if w.startswith("block:")]


@dataclass
class AlignmentReport:
    """输出对齐校验报告（对齐 v0.3 文档 4.2 质量检查）。"""
    canvas_ok: bool = False
    layers_nonempty: bool = True
    composite_mae: float = 0.0          # 合成图 vs 原图的前景区域平均绝对误差
    composite_psnr: Optional[float] = None
    coverage: float = 0.0               # 前景被合成图覆盖的比例 0~1
    layer_count: int = 0
    messages: list = field(default_factory=list)


@dataclass
class PipelineReport:
    """总报告：一个任务从输入到输出的全部可复现信息。"""
    job_id: str = ""
    status: JobStatus = JobStatus.QUEUED
    image_report: Optional[ImageReport] = None
    manifest: Optional[object] = None      # LayerManifest
    alignment: Optional[AlignmentReport] = None
    stages: list = field(default_factory=list)  # list[StageRecord]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
