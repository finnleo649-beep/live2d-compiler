"""任务状态机与阶段记录（对齐 v0.3 文档 3.2 与 7.2）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    SEGMENTING = "segmenting"
    COMPLETING = "completing"
    RIGGING = "rigging"
    EXPORTING = "exporting"
    VALIDATING_OUTPUT = "validating_output"
    SUCCEEDED = "succeeded"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 统一错误码（v0.3 文档 7.2），本原型只用到其中一部分。
ERROR_CODES = (
    "INPUT_INVALID",
    "INPUT_UNSUPPORTED",
    "SEGMENTATION_FAILED",
    "LAYER_SCHEMA_INVALID",
    "COMPLETION_FAILED",
    "RIGGING_TOOL_FAILED",
    "EXPORT_FAILED",
    "OUTPUT_INCOMPLETE",
    "RUNTIME_LOAD_FAILED",
    "RESOURCE_LIMIT_EXCEEDED",
    "CANCELLED_BY_USER",
)


@dataclass
class StageRecord:
    """单个阶段的执行记录：开始/结束时间、时长、日志摘要、错误码。"""
    name: str
    status: str = "pending"  # pending | running | ok | failed
    error_code: Optional[str] = None
    message: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class Job:
    """一个任务的统一记录。中间产物路径、哈希、报告都挂在这里。"""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    stages: list = field(default_factory=list)  # list[StageRecord]
    input_sha256: Optional[str] = None
    input_path: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
