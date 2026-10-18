"""命令行流水线入口：串联 输入校验 → 分析 → 分层 → 输出校验。

用法：
    python -m pipeline.run 立绘.png -o out_dir [-k 8]

输出：
    out_dir/
      layers/unknown_000.png ...   # 分层图层
      preview_composite.png         # 合成预览（人工比对用）
      image_report.json
      layer_manifest.json
      pipeline_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum

import numpy as np

from schemas.job import JobStatus, StageRecord
from schemas.pipeline_report import PipelineReport
from pipeline.imaging import composite_to_uint8
from pipeline.stages.validate_input import validate_input
from pipeline.stages.analyze_image import analyze_image
from pipeline.stages.segment_layers import segment
from pipeline.stages.validate_output import validate_output


def _jsonable(obj):
    """把 dataclass/Enum/tuple/numpy 递归转成可 JSON 序列化的纯 Python。"""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(input_path: str, output_dir: str, k: int = 8,
        backend: str = "color") -> PipelineReport:
    os.makedirs(output_dir, exist_ok=True)
    layers_dir = os.path.join(output_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    sha = _sha256_file(input_path)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + sha[:8]

    report = PipelineReport(job_id=job_id, status=JobStatus.VALIDATING)
    stages: list = []

    def stage(name, fn):
        rec = StageRecord(name=name, status="running", started_at=_now_iso())
        t0 = time.time()
        try:
            result = fn()
            rec.status = "ok"
            rec.finished_at = _now_iso()
            rec.duration_ms = int((time.time() - t0) * 1000)
            rec.message = "ok"
            return result
        except ValueError as exc:
            rec.status = "failed"
            rec.finished_at = _now_iso()
            rec.duration_ms = int((time.time() - t0) * 1000)
            msg = str(exc)
            rec.message = msg
            rec.error_code = msg.split(" ")[0].strip("[]") if msg.startswith("[") else "UNKNOWN"
            report.error_code = rec.error_code
            report.error_message = msg
            raise
        finally:
            stages.append(rec)

    try:
        # 阶段 1：输入校验
        v = stage("validate_input", lambda: validate_input(input_path))
        arr, info = v["arr"], v["info"]

        # 阶段 2：分析
        img_report = stage("analyze_image", lambda: analyze_image(arr, info))

        # 阶段 3：分层（写图层文件）
        if backend == "semantic":
            from pipeline.stages.segment_semantic import segment_semantic
            manifest, layer_arrays = stage(
                "segment_semantic",
                lambda: segment_semantic(arr, output_dir=layers_dir),
            )
        else:
            manifest, layer_arrays = stage(
                "segment_layers",
                lambda: segment(arr, K=k, output_dir=layers_dir),
            )

        # 阶段 5：输出校验
        alignment = stage(
            "validate_output",
            lambda: validate_output(arr, manifest, layer_arrays, output_dir=layers_dir),
        )

        # 合成预览（人工比对用）
        composite_to_uint8(layer_arrays)
        from PIL import Image
        Image.fromarray(composite_to_uint8(layer_arrays), mode="RGBA").save(
            os.path.join(output_dir, "preview_composite.png")
        )

        report.image_report = img_report
        report.manifest = manifest
        report.alignment = alignment
        report.status = JobStatus.NEEDS_REVIEW  # 纯算法初稿：始终需要人工检查
        report.stages = stages

    except ValueError:
        report.status = JobStatus.FAILED
        report.stages = stages
    except Exception as exc:  # noqa: BLE001
        report.status = JobStatus.FAILED
        report.error_code = "INTERNAL"
        report.error_message = str(exc)
        report.stages = stages

    # 写报告
    with open(os.path.join(output_dir, "image_report.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(report.image_report), f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "layer_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(report.manifest), f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "pipeline_report.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, ensure_ascii=False, indent=2)

    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live2D Compiler: 立绘 -> 分层")
    ap.add_argument("input", help="输入 PNG 立绘路径")
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    ap.add_argument("-k", "--clusters", type=int, default=8, help="颜色聚类数（color 后端用，默认 8）")
    ap.add_argument("--backend", choices=["color", "semantic"], default="color",
                    help="分层后端：color=颜色聚类（无需模型）; semantic=AI 语义分割（需 .deps 里的 torch）")
    args = ap.parse_args(argv)

    report = run(args.input, args.output, k=args.clusters, backend=args.backend)
    print("job_id: %s" % report.job_id)
    print("status: %s" % report.status.value)
    if report.alignment:
        print("layers: %d  coverage: %s  mae: %s" % (
            report.alignment.layer_count,
            report.alignment.coverage,
            report.alignment.composite_mae,
        ))
    if report.error_message:
        print("error: %s" % report.error_message)
    return 0 if report.status != JobStatus.FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
