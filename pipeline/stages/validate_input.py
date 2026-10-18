"""阶段 1：输入校验（对齐 v0.3 文档 4.1 的失败处理）。"""
from __future__ import annotations

import os

from pipeline.imaging import load_rgba


MIN_DIM = 64
MAX_DIM = 8192
SUPPORTED_EXTS = (".png", ".psd")  # 本原型实际只支持 PNG 像素读取


def validate_input(path: str) -> dict:
    """校验输入文件，返回 {"arr": 数组, "info": 基础信息}。

    不符合要求时抛 ValueError（携带统一错误码 INPUT_INVALID / INPUT_UNSUPPORTED）。
    """
    if not os.path.exists(path):
        raise ValueError("[INPUT_INVALID] file not found: %s" % path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            "[INPUT_UNSUPPORTED] unsupported extension %r (supported: %s)"
            % (ext, ", ".join(SUPPORTED_EXTS))
        )

    if ext == ".psd":
        # 路 B 不解析 PSD，明确报不支持，而不是假装成功。
        raise ValueError(
            "[INPUT_UNSUPPORTED] PSD input is not supported in the pure-algorithm "
            "prototype yet; please provide a PNG"
        )

    try:
        arr = load_rgba(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("[INPUT_INVALID] cannot open as image: %s" % exc)

    h, w = arr.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        raise ValueError(
            "[INPUT_INVALID] resolution too low (%dx%d), minimum %dpx"
            % (w, h, MIN_DIM)
        )
    if h > MAX_DIM or w > MAX_DIM:
        raise ValueError(
            "[INPUT_INVALID] resolution too high (%dx%d), maximum %dpx"
            % (w, h, MAX_DIM)
        )

    info = {"path": path, "ext": ext, "width": w, "height": h}
    return {"arr": arr, "info": info}
