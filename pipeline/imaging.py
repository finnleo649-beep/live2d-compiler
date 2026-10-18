"""公共成像工具：仅依赖 numpy + Pillow，无网络、无模型、无 cv2。

这些是本原型（路 B，纯算法）的核心原语；后续接入 AI 分割模型（路 A）时，
只有 foreground/segment 的实现被替换，签名保持不变。
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_rgba(path) -> np.ndarray:
    """打开图片并归一为 RGBA uint8 数组 (H, W, 4)。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def estimate_background_color(arr: np.ndarray, patch: int = 24) -> np.ndarray:
    """用四角 patch 的中位数估计纯色背景的 RGB。

    仅用于不透明、纯色背景的立绘；透明背景不调用本函数。
    """
    h, w = arr.shape[:2]
    p = min(patch, h // 4, w // 4)
    p = max(p, 1)
    corners = [
        arr[:p, :p, :3],
        arr[:p, w - p:, :3],
        arr[h - p:, :p, :3],
        arr[h - p:, w - p:, :3],
    ]
    allpix = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.median(allpix, axis=0).astype(np.float64)


def foreground_mask(arr: np.ndarray, alpha_threshold: int = 128,
                    bg_dist_threshold: float = 40.0):
    """返回前景二值掩码 (H, W) 与背景类型。

    背景类型："transparent"（有透明通道）或 "opaque"（纯色背景，靠颜色距离分割）。
    """
    alpha = arr[..., 3]
    if int((alpha < 128).sum()) > 0:
        mask = alpha >= alpha_threshold
        return mask, "transparent"

    rgb = arr[..., :3].astype(np.float64)
    bg = estimate_background_color(arr)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    mask = dist >= bg_dist_threshold
    return mask, "opaque"


def laplacian_variance(arr: np.ndarray) -> float:
    """拉普拉斯方差作为清晰度度量：值越大越清晰（无 cv2 实现）。"""
    gray = arr[..., :3].mean(axis=2).astype(np.float64)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def composite_layers(layers) -> np.ndarray:
    """按顺序 alpha-over 合成一组 RGBA 图层，返回 float RGBA (H, W, 4) ∈ [0,1]。

    layers: list[np.ndarray] 每个 (H, W, 4) uint8。
    """
    if not layers:
        raise ValueError("composite_layers requires at least one layer")
    h, w = layers[0].shape[:2]
    acc = np.zeros((h, w, 4), dtype=np.float64)
    for lay in layers:
        rgba = lay.astype(np.float64) / 255.0
        a = rgba[..., 3:4]
        acc = rgba * a + acc * (1.0 - a)
    return acc


def composite_to_uint8(layers) -> np.ndarray:
    """合成并转回 uint8 RGBA。"""
    acc = composite_layers(layers)
    return np.clip(acc * 255.0, 0, 255).astype(np.uint8)
