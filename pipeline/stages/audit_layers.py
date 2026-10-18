"""分层质量审查：以原图为基准，自动检测 see-through 的分层 / 补全错误。

原理（通用，不针对具体图）：
- see-through 会把图层重新合成一张图；**原图中可见的前景像素，理想情况下合成图应与之一致**。
- 若合成图在原图可见区域出现明显色差 → 说明该区域被补全覆盖或被误分到别的图层
  （例：白色发箍被识别成头发，并被 inpainting 补成深蓝色）。
- 把偏差按图层归因，生成差异热力图，给出 verdict（ok / needs_review / failed）。

局限性（如实说明）：
- 只检测「原图可见区域」的偏差。原图中被遮挡、由 see-through 补全的区域**不参与判定**
  （补全本来就会和原图不同，属于合理行为）。
- 颜色相近的误判（例如浅色头发被归到白色配饰）可能检不出。

用法：
    from pipeline.stages.audit_layers import audit_layers
    report = audit_layers(original_png, layered_psd, output_dir)
"""
from __future__ import annotations

import os

import numpy as np

# 差异判定阈值（0-255 的最大通道差）
DIFF_THRESHOLD = 24
# 差异占比阈值：低于 OK_RATIO 视为 ok，高于 REVIEW_RATIO 视为 failed
OK_RATIO = 0.02
REVIEW_RATIO = 0.10


def _get_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def _align_original(orig_rgba: np.ndarray, target_w: int, target_h: int) -> tuple:
    """复现 see-through 的 center_square_pad_resize（黑色 pad + cv2 缩放）。

    返回 (对齐后的 RGB uint8, 对齐后的前景 mask bool)。
    前景 mask 先在**原始尺寸**上计算，再按同样的 pad/resize 处理，避免 pad 颜色干扰。
    """
    from pipeline.imaging import foreground_mask

    # 1. 原始尺寸下的前景
    fg, _ = foreground_mask(orig_rgba)

    # 2. pad 成正方形（黑色，与 see-through 一致）
    rgb = orig_rgba[..., :3].astype(np.uint8)
    h, w = rgb.shape[:2]
    if h != w:
        sz = max(h, w)
        px1, py1 = (sz - w) // 2, (sz - h) // 2
        padded = np.zeros((sz, sz, 3), dtype=np.uint8)
        padded[py1:py1 + h, px1:px1 + w] = rgb
        rgb = padded
        fg_padded = np.zeros((sz, sz), dtype=np.uint8)
        fg_padded[py1:py1 + h, px1:px1 + w] = fg.astype(np.uint8)
        fg_u8 = fg_padded
    else:
        fg_u8 = fg.astype(np.uint8)

    # 3. 缩放到目标尺寸（cv2 插值与 see-through 一致）
    cv2 = _get_cv2()
    if cv2 is not None:
        interp = cv2.INTER_AREA if target_w * target_h < rgb.shape[0] * rgb.shape[1] else cv2.INTER_LINEAR
        rgb_aligned = cv2.resize(rgb, (target_w, target_h), interpolation=interp)
        fg_aligned = cv2.resize(fg_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        # 无 cv2 时退化为 numpy 最近邻缩放
        ys = (np.arange(target_h) * rgb.shape[0] / target_h).astype(int)
        xs = (np.arange(target_w) * rgb.shape[1] / target_w).astype(int)
        rgb_aligned = rgb[ys][:, xs]
        fg_aligned = fg_u8[ys][:, xs] > 0

    return rgb_aligned, fg_aligned


def _composite_layers(layers, canvas_w: int, canvas_h: int) -> np.ndarray:
    """按堆叠顺序 alpha-over 合成图层，返回 uint8 RGBA (H, W, 4)。

    layers: [(name, rgba_array, left, top), ...]，按从底到顶排列。
    """
    acc = np.zeros((canvas_h, canvas_w, 4), dtype=np.float64)
    for _name, arr, left, top in layers:
        h, w = arr.shape[:2]
        y0, x0 = int(top), int(left)
        y1, x1 = min(canvas_h, y0 + h), min(canvas_w, x0 + w)
        if y0 >= canvas_h or x0 >= canvas_w or y1 <= y0 or x1 <= x0:
            continue
        region = arr[: y1 - y0, : x1 - x0].astype(np.float64) / 255.0
        canvas_layer = np.zeros((canvas_h, canvas_w, 4), dtype=np.float64)
        canvas_layer[y0:y1, x0:x1] = region
        a = canvas_layer[..., 3:4]
        acc = canvas_layer * a + acc * (1.0 - a)
    return np.clip(acc * 255.0, 0, 255).astype(np.uint8)


def _layer_mask(arr: np.ndarray, left: int, top: int, canvas_w: int, canvas_h: int) -> np.ndarray:
    """把图层 alpha 放到全画布坐标系，返回布尔 mask。"""
    h, w = arr.shape[:2]
    m = np.zeros((canvas_h, canvas_w), dtype=bool)
    y0, x0 = int(top), int(left)
    y1, x1 = min(canvas_h, y0 + h), min(canvas_w, x0 + w)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = arr[: y1 - y0, : x1 - x0, 3] > 20
    return m


def _find_suspect_regions(diff: np.ndarray, fg_mask: np.ndarray, block_size: int = 16,
                          percentile: float = 85.0, min_blocks: int = 3) -> list:
    """分块检测高差异区域（成片异常，排除边缘抗锯齿等分散噪声）。

    返回 [{"bbox": [x0,y0,x1,y1], "blocks": n, "avg_diff": float}, ...]（按严重度排序）。
    """
    H, W = diff.shape
    bs = block_size
    bh, bw = H // bs, W // bs
    if bh == 0 or bw == 0:
        return []

    dh, dw = bh * bs, bw * bs
    diff_c = diff[:dh, :dw]
    fg_c = fg_mask[:dh, :dw]

    # 向量化分块统计
    diff_blocks = diff_c.reshape(bh, bs, bw, bs).mean(axis=(1, 3))
    fg_ratio = fg_c.reshape(bh, bs, bw, bs).mean(axis=(1, 3))

    valid = fg_ratio > 0.30  # 块内至少 30% 是前景才参与判定
    if not valid.any():
        return []

    vals = diff_blocks[valid]
    if len(vals) < 10:
        return []

    thr = max(30.0, float(np.percentile(vals, percentile)))
    high = valid & (diff_blocks > thr)
    if not high.any():
        return []

    # 连通域聚类（找成片区域）
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(high)
    except Exception:
        lbl, n = None, 0

    regions = []
    if lbl is not None and n > 0:
        sizes = ndimage.sum(high, lbl, index=np.arange(1, n + 1))
        for i in range(n):
            size = int(sizes[i])
            if size < min_blocks:
                continue
            m = lbl == (i + 1)
            yy, xx = np.nonzero(m)
            regions.append({
                "bbox": [int(xx.min()) * bs, int(yy.min()) * bs,
                         (int(xx.max()) + 1) * bs, (int(yy.max()) + 1) * bs],
                "blocks": size,
                "avg_diff": round(float(diff_blocks[m].mean()), 1),
            })
        regions.sort(key=lambda r: -(r["blocks"] * r["avg_diff"]))
    else:
        # scipy 不可用时退化为整块报告
        yy, xx = np.nonzero(high)
        if len(yy):
            regions.append({
                "bbox": [int(xx.min()) * bs, int(yy.min()) * bs,
                         (int(xx.max()) + 1) * bs, (int(yy.max()) + 1) * bs],
                "blocks": int(high.sum()),
                "avg_diff": round(float(diff_blocks[high].mean()), 1),
            })

    return regions


def audit_layers(original_path: str, psd_path: str, output_dir: str) -> dict:
    """审查分层质量，返回报告（并写出差异热力图）。"""
    from PIL import Image
    from psd_tools import PSDImage

    os.makedirs(output_dir, exist_ok=True)

    # 1. 原图（原始尺寸）
    orig = np.array(Image.open(original_path).convert("RGBA"))

    # 2. PSD 图层（按 PSD 堆叠顺序 = 从底到顶）
    psd = PSDImage.open(psd_path)
    W, H = psd.size
    layers = []
    for layer in psd:
        img = layer.topil()
        if img is None:
            continue
        layers.append((layer.name, np.array(img.convert("RGBA")), int(layer.left), int(layer.top)))

    if not layers:
        raise ValueError("[LAYER_SCHEMA_INVALID] PSD has no readable layers: %s" % psd_path)

    # 3. 原图对齐到 PSD 画布（复现 see-through 的 pad+resize），并得到对齐后的前景 mask
    orig_rgb, fg_mask = _align_original(orig, W, H)

    # 4. 合成
    composite = _composite_layers(layers, W, H)

    # 5. 差异（只看 RGB；只在原图可见前景区域判定）
    orig_rgb_i = orig_rgb.astype(np.int16)
    comp_rgb_i = composite[..., :3].astype(np.int16)
    diff = np.abs(orig_rgb_i - comp_rgb_i).max(axis=2)
    diff_mask = (diff > DIFF_THRESHOLD) & fg_mask

    # 6. 按图层归因
    layer_issues = []
    for name, arr, left, top in layers:
        m = _layer_mask(arr, left, top, W, H)
        overlap = int((diff_mask & m).sum())
        if overlap > 0:
            layer_issues.append({
                "layer": name,
                "diff_pixels": overlap,
                "layer_pixels": int(m.sum()),
                "diff_ratio_in_layer": round(overlap / max(1, int(m.sum())), 4),
            })
    layer_issues.sort(key=lambda x: -x["diff_pixels"])

    # 7. 差异热力图（红 = 与原图偏差大）
    heat = np.zeros((H, W, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(diff * 3, 0, 255).astype(np.uint8)
    heat[~fg_mask] = 0
    heat_path = os.path.join(output_dir, "audit_diff.png")
    Image.fromarray(heat).save(heat_path)

    total_fg = int(fg_mask.sum())
    total_diff = int(diff_mask.sum())
    ratio = total_diff / max(1, total_fg)

    # 8. 分块检测可疑区域（成片差异，排除边缘噪声）
    suspect_regions = _find_suspect_regions(diff, fg_mask)

    # 严重度分级
    for r in suspect_regions:
        r["severity"] = "high" if (r["blocks"] >= 8 or r["avg_diff"] >= 150) else "medium"

    suspect_area = 0
    for r in suspect_regions:
        x0, y0, x1, y1 = r["bbox"]
        suspect_area += (x1 - x0) * (y1 - y0)
    suspect_ratio = suspect_area / max(1, total_fg)

    high_count = sum(1 for r in suspect_regions if r["severity"] == "high")

    # 判定：有高严重度区域 -> failed；只有中低 -> needs_review；无 -> ok
    if not suspect_regions:
        verdict = "ok"
    elif high_count > 0:
        verdict = "failed"
    else:
        verdict = "needs_review"

    # 只保留最可疑的若干个报告，避免噪音
    top_regions = suspect_regions[:10]

    # 9. 在热力图上画出可疑区域边框（便于人工定位）
    try:
        from PIL import ImageDraw
        img = Image.open(heat_path)
        draw = ImageDraw.Draw(img)
        for r in top_regions:
            color = (0, 255, 255) if r["severity"] == "high" else (255, 255, 0)
            draw.rectangle(r["bbox"], outline=color, width=3)
        img.save(heat_path)
    except Exception:
        pass

    return {
        "verdict": verdict,
        "suspect_regions": top_regions,
        "suspect_region_count": len(suspect_regions),
        "high_severity_count": high_count,
        "suspect_area_ratio": round(suspect_ratio, 4),
        "diff_ratio": round(ratio, 4),
        "diff_pixels": total_diff,
        "foreground_pixels": total_fg,
        "heatmap": heat_path,
        "layer_issues": layer_issues,
        "note": "severity=high 的区域建议人工检查；热力图中青色=high、黄色=medium",
    }
