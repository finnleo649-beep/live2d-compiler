"""阶段 3'：语义分层（路 A：AI 语义分割 + 身体颜色聚类混合）。

策略（针对 face-parsing 的实测限制，见 docs/decisions/0006）：
- face-parsing 在真人脸数据上训练，**只擅长头部/上半身**；整张全身图送进去会严重误判。
- 因此：**头部区域（人物 bbox 顶部 head_ratio）用 AI 语义分割**，得到 hair/face/eye/brow/mouth 等标准标签；
- **身体/腿部区域用颜色聚类**兜底，保证零丢失且仍有可用的初始分层。

诚实边界：
- 头发前后未区分（统一 hair_front）；
- 身体图层是颜色聚类，不是语义（命名 body_color_*）；
- 动漫风格 + 训练数据差异会导致部分误判，如实标记 confidence。
- 零丢失铁律：前景中未被覆盖的像素归入 unknown_foreground。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from schemas.layer_manifest import LABEL_VERSION, LayerEntry, LayerManifest
from pipeline.imaging import foreground_mask

# 19 类 -> 标准标签（v0.3 文档 4.2）。None 表示丢弃（background 由前景 mask 兜底）。
FACE_PARSING_TO_LABEL = {
    0: None,                  # background
    1: "face_base",           # skin
    2: "face_base",           # nose（鼻子并入脸）
    3: "accessory_glasses",   # eye_g
    4: "eye_L_base",          # l_eye
    5: "eye_R_base",          # r_eye
    6: "brow_L",              # l_brow
    7: "brow_R",              # r_brow
    8: "ear_L",               # l_ear
    9: "ear_R",               # r_ear
    10: "mouth_cavity",       # mouth
    11: "mouth_upper_lip",    # u_lip
    12: "mouth_lower_lip",    # l_lip
    13: "hair_front",         # hair
    14: "accessory_hat",      # hat
    15: "accessory_earring",  # ear_r
    16: "accessory_necklace", # neck_l
    17: "body",               # neck
    18: "clothes_base",       # cloth
}

_MODEL_CACHE = {}


def _single_character_bbox(fg_mask: np.ndarray):
    """用连通域分析找单个主要角色的 bbox（面积最大的连通分量）。"""
    from scipy import ndimage

    h, w = fg_mask.shape
    labeled, n = ndimage.label(fg_mask)
    if n == 0:
        return 0, 0, w - 1, h - 1
    sizes = ndimage.sum(fg_mask, labeled, index=np.arange(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labeled == best)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _load_model(model_id: str = "jonathandinu/face-parsing", device: Optional[str] = None):
    """懒加载模型（只有语义后端才要求装 torch）。"""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    proc = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    _MODEL_CACHE[model_id] = (proc, model, device)
    return _MODEL_CACHE[model_id]


def _predict(proc, model, device, crop_rgb: np.ndarray):
    """对裁剪后的 RGB 图推理，返回 (pred_cls, confidence)。"""
    import torch
    import torch.nn.functional as F

    inputs = proc(images=crop_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    up = F.interpolate(
        logits, size=(crop_rgb.shape[0], crop_rgb.shape[1]),
        mode="bilinear", align_corners=False,
    )
    prob = torch.softmax(up, dim=1)[0].cpu().numpy()
    return prob.argmax(axis=0), prob.max(axis=0)


def _postprocess_labels(label_mask, label_conf_sum, label_conf_cnt, head_fg_area,
                        head_upper, h, w):
    """修正模型明显误判，返回 (label_mask, notes)。

    规则（依据 docs/decisions/0006 的实测局限）：
    1. 大面积 + 低置信度的 `accessory_hat`：**头部范围**内重分配为 `hair_front`，
       其余部分归 `unknown_foreground`。头部范围优先用 `face_base` 的实际 bbox 精确定位
       （脸顶 - 0.9×脸高 ~ 脸底 + 0.25×脸高），避免把上半身衣服吞进头发层。
    2. 极小面积的配件（glasses / earring / necklace）→ 合并到相邻主图层。
    """
    notes = []

    # 规则 1：hat 误判（上下切分）
    if "accessory_hat" in label_mask:
        area = int(label_mask["accessory_hat"].sum())
        cnt = label_conf_cnt.get("accessory_hat", 0)
        conf = (label_conf_sum.get("accessory_hat", 0.0) / cnt) if cnt else 0.0
        if area > head_fg_area * 0.25 and conf < 0.7:
            hat = label_mask.pop("accessory_hat")
            # 用传入的 head_upper（头部区域的上部）切分：
            # 上部归头发，下部归 unknown，避免把上半身衣服吞进头发层。
            upper = hat & head_upper
            lower = hat & ~head_upper
            moved_up = int(upper.sum())
            moved_low = int(lower.sum())
            if bool(upper.any()):
                if "hair_front" in label_mask:
                    label_mask["hair_front"] = label_mask["hair_front"] | upper
                else:
                    label_mask["hair_front"] = upper
            if bool(lower.any()):
                if "unknown_foreground" in label_mask:
                    label_mask["unknown_foreground"] = label_mask["unknown_foreground"] | lower
                else:
                    label_mask["unknown_foreground"] = lower
            notes.append(
                "accessory_hat: %d px -> hair_front, %d px -> unknown_foreground "
                "(area=%d, conf=%.2f)" % (moved_up, moved_low, area, conf)
            )
            label_conf_sum.pop("accessory_hat", None)
            label_conf_cnt.pop("accessory_hat", None)

    # 规则 2：小配件合并
    small_merge = (
        ("accessory_glasses", "face_base"),
        ("accessory_earring", "hair_front"),
        ("accessory_necklace", "clothes_base"),
    )
    for small_label, target in small_merge:
        if small_label not in label_mask:
            continue
        area = int(label_mask[small_label].sum())
        if area < head_fg_area * 0.01:
            m = label_mask.pop(small_label)
            if target in label_mask:
                label_mask[target] = label_mask[target] | m
            else:
                label_mask[target] = m
            notes.append("%s->%s (small area=%d)" % (small_label, target, area))
            label_conf_sum.pop(small_label, None)
            label_conf_cnt.pop(small_label, None)

    return label_mask, notes


def segment_semantic(arr: np.ndarray, output_dir: Optional[str] = None,
                     model_id: str = "jonathandinu/face-parsing",
                     head_ratio: float = 0.5, body_colors: int = 8):
    """混合分层：头部语义分割 + 身体颜色聚类。

    返回 (LayerManifest, list[np.ndarray])。
    """
    from PIL import Image

    h, w = arr.shape[:2]
    fg_mask, bg_kind = foreground_mask(arr)
    if not bool(fg_mask.any()):
        raise ValueError("[SEGMENTATION_FAILED] no foreground detected")

    x0, y0, x1, y1 = _single_character_bbox(fg_mask)
    char_h = y1 - y0 + 1
    head_y1 = min(y1, y0 + max(1, int(char_h * head_ratio)) - 1)

    label_mask = {}
    label_conf_sum = {}
    label_conf_cnt = {}

    # ---------- 1. 头部：AI 语义分割 ----------
    head_crop = arr[y0:head_y1 + 1, x0:x1 + 1, :3].astype(np.uint8)
    proc, model, device = _load_model(model_id)
    pred, conf = _predict(proc, model, device, head_crop)

    for cls_id, label in FACE_PARSING_TO_LABEL.items():
        if label is None:
            continue
        m = (pred == cls_id)
        if not m.any():
            continue
        full = np.zeros((h, w), dtype=bool)
        full[y0:head_y1 + 1, x0:x1 + 1] = m
        label_mask[label] = label_mask.get(label, np.zeros((h, w), dtype=bool)) | full
        label_conf_sum[label] = label_conf_sum.get(label, 0.0) + float(conf[m].sum())
        label_conf_cnt[label] = label_conf_cnt.get(label, 0) + int(m.sum())

    # 头部区域内未被语义覆盖的前景 -> unknown_foreground
    head_region = np.zeros((h, w), dtype=bool)
    head_region[y0:head_y1 + 1, x0:x1 + 1] = True
    covered = np.zeros((h, w), dtype=bool)
    for m in label_mask.values():
        covered |= m
    unk_head = (fg_mask & head_region) & ~covered
    if bool(unk_head.any()):
        label_mask["unknown_foreground"] = unk_head

    # 后处理：修正模型明显误判（规则见 _postprocess_labels）
    head_fg_area = int((fg_mask & head_region).sum())
    head_upper = np.zeros((h, w), dtype=bool)
    upper_y1 = y0 + max(1, int((head_y1 - y0 + 1) * 0.5)) - 1
    head_upper[y0:upper_y1 + 1, x0:x1 + 1] = True
    label_mask, postprocess_notes = _postprocess_labels(
        label_mask, label_conf_sum, label_conf_cnt, head_fg_area, head_upper, h, w
    )

    # ---------- 2. 身体：颜色聚类 ----------
    body_fg = fg_mask & ~head_region
    if bool(body_fg.any()):
        rgb = Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")
        q = rgb.quantize(colors=body_colors, method=Image.Quantize.MEDIANCUT)
        qarr = np.asarray(q, dtype=np.uint8)
        for idx in np.unique(qarr):
            m = (qarr == idx) & body_fg
            if int(m.sum()) < 64:
                continue
            label = "body_color_%02d" % int(idx)
            label_mask[label] = m
            label_conf_cnt[label] = 0

    # ---------- 3. 生成图层（按面积降序） ----------
    ordered = sorted(label_mask.items(), key=lambda kv: -int(kv[1].sum()))
    layers = []
    layer_arrays = []
    for i, (label, m) in enumerate(ordered):
        layer_id = label
        file_name = layer_id + ".png"
        layer_arr = np.zeros_like(arr)
        layer_arr[m] = arr[m]
        if output_dir:
            Image.fromarray(layer_arr, mode="RGBA").save(
                os.path.join(output_dir, file_name)
            )
        area = int(m.sum())
        ys2, xs2 = np.nonzero(m)
        bbox = (int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max()))
        mean = tuple(float(v) for v in arr[m][:, :3].mean(axis=0))
        cnt = label_conf_cnt.get(label, 0)
        avg_conf = round(label_conf_sum.get(label, 0.0) / cnt, 4) if cnt else None
        if label == "unknown_foreground":
            source = "foreground_fallback"
        elif label.startswith("body_color_"):
            source = "color_cluster"
        else:
            source = "face_parsing"
        layers.append(LayerEntry(
            layer_id=layer_id, label=label, label_version=LABEL_VERSION,
            file=file_name, area_pixels=area, bbox=bbox, mean_color=mean,
            source=source, confidence=avg_conf,
        ))
        layer_arrays.append(layer_arr)

    manifest = LayerManifest(
        version=LABEL_VERSION, canvas_width=w, canvas_height=h,
        background_kind=bg_kind, layers=layers,
        postprocess_notes=postprocess_notes,
    )
    return manifest, layer_arrays


def render_label_map(arr: np.ndarray, output_path: str,
                     model_id: str = "jonathandinu/face-parsing",
                     head_ratio: float = 0.5) -> str:
    """渲染头部语义 label map（调试/可视化用）。"""
    from PIL import Image

    h, w = arr.shape[:2]
    fg_mask, _ = foreground_mask(arr)
    x0, y0, x1, y1 = _single_character_bbox(fg_mask)
    char_h = y1 - y0 + 1
    head_y1 = min(y1, y0 + max(1, int(char_h * head_ratio)) - 1)

    proc, model, device = _load_model(model_id)
    pred, _ = _predict(proc, model, device, arr[y0:head_y1 + 1, x0:x1 + 1, :3].astype(np.uint8))

    rng = np.random.default_rng(42)
    palette = rng.integers(40, 255, size=(32, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[y0:head_y1 + 1, x0:x1 + 1] = palette[pred]
    canvas[~fg_mask] = [0, 0, 0]
    Image.fromarray(canvas).save(output_path)
    return output_path
