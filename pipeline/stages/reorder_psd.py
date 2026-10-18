"""PSD 图层顺序重排：修正 psd2live 固定排序导致的配件遮挡/穿模。

背景：
- psd2live 的 `WorkspaceLayoutEdits.stack()` 按 **PSD 图层的 `order`** 决定绘制顺序
  （见 `_probe_git/src/main/kotlin/io/github/psd2live/application/WorkspaceLayoutEdits.kt` 第 137 行）。
- 但 see-through 输出的 PSD 图层顺序不一定符合视觉层次
  （例如发箍 `headwear` 可能排在 `front hair` 下面，导致被头发遮住 / 转头时分离）。
- 因此在把 PSD 交给 psd2live **之前**，按视觉层次重排图层，即可修正最终模型的 drawOrder，
  **不需要修改 psd2live 源码**。

用法：
    from pipeline.stages.reorder_psd import reorder_psd
    report = reorder_psd(src_psd, dst_psd)
"""
from __future__ import annotations

import os

# 绘制优先级（从底到顶）：数字越大越靠前（越上层）。
# 顺序原则：后发 → 身体 → 四肢衣物 → 五官 → 前发 → 头饰/配饰。
DRAW_PRIORITY = {
    "back hair": 10,
    "tail": 15,
    "wings": 16,
    "body": 20,
    "neck": 22,
    "legwear": 30,
    "footwear": 32,
    "handwear": 40,
    "bottomwear": 50,
    "topwear": 55,
    "objects": 60,
    "ears": 70,
    "face": 80,
    "nose": 82,
    "mouth": 88,
    "eyebrow": 90,
    "eyewhite": 92,
    "irides": 94,
    "eyelash": 96,
    "front hair": 100,
    # 配件默认在头发之上（发箍/帽子/眼镜应当可见）
    "headwear": 110,
    "earwear": 112,
    "eyewear": 114,
}

UNKNOWN_PRIORITY = 500  # 未知图层放最顶，避免被遮挡而"消失"


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def reorder_psd(src: str, dst: str) -> dict:
    """读取 PSD，按 DRAW_PRIORITY 重排图层（从底到顶）并保存到 dst。

    返回报告：{"before": [...], "after": [...], "moved": [...]}

    说明：保持每个图层的名字、位置偏移、透明度、可见性不变；
    只改变图层堆叠顺序。同优先级图层保持原有相对顺序（稳定排序）。
    """
    from psd_tools import PSDImage

    if not os.path.isfile(src):
        raise ValueError("[LAYER_SCHEMA_INVALID] PSD not found: %s" % src)

    psd = PSDImage.open(src)
    layers = list(psd)
    before = [l.name for l in layers]

    indexed = list(enumerate(layers))
    indexed.sort(key=lambda il: (DRAW_PRIORITY.get(_norm(il[1].name), UNKNOWN_PRIORITY), il[0]))
    ordered = [l for _, l in indexed]
    after = [l.name for l in ordered]

    # 重建 PSD：按新顺序逐层添加（先加的在底层）
    new = PSDImage.new(mode="RGBA", size=psd.size)
    added = 0
    for layer in ordered:
        img = layer.topil()
        if img is None:
            continue
        try:
            new_layer = new.create_pixel_layer(
                img,
                name=layer.name,
                left=layer.left,
                top=layer.top,
                opacity=int(layer.opacity),
            )
            # visible 不是构造参数，是属性
            try:
                new_layer.visible = bool(layer.visible)
            except Exception:
                pass
            added += 1
        except Exception:
            # 单个图层失败不应导致整个流程失败
            continue

    if added == 0:
        raise ValueError("[LAYER_SCHEMA_INVALID] no layer could be re-created")

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    new.save(dst)

    moved = [n for i, n in enumerate(after) if i >= len(before) or before[i] != n]
    return {"before": before, "after": after, "moved": moved}


def draw_order_index(name: str) -> int:
    """查询某图层名在绘制顺序中的位置（越小越靠下）。用于测试与诊断。"""
    return DRAW_PRIORITY.get(_norm(name), UNKNOWN_PRIORITY)
