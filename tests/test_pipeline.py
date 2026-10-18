"""路 B 原型的自包含测试（无网络、无模型、合成 fixture）。"""
import os
import shutil
import unittest

import numpy as np
from PIL import Image

from pipeline.imaging import composite_to_uint8, foreground_mask
from pipeline.run import run
from pipeline.stages.analyze_image import analyze_image
from pipeline.stages.segment_layers import segment
from pipeline.stages.validate_input import validate_input
from pipeline.stages.validate_output import validate_output

# 沙箱只放行工作区写；把临时目录固定到 tests/ 下，避免落到系统 Temp 被拒。
_HERE = os.path.dirname(os.path.abspath(__file__))


def make_fixture(w=256, h=256) -> np.ndarray:
    """合成一张「透明背景 + 4 个纯色前景块」的测试立绘。"""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[30:110, 30:110] = (220, 40, 40, 255)      # 红（模拟脸/身体）
    arr[30:110, 130:220] = (40, 80, 220, 255)     # 蓝（模拟头发）
    arr[130:220, 30:110] = (40, 180, 90, 255)     # 绿（模拟衣服）
    yy, xx = np.ogrid[:h, :w]
    circle = (xx - 170) ** 2 + (yy - 170) ** 2 <= 40 ** 2
    arr[circle] = (230, 210, 60, 255)             # 黄（模拟眼睛/装饰）
    return arr


class PipelineTest(unittest.TestCase):
    def setUp(self):
        # 沙箱下 tempfile.mkdtemp 的随机目录会触发 PermissionError，
        # 改用 os.makedirs 的固定目录（probe 验证可写可删）。
        self.dir = os.path.join(_HERE, "_scratch")
        os.makedirs(self.dir, exist_ok=True)
        self.arr = make_fixture()
        self.png = os.path.join(self.dir, "fixture.png")
        Image.fromarray(self.arr, mode="RGBA").save(self.png)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_foreground_mask_transparent(self):
        mask, kind = foreground_mask(self.arr)
        self.assertEqual(kind, "transparent")
        self.assertGreater(int(mask.sum()), 0)

    def test_validate_input_ok(self):
        v = validate_input(self.png)
        self.assertEqual(v["info"]["width"], 256)
        self.assertEqual(v["info"]["height"], 256)

    def test_validate_input_rejects_psd(self):
        psd = os.path.join(self.dir, "fake.psd")
        with open(psd, "w", encoding="utf-8") as f:
            f.write("not a real psd")
        with self.assertRaises(ValueError) as cm:
            validate_input(psd)
        self.assertIn("INPUT_UNSUPPORTED", str(cm.exception))

    def test_validate_input_missing_file(self):
        with self.assertRaises(ValueError) as cm:
            validate_input(os.path.join(self.dir, "nope.png"))
        self.assertIn("INPUT_INVALID", str(cm.exception))

    def test_analyze(self):
        v = validate_input(self.png)
        rep = analyze_image(v["arr"], v["info"])
        self.assertTrue(rep.allowed)
        self.assertTrue(rep.has_alpha)
        self.assertGreater(rep.subject_ratio, 0.1)

    def test_segment_and_validate(self):
        manifest, arrays = segment(self.arr, K=8)
        self.assertGreaterEqual(len(manifest.layers), 3)
        for e in manifest.layers:
            self.assertTrue(e.label.startswith("unknown_"))
        align = validate_output(self.arr, manifest, arrays)
        self.assertTrue(align.canvas_ok)
        self.assertTrue(align.layers_nonempty)
        self.assertGreaterEqual(align.coverage, 0.99)
        self.assertLess(align.composite_mae, 1.0)

    def test_composite_lossless_on_synthetic(self):
        # 合成图在前景区域应与原图几乎一致（路 B 颜色聚类的正确性基准）
        _, arrays = segment(self.arr, K=8)
        comp = composite_to_uint8(arrays)
        mask, _ = foreground_mask(self.arr)
        diff = np.abs(comp.astype(int) - self.arr.astype(int))[mask].mean()
        self.assertLess(diff, 1.0)

    def test_end_to_end(self):
        out = os.path.join(self.dir, "out")
        rep = run(self.png, out, k=8)
        self.assertIn(rep.status.value, ("succeeded", "needs_review"))
        for fn in ("image_report.json", "layer_manifest.json", "pipeline_report.json"):
            self.assertTrue(os.path.exists(os.path.join(out, fn)))
        self.assertTrue(os.path.exists(os.path.join(out, "preview_composite.png")))
        self.assertGreaterEqual(len(os.listdir(os.path.join(out, "layers"))), 3)


if __name__ == "__main__":
    unittest.main()
