# 决策记录 0007：See-through 引擎验证成功（核心突破）

- **日期**：2026-10-17
- **状态**：**已验证通过**——单张动漫立绘 → 17 层专业语义 PSD（含遮挡补全 + 深度图）
- **结论**：`shitagaki-lab/see-through` 是本项目的**核心技术引擎**，取代此前的 face-parsing 方案

---

## 一、这是什么

[`shitagaki-lab/see-through`](https://github.com/shitagaki-lab/see-through)——ACM SIGGRAPH 2026 论文
《Single-image Layer Decomposition for Anime Characters》，4663 stars，2026-10 仍在活跃维护。

> 官方描述：把单张动漫插画分解为**完全补全的、语义独立的图层**，并推断绘制顺序——
> 最多 **23 层**，含 hair / face / eyes / clothing / accessories 等。

其定位与我们 `docs/decisions/0001` 的判断**完全一致**：
> "our decomposition can serve as a useful **starting point** for Live2D artists by eliminating
> some of the most tedious part of the workflow, such as manual segmentation and occluded region inpainting."

---

## 二、环境配置（Windows + RTX 4060 8GB）

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| PyTorch | **2.11.0+cu128**（CUDA 可用，compute 8.9） |
| transformers | **5.0.0**（严格按需求，5.20 会破坏 offload） |
| diffusers | 0.37.0 |
| bitsandbytes | 0.50.2 |
| 其他 | timm 1.0.30 / kornia 0.8.3 / omegaconf / einops / accelerate / psd-tools 1.25 |

**关键配置**：
```powershell
$env:PYTHONPATH = "<项目>\.deps;<see-through>\common;<see-through>\annotators"
$env:HF_HOME    = "<项目>\.tools\hf-cache"
# common/annotators 用命名空间包，PYTHONPATH 即可，无需 pip -e
```

**8GB 显存的正确姿势**：
- ❌ `inference_psd_quantized.py`（NF4 + cpu_offload）→ 失败：
  - 不开启 cpu_offload 时 `CUBLAS_STATUS_ALLOC_FAILED`（显存不足）
  - 开启后 `Expected all tensors to be on the same device`（offload 的设备分片与当前 torch/diffusers 组合不兼容）
- ✅ **`inference_psd_blockswap.py`**（bf16 + block swap，官方为 8GB 卡设计）——**成功**

---

## 三、实测结果（官方示例图 `common/assets/test_image.png`）

```bash
python inference/scripts/inference_psd_blockswap.py \
  --srcp common/assets/test_image.png --save_to_psd --resolution 1024
```

### 耗时（RTX 4060 8GB）
| 阶段 | 耗时 |
|---|---|
| LayerDiff 3D（30 步） | **3 分 32 秒**（7.07 s/it） |
| Marigold 深度（4 步） | **11 秒**（2.85 s/it） |
| PSD 组装 | 数秒 |
| **合计** | **约 4 分钟/张**（首次含模型下载约 20 分钟） |

### 输出：17 层专业语义 PSD

```
test_image.psd (5.0 MB, 1024×1024)
  back hair, front hair, face, neck, ears, nose, mouth,
  eyebrow, eyelash, eyewhite, irides, eyewear,
  topwear, bottomwear, handwear, legwear, footwear
```

**图层命名完全符合 Live2D 标准**（对应 v0.3 文档 4.2 标签体系）：
- `front hair` / `back hair` —— 头发前后已分离
- `eyewhite` / `irides` / `eyelash` —— 眼睛分件
- `topwear` / `bottomwear` / `handwear` / `legwear` / `footwear` —— 服装分类
- `face` / `nose` / `mouth` / `eyebrow` / `ears` —— 五官齐全

### 额外产物
- `test_image_depth.psd` —— 带深度的 PSD（2.5D 排序用）
- 每层独立 PNG + 对应 `*_depth.png`（深度图）
- `reconstruction.png` —— 重建图（已验证与原图一致，零丢失）
- `info.json` —— 元信息

### 与 face-parsing 方案对比

| 维度 | face-parsing（0006） | **see-through（0007）** |
|---|---|---|
| 图层数 | 4–16（多为颜色块） | **17 专业语义层** |
| 前后发分离 | ❌ | ✅ |
| 眼睛分件 | ❌ 只有整眼 | ✅ eyewhite/irides/eyelash |
| 服装分类 | ❌ 只有一个 clothes | ✅ 5 类 |
| 遮挡补全 | ❌ | ✅ **自动补全（fully inpainted）** |
| 深度图 | ❌ | ✅ 每层都有 |
| 输出格式 | PNG + manifest | **PSD（可直接导入编辑器）** |
| 耗时（4060） | 53 ms | 约 4 分钟 |

---

## 四、项目架构的调整

**see-through 成为核心引擎**，我们的代码定位改为**编排层 + 后处理层**：

```
用户立绘
  ↓
[我们的 pipeline] 输入校验 / 任务状态 / 错误码 / 报告
  ↓
[see-through blockswap] 单图 → 17 层语义 PSD（含补全）
  ↓
[我们的后处理] 标签规范化 / 图层清单 / 质量门禁 / 来源追溯
  ↓
交付：分层 PSD + manifest + 报告
```

**我们的价值不在模型**，而在：
1. 把它包装成可批量、可诊断、可重试的生产流水线
2. 标签标准化 + 图层清单 + 质量门禁
3. 面向画师/建模师的交付形态（规范命名、可直接导入）
4. 成本/耗时记录，为定价提供依据

---

## 五、待办与风险

| 项 | 说明 |
|---|---|
| **许可证** | see-through 为研究项目（LICENSE 11.1 KB，需审查）。商用前必须确认 |
| **模型许可** | LayerDiff3D（SDXL 基础）、Marigold、SAM 各有许可，需一并审查 |
| **磁盘** | hf-cache 当前 **15.5 GB**，加上项目其他依赖，需预留 30 GB+ |
| **分辨率** | 1024 稳定；1280 在 8GB 上未验证（官方称 1280 需 8GB 极限） |
| **批量** | 单张约 4 分钟，批量需队列；模型常驻可用时后续更快 |
| **动漫质量** | 官方示例效果好；需用真实立绘验证（白发/白围裙等难例） |

---

## 六、结论

**项目最核心的技术风险已消除**：从"一张立绘 → 可交付的分层素材"这条链路，
现在有**开源、可复现、在 8GB 消费级显卡上跑通**的完整实现。

v0.3 文档第 14 章的核心问题已有答案：
- 分层工具能否稳定生成语义正确的图层？→ **✅ 能（see-through）**
- 分层 PSD 的命名/结构是否满足绑定要求？→ **✅ 17 层标准命名**
- 自动绑定是否支持无 GUI 批处理？→ **✅ psd2live CLI（0005）**
- 输出能否在目标软件打开？→ 待验证（见 0005 待办）

---

## 七、端到端串联验证成功（2026-10-17 追加）★★★

**把 see-through 输出的 17 层 PSD 直接喂给 psd2live，一次成功。**

```powershell
# 1) 立绘 → 17 层语义 PSD（约 4 分钟，见第三节）
python inference/scripts/inference_psd_blockswap.py --srcp 立绘.png --save_to_psd --resolution 1024

# 2) 分层 PSD → 完整 Live2D 模型（14 秒）
gradlew.bat run --args="--input ftv_test.psd --output build/ftv-output"
```

### 结果：12 个模型文件，BUILD SUCCESSFUL in 14s

| 文件 | 大小 | 说明 |
|---|---|---|
| `ftv_test.moc3` | 247 KB | 运行时模型 |
| `ftv_test.cmo3` | 1.9 MB | 可编辑工程 |
| `ftv_test.model3.json` | 741 B | 入口清单 |
| `ftv_test.physics3.json` | 8.6 KB | 物理 |
| `ftv_test.cdi3.json` | 4.1 KB | 参数显示 |
| `ftv_test.4096/texture_00.png` | 872 KB | 纹理图集 |
| `idle/blink/nod/shake.motion3.json` | — | 4 个动作 |

### 模型质量数据（psd2live 诊断报告）

| 指标 | 数值 |
|---|---|
| drawable（图形网格） | **19** |
| deformer（变形器） | **19** |
| parameter（参数） | **20**（AngleXYZ / BodyAngleXYZ / BodyLean / Proportion / EyeLOpen/ROpen / EyeBallXY / EyeBallForm / BrowLY/RY …） |
| 纹理图集 | 1 页 4096 |
| 运行时目标 | Cubism 5.0（mocVersion 5） |
| 面部绑定算法 | `perspective-parallelogram-nine-pose-v2`（**九姿势透视绑定**） |
| 变形器层级 | head / face / frontHair / backHair |
| 物理组 | `PhysicsHairFront` / `PhysicsHairBack` / `PhysicsEyeJelly` |
| 动作 | idle / blink / nod / shake |

**`model3.json` 的 `EyeBlink` / `LipSync` 组均正确填充**（`ParamEyeLOpen`/`ParamEyeROpen`/`ParamMouthOpenY`），
VTube Studio 面捕具备基础条件。

### 关键结论

1. **命名兼容性零问题**：see-through 的图层名（`front hair` 带空格、`eyewhite` 无左右后缀）被 psd2live 直接接受并正确分类为 `ArtMeshBackHair` 等。
2. **无需写命名映射层**——原以为的最大风险点不存在。
3. **端到端耗时**：**约 4 分钟（分层）+ 14 秒（绑定额导出）≈ 4–5 分钟/张**。

### 踩坑记录：Java ImageIO 临时目录

psd2live 首次处理 see-through PSD 时报 `javax.imageio.IIOException: Can't create cache file`
（`AccessDeniedException: AppData\Local\Temp`）。**必须在运行 psd2live 前把 Java 临时目录指到工作区**：

```powershell
$env:JAVA_TOOL_OPTIONS = "-Djava.io.tmpdir=<工作区>\.tools\javatem"
$env:TEMP = "<工作区>\.tools\javatem"
$env:TMP  = "<工作区>\.tools\javatem"
```

否则 `AtlasPagePng` 在生成纹理图集时失败。

### 仍未验证
- **VTube Studio 实际加载**（把 `ftv-output` 放进 `Live2DModels` 测试）
- **绑定质量主观评估**（转头/眨眼/口型是否自然，需人工目视）
- 真实立绘（非示例图）的表现
