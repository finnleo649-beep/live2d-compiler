# Live2D Compiler

**一张动漫立绘 → 完整可加载的 Live2D 模型**（半自动，约 4–5 分钟/张）

面向**画师 / 建模师**的立绘预处理与自动绑定工具：把静态立绘自动拆成语义图层、推断遮挡补全、生成网格与参数绑定，输出可直接导入 Cubism Editor 或 VTube Studio 的模型文件。

> **定位说明**：这是**半自动工具**，不是"一键出商业级皮套"。自动产物是**可用的起点**，精细调整仍需在 Cubism Editor 里完成。详见 [`docs/decisions/0001`](docs/decisions/0001-定位调整-面向专业用户.md)。

---

## 它能做什么

| 输入 | 输出 |
|---|---|
| 一张动漫角色立绘（PNG，单人，清晰） | 17 层语义分层 PSD + 深度图 |
| | `.moc3` 运行时模型 |
| | `.cmo3` 可编辑工程 |
| | `.model3.json` / `.physics3.json` / `.cdi3.json` |
| | 4 个动作（idle / blink / nod / shake） |
| | 物理（前发 / 后发 / 眼珠果冻） |
| | 质量报告（标注需要人工修的地方） |

**耗时参考**（RTX 4060 Laptop 8GB）：分层约 4 分钟 + 绑定导出约 20 秒。

---

## 快速开始

> 前提：已按下方「环境搭建」完成安装。

### 方式 1：拖图（最简单）

把 PNG 立绘拖到 **`生成模型.bat`** 上，松开即可。

### 方式 2：命令行

```bash
python generate_model.py 你的立绘.png
```

跑完会自动：
1. 把模型复制到 VTube Studio 的 `Live2DModels/` 目录（路径见 `generate_model.py` 里的 `VTS`）
2. 弹出结果文件夹
3. 显示质量结论

**VTube Studio 里的模型名**：`gen_XXXXXX`（6 位数字）。

### 输入图要求

- ✅ 单个动漫角色、正面或半侧、清晰
- ✅ 透明背景或纯色背景
- ❌ 多人并排、复杂场景背景、真人照片、草图

---

## 环境搭建（首次）

```bash
python setup.py
```

脚本会自动完成 5 步（已完成的部分会跳过，可反复运行）：

| 步骤 | 内容 | 体积 |
|---|---|---|
| 1 | 下载 JDK 21（免安装版） | ~200 MB |
| 2 | 下载 psd2live 源码 | ~50 MB |
| 3 | 下载 see-through 源码 | ~10 MB |
| 4 | 安装 Python 依赖（torch CUDA + 其余） | ~5 GB |
| 5 | 下载 AI 模型（LayerDiff3D + Marigold） | ~12.5 GB |

**总计约 18 GB 下载**，首次耗时 30–90 分钟（取决于网速）。

> 需要 NVIDIA 显卡（CUDA）。本项目在 RTX 4060 Laptop 8GB 上验证通过。

---

## 技术架构

本项目是**编排层 + 后处理层**，核心能力来自两个开源引擎：

```
立绘 PNG
   ↓
┌─────────────────────────────────────────┐
│ see-through（SIGGRAPH 2026）             │
│ 单图 → 语义分层（LayerDiff3D + Marigold） │
└─────────────────────────────────────────┘
   ↓ 17 层语义 PSD（含遮挡补全 + 深度图）
┌─────────────────────────────────────────┐
│ psd2live 3.5.0（Kotlin）                 │
│ 分层 PSD → 网格 / 变形器 / 参数 / 导出    │
└─────────────────────────────────────────┘
   ↓
完整 Live2D 模型（moc3 / cmo3 / model3.json / physics3 / motion3）
   ↓
本项目的质量门禁 + 交付封装
```

**本项目自己的代码**（`pipeline/`、`schemas/`、`generate_model.py`）负责：
- 一键编排（调用两个引擎）
- 环境适配（8GB 显存、Windows 路径、Java 临时目录）
- 质量门禁（自动标注漏层 / 顺序问题 / 关键层缺失）
- 交付格式（复制到 VTS、生成报告）

---

## 目录结构

```
live2d-compiler/
├─ pipeline/                    # 备选分层后端（颜色聚类 / 语义分割）
│  ├─ imaging.py                # 前景分割 / 合成 等基础图像工具
│  ├─ run.py                    # CLI 流水线
│  └─ stages/
│     ├─ validate_input.py      # 输入校验
│     ├─ analyze_image.py       # 图片分析
│     ├─ segment_layers.py      # 颜色聚类分层（无模型，快）
│     ├─ segment_semantic.py    # face-parsing 语义分层（备选）
│     └─ validate_output.py     # 输出对齐校验
├─ schemas/                     # 数据契约（job / layer_manifest / report）
├─ tests/                       # 单元测试（合成 fixture，自包含）
├─ docs/decisions/              # 设计与实测决策记录（0001–0008）
├─ generate_model.py            # ★ 一键生成（主入口）
├─ 生成模型.bat                  # ★ 双击入口
├─ setup.py                     # 环境搭建
├─ run.bat / test.bat           # 备选后端 / 单元测试
├─ README.md
│
│  ── 以下由 setup.py 下载，不入仓库 ──
├─ .deps/                       # Python 依赖（约 5 GB）
├─ .tools/                      # JDK / Gradle / AI 模型缓存（约 14 GB）
├─ _probe_see_through/          # see-through 源码
└─ _probe_git/                  # psd2live 源码
```

---

## 已知限制

| 限制 | 说明 |
|---|---|
| **绑定精度** | 自动绑定是程序化生成，转头/眨眼精度**低于手工绑定**（行业现状） |
| **眨眼** | see-through 不输出独立眼睑层，眨眼效果为近似 |
| **图层顺序** | psd2live 按固定 tag 顺序排列，少数配饰（如发箍）可能被头发遮挡 |
| **输入敏感** | 多人图、复杂背景会显著降低分层质量 |
| **分辨率** | 验证于 1024px；1280px 在 8GB 显存上未验证 |
| **显存** | 需约 8 GB（使用 blockswap 管线） |

这些限制在 [`docs/decisions/`](docs/decisions/) 中有详细实测记录。

---

## 许可证与商用注意 ⚠️

本项目**自身代码**可自由使用，但依赖的外部引擎各有许可，**商用前必须逐项审查**：

| 依赖 | 许可 | 注意 |
|---|---|---|
| [see-through](https://github.com/shitagaki-lab/see-through) | 研究项目（需审查 LICENSE） | 含 LayerDiff3D / Marigold 模型权重，另有许可 |
| [psd2live](https://github.com/tsunehimatoi/psd2live) | **GPL-3.0** | 作为独立子进程调用；集成进商业服务需法律审查 |
| Live2D Cubism | 官方商业授权 | 模型发布/分发需确认授权条款 |

> 本项目**不包含**上述引擎的源码（由 `setup.py` 下载），以保持仓库干净并隔离许可证。

---

## 设计决策

关键技术决策与实测结果记录在 [`docs/decisions/`](docs/decisions/)：

| 文档 | 内容 |
|---|---|
| [0001](docs/decisions/0001-定位调整-面向专业用户.md) | 定位：面向画师/建模师，`.moc3` 降级为可选 |
| [0002](docs/decisions/0002-参考模型结构分析.md) | 商业成品皮套的结构分析（19 网格 / 76 参数 / 5 物理链） |
| [0003](docs/decisions/0003-验收标准.md) | 验收标准：工时节省率 / 修正率 / 零返工 |
| [0004](docs/decisions/0004-文档版本递增规则.md) | 文档版本规则 |
| [0005](docs/decisions/0005-阶段0尽调实测结果.md) | psd2live CLI 实测（22 秒生成 12 文件） |
| [0006](docs/decisions/0006-路A语义分割能力验证.md) | face-parsing 语义分割验证 + 后处理规则 |
| [0007](docs/decisions/0007-See-through引擎验证成功.md) | see-through 部署与 8GB 适配（含踩坑记录） |
| [0008](docs/decisions/0008-全链路验证结果汇总.md) | 端到端验证：立绘 → 模型全链路 |

---

## 测试

```bash
python -m unittest discover -s tests -v
```

8 个单元测试，覆盖输入校验、前景分割、分层、输出对齐（不依赖 AI 模型，可离线跑）。
