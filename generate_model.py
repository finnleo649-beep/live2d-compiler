"""一键生成 Live2D 模型：一张立绘 → 完整可加载模型。

用法（两种都行）：
  1. 把 PNG 立绘拖到同目录的「生成模型.bat」上
  2. 命令行：python generate_model.py 你的图.png

流程：
  [1/3] 立绘 → see-through → 分层 PSD（约 4 分钟）
  [2/3] 分层 PSD → psd2live → 模型文件（约 20 秒）
  [3/3] 复制到 VTube Studio 的 Live2DModels 目录
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import time
import json

# ---- 路径全部由脚本位置推导（Python 处理中文路径没问题）----
PROJ = os.path.dirname(os.path.abspath(__file__))
PYEXE = sys.executable
SEE = os.path.join(PROJ, "_probe_see_through", "see-through-main")
P2L = os.path.join(PROJ, "_probe_git")
JDK = os.path.join(PROJ, ".tools", "jdk21", "jdk-21.0.12.1+1")
GRADLE_HOME = os.path.join(PROJ, ".tools", "gradle-home")
JAVA_TMP = os.path.join(PROJ, ".tools", "javatem")
DEPS = os.path.join(PROJ, ".deps")
HF_CACHE = os.path.join(PROJ, ".tools", "hf-cache")
VTS = r"D:\steam\steamapps\common\VTube Studio\VTube Studio_Data\StreamingAssets\Live2DModels"

# 网格间距：越小变形越平滑，但文件更大、生成更慢。
# 64 = psd2live 默认（太粗，转头折角）；32 = 明显更平滑；24/16 更精细但更慢。
MESH_SPACING = int(os.environ.get("L2D_MESH_SPACING", "32"))

# 让 .deps 里的第三方库（psd-tools 等）和项目自身包可被导入
for _p in (DEPS, PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEE_SCRIPT = os.path.join("inference", "scripts", "inference_psd_blockswap.py")


def fail(msg: str) -> int:
    print("\n[ERROR] " + msg)
    try:
        input("\n按回车退出...")
    except Exception:
        pass
    return 1


def quality_check(model_dir: str) -> dict:
    """检查生成的模型质量，自动标出需要人工修的地方。

    这是「半自动」定位的核心：不假装成功，把问题明确标出来。
    """
    issues = []
    warns = []
    info = {}

    # 读 psd2live 诊断报告
    rep = None
    for f in os.listdir(model_dir):
        if f.endswith(".psd2live.json"):
            try:
                with open(os.path.join(model_dir, f), encoding="utf-8") as fh:
                    rep = json.load(fh)
            except Exception:
                pass
            break

    # 读 model3.json
    mj = None
    for f in os.listdir(model_dir):
        if f.endswith(".model3.json"):
            try:
                with open(os.path.join(model_dir, f), encoding="utf-8") as fh:
                    mj = json.load(fh)
            except Exception:
                pass
            break

    # 1. 文件完整性
    if mj is None:
        issues.append("model3.json 缺失")
    else:
        refs = [mj["FileReferences"]["Moc"]] + mj["FileReferences"].get("Textures", [])
        missing = [r for r in refs if not os.path.exists(os.path.join(model_dir, r))]
        if missing:
            issues.append("引用的文件缺失: %s" % missing)

    # 2. 图层 / 参数 / 顺序
    if rep:
        layers = rep.get("layers", [])
        summary = rep.get("summary", {})
        n_layers = summary.get("layers", len(layers))
        n_params = summary.get("parameters", 0)
        info["layers"] = n_layers
        info["parameters"] = n_params
        info["drawables"] = summary.get("drawables")
        info["deformers"] = summary.get("deformers")

        if n_layers < 8:
            issues.append("图层数过少（%d），可能漏层，需人工检查" % n_layers)
        elif n_layers < 12:
            warns.append("图层数偏少（%d）" % n_layers)

        if n_params < 15:
            warns.append("参数偏少（%d）" % n_params)

        tags = [l.get("tag") for l in layers]
        for key in ("face", "front hair"):
            if key not in tags:
                issues.append("关键图层缺失：%s" % key)
        if "back hair" not in tags:
            warns.append("没有后发层（长发角色可能需要）")

        # 已知的系统性问题：psd2live 用固定顺序，头饰可能被头发遮住
        if "headwear" in tags and "front hair" in tags:
            if tags.index("headwear") < tags.index("front hair"):
                warns.append("头饰(headwear)被排在头发下面，可能被遮挡/穿模")

    if issues:
        verdict = "needs_review"
    elif warns:
        verdict = "ok_with_notes"
    else:
        verdict = "ok"

    return {"verdict": verdict, "issues": issues, "warnings": warns, "info": info}


def main() -> int:
    if len(sys.argv) < 2:
        print("=" * 60)
        print(" 一键生成 Live2D 模型")
        print("=" * 60)
        print(" 用法：把 PNG 立绘拖到「生成模型.bat」上，或命令行：")
        print("     python generate_model.py 你的图.png")
        print()
        print(" 要求：单个动漫角色、清晰、尽量透明或纯色背景。")
        print(" 耗时：约 4-5 分钟。")
        print("=" * 60)
        try:
            input("\n按回车退出...")
        except Exception:
            pass
        return 0

    img = os.path.abspath(sys.argv[1])
    if not os.path.isfile(img):
        return fail("找不到图片文件：" + img)
    if not img.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        return fail("请提供 PNG / JPG / WEBP 图片")

    tag = str(int(time.time()))[-6:]
    name = "gen_" + tag
    work_png_rel = "work_%s.png" % tag

    print("=" * 60)
    print(" 输入图片 : %s" % img)
    print(" 模型名称 : %s" % name)
    print("=" * 60)

    os.makedirs(JAVA_TMP, exist_ok=True)

    # ---------- Step 1: 复制图片 ----------
    print("\n[1/3] 复制图片到工作目录...")
    try:
        shutil.copy2(img, os.path.join(SEE, work_png_rel))
    except Exception as e:
        return fail("复制失败：%s" % e)

    # ---------- Step 2: AI 分层 ----------
    print("\n[2/3] AI 分层中（约 4 分钟，请稍候）...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        DEPS, os.path.join(SEE, "common"), os.path.join(SEE, "annotators")
    ])
    env["HF_HOME"] = HF_CACHE
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    r = subprocess.run(
        [PYEXE, SEE_SCRIPT, "--srcp", work_png_rel, "--save_to_psd", "--resolution", "1024"],
        cwd=SEE, env=env,
    )
    if r.returncode != 0:
        return fail("AI 分层失败（退出码 %d）。请查看上方日志。" % r.returncode)

    psd = os.path.join(SEE, "workspace", "layerdiff_output_blockswap", "work_%s.psd" % tag)
    if not os.path.isfile(psd):
        return fail("未找到分层 PSD：%s" % psd)
    print("\n[2/3] 分层完成：%s" % psd)

    # ---------- Step 2.5: 图层绘制顺序修正 ----------
    # psd2live 按 PSD 图层顺序决定 drawOrder；see-through 输出的顺序不一定符合视觉层次
    # （例如发箍 headwear 排在头发下面会被遮挡）。这里按语义优先级重排。
    work_psd_name = "work_%s.psd" % tag
    work_psd_path = os.path.join(P2L, work_psd_name)
    print("\n[2.5/3] 修正图层绘制顺序（配件遮挡问题）...")
    try:
        from pipeline.stages.reorder_psd import reorder_psd
        rep = reorder_psd(psd, work_psd_path)
        moved = rep.get("moved", [])
        if moved:
            print("   调整了 %d 个图层位置" % len(moved))
        else:
            print("   顺序已符合规范，无需调整")
    except Exception as e:
        print("   [警告] 重排失败，使用原始顺序：%s" % e)
        shutil.copy2(psd, work_psd_path)

    # ---------- Step 3: 绑定 + 导出 ----------
    print("\n[3/3] 绑定与导出模型（约 20 秒）...")

    env2 = os.environ.copy()
    env2["JAVA_HOME"] = JDK
    env2["GRADLE_USER_HOME"] = GRADLE_HOME
    env2["PATH"] = os.path.join(JDK, "bin") + os.pathsep + env2.get("PATH", "")
    env2["TEMP"] = JAVA_TMP
    env2["TMP"] = JAVA_TMP
    env2["JAVA_TOOL_OPTIONS"] = "-Djava.io.tmpdir=" + JAVA_TMP

    out_name = "build/work_%s" % tag
    args = "--input %s --output %s --mesh-spacing %d" % (work_psd_name, out_name, MESH_SPACING)
    cmd = ["cmd", "/c", "gradlew.bat", "run", "--args=%s" % args]
    r = subprocess.run(cmd, cwd=P2L, env=env2)
    if r.returncode != 0:
        return fail("绑定失败（退出码 %d）。请查看上方日志。" % r.returncode)

    built = os.path.join(P2L, "build", "work_%s" % tag)
    if not os.path.isdir(built):
        return fail("未找到输出目录：%s" % built)

    # ---------- 质量检查（半自动定位：明确标出要修的地方）----------
    print("\n[质检] 检查模型质量...")
    qr = quality_check(built)
    try:
        with open(os.path.join(built, "quality_report.json"), "w", encoding="utf-8") as f:
            json.dump(qr, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print("  结论: %s" % qr["verdict"])
    for it in qr["issues"]:
        print("   [需要修] %s" % it)
    for w in qr["warnings"]:
        print("   [注意] %s" % w)
    if not qr["issues"] and not qr["warnings"]:
        print("   未发现明显问题")

    # ---------- 交付到 VTS ----------
    if os.path.isdir(VTS):
        out_dir = os.path.join(VTS, name)
    else:
        out_dir = os.path.join(PROJ, "storage", "outputs", name)
        print("\n[提示] 未找到 VTS 目录，输出到：%s" % out_dir)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(built, out_dir)

    print("\n" + "=" * 60)
    print(" 完成！")
    print(" 模型目录 : %s" % out_dir)
    print(" 质量结论 : %s（详见 quality_report.json）" % qr["verdict"])
    if os.path.isdir(VTS):
        print(" VTube Studio 里的模型名 : %s" % name)
        print(" 打开 VTS → 模型选择，找到「%s」即可。" % name)
    print("=" * 60)

    try:
        os.startfile(out_dir)
    except Exception:
        pass
    try:
        input("\n按回车退出...")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
