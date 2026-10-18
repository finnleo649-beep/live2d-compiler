"""环境搭建：下载外部引擎、安装 Python 依赖、拉取 AI 模型。

用法（在项目目录下）：
    python setup.py

这个脚本做 5 件事（已完成的部分会自动跳过，可反复运行）：
  1. 下载 JDK 21（免安装版）到 .tools/jdk21
  2. 下载 psd2live 源码到 _probe_git
  3. 下载 see-through 源码到 _probe_see_through
  4. 安装 Python 依赖到 .deps（torch CUDA + 其余）
  5. 下载 AI 模型到 .tools/hf-cache（LayerDiff3D + Marigold，约 12.5 GB）

总下载量约 16 GB，首次耗时取决于网速（30–90 分钟）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

PROJ = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(PROJ, ".tools")
DEPS = os.path.join(PROJ, ".deps")
JDK_DIR = os.path.join(TOOLS, "jdk21")
GRADLE_HOME = os.path.join(TOOLS, "gradle-home")
HF_CACHE = os.path.join(TOOLS, "hf-cache")
JAVA_TMP = os.path.join(TOOLS, "javatem")
P2L = os.path.join(PROJ, "_probe_git")
SEE_ROOT = os.path.join(PROJ, "_probe_see_through")
SEE = os.path.join(SEE_ROOT, "see-through-main")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 固定版本，保证可复现
JDK_URL = "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jdk/hotspot/normal/eclipse"
PSD2LIVE_REPO = "tsunehimatoi/psd2live"
SEE_REPO = "shitagaki-lab/see-through"

DEPS = DEPS  # noqa: 保持名字（下方 pip_install 使用）
PIP_REQUIREMENTS = [
    "transformers==5.0.0",
    "diffusers==0.37.0",
    "accelerate",
    "safetensors",
    "timm",
    "kornia",
    "omegaconf",
    "einops",
    "psd-tools",
    "opencv-python-headless",
    "scikit-image",
    "matplotlib",
    "pandas",
    "scikit-learn",
    "simple-parsing",
    "grad-cam",
    "lpips",
    "pycocotools",
    "bitsandbytes",
    "scipy",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
MODELS = [
    "layerdifforg/seethroughv0.0.2_layerdiff3d",   # ~9.5 GB
    "24yearsold/seethroughv0.0.1_marigold",        # ~3 GB
]


def step(i: int, total: int, msg: str) -> None:
    print("\n" + "=" * 64)
    print(" [%d/%d] %s" % (i, total, msg))
    print("=" * 64)


def download(url: str, dest: str, desc: str = "") -> None:
    """带进度下载（Python 网络栈，不依赖系统 TLS）。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("  下载 %s ..." % (desc or url.split("/")[-1]))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done - last >= 50_000_000:
                last = done
                print("    %.0f / %.0f MB" % (done / 1e6, total / 1e6))
                sys.stdout.flush()
    print("   完成 %.1f MB" % (os.path.getsize(dest) / 1e6))


def github_default_branch(repo: str) -> str:
    try:
        req = urllib.request.Request("https://api.github.com/repos/" + repo, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("default_branch", "main")
    except Exception:
        return "main"


def download_zip_repo(repo: str, dest_dir: str) -> str:
    """下载 GitHub 仓库 zip，解压后把内容直接放到 dest_dir（去掉 zip 的顶层目录）。"""
    branch = github_default_branch(repo)
    url = "https://github.com/%s/archive/refs/heads/%s.zip" % (repo, branch)
    tmp_zip = os.path.join(TOOLS, "_repo.zip")
    download(url, tmp_zip, "%s@%s" % (repo, branch))

    tmp_extract = os.path.join(TOOLS, "_repo_extract")
    if os.path.isdir(tmp_extract):
        shutil.rmtree(tmp_extract)
    os.makedirs(tmp_extract, exist_ok=True)
    with zipfile.ZipFile(tmp_zip) as z:
        z.extractall(tmp_extract)
    os.remove(tmp_zip)

    entries = os.listdir(tmp_extract)
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp_extract, entries[0])):
        src = os.path.join(tmp_extract, entries[0])
    else:
        src = tmp_extract

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.move(src, dest_dir)
    shutil.rmtree(tmp_extract, ignore_errors=True)
    return dest_dir


def pip_install(args: list) -> int:
    py = sys.executable
    cmd = [py, "-m", "pip", "install", "--no-cache-dir", "--target", DEPS] + args
    print("  pip " + " ".join(args[:3]) + (" ..." if len(args) > 3 else ""))
    return subprocess.run(cmd).returncode


def deps_ready() -> bool:
    """快速检查关键依赖是否已装（避免 pip 逐包网络检查，可反复运行）。"""
    if DEPS not in sys.path:
        sys.path.insert(0, DEPS)
    import importlib.util
    required = ["torch", "transformers", "diffusers", "psd_tools", "cv2",
                "huggingface_hub", "scipy", "timm", "kornia"]
    try:
        missing = [m for m in required if importlib.util.find_spec(m) is None]
    except Exception:
        return False
    if missing:
        print("  缺少依赖: %s" % missing)
        return False
    return True


def models_ready() -> bool:
    """检查 AI 模型是否已在本地缓存（避免重复联网检查）。"""
    hub = os.path.join(HF_CACHE, "hub")
    for repo_id in MODELS:
        dirname = "models--" + repo_id.replace("/", "--")
        if not os.path.isdir(os.path.join(hub, dirname)):
            print("  缺少模型: %s" % repo_id)
            return False
    return True


# ============================================================
#  psd2live 本地补丁
# ============================================================
# 作用：给 psd2live 的 CLI 暴露眨眼参数，并把默认闭合度从 72% 改到 95%。
# 背景：psd2live 默认 blinkCorner=34 + blinkDepth=38，眼睛只能闭到 72% 高度，
#       视觉上像"半闭眼"；20 + 75 = 95% 才是真正闭合。
# 说明：完整 diff 见 patches/psd2live-blink.patch（存档 + 人工审查用）。

BLINK_PATCH_DESC = "眨眼参数（blinkDepth/blinkCorner/eyelashSquash）"


def apply_psd2live_patches(p2l_dir: str) -> bool:
    """给 psd2live 源码应用本地补丁（幂等，可反复运行）。"""
    main_kt = os.path.join(p2l_dir, "src", "main", "kotlin", "io", "github", "psd2live", "Main.kt")
    if not os.path.isfile(main_kt):
        print("  [警告] 找不到 %s" % main_kt)
        return False

    with open(main_kt, encoding="utf-8") as f:
        content = f.read()

    # 幂等：已应用则跳过
    if "blinkDepth" in content and "--blink-depth" in content:
        print("  %s: 已应用" % BLINK_PATCH_DESC)
        return True

    original = content

    # 1) import RigTuning
    old = "import io.github.psd2live.agent.AgentMcpController\n"
    new = old + "import io.github.psd2live.core.RigTuning\n"
    if old in content and new not in content:
        content = content.replace(old, new, 1)

    # 2) PipelineConfig 里加 rigTuning
    old = '\t\tbodyStrength = options.float("--body-strength", 1f),\n'
    new = old + (
        "\t\t// Blink rig tuning: closed-eye curve sits at (blinkCorner + blinkDepth)% of the eye-white height.\n"
        "\t\t// The library default (34 + 38 = 72%) leaves the eyes visibly half-open; 20 + 75 = 95% closes them.\n"
        "\t\trigTuning = RigTuning(\n"
        '\t\t\tblinkDepth = options.float("--blink-depth", 75f).coerceIn(0f, 80f),\n'
        '\t\t\tblinkCorner = options.float("--blink-corner", 20f).coerceIn(0f, 80f),\n'
        '\t\t\teyelashSquash = options.float("--eyelash-squash", 80f).coerceIn(0f, 150f),\n'
        "\t\t),\n"
    )
    if old in content:
        content = content.replace(old, new, 1)

    # 3) CLI 参数名列表
    old = '"--head-strength", "--body-strength"'
    new = '"--head-strength", "--body-strength", "--blink-depth", "--blink-corner", "--eyelash-squash"'
    if old in content and new not in content:
        content = content.replace(old, new, 1)

    if content == original:
        print("  [警告] %s 未能应用（源码结构可能已变化）" % BLINK_PATCH_DESC)
        print("         请手动参考 patches/psd2live-blink.patch")
        return False

    with open(main_kt, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("  %s: 已应用" % BLINK_PATCH_DESC)
    return True


def main() -> int:
    total = 5
    print("=" * 64)
    print(" Live2D Compiler  环境搭建")
    print(" 项目目录: %s" % PROJ)
    print("=" * 64)
    print(" 提示：首次需要下载约 16 GB，请确保网络畅通、磁盘充足。")

    os.makedirs(TOOLS, exist_ok=True)
    os.makedirs(JAVA_TMP, exist_ok=True)

    # ---------- 1. JDK 21 ----------
    step(1, total, "JDK 21")
    java_exe = os.path.join(JDK_DIR, "jdk-21.0.12.1+1", "bin", "java.exe")
    if os.path.isfile(java_exe):
        print("  已存在，跳过")
    else:
        zp = os.path.join(TOOLS, "jdk21.zip")
        download(JDK_URL, zp, "JDK 21 (约 200MB)")
        with zipfile.ZipFile(zp) as z:
            z.extractall(JDK_DIR)
        os.remove(zp)
        found = []
        for root, _, files in os.walk(JDK_DIR):
            if "java.exe" in files and root.endswith("bin"):
                found.append(os.path.join(root, "java.exe"))
        if not found:
            print("  [错误] JDK 解压后找不到 java.exe")
            return 1
        java_exe = found[0]
    print("  java: %s" % java_exe)

    # ---------- 2. psd2live ----------
    step(2, total, "psd2live 源码（分层 → 模型）")
    if os.path.isfile(os.path.join(P2L, "gradlew.bat")):
        print("  已存在，跳过下载")
    else:
        if os.path.isdir(P2L):
            shutil.rmtree(P2L)
        download_zip_repo(PSD2LIVE_REPO, P2L)
        if not os.path.isfile(os.path.join(P2L, "gradlew.bat")):
            print("  [错误] 解压后找不到 gradlew.bat")
            return 1
    apply_psd2live_patches(P2L)
    print("  psd2live 就绪")

    # ---------- 3. see-through ----------
    step(3, total, "see-through 源码（立绘 → 分层）")
    if os.path.isfile(os.path.join(SEE, "inference", "scripts", "inference_psd_blockswap.py")):
        print("  已存在，跳过")
    else:
        # see-through 保留 zip 的顶层目录（see-through-main），与 generate_model.py 的路径一致
        if os.path.isdir(SEE_ROOT):
            shutil.rmtree(SEE_ROOT)
        os.makedirs(SEE_ROOT, exist_ok=True)
        branch = github_default_branch(SEE_REPO)
        url = "https://github.com/%s/archive/refs/heads/%s.zip" % (SEE_REPO, branch)
        tmp_zip = os.path.join(TOOLS, "_see.zip")
        download(url, tmp_zip, "see-through@%s" % branch)
        with zipfile.ZipFile(tmp_zip) as z:
            z.extractall(SEE_ROOT)
        os.remove(tmp_zip)
        if not os.path.isdir(SEE):
            print("  [错误] 解压后目录不对（期望 see-through-main）")
            return 1
    print("  see-through 就绪")

    # ---------- 4. Python 依赖 ----------
    step(4, total, "Python 依赖（torch CUDA + 其余，约 5 GB）")
    if deps_ready():
        print("  关键依赖已就绪，跳过")
    else:
        if os.path.isdir(os.path.join(DEPS, "torch")):
            print("  torch 已存在，跳过 torch 安装")
        else:
            rc = pip_install(["torch", "torchvision", "--index-url", TORCH_INDEX])
            if rc != 0:
                print("  [错误] torch 安装失败")
                return 1
        rc = pip_install(PIP_REQUIREMENTS)
        if rc != 0:
            print("  [警告] 部分依赖安装返回非零（可能已存在），继续")

    # ---------- 5. AI 模型 ----------
    step(5, total, "AI 模型（约 12.5 GB，最耗时）")
    if models_ready():
        print("  模型已存在，跳过")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("  [错误] huggingface_hub 未安装（第 4 步应已安装）")
            return 1

        os.makedirs(HF_CACHE, exist_ok=True)
        for repo_id in MODELS:
            print("\n  拉取模型 %s ..." % repo_id)
            try:
                snapshot_download(repo_id=repo_id, cache_dir=HF_CACHE)
                print("  完成")
            except Exception as e:
                print("  [错误] %s 下载失败: %s" % (repo_id, e))
                return 1

    # ---------- 完成 ----------
    print("\n" + "=" * 64)
    print(" 环境搭建完成！")
    print(" 现在可以：把 PNG 立绘拖到「生成模型.bat」上，或运行：")
    print("   python generate_model.py 你的图.png")
    print("=" * 64)
    try:
        input("\n按回车退出...")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
