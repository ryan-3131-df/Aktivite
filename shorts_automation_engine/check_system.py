#!/usr/bin/env python3
"""
System dependency check for the Shorts Automation Engine.

    python check_system.py            # report
    python check_system.py --install  # also try to install ffmpeg / imagemagick with your package manager

Only uses the standard library, so it runs before `pip install -r requirements.txt`.
Exit code 0 = ready to render, 1 = something required is missing.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK, WARN, FAIL = "✅", "⚠️ ", "❌"

REQUIRED_PACKAGES = [  # (import name, pip name)
    ("anthropic", "anthropic"), ("moviepy", "moviepy"), ("numpy", "numpy"), ("PIL", "pillow"),
    ("cv2", "opencv-python-headless"), ("pydantic", "pydantic"), ("requests", "requests"),
    ("dotenv", "python-dotenv"),
]
OPTIONAL_PACKAGES = [
    ("faster_whisper", "faster-whisper", "word-level timestamps (falls back to ElevenLabs alignment)"),
    ("yt_dlp", "yt-dlp", "YouTube Creative-Commons fallback when Pexels has no footage"),
]
REQUIRED_ENCODERS = ["libx264", "aac"]
REQUIRED_FILTERS = ["loudnorm", "amix", "adelay", "alimiter", "boxblur", "overlay", "atempo"]
HW_ENCODERS = ["h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_amf", "h264_vaapi"]
API_KEYS = [
    ("ANTHROPIC_API_KEY", True, "script generation (Claude)"),
    ("ELEVENLABS_API_KEY", True, "voiceover"),
    ("PEXELS_API_KEY", True, "stock footage"),
    ("OPENAI_API_KEY", False, "optional: Whisper API timestamps"),
]


def sh(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout)
        return p.returncode, p.stdout or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def install_hint(tool: str) -> str:
    system = platform.system()
    if system == "Darwin":
        return f"brew install {tool}"
    if system == "Windows":
        pkg = {"ffmpeg": "Gyan.FFmpeg", "imagemagick": "ImageMagick.ImageMagick"}[tool]
        return f"winget install {pkg}      (or: choco install {tool})"
    if shutil.which("apt-get"):
        return f"sudo apt-get update && sudo apt-get install -y {tool}"
    if shutil.which("dnf"):
        return f"sudo dnf install -y {'ImageMagick' if tool == 'imagemagick' else tool}"
    if shutil.which("pacman"):
        return f"sudo pacman -S --noconfirm {tool}"
    return f"install '{tool}' with your package manager"


def try_install(tool: str) -> bool:
    cmd = install_hint(tool).split("      ")[0]
    print(f"   → running: {cmd}")
    return subprocess.run(cmd, shell=True).returncode == 0


def load_env_file() -> dict[str, str]:
    env = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.split(" #", 1)[0].strip().strip("'\"")
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true", help="try to install missing system tools")
    args = ap.parse_args(argv)
    failures = 0
    print(f"\nShorts Automation Engine · system check  ({platform.system()} {platform.machine()})\n")

    # ── Python ─────────────────────────────────────────────────────────── #
    py_ok = sys.version_info >= (3, 10)
    print(f"{OK if py_ok else FAIL} Python {platform.python_version()}  (need ≥ 3.10)")
    failures += not py_ok

    # ── FFmpeg ─────────────────────────────────────────────────────────── #
    print("\nSystem tools")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if not path and args.install and tool == "ffmpeg":
            try_install("ffmpeg")
            path = shutil.which(tool)
        if path:
            _, out = sh([tool, "-version"])
            print(f"{OK} {tool:<12} {out.splitlines()[0][:70] if out else path}")
        else:
            failures += 1
            print(f"{FAIL} {tool:<12} NOT FOUND   →  {install_hint('ffmpeg')}")

    if shutil.which("ffmpeg"):
        _, enc = sh(["ffmpeg", "-hide_banner", "-encoders"])
        _, flt = sh(["ffmpeg", "-hide_banner", "-filters"])
        enc_names = {l.split()[1] for l in enc.splitlines() if len(l.split()) > 1}
        flt_names = {l.split()[1] for l in flt.splitlines() if len(l.split()) > 2}
        missing = [e for e in REQUIRED_ENCODERS if e not in enc_names] + [f for f in REQUIRED_FILTERS if f not in flt_names]
        if missing:
            failures += 1
            print(f"{FAIL} ffmpeg build is missing: {', '.join(missing)}  (install a full build)")
        else:
            print(f"{OK} ffmpeg build has libx264, aac and all required filters")
        if "libmp3lame" not in enc_names:
            print(f"{WARN} libmp3lame missing - starter SFX/music can't be generated (bring your own files)")
        usable = []
        for hw in HW_ENCODERS:
            if hw not in enc_names:
                continue
            pre = ["-vaapi_device", "/dev/dri/renderD128"] if hw == "h264_vaapi" else []
            vf = ["-vf", "format=nv12,hwupload"] if hw == "h264_vaapi" else []
            code, _ = sh(["ffmpeg", "-hide_banner", "-loglevel", "error", *pre, "-f", "lavfi", "-i",
                          "testsrc2=s=320x568:r=30:d=0.3", *vf, "-c:v", hw, "-f", "null", "-"])
            if code == 0:
                usable.append(hw)
        if usable:
            print(f"{OK} hardware encoder available: {', '.join(usable)}  (used automatically)")
        else:
            print(f"{WARN} no working hardware encoder - rendering with libx264 (slower, same quality)")

    magick = shutil.which("magick") or shutil.which("convert")
    if not magick and args.install:
        try_install("imagemagick")
        magick = shutil.which("magick") or shutil.which("convert")
    if magick:
        _, out = sh([magick, "-version"])
        print(f"{OK} {'imagemagick':<12} {out.splitlines()[0][:70] if out else magick}")
    else:
        print(f"{WARN} imagemagick  not found (optional: captions are rendered with Pillow, not ImageMagick)"
              f"\n     install: {install_hint('imagemagick')}")

    # ── Python packages ────────────────────────────────────────────────── #
    print("\nPython packages")
    missing_pkgs = []
    for mod, pip_name in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(mod)
            print(f"{OK} {pip_name:<24} {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            missing_pkgs.append(pip_name)
            print(f"{FAIL} {pip_name:<24} {type(exc).__name__}: {exc}")
    for mod, pip_name, why in OPTIONAL_PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "") or getattr(getattr(m, "version", None), "__version__", "")
            print(f"{OK} {pip_name:<24} {ver}")
        except Exception:  # noqa: BLE001
            print(f"{WARN} {pip_name:<24} not installed - {why}")
    if missing_pkgs:
        print(f"\n   → pip install -r {ROOT / 'requirements.txt'}")
    try:
        import moviepy
        if int(moviepy.__version__.split(".")[0]) < 2:
            failures += 1
            print(f"{FAIL} moviepy {moviepy.__version__} is too old - this project needs MoviePy 2.x")
    except Exception:  # noqa: BLE001
        pass

    # ── API keys ───────────────────────────────────────────────────────── #
    print("\nAPI keys (.env)")
    env = {**load_env_file(), **{k: v for k, v in os.environ.items() if v}}
    if not (ROOT / ".env").exists():
        print(f"{WARN} .env not found  →  cp .env.example .env  and fill in your keys")
    missing_keys = False
    for key, required, why in API_KEYS:
        val = env.get(key, "")
        if val and not val.lower().startswith(("your_", "sk-xxx", "xxx")):
            print(f"{OK} {key:<22} set ({val[:4]}…{val[-2:]})")
        elif required:
            missing_keys = True
            print(f"{WARN} {key:<22} missing - needed for {why}")
        else:
            print(f"   {key:<22} not set ({why})")
    if missing_keys:
        print("   (without keys you can still run:  python main.py \"any topic\" --dry-run)")

    # ── Assets ─────────────────────────────────────────────────────────── #
    print("\nAssets")
    for sub, exts in (("fonts", (".ttf", ".otf")), ("sfx", (".mp3", ".wav")), ("music", (".mp3", ".wav", ".m4a"))):
        d = ROOT / "assets" / sub
        files = [f.name for f in d.glob("*") if f.suffix.lower() in exts] if d.exists() else []
        mark = OK if files else WARN
        extra = "" if files else "  →  python main.py --init-assets"
        print(f"{mark} assets/{sub:<6} {len(files)} file(s){extra}")

    print()
    if failures:
        print(f"{FAIL} {failures} required item(s) missing - fix the ❌ lines above.\n")
        return 1
    print(f"{OK} Ready. Try:  python main.py \"most dangerous animals\" --dry-run\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
