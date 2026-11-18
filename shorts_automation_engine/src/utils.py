"""Shared helpers: logging, subprocess wrappers, text normalisation, easing curves."""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger("shorts")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Third-party chatter.
    for noisy in ("httpx", "httpx2", "urllib3", "faster_whisper", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def stage(title: str):
    """Log a pipeline stage with its wall-clock duration."""
    log.info("━━━ %s ━━━", title)
    t0 = time.perf_counter()
    yield
    log.info("✓ %s (%.1fs)", title, time.perf_counter() - t0)


class PipelineError(RuntimeError):
    """A user-facing, actionable pipeline failure."""


# --------------------------------------------------------------------------- #
# Subprocess / FFmpeg plumbing
# --------------------------------------------------------------------------- #
def which_or_raise(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise PipelineError(
            f"'{binary}' was not found on PATH. Run `python check_system.py` for install instructions."
        )
    return path


def run(cmd: Sequence[str], *, capture: bool = True, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command, raising a readable error (with stderr tail) on failure."""
    log.debug("$ %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-15:]
        raise PipelineError(
            f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd[:6])} …\n" + "\n".join(tail)
        )
    return proc


def ffprobe(path: Path | str) -> dict:
    """Return basic stream info for a media file: duration, width, height, fps, has_audio, rotation."""
    proc = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
    )
    data = json.loads(proc.stdout or "{}")
    info: dict = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "has_audio": False, "has_video": False}
    fmt_dur = float(data.get("format", {}).get("duration", 0) or 0)
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and not info["has_video"]:
            info["has_video"] = True
            w, h = int(st.get("width", 0)), int(st.get("height", 0))
            rotation = 0
            tags = st.get("tags", {}) or {}
            if "rotate" in tags:
                rotation = int(float(tags["rotate"]))
            for sd in st.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rotation = int(float(sd["rotation"]))
            if abs(rotation) % 180 == 90:
                w, h = h, w
            info.update(width=w, height=h, rotation=rotation)
            rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
            try:
                num, den = rate.split("/")
                info["fps"] = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0.0
            sdur = float(st.get("duration", 0) or 0)
            info["duration"] = sdur or fmt_dur
        elif st.get("codec_type") == "audio":
            info["has_audio"] = True
            if not info["duration"]:
                info["duration"] = float(st.get("duration", 0) or 0) or fmt_dur
    if not info["duration"]:
        info["duration"] = fmt_dur
    return info


def media_duration(path: Path | str) -> float:
    return float(ffprobe(path)["duration"])


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def slugify(text: str, max_len: int = 48) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:max_len].rstrip("-")) or "short"


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_to_words(n: int) -> str:
    """Tiny English number speller (0 … 999,999,999) used to align '6' with 'six'."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    for div, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div:
            rest = n % div
            return number_to_words(n // div) + f" {name}" + ("" if rest == 0 else " " + number_to_words(rest))
    return str(n)


def normalize_token(token: str) -> list[str]:
    """Normalise a spoken/written token for alignment: lowercase alnum, digits spelled out."""
    t = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode().lower()
    t = t.replace("%", " percent")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 1,000 -> 1000
    parts = re.findall(r"[a-z]+|\d+", t)
    out: list[str] = []
    for p in parts:
        if p.isdigit() and len(p) <= 9:
            out.extend(number_to_words(int(p)).split())
        else:
            out.append(p)
    return out


def has_alnum(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def caption_text(token: str) -> str:
    """Display form of a spoken word for Hormozi-style captions."""
    t = token.strip()
    t = t.strip("\"“”‘’()[]{}*_")
    keep_end = t[-1] if t and t[-1] in "?!" else ""
    t = t.rstrip(".,;:!?…—–-").lstrip("¿¡-—–")
    return (t + keep_end).upper()


# --------------------------------------------------------------------------- #
# Maths / easing
# --------------------------------------------------------------------------- #
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return -(math.cos(math.pi * t) - 1) / 2


def ease_out_back(t: float, overshoot: float = 1.9) -> float:
    t = clamp(t)
    c3 = overshoot + 1
    return 1 + c3 * (t - 1) ** 3 + overshoot * (t - 1) ** 2


def quantize(t: float, fps: int) -> float:
    """Snap a time to the nearest frame boundary."""
    return round(t * fps) / fps


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def human_list(items: Iterable[str]) -> str:
    items = list(items)
    return ", ".join(items) if items else "—"
