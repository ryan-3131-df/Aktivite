"""
FFmpeg building blocks: hardware-encoder detection, clip normalisation
(crop / blurred-background pad / trim / 60 fps), loudness normalisation and a
raw-frame pipe writer used by the MoviePy renderer.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import Settings, VideoSettings
from .models import SourceClip
from .utils import PipelineError, log, run

# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
HW_ENCODER_PRIORITY = ("h264_nvenc", "h264_videotoolbox", "h264_qsv", "h264_amf", "h264_vaapi")
VAAPI_DEVICE = "/dev/dri/renderD128"


@functools.lru_cache(maxsize=1)
def available_encoders() -> frozenset[str]:
    proc = run(["ffmpeg", "-hide_banner", "-encoders"], check=False)
    names = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            names.add(parts[1])
    return frozenset(names)


@functools.lru_cache(maxsize=1)
def available_filters() -> frozenset[str]:
    proc = run(["ffmpeg", "-hide_banner", "-filters"], check=False)
    names = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"[TSC.|]{3}", parts[0] or ""):
            names.add(parts[1])
    return frozenset(names)


def encoder_args(encoder: str, video: VideoSettings) -> tuple[list[str], list[str]]:
    """Return (global/pre-input args, output args) for the final H.264 encode."""
    crf, fps = video.crf, video.fps
    gop = ["-g", str(fps * 2)]
    if encoder == "h264_nvenc":
        return [], ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(crf + 1),
                    "-b:v", "0", "-maxrate", "30M", "-bufsize", "60M", "-profile:v", "high",
                    "-pix_fmt", "yuv420p", *gop]
    if encoder == "h264_videotoolbox":
        return [], ["-c:v", "h264_videotoolbox", "-b:v", "16M", "-maxrate", "24M", "-profile:v", "high",
                    "-allow_sw", "1", "-pix_fmt", "yuv420p", *gop]
    if encoder == "h264_qsv":
        return [], ["-c:v", "h264_qsv", "-preset", "slow", "-global_quality", str(crf + 2),
                    "-profile:v", "high", "-pix_fmt", "nv12", *gop]
    if encoder == "h264_amf":
        return [], ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", str(crf),
                    "-qp_p", str(crf + 2), "-qp_b", str(crf + 4), "-pix_fmt", "yuv420p", *gop]
    if encoder == "h264_vaapi":
        return ["-vaapi_device", VAAPI_DEVICE], ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi",
                                                 "-qp", str(crf + 2), "-profile:v", "high", *gop]
    # Software fallback - best quality per bit.
    return [], ["-c:v", "libx264", "-preset", video.x264_preset, "-crf", str(crf), "-profile:v", "high",
                "-pix_fmt", "yuv420p", *gop, "-bf", "2"]


@functools.lru_cache(maxsize=8)
def encoder_works(encoder: str) -> bool:
    """Actually try a tiny encode - an encoder can be compiled in but have no GPU/driver behind it."""
    if encoder not in available_encoders():
        return False
    pre, out = encoder_args(encoder, VideoSettings(fps=30))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *pre, "-f", "lavfi", "-i",
           "testsrc2=s=320x568:r=30:d=0.3", *out, "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def pick_encoder(requested: str = "auto") -> str:
    if requested and requested != "auto":
        if not encoder_works(requested):
            log.warning("Encoder '%s' is not usable on this machine - falling back to libx264", requested)
            return "libx264"
        return requested
    for enc in HW_ENCODER_PRIORITY:
        if encoder_works(enc):
            return enc
    return "libx264"


# --------------------------------------------------------------------------- #
# Probing helpers
# --------------------------------------------------------------------------- #
def detect_leading_silence(path: Path, threshold_db: float = -45.0, min_silence: float = 0.05) -> float:
    """Seconds of silence at the very start of an audio file."""
    proc = run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
                f"silencedetect=noise={threshold_db}dB:d={min_silence}", "-f", "null", "-"], check=False)
    starts = re.findall(r"silence_start: (-?[\d.]+)", proc.stderr or "")
    ends = re.findall(r"silence_end: ([\d.]+)", proc.stderr or "")
    if starts and ends and float(starts[0]) <= 0.01:
        return float(ends[0])
    return 0.0


def measure_loudness(path: Path) -> dict:
    """EBU R128 measurement via loudnorm's analysis pass."""
    proc = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], check=False)
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr or "", re.S)
    if not match:
        raise PipelineError(f"Could not measure loudness of {path}")
    return json.loads(match.group(0))


def loudnorm_two_pass(src: Path, dst: Path, target_i: float, target_tp: float = -1.0, lra: float = 11.0,
                      sample_rate: int = 48000, channels: int = 2, extra_filters: str = "") -> Path:
    """Two-pass EBU R128 normalisation (pass 1 measures, pass 2 applies linearly where possible)."""
    pre = f"{extra_filters}," if extra_filters else ""
    proc = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-af",
                f"{pre}loudnorm=I={target_i}:TP={target_tp}:LRA={lra}:print_format=json",
                "-f", "null", "-"], check=False)
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr or "", re.S)
    if not match:
        raise PipelineError(f"loudnorm analysis failed for {src}:\n{(proc.stderr or '')[-800:]}")
    m = json.loads(match.group(0))
    if m.get("input_i") in ("-inf", None) or float(m["input_i"]) < -70:
        # Silent input (e.g. dry-run placeholder voice) - just resample.
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
             *(["-af", extra_filters] if extra_filters else []),
             "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)])
        return dst
    af = (f"{pre}loudnorm=I={target_i}:TP={target_tp}:LRA={lra}:measured_I={m['input_i']}:"
          f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
          f"offset={m['target_offset']}:linear=true:print_format=summary,aresample={sample_rate}")
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-af", af,
         "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)])
    return dst


def to_wav(src: Path, dst: Path, sample_rate: int = 48000, channels: int = 1, filters: str = "") -> Path:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         *(["-af", filters] if filters else []),
         "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)])
    return dst


def extract_frame_jpegs(path: Path, times: Sequence[float], width: int = 384) -> list[bytes]:
    """Grab small JPEG stills (used to show local clips to Claude's vision)."""
    frames: list[bytes] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(times):
            out = Path(tmp) / f"f{i}.jpg"
            run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
                 "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out)], check=False)
            if out.exists() and out.stat().st_size > 0:
                frames.append(out.read_bytes())
    return frames


# --------------------------------------------------------------------------- #
# Clip normalisation (crop / blur-pad / trim / fps)
# --------------------------------------------------------------------------- #
def _fit_filter(src: SourceClip, video: VideoSettings) -> tuple[str, str]:
    """Return (filter_complex_body, mode) mapping [0:v] -> [v] at WxH."""
    W, H = video.width, video.height
    mode = video.fit_mode
    if mode == "smart":
        mode = "crop" if src.aspect <= 0.75 else "blur"
    if mode == "crop":
        return (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={W}:{H},setsar=1[v]"), mode
    # Blurred background: blur at 1/4 resolution (fast), then scale back up.
    fg_w = int(round(W * video.blur_pad_zoom / 2) * 2)
    return (
        f"[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
        f"boxblur=luma_radius=18:luma_power=2:chroma_radius=10:chroma_power=2,"
        f"scale={W}:{H}:flags=bicubic,eq=brightness=-0.10:saturation=1.25[bg];"
        f"[fgsrc]scale={fg_w}:-2:flags=lanczos[fg];"
        f"[bg][fg]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1,setsar=1[v]"
    ), "blur"


def normalize_shot(src: SourceClip, dst: Path, start: float, n_frames: int, video: VideoSettings) -> Path:
    """
    Cut `n_frames` frames (at video.fps) from `src` starting at `start` seconds and
    conform them to WxH @ fps, no audio. Short sources are slowed down slightly
    (if within 30%) or looped, so a shot is never frozen or black.
    """
    fps = video.fps
    need = n_frames / fps
    src_dur = max(src.duration, 0.04)
    start = max(0.0, min(start, max(0.0, src_dur - need)))
    available = src_dur - start

    pre_input: list[str] = []
    speed_filter = ""
    if available >= need:
        pre_input += ["-ss", f"{start:.3f}"]
    elif src_dur >= need * 0.7:
        pre_input += ["-ss", "0"]
        stretch = need / src_dur * 1.01
        speed_filter = f"setpts={stretch:.4f}*PTS,"
    else:
        pre_input += ["-stream_loop", "-1"]
    if video.hwaccel_decode:
        pre_input = ["-hwaccel", "auto", *pre_input]

    fit, mode = _fit_filter(src, video)
    # Insert speed + fps conversion before the fit graph's output label.
    graph = fit.replace("[0:v]", f"[0:v]{speed_filter}fps={fps},", 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *pre_input, "-i", str(src.path),
        "-filter_complex", graph, "-map", "[v]", "-an", "-frames:v", str(n_frames),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-video_track_timescale", str(fps * 1000), str(dst),
    ])
    log.debug("normalised %s → %s (%s, %d frames from %.2fs)", src.path.name, dst.name, mode, n_frames, start)
    return dst


def concat_shots(shots: Sequence[Path], dst: Path) -> Path:
    """Losslessly join identically-encoded shot files."""
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in shots), encoding="utf-8")
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", str(dst)])
    return dst


def make_placeholder_clip(dst: Path, label: str, seconds: float, index: int, font_path: Path | None,
                          video: VideoSettings) -> Path:
    """Animated gradient + label - used for --dry-run or when every media source fails."""
    palettes = [("0x0f2027", "0x2c5364"), ("0x41295a", "0x2f0743"), ("0x1d4350", "0xa43931"),
                ("0x134e5e", "0x71b280"), ("0x283c86", "0x45a247"), ("0x3a1c71", "0xd76d77"),
                ("0x232526", "0x414345")]
    c0, c1 = palettes[index % len(palettes)]
    W, H = video.width, video.height
    vf = f"gradients=s={W}x{H}:c0={c0}:c1={c1}:x0=0:y0=0:x1={W}:y1={H}:speed=0.02:d={seconds:.2f}:r=30"
    filters = []
    if font_path and "drawtext" in available_filters():
        txt = dst.with_suffix(".txt")
        txt.write_text(label.upper(), encoding="utf-8")
        ff = font_path.as_posix().replace(":", r"\:")
        tf = txt.as_posix().replace(":", r"\:")
        filters.append(f"drawtext=fontfile='{ff}':textfile='{tf}':expansion=none:fontsize=84:fontcolor=white@0.35:"
                       f"x=(w-text_w)/2:y=h*0.62")
    lavfi = vf if "gradients" in available_filters() else f"color=c={c0}:s={W}x{H}:d={seconds:.2f}:r=30"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", lavfi,
         *(["-vf", ",".join(filters)] if filters else []),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)])
    return dst


# --------------------------------------------------------------------------- #
# Raw frame writer (MoviePy frames -> FFmpeg encoder, with audio mux)
# --------------------------------------------------------------------------- #
class FFmpegPipeWriter:
    """Streams RGB24 frames into FFmpeg and muxes the final audio in the same pass."""

    def __init__(self, out_path: Path, size: tuple[int, int], fps: int, encoder: str,
                 video: VideoSettings, audio_path: Path | None = None, audio_bitrate: str = "320k"):
        pre, vargs = encoder_args(encoder, video)
        w, h = size
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *pre,
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if audio_path:
            cmd += ["-i", str(audio_path)]
        cmd += ["-map", "0:v:0"]
        if audio_path:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000"]
        cmd += [*vargs, "-r", str(fps), "-movflags", "+faststart", str(out_path)]
        log.debug("$ %s", " ".join(cmd))
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr)
        self.size = size

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            self._raise(f"FFmpeg encoder pipe closed: {exc}")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            self._raise(f"FFmpeg exited with code {code}")
        self._stderr.close()

    def _raise(self, msg: str) -> None:
        self.proc.kill()
        self._stderr.seek(0)
        tail = self._stderr.read().decode(errors="replace")[-1500:]
        raise PipelineError(f"{msg}\n{tail}")


def resolve_encoder(settings: Settings) -> str:
    enc = pick_encoder(settings.video.encoder)
    log.info("Video encoder: %s", enc)
    return enc
