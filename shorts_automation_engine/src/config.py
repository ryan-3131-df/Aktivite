"""
Central configuration for the Shorts Automation Engine.

Every tunable lives here. Values are read from environment variables (loaded
from `.env` at the project root) and can be overridden per-run from the CLI
(see `main.py`). Nothing in the rest of the codebase should read `os.environ`
directly - pass a `Settings` object instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time so `check_system.py` can run first
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SFX_DIR = ASSETS_DIR / "sfx"
FONTS_DIR = ASSETS_DIR / "fonts"
MUSIC_DIR = ASSETS_DIR / "music"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_CLIPS_DIR = PROJECT_ROOT / "input_clips"
CACHE_DIR = PROJECT_ROOT / "cache"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #
def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value.strip() if value is not None and value.strip() != "" else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "")
    if raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


# --------------------------------------------------------------------------- #
# Settings groups
# --------------------------------------------------------------------------- #
@dataclass
class VideoSettings:
    width: int = 1080
    height: int = 1920
    fps: int = field(default_factory=lambda: _env_int("VIDEO_FPS", 60))

    # Ken Burns: every shot continuously zooms from zoom_start -> zoom_end.
    zoom_start: float = 1.0
    zoom_end: float = field(default_factory=lambda: _env_float("KEN_BURNS_ZOOM", 1.08))
    # Extra "punch" zoom applied at the first frames of every rank reveal (0.06 = +6%).
    punch_zoom: float = 0.06
    punch_seconds: float = 0.22
    # White flash on rank cuts (alpha at the cut, fades out over flash_seconds).
    flash_on_cut: bool = True
    flash_alpha: float = 0.45
    flash_seconds: float = 0.12

    # How non-9:16 clips are framed: smart | crop | blur
    #   crop  -> scale to cover 1080x1920 and center-crop
    #   blur  -> fit inside a blurred, darkened copy of itself (letterbox killer)
    #   smart -> crop when the clip is already portrait-ish, otherwise blur-pad
    fit_mode: str = field(default_factory=lambda: _env("FIT_MODE", "smart"))
    # Foreground scale for blur-pad, relative to "fit to width" (1.15 crops the sides slightly
    # so landscape footage reads bigger on a phone).
    blur_pad_zoom: float = field(default_factory=lambda: _env_float("BLUR_PAD_ZOOM", 1.15))
    # Rapid cuts: if a rank has several source clips, cut between them every ~N seconds.
    target_shot_seconds: float = 2.4
    # Hook visuals: "montage" (rapid teaser of the ranked clips) or "clip" (hook keywords clip).
    hook_style: str = field(default_factory=lambda: _env("HOOK_STYLE", "montage"))

    # Final encode.
    encoder: str = field(default_factory=lambda: _env("VIDEO_ENCODER", "auto"))
    x264_preset: str = field(default_factory=lambda: _env("X264_PRESET", "medium"))
    crf: int = field(default_factory=lambda: _env_int("VIDEO_CRF", 18))
    hwaccel_decode: bool = field(default_factory=lambda: _env_bool("HWACCEL_DECODE", False))


@dataclass
class CaptionSettings:
    max_words: int = 3
    max_chars: int = 16           # soft limit per chunk (uppercase chars, excluding spaces)
    pause_break: float = 0.35     # start a new chunk after a pause this long (s)
    font_size: int = 118
    min_font_size: int = 78
    max_width: int = 960
    stroke_width: int = 12
    y_center: int = 1200          # caption block vertical center (px)
    base_color: tuple[int, int, int] = (255, 255, 255)
    # Active-word highlight colors, alternating per chunk (Yellow / Green).
    highlight_colors: tuple[tuple[int, int, int], ...] = ((255, 226, 0), (60, 255, 80))
    shadow_offset: tuple[int, int] = (5, 9)
    shadow_blur: int = 7
    shadow_alpha: int = 190
    pop_seconds: float = 0.13
    pop_from: float = 0.72


@dataclass
class AudioSettings:
    voice_lufs: float = field(default_factory=lambda: _env_float("VOICE_LUFS", -14.0))
    music_lufs: float = field(default_factory=lambda: _env_float("MUSIC_LUFS", -22.0))
    true_peak_db: float = -1.0
    # Optional dynamic ducking of the music under the voice (on top of the -22 LUFS bed).
    music_sidechain: bool = field(default_factory=lambda: _env_bool("MUSIC_SIDECHAIN", False))
    music_fade_in: float = 0.15
    sample_rate: int = 48000
    # Gain (dB) applied to each SFX file in the mix. SFX files are peak-normalised to -1 dBFS.
    sfx_gain_db: dict[str, float] = field(
        default_factory=lambda: {
            "whoosh": -12.0,
            "pop": -11.0,
            "ding": -13.0,
            "bass_drop": -9.0,
            "boom": -10.0,
            "riser": -15.0,
        }
    )


@dataclass
class Settings:
    # --- API keys ---------------------------------------------------------- #
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))
    pexels_api_key: str = field(default_factory=lambda: _env("PEXELS_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))  # optional (Whisper API)

    # --- Script generation (Claude) --------------------------------------- #
    claude_model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-opus-5-5"))
    claude_effort: str = field(default_factory=lambda: _env("CLAUDE_EFFORT", "high"))
    # Server-side refusal fallback (re-runs a declined request on a fallback model).
    claude_fallbacks: bool = field(default_factory=lambda: _env_bool("CLAUDE_FALLBACKS", True))
    language: str = field(default_factory=lambda: _env("LANGUAGE", "en"))
    target_min_seconds: float = 40.0
    target_max_seconds: float = 50.0
    words_per_second: float = field(default_factory=lambda: _env_float("WORDS_PER_SECOND", 2.9))

    # --- Voice (ElevenLabs) ------------------------------------------------ #
    # Default: "Adam" premade voice. Any voice ID from your ElevenLabs library works.
    elevenlabs_voice_id: str = field(default_factory=lambda: _env("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB"))
    elevenlabs_model_id: str = field(default_factory=lambda: _env("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"))
    elevenlabs_output_format: str = field(default_factory=lambda: _env("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"))
    # High-energy, fast-paced delivery.
    voice_stability: float = field(default_factory=lambda: _env_float("VOICE_STABILITY", 0.32))
    voice_similarity: float = field(default_factory=lambda: _env_float("VOICE_SIMILARITY", 0.80))
    voice_style: float = field(default_factory=lambda: _env_float("VOICE_STYLE", 0.55))
    voice_speed: float = field(default_factory=lambda: _env_float("VOICE_SPEED", 1.10))
    voice_speaker_boost: bool = True
    # Max time-stretch applied if the voiceover runs longer than target_max_seconds.
    max_tempo_fit: float = 1.15

    # --- Word timestamps --------------------------------------------------- #
    # auto | faster-whisper | openai | elevenlabs | estimate
    timestamp_backend: str = field(default_factory=lambda: _env("TIMESTAMP_BACKEND", "auto"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base.en"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "auto"))

    # --- Media sourcing ---------------------------------------------------- #
    clips_per_rank: int = field(default_factory=lambda: _env_int("CLIPS_PER_RANK", 2))
    allow_ytdlp: bool = field(default_factory=lambda: _env_bool("ALLOW_YTDLP", True))
    ytdlp_creative_commons_only: bool = field(default_factory=lambda: _env_bool("YTDLP_CC_ONLY", True))
    ytdlp_age_limit: int = 13  # safe-search equivalent: skip age-restricted videos
    max_download_height: int = 2160

    # --- Music ------------------------------------------------------------- #
    music_path: str = field(default_factory=lambda: _env("MUSIC_PATH", ""))
    use_music: bool = field(default_factory=lambda: _env_bool("USE_MUSIC", True))

    seed: int | None = None

    video: VideoSettings = field(default_factory=VideoSettings)
    captions: CaptionSettings = field(default_factory=CaptionSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.video.width, self.video.height

    def target_word_range(self) -> tuple[int, int]:
        lo = int(self.target_min_seconds * self.words_per_second * 0.92)
        hi = int(self.target_max_seconds * self.words_per_second * 0.97)
        return lo, hi
