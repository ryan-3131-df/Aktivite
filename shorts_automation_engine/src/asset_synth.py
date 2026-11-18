"""
Procedural, royalty-free starter assets.

So the engine works out of the box with zero downloads, this module
synthesises the SFX pack (whoosh, pop, ding, bass_drop, boom, riser) and two
background beds (a phonk loop and a suspense loop) with NumPy, then encodes
them to MP3 with FFmpeg. Drop your own files into assets/sfx or assets/music
with the same names to override them.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .config import FONTS_DIR, MUSIC_DIR, SFX_DIR
from .utils import log, run

SR = 48000
SFX_NAMES = ("whoosh", "pop", "ding", "bass_drop", "boom", "riser")
FONT_URLS = {
    "Anton-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Montserrat-Black.ttf": "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Black.ttf",
    "Montserrat-ExtraBold.ttf": "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-ExtraBold.ttf",
}


# --------------------------------------------------------------------------- #
# DSP helpers (vectorised, no SciPy needed)
# --------------------------------------------------------------------------- #
def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def _osc(freq: np.ndarray | float, n: int, shape: str = "sine") -> np.ndarray:
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    phase = 2 * np.pi * np.cumsum(f) / SR
    if shape == "square":
        return np.sign(np.sin(phase))
    if shape == "saw":
        return 2 * ((phase / (2 * np.pi)) % 1.0) - 1
    return np.sin(phase)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Zero-phase FFT band-pass with soft edges."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    lo_w = 1 / (1 + (lo / np.maximum(f, 1e-6)) ** 4) if lo > 0 else 1.0
    hi_w = 1 / (1 + (f / hi) ** 4) if hi < SR / 2 else 1.0
    return np.fft.irfft(X * lo_w * hi_w, n)


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, n)


def _env_ad(n: int, attack: float, decay_tau: float) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / decay_tau)


def _norm(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(x)) or 1.0
    return x / peak * (10 ** (peak_db / 20))


def _stereo(x: np.ndarray, pan: np.ndarray | float = 0.0) -> np.ndarray:
    pan = np.broadcast_to(np.asarray(pan, dtype=np.float64), x.shape)
    left = x * np.cos((pan + 1) * np.pi / 4)
    right = x * np.sin((pan + 1) * np.pi / 4)
    return np.stack([left, right], axis=1)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    if audio.ndim == 1:
        audio = _stereo(audio)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _to_mp3(audio: np.ndarray, dst: Path, bitrate: str = "192k") -> Path:
    tmp = dst.with_suffix(".tmp.wav")
    _write_wav(tmp, audio)
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp), "-c:a", "libmp3lame",
         "-b:a", bitrate, str(dst)])
    tmp.unlink(missing_ok=True)
    return dst


# --------------------------------------------------------------------------- #
# SFX
# --------------------------------------------------------------------------- #
def synth_whoosh(seconds: float = 0.7) -> np.ndarray:
    n = int(seconds * SR)
    noise = _noise(n, 1)
    # Time-varying band-pass via overlap-add of short FFT-filtered blocks.
    block, hop = 2048, 512
    out = np.zeros(n + block)
    win = np.hanning(block)
    for start in range(0, n, hop):
        p = start / n
        center = 300 + 4200 * np.sin(np.pi * p) ** 1.5
        seg = np.zeros(block)
        chunk = noise[start:start + block]
        seg[: len(chunk)] = chunk
        out[start:start + block] += _bandpass(seg * win, center * 0.45, center * 1.8)
    out = out[:n]
    t = np.arange(n) / n
    env = np.sin(np.pi * np.clip(t, 0, 1)) ** 2.2 * np.where(t < 0.55, 1.0, np.exp(-(t - 0.55) * 4))
    return _norm(_stereo(out * env, pan=np.linspace(-0.8, 0.8, n)))


def synth_pop(seconds: float = 0.16) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    freq = 420 + 900 * np.exp(-t * 55)
    body = _osc(freq, n) * _env_ad(n, 0.002, 0.035)
    click = _bandpass(_noise(n, 2), 2000, 9000) * _env_ad(n, 0.0005, 0.004) * 0.5
    return _norm(body + click)


def synth_ding(seconds: float = 1.6) -> np.ndarray:
    n = int(seconds * SR)
    base = 1318.5  # E6
    partials = [(1.0, 1.0, 0.9), (2.756, 0.45, 0.45), (5.404, 0.25, 0.25), (8.933, 0.12, 0.12)]
    x = sum(a * _osc(base * r, n) * _env_ad(n, 0.002, tau) for r, a, tau in partials)
    return _norm(x)


def synth_bass_drop(seconds: float = 1.8) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    freq = 32 + 110 * np.exp(-t * 3.2)
    sub = np.tanh(2.2 * _osc(freq, n)) * _env_ad(n, 0.004, 0.75)
    hit = _bandpass(_noise(n, 3), 40, 900) * _env_ad(n, 0.001, 0.06) * 0.9
    return _norm(sub + hit)


def synth_boom(seconds: float = 2.2) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    low = _osc(38 + 50 * np.exp(-t * 6), n) * _env_ad(n, 0.003, 0.9)
    rumble = _bandpass(_noise(n, 4), 25, 260) * _env_ad(n, 0.002, 0.5) * 2.5
    crack = _bandpass(_noise(n, 5), 800, 6000) * _env_ad(n, 0.0005, 0.03) * 0.6
    return _norm(np.tanh(1.6 * (low + rumble)) + crack)


def synth_riser(seconds: float = 1.6) -> np.ndarray:
    n = int(seconds * SR)
    p = np.linspace(0, 1, n)
    tone = _osc(180 * (12 ** p), n, "saw") * 0.25
    hiss = _bandpass(_noise(n, 6), 1500, 12000) * 0.6
    env = p ** 2.4
    return _norm(_stereo((tone + hiss * p) * env, pan=np.sin(p * 18) * 0.35))


SFX_BUILDERS = {
    "whoosh": synth_whoosh, "pop": synth_pop, "ding": synth_ding,
    "bass_drop": synth_bass_drop, "boom": synth_boom, "riser": synth_riser,
}


# --------------------------------------------------------------------------- #
# Music beds
# --------------------------------------------------------------------------- #
def _place(track: np.ndarray, sample: np.ndarray, at: int) -> None:
    """Add `sample` into `track` at sample index `at`, wrapping around for seamless loops."""
    n = len(track)
    at %= n
    sample = sample[:n]
    end = at + len(sample)
    if end <= n:
        track[at:end] += sample
    else:
        cut = n - at
        track[at:] += sample[:cut]
        track[: end - n] += sample[cut:]


def synth_phonk(bars: int = 16, bpm: float = 140.0) -> np.ndarray:
    """Dark drift-phonk loop: cowbell riff, distorted 808, half-time drums, triplet hats."""
    beat = 60.0 / bpm
    step = beat / 4  # 16th notes
    n = int(round(bars * 4 * beat * SR))
    drums, bass, bell = np.zeros(n), np.zeros(n), np.zeros(n)

    def note_hz(semis_from_a4: int) -> float:
        return 440.0 * 2 ** (semis_from_a4 / 12)

    kick_n = int(0.45 * SR)
    kt = np.arange(kick_n) / SR
    kick = np.tanh(3 * _osc(45 + 140 * np.exp(-kt * 35), kick_n) * _env_ad(kick_n, 0.001, 0.16))
    clap_n = int(0.25 * SR)
    clap = _bandpass(_noise(clap_n, 7), 900, 7000) * _env_ad(clap_n, 0.001, 0.06)
    hat_n = int(0.06 * SR)
    hat = _bandpass(_noise(hat_n, 8), 7000, 16000) * _env_ad(hat_n, 0.0005, 0.012)

    # E minor riff (semitones relative to A4): E5=7, G5=10, F#5=9, D5=5, B4=2
    riff_a = [7, None, 7, 10, None, 9, None, 7, 5, None, 7, None, 2, None, 5, None]
    riff_b = [7, None, 7, 10, None, 12, None, 10, 9, None, 7, None, 5, None, 2, None]
    bass_roots = [-29, -29, -26, -31]  # E1, E1, G1, D1 (per bar, cycling)

    for bar in range(bars):
        bar0 = bar * 4 * beat
        for s in range(16):
            t0 = int((bar0 + s * step) * SR)
            # Half-time drums: kick on 1 and the "and" of 3, clap on 3.
            if s in (0, 10) or (bar % 4 == 3 and s == 14):
                _place(drums, kick, t0)
            if s == 8:
                _place(drums, clap * 0.9, t0)
            if s % 2 == 0:
                _place(drums, hat * (0.55 if s % 4 else 0.8), t0)
            if bar % 2 == 1 and s in (12, 13):  # hat roll
                for k in range(3):
                    _place(drums, hat * 0.4, t0 + int(k * step / 3 * SR))
            riff = riff_a if (bar // 4) % 2 == 0 else riff_b
            semi = riff[s]
            if semi is not None:
                f = note_hz(semi)
                cn = int(0.22 * SR)
                cb = (_osc(f, cn, "square") + 0.7 * _osc(f * 1.4836, cn, "square"))
                cb = _bandpass(cb, 500, 4200) * _env_ad(cn, 0.001, 0.06)
                _place(bell, cb * (0.9 if s % 4 == 0 else 0.7), t0)
        # 808: long note per bar with a glide into the next root.
        root = note_hz(bass_roots[bar % 4])
        nxt = note_hz(bass_roots[(bar + 1) % 4])
        bn = int(4 * beat * SR)
        p = np.arange(bn) / bn
        freq = np.where(p < 0.85, root, root + (nxt - root) * (p - 0.85) / 0.15)
        b808 = np.tanh(2.5 * _osc(freq, bn)) * _env_ad(bn, 0.003, 1.4)
        _place(bass, b808, int(bar0 * SR))

    # Slap-back delay on the cowbell (wraps for a seamless loop).
    d = int(beat * 0.75 * SR)
    bell = bell + 0.35 * np.roll(bell, d) + 0.12 * np.roll(bell, 2 * d)
    mix = 0.9 * drums + 0.75 * bass + 0.42 * bell
    return _norm(np.tanh(1.2 * mix), -1.0)


def synth_suspense(seconds: float = 32.0, bpm: float = 96.0) -> np.ndarray:
    """Dark cinematic bed: detuned drone, heartbeat pulse, ticking clock."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    drone = sum(_osc(f, n, "saw") for f in (55.0, 55.4, 82.4, 110.2)) / 4
    drone = _bandpass(drone, 30, 900) * (0.6 + 0.4 * np.sin(2 * np.pi * t / seconds * 2) ** 2)
    pulse = np.zeros(n)
    beat = 60 / bpm
    hb_n = int(0.3 * SR)
    hbt = np.arange(hb_n) / SR
    thump = np.tanh(2 * _osc(50 + 40 * np.exp(-hbt * 30), hb_n) * _env_ad(hb_n, 0.002, 0.09))
    tick_n = int(0.03 * SR)
    tick = _bandpass(_noise(tick_n, 9), 3000, 9000) * _env_ad(tick_n, 0.0003, 0.004)
    k = 0
    while k * beat < seconds:
        at = int(k * beat * SR)
        _place(pulse, thump, at)
        _place(pulse, thump * 0.6, at + int(0.22 * SR))
        _place(pulse, tick * 0.5, at + int(beat / 2 * SR))
        k += 1
    return _norm(np.tanh(0.8 * drone + pulse), -1.0)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def ensure_assets(force: bool = False) -> None:
    """Create any missing SFX / music files (and try to fetch fonts)."""
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in SFX_BUILDERS.items():
        dst = SFX_DIR / f"{name}.mp3"
        if force or not dst.exists():
            log.info("Synthesising SFX → %s", dst.name)
            _to_mp3(builder(), dst)
    beds = {"phonk_default.mp3": synth_phonk, "suspense_default.mp3": synth_suspense}
    for fname, builder in beds.items():
        dst = MUSIC_DIR / fname
        if force or not dst.exists():
            log.info("Synthesising music bed → %s", dst.name)
            _to_mp3(builder(), dst, bitrate="192k")
    ensure_fonts()


def ensure_fonts() -> None:
    """Download the open-licensed (OFL) fonts if they're missing. Failure is non-fatal."""
    missing = [f for f in FONT_URLS if not (FONTS_DIR / f).exists()]
    if not missing:
        return
    try:
        import requests
    except ImportError:
        return
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in missing:
        try:
            r = requests.get(FONT_URLS[fname], timeout=30)
            r.raise_for_status()
            (FONTS_DIR / fname).write_bytes(r.content)
            log.info("Downloaded font %s", fname)
        except Exception as exc:  # noqa: BLE001 - fonts are a nice-to-have
            log.warning("Could not download %s (%s); falling back to system fonts.", fname, exc)


def sfx_path(name: str) -> Path | None:
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        p = SFX_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None
