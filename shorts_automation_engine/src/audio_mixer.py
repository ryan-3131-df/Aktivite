"""
Audio layering:

  voice  → two-pass EBU R128 loudnorm to -14 LUFS (primary)
  music  → looped/trimmed bed normalised to -22 LUFS (8 LU under the voice),
           optional sidechain ducking under speech
  SFX    → whoosh on rank transitions, pop on reveals, script cues, placed
           with sample-accurate `adelay`
  master → true-peak limiter at -1 dBTP, 48 kHz stereo WAV
"""

from __future__ import annotations

import random
from pathlib import Path

from .asset_synth import sfx_path
from .config import MUSIC_DIR, Settings
from .ffmpeg_tools import loudnorm_two_pass, measure_loudness
from .timeline import Timeline
from .utils import log, media_duration, run

MUSIC_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")


def pick_music(settings: Settings, mood: str, rng: random.Random) -> Path | None:
    if not settings.use_music:
        return None
    if settings.music_path:
        p = Path(settings.music_path)
        if p.is_file():
            return p
        log.warning("MUSIC_PATH %s not found - choosing from assets/music", p)
    tracks = sorted(p for p in MUSIC_DIR.glob("*") if p.suffix.lower() in MUSIC_EXTS)
    if not tracks:
        log.warning("No music in assets/music - rendering without a beat")
        return None
    # Prefer user tracks over the generated defaults, then tracks matching the script's mood.
    user = [t for t in tracks if "_default" not in t.stem] or tracks
    mood_match = [t for t in user if mood.lower() in t.stem.lower()]
    return rng.choice(mood_match or user)


def build_mix(timeline: Timeline, voice_wav: Path, settings: Settings, work_dir: Path, mood: str = "phonk",
              seed: int | None = None) -> Path:
    a = settings.audio
    sr = a.sample_rate
    D = timeline.duration
    rng = random.Random(seed)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Voice → -14 LUFS, padded/trimmed to the exact video length.
    voice_norm = loudnorm_two_pass(voice_wav, work_dir / "voice_norm.wav", a.voice_lufs, a.true_peak_db,
                                   sample_rate=sr, channels=2, extra_filters=f"apad,atrim=0:{D:.3f}")
    inputs: list[str] = ["-i", str(voice_norm)]
    graph: list[str] = []
    mix_labels: list[str] = []

    if a.music_sidechain:
        graph.append("[0:a]asplit=2[voice][vkey]")
    else:
        graph.append("[0:a]anull[voice]")
    mix_labels.append("[voice]")

    # 2) Music bed → loop, random start, trim, -22 LUFS.
    music = pick_music(settings, mood, rng)
    if music:
        mdur = media_duration(music)
        offset = rng.uniform(0, max(0.0, mdur - D)) if mdur > D + 2 else 0.0
        bed_raw = work_dir / "music_bed_raw.wav"
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stream_loop", "-1", "-ss", f"{offset:.2f}",
             "-i", str(music), "-t", f"{D:.3f}", "-af", f"afade=t=in:d={a.music_fade_in}",
             "-ar", str(sr), "-ac", "2", "-c:a", "pcm_s16le", str(bed_raw)])
        bed = loudnorm_two_pass(bed_raw, work_dir / "music_bed.wav", a.music_lufs, a.true_peak_db - 2,
                                sample_rate=sr, channels=2)
        inputs += ["-i", str(bed)]
        idx = len(inputs) // 2 - 1
        if a.music_sidechain:
            graph.append(f"[{idx}:a][vkey]sidechaincompress=threshold=0.03:ratio=3:attack=15:release=280:makeup=1[music]")
        else:
            graph.append(f"[{idx}:a]anull[music]")
        mix_labels.append("[music]")
        log.info("Music: %s (offset %.1fs) at %.0f LUFS%s", music.name, offset, a.music_lufs,
                 " + sidechain ducking" if a.music_sidechain else "")

    # 3) SFX events.
    placed = 0
    for ev in timeline.sfx:
        path = sfx_path(ev.name)
        if path is None:
            log.debug("SFX '%s' missing - skipped", ev.name)
            continue
        start = ev.time - (media_duration(path) if ev.align == "end" else 0.0)
        if start < 0 or start >= D:
            continue
        inputs += ["-i", str(path)]
        idx = len(inputs) // 2 - 1
        delay_ms = int(round(start * 1000))
        gain = a.sfx_gain_db.get(ev.name, -10.0)
        label = f"[s{placed}]"
        graph.append(f"[{idx}:a]aformat=sample_rates={sr}:channel_layouts=stereo,volume={gain}dB,"
                     f"adelay=delays={delay_ms}:all=1{label}")
        mix_labels.append(label)
        placed += 1

    # 4) Mix + true-peak limiter.
    limit = 10 ** (a.true_peak_db / 20)
    graph.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0:dropout_transition=0,"
                 f"alimiter=limit={limit:.4f}:attack=3:release=60:level=false,"
                 f"atrim=0:{D:.3f},aformat=sample_rates={sr}:channel_layouts=stereo[out]")
    out = work_dir / "mix.wav"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs, "-filter_complex", ";".join(graph),
         "-map", "[out]", "-c:a", "pcm_s16le", str(out)])

    try:
        m = measure_loudness(out)
        log.info("Mix: %d SFX placed · integrated %.1f LUFS · true peak %.1f dBTP",
                 placed, float(m["input_i"]), float(m["input_tp"]))
    except Exception:  # noqa: BLE001 - measurement is informational only
        log.info("Mix: %d SFX placed", placed)
    return out
