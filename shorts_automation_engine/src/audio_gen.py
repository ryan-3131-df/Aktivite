"""
Stage 2 - Voiceover + word-level timestamp engine.

1. ElevenLabs text-to-speech (high-energy settings) → voiceover.mp3
2. Leading-silence trim + optional tempo fit so the Short lands in 40-50 s
3. Word timestamps from Whisper (faster-whisper locally, or the OpenAI
   Whisper API), with ElevenLabs' own character alignment and a heuristic
   estimate as fallbacks
4. Alignment of the recognised words back onto the *script* words, so captions
   always show the exact script text with ASR-accurate timing.
"""

from __future__ import annotations

import base64
import difflib
import importlib.util
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import Settings
from .ffmpeg_tools import detect_leading_silence, to_wav
from .models import ShortScript, WordStamp
from .utils import PipelineError, has_alnum, log, media_duration, normalize_token, read_json, write_json

ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"

RawWord = tuple[str, float, float]  # (text, start, end)


@dataclass
class VoiceoverResult:
    audio_path: Path          # processed mono WAV used for the mix
    words: list[WordStamp]    # one entry per spoken script token
    duration: float
    backend: str


# --------------------------------------------------------------------------- #
# Script tokenisation
# --------------------------------------------------------------------------- #
def script_tokens(script: ShortScript) -> list[tuple[str, int]]:
    """(token, segment_index) for every spoken token in the script."""
    out: list[tuple[str, int]] = []
    for si, seg in enumerate(script.segments):
        out.extend((tok, si) for tok in seg.voiceover_text.split() if has_alnum(tok))
    return out


# --------------------------------------------------------------------------- #
# TTS backends
# --------------------------------------------------------------------------- #
def elevenlabs_tts(text: str, settings: Settings, mp3_path: Path) -> list[RawWord]:
    """Synthesize with ElevenLabs; returns word timings derived from its character alignment."""
    if not settings.elevenlabs_api_key:
        raise PipelineError("ELEVENLABS_API_KEY is missing - add it to .env (or use --dry-run).")
    url = ELEVEN_TTS_URL.format(voice_id=settings.elevenlabs_voice_id)
    payload = {
        "text": text,
        "model_id": settings.elevenlabs_model_id,
        "voice_settings": {
            "stability": settings.voice_stability,
            "similarity_boost": settings.voice_similarity,
            "style": settings.voice_style,
            "use_speaker_boost": settings.voice_speaker_boost,
            "speed": settings.voice_speed,
        },
    }
    headers = {"xi-api-key": settings.elevenlabs_api_key, "Content-Type": "application/json", "Accept": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.post(url, params={"output_format": settings.elevenlabs_output_format},
                                 json=payload, headers=headers, timeout=180)
        except requests.RequestException as exc:
            if attempt == 3:
                raise PipelineError(f"Could not reach ElevenLabs: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            break
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            wait = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
            log.warning("ElevenLabs %s - retrying in %.0fs", resp.status_code, wait)
            time.sleep(wait)
            continue
        detail = resp.text[:400]
        if resp.status_code == 401:
            raise PipelineError(f"ElevenLabs rejected the API key (401). {detail}")
        raise PipelineError(f"ElevenLabs TTS failed ({resp.status_code}): {detail}")

    data = resp.json()
    mp3_path.write_bytes(base64.b64decode(data["audio_base64"]))
    align = data.get("alignment") or data.get("normalized_alignment") or {}
    return _chars_to_words(align.get("characters") or [],
                           align.get("character_start_times_seconds") or [],
                           align.get("character_end_times_seconds") or [])


def _chars_to_words(chars: list[str], starts: list[float], ends: list[float]) -> list[RawWord]:
    words: list[RawWord] = []
    buf, w_start, w_end = "", None, None
    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            if buf.strip():
                words.append((buf, w_start, w_end))
            buf, w_start = "", None
            continue
        if w_start is None:
            w_start = s
        buf += ch
        w_end = e
    if buf.strip():
        words.append((buf, w_start, w_end))
    return [w for w in words if has_alnum(w[0])]


def offline_tts(text: str, wav_path: Path, settings: Settings) -> bool:
    """Best-effort free local TTS for --dry-run. Returns False if only silence could be produced."""
    tmp = wav_path.with_suffix(".raw.wav")
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            subprocess.run([exe, "-s", "190", "-w", str(tmp), text], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            break
    else:
        if shutil.which("say"):  # macOS
            aiff = wav_path.with_suffix(".aiff")
            subprocess.run(["say", "-r", "210", "-o", str(aiff), text], check=False)
            if aiff.exists():
                to_wav(aiff, tmp)
                aiff.unlink(missing_ok=True)
    if tmp.exists() and tmp.stat().st_size > 1000:
        to_wav(tmp, wav_path)
        tmp.unlink(missing_ok=True)
        return True
    # Silence with the estimated narration length.
    seconds = len([w for w in text.split() if has_alnum(w)]) / settings.words_per_second + 0.3
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=mono", "-t", f"{seconds:.2f}", "-c:a", "pcm_s16le", str(wav_path)],
                   check=True)
    return False


# --------------------------------------------------------------------------- #
# Timestamp backends
# --------------------------------------------------------------------------- #
def whisper_words(wav: Path, settings: Settings, prompt: str) -> list[RawWord]:
    from faster_whisper import WhisperModel  # heavy import, keep local

    model_name = settings.whisper_model
    if not settings.language.startswith("en") and model_name.endswith(".en"):
        model_name = model_name[:-3]
    log.info("faster-whisper: loading '%s' (first run downloads the model)…", model_name)
    model = WhisperModel(model_name, device=settings.whisper_device, compute_type="auto")
    segments, _info = model.transcribe(
        str(wav), language=settings.language[:2], word_timestamps=True, beam_size=5,
        vad_filter=False, condition_on_previous_text=False, initial_prompt=prompt[-600:] or None,
    )
    return [(w.word.strip(), float(w.start), float(w.end)) for seg in segments for w in (seg.words or [])
            if has_alnum(w.word)]


def openai_whisper_words(wav: Path, settings: Settings, prompt: str) -> list[RawWord]:
    if not settings.openai_api_key:
        raise PipelineError("OPENAI_API_KEY is required for the 'openai' timestamp backend")
    with open(wav, "rb") as fh:
        resp = requests.post(
            OPENAI_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            data={"model": "whisper-1", "response_format": "verbose_json", "timestamp_granularities[]": "word",
                  "language": settings.language[:2], "prompt": prompt[-600:]},
            files={"file": (wav.name, fh, "audio/wav")}, timeout=300)
    if resp.status_code != 200:
        raise PipelineError(f"OpenAI Whisper API failed ({resp.status_code}): {resp.text[:300]}")
    return [(w["word"], float(w["start"]), float(w["end"])) for w in resp.json().get("words", []) if has_alnum(w["word"])]


def estimate_words(tokens: list[str], duration: float, lead: float = 0.05) -> list[RawWord]:
    """Heuristic timing: distribute time by syllable-ish weight with extra pause after punctuation."""
    def weight(tok: str) -> float:
        core = sum(ch.isalnum() for ch in tok)
        w = 0.55 + 0.11 * core
        if tok[-1:] in ",;:—":
            w += 0.35
        elif tok[-1:] in ".!?":
            w += 0.6
        return w
    weights = [weight(t) for t in tokens]
    total = sum(weights) or 1.0
    span = max(duration - lead - 0.1, 0.5)
    out, t = [], lead
    for tok, w in zip(tokens, weights):
        dur = span * w / total
        out.append((tok, t, t + dur * 0.82))
        t += dur
    return out


# --------------------------------------------------------------------------- #
# Alignment: ASR words -> script tokens
# --------------------------------------------------------------------------- #
def align_to_script(tokens: list[tuple[str, int]], asr: list[RawWord], duration: float) -> list[WordStamp]:
    """
    Map every script token onto ASR timings with difflib on normalised words.
    Replaced / merged / split spans share their time range proportionally to
    length; tokens the ASR missed are interpolated between their neighbours.
    """
    # Flatten normalised sub-words, remembering which original token each came from.
    a_norm, a_owner = [], []
    for i, (tok, _) in enumerate(tokens):
        for piece in normalize_token(tok) or ["_"]:
            a_norm.append(piece)
            a_owner.append(i)
    b_norm, b_owner = [], []
    for j, (w, _, _) in enumerate(asr):
        for piece in normalize_token(w) or ["_"]:
            b_norm.append(piece)
            b_owner.append(j)

    starts: list[float | None] = [None] * len(tokens)
    ends: list[float | None] = [None] * len(tokens)

    def assign(a_lo: int, a_hi: int, t0: float, t1: float) -> None:
        owners = sorted(set(a_owner[a_lo:a_hi]))
        lengths = [max(len(tokens[o][0]), 1) for o in owners]
        total = sum(lengths)
        cur = t0
        for o, ln in zip(owners, lengths):
            nxt = cur + (t1 - t0) * ln / total
            starts[o] = cur if starts[o] is None else min(starts[o], cur)
            ends[o] = nxt if ends[o] is None else max(ends[o], nxt)
            cur = nxt

    matcher = difflib.SequenceMatcher(None, a_norm, b_norm, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op in ("equal", "replace") and j2 > j1:
            if op == "equal":
                for k in range(i2 - i1):
                    j = b_owner[j1 + k]
                    assign(i1 + k, i1 + k + 1, asr[j][1], asr[j][2])
            else:
                js = sorted(set(b_owner[j1:j2]))
                assign(i1, i2, asr[js[0]][1], asr[js[-1]][2])

    # Interpolate gaps (tokens the recogniser never produced).
    n = len(tokens)
    i = 0
    while i < n:
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < n and starts[j] is None:
            j += 1
        left = ends[i - 1] if i > 0 and ends[i - 1] is not None else 0.0
        right = starts[j] if j < n and starts[j] is not None else max(duration - 0.05, left + 0.2 * (j - i))
        step = (right - left) / (j - i)
        for k in range(i, j):
            starts[k] = left + step * (k - i)
            ends[k] = starts[k] + step * 0.9
        i = j

    # Enforce monotonic, non-overlapping timings.
    words: list[WordStamp] = []
    prev_end = 0.0
    for (tok, seg), s, e in zip(tokens, starts, ends):
        s = max(float(s), prev_end)
        e = max(float(e), s + 0.04)
        words.append(WordStamp(word=tok, start=round(s, 3), end=round(e, 3), segment_index=seg))
        prev_end = e
    return words


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _choose_backend(settings: Settings, have_eleven_alignment: bool, silent: bool) -> str:
    if silent:
        return "estimate"
    wanted = settings.timestamp_backend
    if wanted == "auto":
        if importlib.util.find_spec("faster_whisper") is not None:
            return "faster-whisper"
        if settings.openai_api_key:
            return "openai"
        return "elevenlabs" if have_eleven_alignment else "estimate"
    return wanted


def generate_voiceover(script: ShortScript, settings: Settings, out_dir: Path, *, dry_run: bool = False,
                       existing_audio: Path | None = None) -> VoiceoverResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = script.full_voiceover()
    tokens = script_tokens(script)
    raw_wav = out_dir / "voice_raw.wav"
    eleven_words: list[RawWord] = []
    silent = False

    if existing_audio:
        log.info("Using existing voiceover %s", existing_audio)
        to_wav(existing_audio, raw_wav)
    elif dry_run:
        spoke = offline_tts(text, raw_wav, settings)
        silent = not spoke
        log.info("Dry-run voice: %s", "local TTS" if spoke else "silent placeholder (no espeak/say found)")
    else:
        mp3 = out_dir / "voiceover.mp3"
        log.info("ElevenLabs TTS: voice=%s model=%s speed=%.2f", settings.elevenlabs_voice_id,
                 settings.elevenlabs_model_id, settings.voice_speed)
        eleven_words = elevenlabs_tts(text, settings, mp3)
        write_json(out_dir / "elevenlabs_alignment.json", eleven_words)
        to_wav(mp3, raw_wav)

    # Trim leading silence so the hook hits on frame 1, then fit into the target length.
    lead = 0.0 if silent else max(0.0, detect_leading_silence(raw_wav) - 0.03)
    raw_dur = media_duration(raw_wav) - lead
    tempo = 1.0
    if raw_dur > settings.target_max_seconds - 0.5:
        tempo = min(raw_dur / (settings.target_max_seconds - 0.8), settings.max_tempo_fit)
        log.info("Voiceover is %.1fs - speeding up ×%.3f to fit the %ds target", raw_dur,
                 tempo, settings.target_max_seconds)
    elif raw_dur < settings.target_min_seconds:
        log.warning("Voiceover is only %.1fs (target %.0f-%.0fs). Consider a longer script.",
                    raw_dur, settings.target_min_seconds, settings.target_max_seconds)
    filters = [f"atrim=start={lead:.3f}", "asetpts=PTS-STARTPTS"]
    if abs(tempo - 1.0) > 1e-3:
        filters.append(f"atempo={tempo:.4f}")
    voice_wav = out_dir / "voice.wav"
    to_wav(raw_wav, voice_wav, filters=",".join(filters))
    duration = media_duration(voice_wav)

    backend = _choose_backend(settings, bool(eleven_words), silent)
    asr: list[RawWord] = []
    try:
        if backend == "faster-whisper":
            asr = whisper_words(voice_wav, settings, text)
        elif backend == "openai":
            asr = openai_whisper_words(voice_wav, settings, text)
        elif backend == "elevenlabs":
            if not eleven_words:
                raise PipelineError("ElevenLabs alignment is only available for fresh ElevenLabs TTS")
            asr = [(w, (s - lead) / tempo, (e - lead) / tempo) for w, s, e in eleven_words]
    except Exception as exc:  # noqa: BLE001 - timing has safe fallbacks; never fail the run here
        fallback = "elevenlabs" if eleven_words else "estimate"
        log.warning("Timestamp backend '%s' failed (%s) - falling back to '%s'", backend, exc, fallback)
        backend = fallback
        asr = [(w, (s - lead) / tempo, (e - lead) / tempo) for w, s, e in eleven_words] if eleven_words else []
    if backend == "estimate" or not asr:
        backend = "estimate"
        asr = estimate_words([t for t, _ in tokens], duration)

    words = align_to_script(tokens, asr, duration)
    write_json(out_dir / "words.json", [w.model_dump() for w in words])
    log.info("Voiceover %.2fs, %d words timed via %s", duration, len(words), backend)
    return VoiceoverResult(audio_path=voice_wav, words=words, duration=duration, backend=backend)


def load_voiceover(out_dir: Path) -> VoiceoverResult | None:
    """Resume support: reuse voice.wav + words.json from a previous run directory."""
    wav, wj = out_dir / "voice.wav", out_dir / "words.json"
    if not (wav.exists() and wj.exists()):
        return None
    words = [WordStamp.model_validate(w) for w in read_json(wj)]
    return VoiceoverResult(audio_path=wav, words=words, duration=media_duration(wav), backend="cached")
