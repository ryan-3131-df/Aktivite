#!/usr/bin/env python3
"""
Shorts Automation Engine - CLI orchestrator.

    python main.py "deadliest animals on earth"            # full auto: Claude → ElevenLabs → Pexels → render
    python main.py --local-clips                           # script/voice/caption/render around ./input_clips
    python main.py "street food" --local-clips ~/clips --keep-order
    python main.py "space facts" --dry-run                 # no API keys needed: template script, offline voice
    python main.py --check                                 # verify ffmpeg / packages / keys

Pipeline: script_gen → audio_gen → media_fetcher → timeline → audio_mixer → video_builder
Every run writes to output/<timestamp>_<slug>/ (script.json, words.json, timeline.json, short.mp4 …).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py", description="Generate a viral 'Top 6 Ranking' YouTube Short (1080x1920, 60 fps) from one topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("topic", nargs="?", default="", help="Ranking topic, e.g. \"most dangerous roads\" (optional with --local-clips)")

    src = p.add_argument_group("sources")
    src.add_argument("--local-clips", nargs="?", const="input_clips", metavar="DIR",
                     help="Use your own 6 raw clips (default folder: ./input_clips) instead of fetching footage")
    src.add_argument("--keep-order", action="store_true",
                     help="With --local-clips: pin ranks by filename (\"1_x.mp4\"=#1 …) or by name order (first=#6)")
    src.add_argument("--dry-run", action="store_true",
                     help="No paid API calls: template script, offline/placeholder voice, local or placeholder clips")
    src.add_argument("--script-json", metavar="PATH", help="Reuse an existing script.json (skips Claude)")
    src.add_argument("--voiceover", metavar="PATH", help="Reuse an existing voiceover audio file (skips ElevenLabs)")
    src.add_argument("--resume", metavar="RUN_DIR", help="Continue/re-render a previous run directory, reusing its artifacts")
    src.add_argument("--script-only", action="store_true", help="Stop after writing script.json (review before spending TTS credits)")

    tune = p.add_argument_group("tuning")
    tune.add_argument("--voice-id", help="ElevenLabs voice ID (overrides ELEVENLABS_VOICE_ID)")
    tune.add_argument("--timestamps", choices=["auto", "faster-whisper", "openai", "elevenlabs", "estimate"],
                      help="Word-timestamp backend (default: auto → faster-whisper)")
    tune.add_argument("--whisper-model", help="faster-whisper model size (tiny.en, base.en, small.en, …)")
    tune.add_argument("--clips-per-rank", type=int, help="Stock clips fetched per rank for rapid cuts (default 2)")
    tune.add_argument("--no-ytdlp", action="store_true", help="Disable the yt-dlp fallback")
    tune.add_argument("--fit", choices=["smart", "crop", "blur"], help="How non-9:16 clips are framed")
    tune.add_argument("--hook-style", choices=["montage", "clip"], help="Hook visuals: rapid teaser montage or a dedicated clip")
    tune.add_argument("--music", metavar="PATH", help="Background track (default: random file in assets/music)")
    tune.add_argument("--no-music", action="store_true", help="Render without background music")
    tune.add_argument("--sidechain", action="store_true", help="Also duck the music dynamically under the voice")
    tune.add_argument("--seed", type=int, help="Seed for reproducible edit decisions")

    out = p.add_argument_group("output")
    out.add_argument("--fps", type=int, help="Output frame rate (default 60)")
    out.add_argument("--encoder", help="auto | libx264 | h264_nvenc | h264_videotoolbox | h264_qsv | h264_amf | h264_vaapi")
    out.add_argument("--preset", help="libx264 preset (default medium)")
    out.add_argument("--crf", type=int, help="Quality (lower = better, default 18)")
    out.add_argument("--out", metavar="PATH", help="Copy the final MP4 to this path as well")
    out.add_argument("--keep-temp", action="store_true", help="Keep intermediate shots / base video")

    misc = p.add_argument_group("misc")
    misc.add_argument("--check", action="store_true", help="Run the system dependency check and exit")
    misc.add_argument("--init-assets", action="store_true", help="(Re)generate starter SFX/music and download fonts, then exit")
    misc.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def apply_overrides(settings, args) -> None:
    if args.voice_id:
        settings.elevenlabs_voice_id = args.voice_id
    if args.timestamps:
        settings.timestamp_backend = args.timestamps
    if args.whisper_model:
        settings.whisper_model = args.whisper_model
    if args.clips_per_rank:
        settings.clips_per_rank = max(1, args.clips_per_rank)
    if args.no_ytdlp:
        settings.allow_ytdlp = False
    if args.fit:
        settings.video.fit_mode = args.fit
    if args.hook_style:
        settings.video.hook_style = args.hook_style
    if args.music:
        settings.music_path = args.music
    if args.no_music:
        settings.use_music = False
    if args.sidechain:
        settings.audio.music_sidechain = True
    if args.fps:
        settings.video.fps = args.fps
    if args.encoder:
        settings.video.encoder = args.encoder
    if args.preset:
        settings.video.x264_preset = args.preset
    if args.crf is not None:
        settings.video.crf = args.crf
    settings.seed = args.seed


def write_metadata(script, media, run_dir: Path, video_path: Path) -> Path:
    """Upload-ready title / description / hashtags + footage credits."""
    credits = sorted({c.credit for clips in media.values() for c in clips if c.credit})
    lines = [
        f"TITLE: {script.youtube_title}", "",
        "DESCRIPTION:", script.youtube_description, "", " ".join(script.hashtags), "",
        f"VIDEO: {video_path}", "",
        "RANKING:", *[f"  #{s.rank}  {s.on_screen_title}" for s in script.segments if s.segment_type == "rank"],
    ]
    if credits:
        lines += ["", "FOOTAGE CREDITS:", *[f"  - {c}" for c in credits]]
    path = run_dir / "metadata.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_pipeline(args: argparse.Namespace) -> Path:
    # Imports are deferred so `--check` works even when Python packages are missing.
    from src import asset_synth
    from src.audio_gen import generate_voiceover, load_voiceover
    from src.audio_mixer import build_mix
    from src.config import OUTPUT_DIR, Settings
    from src.media_fetcher import MediaFetcher, assign_local_clips, load_local_clips
    from src.script_gen import generate_script, load_script, offline_script, save_script
    from src.timeline import build_timeline
    from src.utils import PipelineError, log, slugify, stage, which_or_raise
    from src.video_builder import VideoBuilder

    settings = Settings()
    apply_overrides(settings, args)
    which_or_raise("ffmpeg")
    which_or_raise("ffprobe")
    asset_synth.ensure_assets()

    local_dir = None
    if args.local_clips:
        local_dir = Path(args.local_clips).expanduser()
        if not local_dir.is_absolute():
            local_dir = (ROOT / local_dir) if (ROOT / local_dir).exists() else local_dir.resolve()
    if not args.topic and not local_dir and not args.script_json and not args.resume:
        raise PipelineError("Give a topic, e.g.  python main.py \"most dangerous animals\"  (or use --local-clips)")

    if args.resume:
        run_dir = Path(args.resume).expanduser().resolve()
        if not run_dir.is_dir():
            raise PipelineError(f"--resume directory not found: {run_dir}")
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = OUTPUT_DIR / f"{stamp}_{slugify(args.topic or 'local-clips')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)
    t_start = time.perf_counter()

    # ── 0. Local clips (optional) ─────────────────────────────────────────── #
    local_clips = None
    if local_dir:
        with stage("Scan local clips"):
            local_clips = load_local_clips(local_dir, keep_order=args.keep_order)

    # ── 1. Script & metadata ──────────────────────────────────────────────── #
    script_path = run_dir / "script.json"
    with stage("1/4 Script & metadata"):
        if args.script_json:
            script = load_script(Path(args.script_json))
        elif script_path.exists():
            script = load_script(script_path)
            log.info("Reusing %s", script_path.name)
        elif args.dry_run:
            script = offline_script(args.topic, local_clips)
        else:
            script = generate_script(args.topic, settings, local_clips, keep_order=args.keep_order)
        problems = script.structure_errors(clip_ids=[c.clip_id for c in local_clips] if local_clips else None)
        if problems:
            raise PipelineError("Invalid script: " + "; ".join(problems))
        save_script(script, script_path)
        for seg in script.segments:
            log.info("  %-5s %s", seg.label, seg.voiceover_text)
    if args.script_only:
        log.info("Script written to %s (--script-only)", script_path)
        return script_path

    # ── 2. Voiceover + word timestamps ────────────────────────────────────── #
    with stage("2/4 Voiceover & word timestamps"):
        voice = None if args.voiceover else load_voiceover(run_dir)
        if voice:
            log.info("Reusing voice.wav + words.json")
        else:
            voice = generate_voiceover(script, settings, run_dir, dry_run=args.dry_run,
                                       existing_audio=Path(args.voiceover) if args.voiceover else None)

    # ── 3. Footage ────────────────────────────────────────────────────────── #
    with stage("3/4 Footage"):
        if local_clips:
            media = assign_local_clips(script, local_clips)
        else:
            media = MediaFetcher(settings, run_dir, dry_run=args.dry_run).fetch_for_script(script)

    # ── 4. Edit + mix + render ────────────────────────────────────────────── #
    with stage("4/4 Compose & render"):
        timeline = build_timeline(script, voice.words, voice.duration, media, settings, seed=args.seed)
        timeline.to_json(run_dir / "timeline.json")
        mix = build_mix(timeline, voice.audio_path, settings, run_dir / "audio", mood=script.music_mood, seed=args.seed)
        final = run_dir / "short.mp4"
        VideoBuilder(settings, run_dir).render(timeline, script, mix, final)

    meta = write_metadata(script, media, run_dir, final)
    if not args.keep_temp:
        shutil.rmtree(run_dir / "shots", ignore_errors=True)
        for tmp in ("base.mp4", "base.txt", "voice_raw.wav"):
            (run_dir / tmp).unlink(missing_ok=True)
    if args.out:
        dest = Path(args.out).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, dest)
        log.info("Copied to %s", dest)

    log.info("━" * 60)
    log.info("DONE in %.0fs  →  %s", time.perf_counter() - t_start, final)
    log.info("Title: %s", script.youtube_title)
    log.info("Metadata: %s", meta)
    return final


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        import check_system
        return check_system.main([])

    from src.utils import PipelineError, log, setup_logging
    setup_logging(args.verbose)
    if args.init_assets:
        from src import asset_synth
        asset_synth.ensure_assets(force=True)
        log.info("Assets ready in ./assets")
        return 0
    try:
        run_pipeline(args)
        return 0
    except PipelineError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
