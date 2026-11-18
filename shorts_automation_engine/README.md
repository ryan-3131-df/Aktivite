# Shorts Automation Engine

One topic in → one ready-to-upload **"Top 6 Ranking" YouTube Short** out
(1080×1920, 9:16, 60 fps, H.264 + AAC), fully automated:

```
topic ─▶ script_gen ─▶ audio_gen ─▶ media_fetcher ─▶ timeline ─▶ audio_mixer ─▶ video_builder ─▶ short.mp4
         Claude        ElevenLabs    Pexels / yt-dlp   edit list   -14 LUFS mix    MoviePy + FFmpeg
                       + Whisper     / your clips
```

## Quick start

```bash
cd shorts_automation_engine
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check_system.py                                  # verifies ffmpeg, packages, keys, GPU encoder
cp .env.example .env                                    # add ANTHROPIC / ELEVENLABS / PEXELS keys

python main.py "most dangerous animals" --dry-run       # offline smoke test, no keys, no cost
python main.py "most dangerous animals"                 # the real thing
```

FFmpeg is the only system requirement (`brew install ffmpeg`, `sudo apt install ffmpeg`,
`winget install Gyan.FFmpeg`). `python check_system.py --install` will try to install it for you.
ImageMagick is **not** needed: text is rendered with Pillow (MoviePy 2.x no longer uses ImageMagick);
the checker only reports whether it is present.

Each run writes `output/<timestamp>_<topic>/`:

| File | What |
|---|---|
| `short.mp4` | the final Short |
| `cover.jpg` | a frame from the first reveal (thumbnail candidate) |
| `metadata.txt` | YouTube title, description, hashtags, ranking, footage credits |
| `script.json` · `words.json` · `timeline.json` | script, word timestamps, full edit decision list |

## Usage

```bash
python main.py "TOPIC"                           # full auto
python main.py "TOPIC" --script-only             # write script.json and stop (review before paying for TTS)
python main.py --resume output/<run_dir>         # re-render a run, reusing script / voice / footage
python main.py "TOPIC" --script-json my.json     # bring your own script
python main.py "TOPIC" --voiceover take.mp3      # bring your own narration (timestamps are re-extracted)

# Your own clips: drop 6 files into input_clips/
python main.py --local-clips                     # Claude looks at the clips, picks the topic and the ranking
python main.py "craziest goals" --local-clips ~/Downloads/goals --keep-order

# Offline test of the whole render path (template script, silent/espeak voice, placeholder or local clips)
python main.py "anything" --dry-run
python main.py "anything" --dry-run --local-clips
```

Useful flags: `--voice-id`, `--timestamps {auto,faster-whisper,openai,elevenlabs,estimate}`,
`--clips-per-rank N`, `--fit {smart,crop,blur}`, `--hook-style {montage,clip}`, `--music FILE`,
`--no-music`, `--sidechain`, `--encoder`, `--preset`, `--crf`, `--fps`, `--seed`, `--out FILE`,
`--keep-temp`, `-v`. Run `python main.py -h` for all of them; every default can also be set in `.env`.

## What each stage does

**1. `src/script_gen.py` — script & metadata (Claude).**
Calls `claude-opus-5-5` with JSON-schema structured outputs, so the response is always valid JSON:
`hook` (≈0–3 s pattern interrupt) → `rank 6 … 1` (≈6 s each, micro open loops) → `outro` that ends on
an unfinished clause the hook completes (seamless loop). Per segment: `rank`, `voiceover_text`,
`on_screen_title`, `search_keywords`, `sound_effect_cue`; plus `youtube_title`, description, hashtags.
The script is validated (order, countdown, word budget for 40–50 s) and re-prompted with the errors if
it fails. Server-side refusal fallback is enabled (`CLAUDE_FALLBACKS=false` to turn off).
With `--local-clips`, three frames per clip are sent to Claude's vision so it narrates what is on screen.

**2. `src/audio_gen.py` — voiceover & word timestamps.**
ElevenLabs TTS with high-energy settings (stability 0.32, style 0.55, speed 1.10; voice "Adam" by
default) → `voiceover.mp3`. Leading silence is trimmed and the take is sped up (≤ ×1.15) only if it
exceeds 50 s. Word timings come from **faster-whisper** (local; `TIMESTAMP_BACKEND=openai` for the
Whisper API), then are aligned back onto the *script* words, so captions show the exact script text
with ASR timing. Fallbacks: ElevenLabs' own character alignment → heuristic estimate.

**3. `src/media_fetcher.py` — footage.**
Per rank: Pexels portrait → Pexels any orientation → yt-dlp YouTube search (Creative-Commons filter,
age limit, < 15 min, downloads only the needed seconds) → generated placeholder. Downloads are cached in
`cache/media/`. Clips are conformed to 1080×1920 @ 60 fps by FFmpeg: portrait-ish sources are
center-cropped, landscape/square sources sit on a blurred copy of themselves; sources that are too short
are slowed slightly or looped.

**4. `src/video_builder.py` — compositor & renderer (MoviePy + FFmpeg).**
- *Base layer:* rapid cuts (hook teaser montage, 2–3 shots per rank, outro recap) with a continuous
  1.00→1.08 Ken Burns zoom (sub-pixel warps, so no jitter), punch-zoom + white flash on every reveal.
- *Ranking HUD (top-left):* glowing `#6 … #1` badge and a 1–6 list tracker that fills in as the
  countdown progresses, plus the topic header and a "TOP 6" title card during the hook.
- *Captions:* 2–3 uppercase words at a time, Montserrat Black, thick black stroke, drop shadow, pop-in,
  active word highlighted (yellow / green alternating, white base).
- *Audio (`src/audio_mixer.py`):* voice two-pass loudnorm to **−14 LUFS**; music bed at **−22 LUFS**
  (`--sidechain` adds dynamic ducking); `whoosh` on every transition, `pop` on every reveal, plus the
  script's cues (`bass_drop`, `boom`, `ding`, `riser`); −1 dBTP limiter.
- *Encode:* NVENC / VideoToolbox / QuickSync / AMF / VAAPI are probed and used automatically, otherwise
  libx264 (CRF 18, High profile, `+faststart`).

## Project layout

```
main.py              CLI orchestrator
check_system.py      dependency / key / encoder check (stdlib only)
requirements.txt     .env.example
src/
  config.py          every tunable (env-overridable)        models.py      pydantic contracts
  script_gen.py      Claude script + metadata                audio_gen.py   ElevenLabs + Whisper + alignment
  media_fetcher.py   Pexels / yt-dlp / local clips           timeline.py    frame-accurate edit list
  audio_mixer.py     LUFS mix + SFX                          video_builder.py  compositor & renderer
  graphics.py        captions, badge, list tracker (Pillow)  ffmpeg_tools.py   FFmpeg helpers
  asset_synth.py     starter SFX / music generator
assets/sfx   assets/music   assets/fonts      input_clips/   output/   tests/
```

## Assets

- `assets/sfx/` — `whoosh, pop, ding, bass_drop, boom, riser` (.mp3). The bundled files are synthesised
  by `src/asset_synth.py`, so they are free to use; replace any of them with your own file of the same name.
- `assets/music/` — a generated phonk loop and a suspense loop are included as starters. Add your own
  licensed tracks here; yours are preferred over the `*_default` ones, and a filename containing the
  script's mood (`phonk`, `suspense`, `epic`, `upbeat`) is picked first.
- `assets/fonts/` — Anton and Montserrat (SIL Open Font License, licence files included). Override with
  `CAPTION_FONT` / `DISPLAY_FONT` in `.env`.
- `python main.py --init-assets` regenerates the SFX/music and re-downloads fonts.

## Tests

```bash
python -m pytest tests -q      # or: python tests/test_core.py
```

Covers word alignment (digits, split/merged words, dropped words), ElevenLabs character alignment,
script validation / schema, and frame-accurate timeline construction.

## Notes & limits

- **Footage rights are your responsibility.** Pexels clips are free to use; the yt-dlp fallback is
  limited to Creative-Commons results by default (`YTDLP_CC_ONLY`), which usually require attribution —
  credits are written to `metadata.txt`. Disable the fallback with `--no-ytdlp`.
- Claude is told to use verifiable facts, but **review `script.json` before publishing**
  (`--script-only`, edit, then `--resume`).
- First faster-whisper run downloads the model (~150 MB for `base.en`).
- Render time: roughly 1–2× real-time per pass on a 4-core CPU with libx264; much faster with a GPU encoder.
