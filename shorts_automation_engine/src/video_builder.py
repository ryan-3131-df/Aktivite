"""
Stage 4 - Viral video compositor & renderer (MoviePy + FFmpeg).

Layer stack (bottom → top), all at 1080x1920 / 60 fps:

  1. Base footage   - FFmpeg-normalised shots (crop or blurred-pad), joined
                      losslessly, read by MoviePy; every shot gets a continuous
                      1.00 → 1.08 Ken Burns zoom with sub-pixel cv2 warps, plus
                      a punch-zoom + white flash on each rank reveal.
  2. Ranking HUD    - glowing "#N" badge + dynamic 1-6 list tracker (top-left)
                      and the topic header; hook title card during the hook.
  3. Captions       - 2-3 word uppercase pop-ups, thick stroke, drop shadow,
                      active word highlighted yellow/green.

Frames are streamed straight into an FFmpeg encoder (NVENC / VideoToolbox /
QSV / AMF / VAAPI when available, libx264 otherwise) which also muxes the
final audio mix.
"""

from __future__ import annotations

import bisect
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .ffmpeg_tools import FFmpegPipeWriter, concat_shots, normalize_shot, resolve_encoder
from .graphics import (RANK_COLORS, Sprite, blend_white, render_badge, render_caption_states, render_header,
                       render_list_panel, render_title_card)
from .models import RANKS, ShortScript
from .timeline import Timeline
from .utils import PipelineError, clamp, ease_in_out_sine, ease_out_back, ease_out_cubic, lerp, log

# HUD layout (px, 1080x1920 canvas; keeps clear of the Shorts UI on the right/bottom).
HEADER_TOP = 138
BADGE_POS = (-12, 186)      # sprite origin (sprite has a 52px glow margin)
LIST_POS = (18, 372)        # sprite origin (sprite has a 22px glow margin)
TITLE_CARD_CY = 660
REVEAL_SECONDS = 0.28
BADGE_POP_SECONDS = 0.26


class Hud:
    """Pre-rendered overlay sprites + the per-frame overlay logic."""

    def __init__(self, timeline: Timeline, script: ShortScript, settings: Settings):
        self.tl = timeline
        self.s = settings
        self.W, self.H = settings.frame_size
        self.cuts = [seg.start for seg in timeline.segments]
        self.rank_starts = [seg.start for seg in timeline.segments if seg.kind == "rank"]

        titles = {seg.rank: seg.on_screen_title for seg in script.segments if seg.segment_type == "rank"}
        self.header = render_header(script.topic_title or script.topic)
        self.title_card = render_title_card("TOP 6", script.hook.on_screen_title or script.topic_title)
        self.badges = {r: render_badge(f"#{r}", RANK_COLORS[r]) for r in RANKS}
        self.badge_top = render_badge("TOP 6", RANK_COLORS[1])

        # List tracker states: before/after each reveal, and the full list for the outro recap.
        self.list_before: dict[int, Sprite] = {}
        self.list_after: dict[int, Sprite] = {}
        for r in RANKS:
            before = {k: titles[k] for k in RANKS if k > r}
            after = {k: titles[k] for k in RANKS if k >= r}
            self.list_before[r] = render_list_panel(before, r)
            self.list_after[r] = render_list_panel(after, r)
        self.list_full = {r: render_list_panel(titles, r) for r in RANKS}
        self.list_full[0] = render_list_panel(titles, None)

        # Captions: one sprite per active-word state, per chunk.
        cfg = settings.captions
        self.cap_starts = [c.start for c in timeline.captions]
        self.cap_sprites = [render_caption_states(c.words, cfg.highlight_colors[c.color_index], cfg)
                            for c in timeline.captions]
        self.shot_starts = [sh.start for sh in timeline.shots]
        log.info("HUD: %d caption chunks pre-rendered", len(self.cap_sprites))

    # ------------------------------------------------------------------ #
    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        seg = self.tl.segment_at(t)
        v = self.s.video
        if not frame.flags.writeable:
            frame = frame.copy()

        # White flash on rank reveals.
        if v.flash_on_cut:
            k = bisect.bisect_right(self.rank_starts, t) - 1
            if k >= 0:
                dt = t - self.rank_starts[k]
                if dt < v.flash_seconds:
                    frame = blend_white(frame, v.flash_alpha * (1 - dt / v.flash_seconds))

        local = t - seg.start
        if seg.kind == "hook":
            self._title_card(frame, local, seg.end - seg.start)
        else:
            first_rank_start = self.rank_starts[0] if self.rank_starts else 0.0
            since_first = t - first_rank_start
            intro = ease_out_cubic(since_first / 0.35)
            self.header.blit_centered(frame, self.W / 2, HEADER_TOP + self.header.h / 2,
                                      opacity=clamp(since_first / 0.25))
            if seg.kind == "rank":
                self._rank_hud(frame, seg.rank, local, intro)
            else:
                self._outro_hud(frame, t, local)
        self._captions(frame, t)
        return frame

    def _title_card(self, frame: np.ndarray, local: float, length: float) -> None:
        slam = ease_out_cubic(local / 0.22)
        scale = lerp(1.28, 1.0, slam)
        fade_out = clamp((length - local) / 0.15)
        self.title_card.blit_centered(frame, self.W / 2, TITLE_CARD_CY, scale=scale,
                                      opacity=min(0.55 + 0.45 * slam, fade_out))

    def _rank_hud(self, frame: np.ndarray, rank: int, local: float, intro: float) -> None:
        # List tracker: slide in on first appearance, then crossfade the newly revealed title.
        x_off = int((1 - intro) * -620)
        p = clamp((local - 0.05) / REVEAL_SECONDS)
        if p < 1.0:
            self.list_before[rank].blit(frame, LIST_POS[0] + x_off, LIST_POS[1], opacity=intro)
            self.list_after[rank].blit(frame, LIST_POS[0] + x_off, LIST_POS[1], opacity=ease_in_out_sine(p) * intro)
        else:
            self.list_after[rank].blit(frame, LIST_POS[0] + x_off, LIST_POS[1], opacity=intro)
        # Badge pop.
        b = self.badges[rank]
        scale = lerp(0.55, 1.0, ease_out_back(local / BADGE_POP_SECONDS))
        b.blit_anchored(frame, BADGE_POS[0], BADGE_POS[1], scale=max(scale, 0.05), anchor=(0.5, 0.5),
                        opacity=clamp(local / 0.08))

    def _outro_hud(self, frame: np.ndarray, t: float, local: float) -> None:
        i = bisect.bisect_right(self.shot_starts, t) - 1
        hl = self.tl.shots[i].highlight_rank if i >= 0 else None
        self.list_full.get(hl or 0, self.list_full[0]).blit(frame, *LIST_POS)
        scale = lerp(0.7, 1.0, ease_out_back(local / BADGE_POP_SECONDS))
        self.badge_top.blit_anchored(frame, BADGE_POS[0], BADGE_POS[1], scale=scale, anchor=(0.5, 0.5))

    def _captions(self, frame: np.ndarray, t: float) -> None:
        i = bisect.bisect_right(self.cap_starts, t) - 1
        if i < 0:
            return
        chunk = self.tl.captions[i]
        if t >= chunk.end or not self.cap_sprites[i]:
            return
        w = max(0, bisect.bisect_right(chunk.word_starts, t) - 1)
        sprite = self.cap_sprites[i][min(w, len(self.cap_sprites[i]) - 1)]
        cfg = self.s.captions
        age = t - chunk.start
        scale = 1.0
        if age < cfg.pop_seconds:
            scale = lerp(cfg.pop_from, 1.0, ease_out_back(age / cfg.pop_seconds))
        sprite.blit_centered(frame, self.W / 2, cfg.y_center, scale=scale)


class KenBurns:
    """Per-shot continuous zoom (+ punch-in on reveals) with sub-pixel affine warps."""

    def __init__(self, timeline: Timeline, settings: Settings):
        self.shots = timeline.shots
        self.starts = [s.start for s in self.shots]
        self.v = settings.video
        self.W, self.H = settings.frame_size

    def __call__(self, get_frame, t: float) -> np.ndarray:
        frame = get_frame(t)
        i = max(0, bisect.bisect_right(self.starts, t + 1e-6) - 1)
        shot = self.shots[i]
        local = t - shot.start
        p = clamp(local / max(shot.duration, 1e-3))
        s = lerp(shot.zoom_from, shot.zoom_to, p)
        if shot.punch and local < self.v.punch_seconds:
            s *= 1 + self.v.punch_zoom * (1 - ease_out_cubic(local / self.v.punch_seconds))
        e = ease_in_out_sine(p)
        ax = lerp(shot.anchor_from[0], shot.anchor_to[0], e) * self.W
        ay = lerp(shot.anchor_from[1], shot.anchor_to[1], e) * self.H
        if abs(s - 1.0) < 1e-4:
            return frame
        m = np.float32([[s, 0, (1 - s) * ax], [0, s, (1 - s) * ay]])
        return cv2.warpAffine(frame, m, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class VideoBuilder:
    def __init__(self, settings: Settings, run_dir: Path):
        self.s = settings
        self.run_dir = run_dir
        self.shots_dir = run_dir / "shots"

    # ------------------------------------------------------------------ #
    def prepare_base(self, timeline: Timeline) -> Path:
        """Normalise every shot with FFmpeg (in parallel) and join them into one base track."""
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        fps = timeline.fps
        jobs = []
        for i, sh in enumerate(timeline.shots):
            n = int(round(sh.end * fps)) - int(round(sh.start * fps))
            jobs.append((sh, self.shots_dir / f"shot_{i:03d}.mp4", n))
        total = sum(n for _, _, n in jobs)
        if total != timeline.n_frames:
            raise PipelineError(f"Shot frames ({total}) ≠ timeline frames ({timeline.n_frames})")

        workers = max(1, min(4, (os.cpu_count() or 4) // 2))
        log.info("Normalising %d shots to %dx%d@%d (%d workers)…", len(jobs), *self.s.frame_size, fps, workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(normalize_shot, sh.source, dst, sh.src_offset, n, self.s.video)
                       for sh, dst, n in jobs]
            for f in futures:
                f.result()
        return concat_shots([dst for _, dst, _ in jobs], self.run_dir / "base.mp4")

    # ------------------------------------------------------------------ #
    def render(self, timeline: Timeline, script: ShortScript, audio_mix: Path, out_path: Path) -> Path:
        from moviepy import VideoFileClip  # MoviePy 2.x

        base_path = self.prepare_base(timeline)
        encoder = resolve_encoder(self.s)
        hud = Hud(timeline, script, self.s)

        base = VideoFileClip(str(base_path), audio=False)
        clip = base.transform(KenBurns(timeline, self.s)).transform(lambda gf, t: hud.apply(gf(t), t))

        fps, n = timeline.fps, timeline.n_frames
        cover_frame = int(round(((timeline.segments[1].start + 0.6) if len(timeline.segments) > 1 else 1.0) * fps))
        writer = FFmpegPipeWriter(out_path, self.s.frame_size, fps, encoder, self.s.video, audio_path=audio_mix)
        log.info("Rendering %d frames (%.2fs @ %d fps) → %s", n, timeline.duration, fps, out_path.name)
        t0 = last = time.perf_counter()
        try:
            for i in range(n):
                frame = clip.get_frame(i / fps)
                writer.write(frame)
                if i == cover_frame:
                    cv2.imwrite(str(out_path.with_name("cover.jpg")), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                now = time.perf_counter()
                if now - last > 1.0 or i == n - 1:
                    last = now
                    rate = (i + 1) / (now - t0)
                    eta = (n - i - 1) / max(rate, 1e-6)
                    bar = "█" * int(30 * (i + 1) / n)
                    sys.stderr.write(f"\r    [{bar:<30}] {i + 1}/{n} frames · {rate:5.1f} fps · ETA {eta:5.1f}s ")
                    sys.stderr.flush()
            sys.stderr.write("\n")
            writer.close()
        finally:
            base.close()
        log.info("Rendered in %.1fs", time.perf_counter() - t0)
        return out_path
