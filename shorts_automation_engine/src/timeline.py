"""
Edit decision list: turns (script + word timestamps + source clips) into a
frame-accurate timeline of segments, shots, caption chunks and SFX events.

All cut points are snapped to frame boundaries so FFmpeg-normalised shots
concatenate to exactly the voiceover length.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings
from .models import RANKS, ShortScript, SourceClip, WordStamp
from .utils import caption_text, log, quantize, write_json

CUT_LEAD = 0.08     # cut a hair before the first word of a segment (visual leads audio)
TAIL = 0.22         # keep this much after the final word (tight = seamless loop)
MIN_SEG = 0.6
MONTAGE_MIN = 0.32  # shortest montage snippet
PUNCTUATION_BREAK = (".", ",", "!", "?", ";", ":", "—", "–")


@dataclass
class SegmentSpan:
    index: int
    kind: str          # hook | rank | outro
    rank: int
    start: float
    end: float
    title: str


@dataclass
class Shot:
    start: float
    end: float
    source: SourceClip
    src_offset: float
    segment_index: int
    zoom_from: float
    zoom_to: float
    anchor_from: tuple[float, float]
    anchor_to: tuple[float, float]
    punch: bool = False
    highlight_rank: int | None = None   # outro recap: which list row to light up

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class CaptionChunk:
    start: float
    end: float
    words: list[str]
    word_starts: list[float]
    color_index: int
    segment_index: int


@dataclass
class SfxEvent:
    time: float
    name: str
    align: str = "start"    # "end" → the sound finishes at `time` (risers)


@dataclass
class Timeline:
    duration: float
    fps: int
    segments: list[SegmentSpan]
    shots: list[Shot]
    captions: list[CaptionChunk]
    sfx: list[SfxEvent]
    cuts: list[float] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(round(self.duration * self.fps))

    def segment_at(self, t: float) -> SegmentSpan:
        for seg in self.segments:
            if t < seg.end:
                return seg
        return self.segments[-1]

    def to_json(self, path: Path) -> None:
        def ser(o):
            d = asdict(o)
            if "source" in d:
                d["source"] = str(o.source.path.name)
            return d
        write_json(path, {
            "duration": self.duration, "fps": self.fps, "cuts": self.cuts,
            "segments": [asdict(s) for s in self.segments],
            "shots": [ser(s) for s in self.shots],
            "captions": [asdict(c) for c in self.captions],
            "sfx": [asdict(e) for e in self.sfx],
        })


# --------------------------------------------------------------------------- #
def build_timeline(script: ShortScript, words: list[WordStamp], voice_duration: float,
                   media: dict[int, list[SourceClip]], settings: Settings, seed: int | None = None) -> Timeline:
    fps = settings.video.fps
    q = lambda t: quantize(t, fps)  # noqa: E731
    rng = random.Random(seed)

    # ---- Segment boundaries from the first word of each segment -------------- #
    first: dict[int, float] = {}
    for w in words:
        first.setdefault(w.segment_index, w.start)
    last_end = max((w.end for w in words), default=voice_duration)
    cuts = [0.0]
    for si in range(1, len(script.segments)):
        guess = first.get(si, cuts[-1] + 1.0)
        cuts.append(q(max(guess - CUT_LEAD, cuts[-1] + MIN_SEG)))
    duration = q(max(last_end + TAIL, cuts[-1] + MIN_SEG))
    bounds = cuts + [duration]
    segments = [
        SegmentSpan(i, seg.segment_type, seg.rank, bounds[i], bounds[i + 1],
                    seg.on_screen_title if seg.segment_type != "rank" else seg.on_screen_title.strip())
        for i, seg in enumerate(script.segments)
    ]

    # ---- Shots ------------------------------------------------------------- #
    shots: list[Shot] = []
    cursor: dict[Path, float] = {}

    def kb() -> tuple[tuple[float, float], tuple[float, float]]:
        a0 = (0.5 + rng.uniform(-0.06, 0.06), 0.46 + rng.uniform(-0.05, 0.05))
        a1 = (0.5 + rng.uniform(-0.14, 0.14), 0.46 + rng.uniform(-0.08, 0.08))
        return a0, a1

    def add(start: float, end: float, src: SourceClip, offset: float, seg_idx: int,
            punch: bool = False, highlight: int | None = None) -> None:
        start, end = q(start), q(end)
        if end - start < 1.0 / fps:
            return
        a0, a1 = kb()
        shots.append(Shot(start, end, src, max(0.0, offset), seg_idx, settings.video.zoom_start,
                          settings.video.zoom_end, a0, a1, punch, highlight))

    def montage(span: SegmentSpan, ranks: list[int], pos: float, highlight: bool) -> None:
        length = span.end - span.start
        n = max(1, min(len(ranks), int(length / MONTAGE_MIN)))
        ranks = ranks[-n:] if highlight else ranks[:n]
        edges = [span.start + length * k / n for k in range(n + 1)]
        for k, rank in enumerate(ranks):
            src = media[rank][0]
            snip = edges[k + 1] - edges[k]
            offset = min(src.duration * pos, max(0.0, src.duration - snip))
            add(edges[k], edges[k + 1], src, offset, span.index, punch=(k == 0 and span.kind == "hook"),
                highlight=rank if highlight else None)

    for span in segments:
        length = span.end - span.start
        if span.kind == "hook":
            if settings.video.hook_style == "clip" and media.get(0):
                add(span.start, span.end, media[0][0], media[0][0].duration * 0.1, span.index, punch=True)
            else:
                teaser = list(RANKS)
                rng.shuffle(teaser)
                montage(span, teaser, 0.72, highlight=False)
        elif span.kind == "rank":
            srcs = media[span.rank]
            if len(srcs) == 1:
                src = srcs[0]
                # Single (e.g. user-provided) clip: play it continuously from just before its middle.
                offset = max(0.0, (src.duration - length) * 0.45)
                add(span.start, span.end, src, offset, span.index, punch=True)
            else:
                n = max(len(srcs) if length >= 1.5 * len(srcs) else 1,
                        round(length / settings.video.target_shot_seconds))
                edges = [span.start + length * k / n for k in range(n + 1)]
                for k in range(n):
                    src = srcs[k % len(srcs)]
                    snip = edges[k + 1] - edges[k]
                    off = cursor.get(src.path, src.duration * 0.08)
                    if off + snip > src.duration:
                        off = src.duration * 0.08
                    cursor[src.path] = off + snip
                    add(edges[k], edges[k + 1], src, off, span.index, punch=(k == 0))
        else:  # outro: recap montage #6 → #1, lighting up each list row
            montage(span, list(RANKS), 0.3, highlight=True)

    # Close any rounding gaps so shots tile [0, duration] exactly.
    for a, b in zip(shots, shots[1:]):
        b.start = a.end
    if shots:
        shots[0].start = 0.0
        shots[-1].end = duration

    # ---- Captions ---------------------------------------------------------- #
    cfg = settings.captions
    chunks: list[CaptionChunk] = []
    for si in range(len(script.segments)):
        seg_words = [w for w in words if w.segment_index == si]
        cur: list[tuple[str, WordStamp]] = []

        def flush() -> None:
            if cur:
                chunks.append(CaptionChunk(cur[0][1].start, cur[-1][1].end, [d for d, _ in cur],
                                           [w.start for _, w in cur], len(chunks) % len(cfg.highlight_colors), si))
                cur.clear()

        for w in seg_words:
            disp = caption_text(w.word)
            if not disp:
                continue
            chars = sum(len(d) for d, _ in cur) + len(disp)
            if cur and (len(cur) >= cfg.max_words or chars > cfg.max_chars or w.start - cur[-1][1].end > cfg.pause_break):
                flush()
            cur.append((disp, w))
            if w.word.rstrip()[-1:] in PUNCTUATION_BREAK:
                flush()
        flush()
    for c, nxt in zip(chunks, chunks[1:]):
        gap = nxt.start - c.end
        c.end = nxt.start if gap < 0.6 else c.end + 0.25
    if chunks:
        chunks[-1].end = duration

    # ---- SFX --------------------------------------------------------------- #
    sfx: list[SfxEvent] = []
    hook_cue = script.hook.sound_effect_cue
    if hook_cue == "riser" and len(cuts) > 1:
        sfx.append(SfxEvent(cuts[1], "riser", align="end"))
    elif hook_cue not in ("none", "whoosh", "pop"):
        sfx.append(SfxEvent(0.0, hook_cue))
    for span in segments[1:]:
        c = span.start
        sfx.append(SfxEvent(max(0.0, c - 0.2), "whoosh"))
        cue = script.segments[span.index].sound_effect_cue
        if span.kind == "rank":
            sfx.append(SfxEvent(c + 0.04, "pop"))
        if cue == "riser":
            sfx.append(SfxEvent(c, "riser", align="end"))
        elif cue in ("ding", "boom", "bass_drop"):
            sfx.append(SfxEvent(c + 0.02, cue))

    tl = Timeline(duration, fps, segments, shots, chunks, sfx, cuts)
    log.info("Timeline: %.2fs, %d segments, %d shots, %d caption chunks, %d SFX",
             duration, len(segments), len(shots), len(chunks), len(sfx))
    for s in segments:
        log.debug("  %-6s #%d %6.2f → %6.2f  %s", s.kind, s.rank, s.start, s.end, s.title)
    return tl
