"""Typed data contracts passed between pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .utils import has_alnum

SFX_CUES: tuple[str, ...] = ("whoosh", "bass_drop", "ding", "pop", "riser", "boom", "none")
MUSIC_MOODS: tuple[str, ...] = ("phonk", "suspense", "epic", "upbeat")
RANKS: tuple[int, ...] = (6, 5, 4, 3, 2, 1)


class ScriptSegment(BaseModel):
    """One spoken block of the Short (hook, a ranked item, or the loop outro)."""

    segment_type: Literal["hook", "rank", "outro"]
    rank: int = Field(description="6..1 for ranked items, 0 for hook and outro")
    voiceover_text: str
    on_screen_title: str
    search_keywords: list[str] = Field(default_factory=list)
    sound_effect_cue: str = "none"
    clip_id: str = ""  # only used with --local-clips

    @property
    def label(self) -> str:
        return f"#{self.rank}" if self.segment_type == "rank" else self.segment_type

    def spoken_words(self) -> list[str]:
        return [w for w in self.voiceover_text.split() if has_alnum(w)]


class ShortScript(BaseModel):
    topic: str
    topic_title: str = Field(description="Short uppercase header shown on screen")
    youtube_title: str
    youtube_description: str
    hashtags: list[str] = Field(default_factory=list)
    music_mood: str = "phonk"
    segments: list[ScriptSegment]

    # ------------------------------------------------------------------ #
    @property
    def hook(self) -> ScriptSegment:
        return self.segments[0]

    @property
    def outro(self) -> ScriptSegment:
        return self.segments[-1]

    def rank_segment(self, rank: int) -> ScriptSegment:
        for seg in self.segments:
            if seg.segment_type == "rank" and seg.rank == rank:
                return seg
        raise KeyError(rank)

    def full_voiceover(self) -> str:
        return " ".join(s.voiceover_text.strip() for s in self.segments)

    def word_count(self) -> int:
        return sum(len(s.spoken_words()) for s in self.segments)

    def structure_errors(self, *, word_range: tuple[int, int] | None = None, clip_ids: list[str] | None = None) -> list[str]:
        """Return human-readable problems; empty list means the script is valid."""
        errors: list[str] = []
        types = [s.segment_type for s in self.segments]
        expected = ["hook"] + ["rank"] * 6 + ["outro"]
        if types != expected:
            errors.append(f"segments must be exactly [hook, rank x6, outro] in that order, got {types}")
            return errors
        ranks = [s.rank for s in self.segments[1:7]]
        if ranks != list(RANKS):
            errors.append(f"rank segments must count down 6,5,4,3,2,1 - got {ranks}")
        for seg in self.segments:
            if not seg.voiceover_text.strip():
                errors.append(f"{seg.label}: voiceover_text is empty")
            if seg.sound_effect_cue not in SFX_CUES:
                errors.append(f"{seg.label}: sound_effect_cue '{seg.sound_effect_cue}' not in {SFX_CUES}")
            if seg.segment_type == "rank" and len(seg.on_screen_title) > 30:
                errors.append(f"{seg.label}: on_screen_title is too long ({len(seg.on_screen_title)} chars, max 30)")
        if word_range:
            n = self.word_count()
            lo, hi = word_range
            if not (lo <= n <= hi):
                errors.append(f"total spoken words = {n}; must be between {lo} and {hi} for a 40-50 s Short")
            hook_words = len(self.hook.spoken_words())
            if hook_words > 13:
                errors.append(f"hook has {hook_words} words; keep it to 7-10 so it lands in ~3 s")
        if clip_ids is not None:
            used = [s.clip_id for s in self.segments if s.segment_type == "rank"]
            bad = [c for c in used if c not in clip_ids]
            if bad:
                errors.append(f"unknown clip_id(s) {bad}; valid ids: {clip_ids}")
            if len(set(used)) != len(used):
                errors.append(f"each rank needs a different clip_id, got {used}")
        return errors


class WordStamp(BaseModel):
    word: str           # original script token (e.g. "6,")
    start: float
    end: float
    segment_index: int  # index into ShortScript.segments


class SourceClip(BaseModel):
    """A downloaded / local / generated video that shots can be cut from."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float = 30.0
    origin: str = "local"   # pexels | ytdlp | local | placeholder
    label: str = ""
    credit: str = ""

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.5625
