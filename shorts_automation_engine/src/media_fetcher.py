"""
Stage 3 - Asset sourcing.

Fallback chain per ranked item (first success wins, until `clips_per_rank`
distinct clips are collected):

    Pexels (portrait)  →  Pexels (any orientation)  →  yt-dlp (YouTube search,
    Creative-Commons filter, age-limit, short-duration filter, partial download)
    →  generated placeholder

Downloads are cached in ./cache/media so re-runs don't re-download. Local
clips (`--local-clips`) bypass all of this. Framing/trimming to 1080x1920 @
60 fps happens in `ffmpeg_tools.normalize_shot` once the timeline is known.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import quote_plus

import requests

from .config import CACHE_DIR, Settings
from .ffmpeg_tools import extract_frame_jpegs, make_placeholder_clip
from .graphics import font_path
from .models import RANKS, ShortScript, SourceClip
from .script_gen import LocalClip
from .utils import PipelineError, ffprobe, log, read_json, slugify, write_json

PEXELS_SEARCH = "https://api.pexels.com/videos/search"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
MEDIA_CACHE = CACHE_DIR / "media"
YT_CC_FILTER = "EgIwAQ%3D%3D"  # YouTube search filter: Creative Commons


def _probe_clip(path: Path, origin: str, label: str, credit: str = "") -> SourceClip | None:
    try:
        info = ffprobe(path)
    except PipelineError as exc:
        log.warning("Skipping unreadable clip %s: %s", path.name, exc)
        return None
    if not info["has_video"] or info["duration"] < 0.3:
        log.warning("Skipping %s (no video stream or too short)", path.name)
        return None
    return SourceClip(path=path, duration=info["duration"], width=info["width"], height=info["height"],
                      fps=info["fps"] or 30.0, origin=origin, label=label, credit=credit)


# --------------------------------------------------------------------------- #
# Local clips
# --------------------------------------------------------------------------- #
def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load_local_clips(folder: Path, keep_order: bool = False, max_clips: int = 10) -> list[LocalClip]:
    """
    Read the user's clips. With --keep-order: a leading number 1-6 in the filename
    ("1_dunk.mp4") pins that rank; otherwise files play in name order (first = #6).
    """
    if not folder.is_dir():
        raise PipelineError(f"Local clips folder not found: {folder}")
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS), key=_natural_key)
    if len(files) < 6:
        raise PipelineError(f"Need at least 6 clips in {folder}, found {len(files)} ({', '.join(f.name for f in files)})")
    if len(files) > max_clips:
        log.warning("Found %d clips; only the first %d are considered", len(files), max_clips)
        files = files[:max_clips]

    forced: dict[Path, int] = {}
    if keep_order:
        files = files[:6]
        nums = {}
        for f in files:
            m = re.match(r"^\s*#?(\d+)", f.stem)
            if m and 1 <= int(m.group(1)) <= 6:
                nums[f] = int(m.group(1))
        if len(nums) == 6 and len(set(nums.values())) == 6:
            forced = nums
        else:
            forced = {f: r for f, r in zip(files, RANKS)}

    clips: list[LocalClip] = []
    for idx, f in enumerate(files):
        src = _probe_clip(f, "local", f.stem)
        if src is None:
            continue
        cid = chr(ord("A") + idx)
        times = [src.duration * p for p in (0.15, 0.5, 0.85)]
        clips.append(LocalClip(clip_id=cid, source=src, frames=extract_frame_jpegs(f, times),
                               forced_rank=forced.get(f)))
        log.info("Clip %s: %s (%.1fs, %dx%d)%s", cid, f.name, src.duration, src.width, src.height,
                 f" → pinned #{forced[f]}" if f in forced else "")
    if len(clips) < 6:
        raise PipelineError("Fewer than 6 readable clips in the local clips folder")
    return clips


def assign_local_clips(script: ShortScript, clips: list[LocalClip]) -> dict[int, list[SourceClip]]:
    by_id = {c.clip_id: c.source for c in clips}
    out: dict[int, list[SourceClip]] = {}
    spare = [c.source for c in clips]
    for rank in RANKS:
        cid = script.rank_segment(rank).clip_id
        if cid in by_id:
            out[rank] = [by_id[cid]]
            if by_id[cid] in spare:
                spare.remove(by_id[cid])
    for rank in RANKS:  # safety net if the script referenced an unknown id
        if rank not in out:
            out[rank] = [spare.pop(0) if spare else clips[0].source]
            log.warning("Rank #%d had no valid clip_id - using %s", rank, out[rank][0].path.name)
    return out


# --------------------------------------------------------------------------- #
# Remote sourcing
# --------------------------------------------------------------------------- #
class MediaFetcher:
    def __init__(self, settings: Settings, run_dir: Path, *, dry_run: bool = False):
        self.s = settings
        self.run_dir = run_dir
        self.dry_run = dry_run
        self.used_ids: set[str] = set()
        self.session = requests.Session()
        MEDIA_CACHE.mkdir(parents=True, exist_ok=True)
        (run_dir / "placeholders").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def fetch_for_script(self, script: ShortScript, min_seconds: float = 7.0) -> dict[int, list[SourceClip]]:
        manifest_path = self.run_dir / "media_manifest.json"
        if manifest_path.exists():
            cached = self._load_manifest(manifest_path)
            if cached:
                log.info("Reusing media from %s", manifest_path.name)
                return cached

        result: dict[int, list[SourceClip]] = {}
        for i, rank in enumerate(RANKS):
            seg = script.rank_segment(rank)
            clips: list[SourceClip] = []
            if not self.dry_run:
                clips = self._fetch_rank(seg.search_keywords or [seg.on_screen_title], min_seconds)
            if not clips:
                if not self.dry_run:
                    log.warning("No footage found for #%d '%s' - using a placeholder", rank, seg.on_screen_title)
                clips = [self._placeholder(seg.on_screen_title, i)]
            result[rank] = clips
            log.info("#%d %-22s ← %s", rank, seg.on_screen_title[:22], ", ".join(c.origin for c in clips))

        hook_kw = script.hook.search_keywords
        if self.s.video.hook_style == "clip" and hook_kw and not self.dry_run:
            hook = self._fetch_rank(hook_kw, 3.0, want=1)
            if hook:
                result[0] = hook
        write_json(manifest_path, {str(k): [c.model_dump(mode="json") for c in v] for k, v in result.items()})
        return result

    def _load_manifest(self, path: Path) -> dict[int, list[SourceClip]] | None:
        try:
            data = read_json(path)
            out = {int(k): [SourceClip.model_validate(c) for c in v] for k, v in data.items()}
            if all(c.path.exists() for v in out.values() for c in v):
                return out
        except Exception:  # noqa: BLE001 - stale manifest, refetch
            pass
        return None

    def _fetch_rank(self, keywords: list[str], min_seconds: float, want: int | None = None) -> list[SourceClip]:
        want = want or max(1, self.s.clips_per_rank)
        found: list[SourceClip] = []
        if self.s.pexels_api_key:
            for orientation in ("portrait", None):
                for kw in keywords:
                    for clip in self._pexels(kw, orientation, min_seconds, want - len(found)):
                        found.append(clip)
                    if len(found) >= want:
                        return found
        else:
            log.debug("PEXELS_API_KEY not set - skipping Pexels")
        if not found and self.s.allow_ytdlp:
            for kw in keywords[:2]:
                clip = self._ytdlp(kw, min_seconds)
                if clip:
                    found.append(clip)
                    break
        return found

    # ------------------------------------------------------------------ #
    def _pexels(self, query: str, orientation: str | None, min_seconds: float, limit: int) -> list[SourceClip]:
        if limit <= 0:
            return []
        params = {"query": query, "per_page": 15, "size": "medium"}
        if orientation:
            params["orientation"] = orientation
        try:
            r = self.session.get(PEXELS_SEARCH, params=params, headers={"Authorization": self.s.pexels_api_key},
                                 timeout=30)
        except requests.RequestException as exc:
            log.warning("Pexels request failed: %s", exc)
            return []
        if r.status_code == 401:
            raise PipelineError("Pexels rejected the API key (401) - check PEXELS_API_KEY")
        if r.status_code != 200:
            log.warning("Pexels search '%s' → HTTP %s", query, r.status_code)
            return []

        videos = r.json().get("videos", [])
        ranked = sorted(videos, key=lambda v: self._score_video(v, min_seconds), reverse=True)
        out: list[SourceClip] = []
        for v in ranked:
            vid = f"pexels:{v['id']}"
            if vid in self.used_ids or (v.get("duration") or 0) < 2:
                continue
            f = self._best_file(v.get("video_files", []))
            if not f:
                continue
            dest = MEDIA_CACHE / f"pexels_{v['id']}_{f['id']}.mp4"
            if not dest.exists() and not self._download(f["link"], dest):
                continue
            credit = f"Video by {v.get('user', {}).get('name', 'unknown')} on Pexels ({v.get('url', '')})"
            clip = _probe_clip(dest, "pexels", query, credit)
            if clip:
                self.used_ids.add(vid)
                out.append(clip)
                if len(out) >= limit:
                    break
        return out

    def _score_video(self, v: dict, min_seconds: float) -> float:
        w, h, dur = v.get("width", 1), v.get("height", 1), v.get("duration", 0)
        score = 0.0
        score += 3.0 if h > w else 0.0                       # portrait first
        score += 2.0 if dur >= min_seconds else dur / min_seconds
        score -= 0.02 * max(0, dur - 40)                     # very long = slow to download
        score += 1.0 if min(w, h) >= 1080 else 0.0
        return score

    def _best_file(self, files: list[dict]) -> dict | None:
        mp4 = [f for f in files if f.get("file_type") == "video/mp4" and f.get("link")
               and (f.get("height") or 0) <= self.s.max_download_height and (f.get("width") or 0) <= self.s.max_download_height]
        if not mp4:
            return None
        # Closest to 1080p short side without going below it when possible.
        def key(f):
            short = min(f.get("width") or 0, f.get("height") or 0)
            return (short >= 1080, -abs(short - 1080), f.get("fps") or 0)
        return max(mp4, key=key)

    def _download(self, url: str, dest: Path) -> bool:
        tmp = dest.with_suffix(".part")
        try:
            with self.session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            return True
        except requests.RequestException as exc:
            log.warning("Download failed (%s): %s", url[:80], exc)
            tmp.unlink(missing_ok=True)
            return False

    # ------------------------------------------------------------------ #
    def _ytdlp(self, query: str, min_seconds: float) -> SourceClip | None:
        try:
            import yt_dlp
            from yt_dlp.utils import download_range_func, match_filter_func
        except ImportError:
            log.debug("yt-dlp not installed - skipping")
            return None
        sp = f"&sp={YT_CC_FILTER}" if self.s.ytdlp_creative_commons_only else ""
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}{sp}"
        base_opts = {"quiet": True, "no_warnings": True, "noprogress": True, "age_limit": self.s.ytdlp_age_limit}
        try:
            with yt_dlp.YoutubeDL({**base_opts, "extract_flat": "in_playlist", "playlistend": 12}) as ydl:
                listing = ydl.extract_info(search_url, download=False) or {}
        except Exception as exc:  # noqa: BLE001 - network / extractor errors are non-fatal
            log.warning("yt-dlp search failed for '%s': %s", query, exc)
            return None

        need = min_seconds + 1.0
        for entry in listing.get("entries") or []:
            vid, dur = entry.get("id"), entry.get("duration") or 0
            if not vid or f"yt:{vid}" in self.used_ids or (dur and not (need <= dur <= 900)):
                continue
            start = max(0.0, (dur or 60) * 0.25)
            key = hashlib.sha1(f"{vid}{start:.1f}{need:.1f}".encode()).hexdigest()[:10]
            stem = MEDIA_CACHE / f"yt_{vid}_{key}"
            cached = next(iter(MEDIA_CACHE.glob(stem.name + ".*")), None)
            if cached is None:
                opts = {
                    **base_opts,
                    "format": "bv*[height<=1920][ext=mp4]/bv*[height<=1920]/b[height<=1920]",
                    "outtmpl": str(stem) + ".%(ext)s",
                    "noplaylist": True,
                    "download_ranges": download_range_func(None, [(start, start + need)]),
                    "match_filter": match_filter_func("!is_live & duration < 900"),
                    "retries": 2,
                }
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={vid}"])
                except Exception as exc:  # noqa: BLE001
                    log.debug("yt-dlp download failed for %s: %s", vid, exc)
                    continue
                cached = next(iter(MEDIA_CACHE.glob(stem.name + ".*")), None)
            if cached:
                credit = f"\"{entry.get('title', vid)}\" (YouTube, Creative Commons) https://youtu.be/{vid}"
                clip = _probe_clip(cached, "ytdlp", query, credit)
                if clip:
                    self.used_ids.add(f"yt:{vid}")
                    return clip
        return None

    # ------------------------------------------------------------------ #
    def _placeholder(self, label: str, index: int) -> SourceClip:
        dest = self.run_dir / "placeholders" / f"{index:02d}_{slugify(label, 24)}.mp4"
        if not dest.exists():
            make_placeholder_clip(dest, label, 9.0, index, font_path("display"), self.s.video)
        clip = _probe_clip(dest, "placeholder", label)
        if clip is None:
            raise PipelineError("Could not create a placeholder clip (is FFmpeg working?)")
        return clip
