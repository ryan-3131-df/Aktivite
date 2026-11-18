"""
Stage 1 - Script & metadata generation with Claude.

Produces a strictly-validated `ShortScript` (hook → #6…#1 → loop outro) via
the Anthropic Messages API with JSON-schema structured outputs. In
`--local-clips` mode, sampled frames of your clips are sent to Claude's
vision so it can rank them and write narration about what is actually on
screen.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .models import MUSIC_MOODS, RANKS, SFX_CUES, ScriptSegment, ShortScript, SourceClip
from .utils import PipelineError, log

# Models that accept the server-side refusal fallback (`fallbacks="default"`).
_FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """\
You are the head writer of a viral YouTube Shorts channel that posts fast "Top 6" countdown rankings: \
vertical 9:16 videos, 40-50 seconds long, narrated by a high-energy AI voice over rapid-cut footage with \
bold word-by-word captions and a ranking tracker on screen. Your scripts are known for near-100% \
retention and for looping so seamlessly that viewers watch twice without noticing.

STRUCTURE (exactly 8 segments, in this order)
1. hook (≈0-3 s, 7-10 words): a pattern interrupt. Open mid-thought with a bold claim, a shocking \
specific, or a direct challenge to the viewer. Zero fluff: never "welcome", "in this video", "let's", \
"today", "here are". The hook's FIRST words must also be the grammatical continuation of the outro's \
last words (see LOOP).
2-7. rank 6 → rank 1 (≈6 s each, 14-19 words each): start every rank with "Number N," followed \
immediately by the item name, then ONE punchy, concrete, verifiable detail that justifies the spot \
(numbers, vivid verbs, sensory specifics). Escalate - each item must feel bigger than the last, and #1 \
is the undeniable payoff. Sprinkle micro open loops on 2-3 of the ranks (e.g. "…but wait until you \
see number one."), never on two consecutive ranks.
8. outro (≈5-6 s, 12-16 words): a fast final twist or payoff that ends on an UNFINISHED clause, so the \
hook completes the sentence when the video loops. No "subscribe", "like", "comment", or goodbyes.

LOOP (critical)
Read outro → hook as one sentence. Example: outro "…and the scariest part is that this is exactly why" \
+ hook "these 6 animals could end you in seconds." loops as "…this is exactly why these 6 animals could \
end you in seconds." The outro must not end with a full stop that closes the thought. End the outro's \
voiceover_text with no terminal punctuation or with an ellipsis-free trailing clause.

STYLE
- Write for the ear: short sentences, contractions, present tense, no parentheses, no emojis, no \
hashtags, no lists inside sentences.
- Spell things the way they should be spoken: "miles per hour" not "mph", "percent" not "%". Plain \
digits are fine for numbers ("Number 6," "over 300 people").
- Total spoken words across all segments: {min_words}-{max_words}.
- Facts must be accurate and widely verifiable. Never invent statistics; if unsure, pick a different item \
or phrase it as a widely reported claim.

ON-SCREEN TEXT
- topic_title: the persistent header, 2-4 words, max 24 characters (e.g. "DEADLIEST ANIMALS").
- hook on_screen_title: the big title card, 2-5 words.
- rank on_screen_title: the item's name only, max 22 characters.
- outro on_screen_title: 2-4 words.

SEARCH KEYWORDS (stock footage)
3 short, concrete, visual phrases per segment, most specific first, e.g. ["box jellyfish", \
"jellyfish underwater", "ocean tentacles"]. Describe what a camera would literally see; avoid abstract \
ideas, logos, and specific private people.

SOUND EFFECT CUES
sound_effect_cue is one of: {sfx}. Hook: usually "bass_drop" or "riser". Ranks: "whoosh" or "pop" \
(whoosh/pop are also added automatically on every transition, so use "ding" or "boom" for emphasis). \
#1: "boom" or "bass_drop". Outro: "ding" or "none".

METADATA
youtube_title: <= 70 characters, curiosity-driven, no clickbait lies. youtube_description: 2-3 short \
lines. hashtags: 5-8 tags including "#shorts". music_mood: one of {moods}.
"""

LOCAL_CLIPS_ADDENDUM = """\

LOCAL CLIPS MODE
The user supplied their own raw clips. You will see sampled frames (early / middle / late) of each clip, \
labelled with a clip ID. Assign exactly one distinct clip to every rank via clip_id and write each rank's \
narration about what is actually visible in that clip. {order_rule} Use "" as clip_id for the hook and \
outro. If no topic is given, infer the most compelling ranking topic that ties the clips together. \
search_keywords may describe the clip's content (they are only used for metadata).
"""


@dataclass
class LocalClip:
    clip_id: str
    source: SourceClip
    frames: list[bytes]
    forced_rank: int | None = None


# --------------------------------------------------------------------------- #
# JSON schema for structured outputs
# --------------------------------------------------------------------------- #
def build_schema(clip_ids: list[str] | None) -> dict:
    segment_props: dict = {
        "segment_type": {"type": "string", "enum": ["hook", "rank", "outro"]},
        "rank": {"type": "integer", "description": "6..1 for ranked items; 0 for hook and outro"},
        "voiceover_text": {"type": "string"},
        "on_screen_title": {"type": "string"},
        "search_keywords": {"type": "array", "items": {"type": "string"}},
        "sound_effect_cue": {"type": "string", "enum": list(SFX_CUES)},
    }
    required = list(segment_props)
    if clip_ids is not None:
        segment_props["clip_id"] = {"type": "string", "enum": [*clip_ids, ""]}
        required.append("clip_id")
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "topic_title": {"type": "string"},
            "youtube_title": {"type": "string"},
            "youtube_description": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "music_mood": {"type": "string", "enum": list(MUSIC_MOODS)},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": segment_props,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["topic", "topic_title", "youtube_title", "youtube_description", "hashtags",
                     "music_mood", "segments"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Claude call
# --------------------------------------------------------------------------- #
def _user_content(topic: str, settings: Settings, local_clips: list[LocalClip] | None,
                  feedback: list[str] | None) -> list[dict]:
    content: list[dict] = []
    if local_clips:
        for lc in local_clips:
            src = lc.source
            header = (f"Clip {lc.clip_id} — file '{src.path.name}', {src.duration:.1f}s, "
                      f"{src.width}x{src.height}")
            if lc.forced_rank:
                header += f" — MUST be rank #{lc.forced_rank}"
            content.append({"type": "text", "text": header})
            for jpg in lc.frames:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                            "data": base64.b64encode(jpg).decode()}})
    lang = "" if settings.language.startswith("en") else f"\nWrite every spoken and on-screen line in language code '{settings.language}'."
    ask = f"Topic: {topic}" if topic else "Topic: (infer it from the clips)"
    ask += lang + "\nWrite the complete Top 6 Short now."
    if feedback:
        ask += ("\n\nA previous draft was rejected for these reasons - fix all of them:\n- "
                + "\n- ".join(feedback))
    content.append({"type": "text", "text": ask})
    return content


def _extract_text(message) -> str:
    """Text of the final answer; ignores anything a declined model wrote before a fallback block."""
    blocks = list(message.content)
    last_fallback = max((i for i, b in enumerate(blocks) if getattr(b, "type", "") == "fallback"), default=-1)
    return "".join(b.text for b in blocks[last_fallback + 1:] if getattr(b, "type", "") == "text")


def _call_claude(system: str, content: list[dict], schema: dict, settings: Settings) -> ShortScript:
    try:
        import anthropic
    except ImportError as exc:
        raise PipelineError("The 'anthropic' package is not installed: pip install -r requirements.txt") from exc

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None, max_retries=3)
    kwargs: dict = dict(
        model=settings.claude_model,
        max_tokens=32000,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": settings.claude_effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if settings.claude_fallbacks and settings.claude_model in _FALLBACK_MODELS:
        kwargs.update(betas=[_FALLBACK_BETA], fallbacks="default")

    try:
        # Streaming keeps long thinking + output safely under HTTP timeouts.
        with client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise PipelineError("Anthropic rejected the API key - check ANTHROPIC_API_KEY in .env") from exc
    except anthropic.PermissionDeniedError as exc:
        raise PipelineError(f"Anthropic API key lacks permission for {settings.claude_model}: {exc.message}") from exc
    except anthropic.NotFoundError as exc:
        raise PipelineError(f"Model '{settings.claude_model}' not found - check CLAUDE_MODEL") from exc
    except anthropic.BadRequestError as exc:
        raise PipelineError(f"Anthropic rejected the request: {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise PipelineError("Anthropic rate limit hit even after retries - wait a minute and re-run") from exc
    except anthropic.APIStatusError as exc:
        raise PipelineError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise PipelineError("Could not reach the Anthropic API - check your internet connection") from exc

    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        cat = getattr(details, "category", None) if details else None
        raise PipelineError(f"Claude declined this topic (category: {cat}). Try rephrasing the topic.")
    if message.stop_reason == "max_tokens":
        raise PipelineError("Claude ran out of output tokens before finishing the script - re-run.")
    if any(getattr(b, "type", "") == "fallback" for b in message.content):
        log.info("Primary model declined; script served by fallback model %s", message.model)

    text = _extract_text(message)
    try:
        return ShortScript.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelineError(f"Claude returned JSON that does not match the script schema: {exc}") from exc


def generate_script(topic: str, settings: Settings, local_clips: list[LocalClip] | None = None,
                    keep_order: bool = False) -> ShortScript:
    """Generate and validate a script, re-prompting with the validation errors if needed."""
    if not settings.anthropic_api_key:
        log.info("ANTHROPIC_API_KEY not set - relying on the SDK's default credential chain")
    lo, hi = settings.target_word_range()
    system = SYSTEM_PROMPT.format(min_words=lo, max_words=hi, sfx=", ".join(SFX_CUES), moods=", ".join(MUSIC_MOODS))
    clip_ids = None
    if local_clips:
        clip_ids = [lc.clip_id for lc in local_clips]
        order_rule = ("Respect the fixed rank noted next to each clip." if keep_order else
                      "Choose the 6 strongest clips and order them so the most jaw-dropping clip is #1.")
        system += LOCAL_CLIPS_ADDENDUM.format(order_rule=order_rule)
    schema = build_schema(clip_ids)

    feedback: list[str] | None = None
    last_script: ShortScript | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log.info("Asking %s for a script (attempt %d/%d)…", settings.claude_model, attempt, MAX_ATTEMPTS)
        script = _call_claude(system, _user_content(topic, settings, local_clips, feedback), schema, settings)
        errors = script.structure_errors(word_range=(lo - 6, hi + 8), clip_ids=clip_ids)
        if keep_order and local_clips and not errors:
            for lc in local_clips:
                if lc.forced_rank and script.rank_segment(lc.forced_rank).clip_id != lc.clip_id:
                    errors.append(f"clip {lc.clip_id} must be rank #{lc.forced_rank}")
        if not errors:
            log.info("Script OK: %d words, topic '%s'", script.word_count(), script.topic)
            return _tidy(script)
        log.warning("Script needs fixes: %s", "; ".join(errors))
        feedback, last_script = errors, script

    structural = last_script.structure_errors(clip_ids=clip_ids) if last_script else ["no script"]
    if last_script and not structural:
        log.warning("Using the last draft despite soft issues (word count) after %d attempts", MAX_ATTEMPTS)
        return _tidy(last_script)
    raise PipelineError("Claude could not produce a valid script: " + "; ".join(structural))


def _tidy(script: ShortScript) -> ShortScript:
    tags = []
    for t in script.hashtags:
        t = "#" + t.strip().lstrip("#").replace(" ", "")
        if len(t) > 1 and t.lower() not in {x.lower() for x in tags}:
            tags.append(t)
    if "#shorts" not in {t.lower() for t in tags}:
        tags.insert(0, "#shorts")
    script.hashtags = tags
    script.topic_title = script.topic_title.strip().upper()
    return script


# --------------------------------------------------------------------------- #
# Offline template (used by --dry-run)
# --------------------------------------------------------------------------- #
_DRY_LINES = [
    "looks harmless at first, but the numbers behind it are honestly hard to believe.",
    "broke every expectation, and the internet still argues about how it even happened.",
    "is the one most people get wrong, and the real story is even wilder.",
    "turns up the intensity so fast you will want to watch it twice.",
    "almost took the top spot, but number one is on a completely different level.",
    "is the undisputed king, because nothing else even comes close to this.",
]


def offline_script(topic: str, local_clips: list[LocalClip] | None = None) -> ShortScript:
    """A deterministic placeholder script so the full render path can be tested with no API keys."""
    topic = topic.strip() or "Viral Moments"
    clips = list(local_clips or [])[:6]
    if any(lc.forced_rank for lc in clips):
        clips.sort(key=lambda lc: -(lc.forced_rank or 0))
    names: dict[int, str] = {}
    ids: dict[int, str] = {}
    for rank, i in zip(RANKS, range(6)):
        if i < len(clips):
            stem = clips[i].source.path.stem
            pretty = " ".join(w for w in stem.replace("_", " ").replace("-", " ").split() if not w.isdigit())
            names[rank] = (pretty or f"Clip {clips[i].clip_id}").title()[:22]
            ids[rank] = clips[i].clip_id
        else:
            names[rank] = f"Contender {rank}"
            ids[rank] = ""
    segs = [ScriptSegment(segment_type="hook", rank=0, voiceover_text=f"nobody is ready for these 6 {topic}, and number one is unreal.",
                          on_screen_title=topic, search_keywords=[topic], sound_effect_cue="bass_drop")]
    for i, rank in enumerate(RANKS):
        segs.append(ScriptSegment(
            segment_type="rank", rank=rank,
            voiceover_text=f"Number {rank}, {names[rank]}. This one {_DRY_LINES[i]}",
            on_screen_title=names[rank], search_keywords=[names[rank], topic],
            sound_effect_cue="boom" if rank == 1 else "whoosh", clip_id=ids[rank]))
    segs.append(ScriptSegment(segment_type="outro", rank=0,
                              voiceover_text="Now you know the real ranking, and honestly, that is exactly why",
                              on_screen_title="Watch Again", search_keywords=[topic], sound_effect_cue="ding"))
    return ShortScript(topic=topic, topic_title=topic.upper()[:24], youtube_title=f"Top 6 {topic} 🤯",
                       youtube_description=f"Ranking the top 6 {topic}. Which one surprised you?",
                       hashtags=["#shorts", "#top6", "#ranking"], music_mood="phonk", segments=segs)


def save_script(script: ShortScript, path: Path) -> None:
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")


def load_script(path: Path) -> ShortScript:
    return ShortScript.model_validate_json(Path(path).read_text(encoding="utf-8"))
