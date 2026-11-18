"""
Offline unit tests for the logic that doesn't need API keys or FFmpeg.

    python -m pytest tests -q        # or:  python tests/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_gen import _chars_to_words, align_to_script, estimate_words, script_tokens  # noqa: E402
from src.config import Settings  # noqa: E402
from src.models import RANKS, SourceClip  # noqa: E402
from src.script_gen import build_schema, offline_script  # noqa: E402
from src.timeline import build_timeline  # noqa: E402
from src.utils import caption_text, normalize_token, number_to_words  # noqa: E402


def test_number_and_token_normalisation():
    assert number_to_words(6) == "six"
    assert number_to_words(300) == "three hundred"
    assert number_to_words(1250) == "one thousand two hundred fifty"
    assert normalize_token("6,") == ["six"]
    assert normalize_token("1,000") == ["one", "thousand"]
    assert normalize_token("Wi-Fi!") == ["wi", "fi"]
    assert caption_text("unreal.") == "UNREAL"
    assert caption_text("why?") == "WHY?"
    assert caption_text("don't,") == "DON'T"


def test_alignment_handles_digits_splits_and_missing_words():
    tokens = [("Number", 0), ("6,", 0), ("the", 0), ("box-jellyfish.", 0), ("It", 1), ("kills", 1),
              ("in", 1), ("1,000", 1), ("seconds!", 1)]
    asr = [("Number", 0.10, 0.40), ("six", 0.42, 0.70), ("the", 0.75, 0.85), ("box", 0.86, 1.10),
           ("jellyfish", 1.10, 1.70), ("It", 2.00, 2.10), ("kills", 2.12, 2.50),  # "in" dropped by ASR
           ("1000", 2.70, 3.30), ("seconds", 3.32, 3.90)]
    words = align_to_script(tokens, asr, duration=4.0)
    assert [w.word for w in words] == [t for t, _ in tokens]
    by = {w.word: w for w in words}
    assert abs(by["6,"].start - 0.42) < 1e-6 and abs(by["6,"].end - 0.70) < 1e-6
    assert abs(by["box-jellyfish."].start - 0.86) < 1e-6 and abs(by["box-jellyfish."].end - 1.70) < 1e-6
    assert 2.50 <= by["in"].start <= 2.70                       # interpolated into the gap
    assert abs(by["1,000"].start - 2.70) < 1e-6
    assert all(a.end <= b.start + 1e-6 for a, b in zip(words, words[1:]))   # monotonic, no overlap
    assert [w.segment_index for w in words] == [0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_elevenlabs_character_alignment_to_words():
    text = "Hi there, 6!"
    starts = [i * 0.1 for i in range(len(text))]
    ends = [s + 0.1 for s in starts]
    words = _chars_to_words(list(text), starts, ends)
    assert [w[0] for w in words] == ["Hi", "there,", "6!"]
    assert abs(words[1][1] - 0.3) < 1e-9 and abs(words[1][2] - 0.9) < 1e-9


def test_offline_script_is_structurally_valid_and_loops():
    script = offline_script("street foods")
    assert script.structure_errors() == []
    assert [s.rank for s in script.segments[1:7]] == list(RANKS)
    # Loop: outro must end on an open clause that the hook completes.
    assert not script.outro.voiceover_text.rstrip().endswith((".", "!", "?"))
    assert script.hook.voiceover_text[0].islower()


def test_schema_is_strict():
    schema = build_schema(["A", "B", "C", "D", "E", "F"])
    seg = schema["properties"]["segments"]["items"]
    assert schema["additionalProperties"] is False and seg["additionalProperties"] is False
    assert set(seg["required"]) == set(seg["properties"])
    assert seg["properties"]["clip_id"]["enum"][-1] == ""
    assert "clip_id" not in build_schema(None)["properties"]["segments"]["items"]["properties"]


def _fake_media() -> dict[int, list[SourceClip]]:
    return {r: [SourceClip(path=Path(f"/tmp/clip{r}_{k}.mp4"), duration=9.0, width=1080, height=1920)
                for k in range(2)] for r in RANKS}


def test_timeline_is_frame_accurate_and_captions_are_short():
    settings = Settings()
    script = offline_script("street foods")
    tokens = script_tokens(script)
    raw = estimate_words([t for t, _ in tokens], duration=46.0)
    words = align_to_script(tokens, raw, 46.0)
    tl = build_timeline(script, words, 46.0, _fake_media(), settings, seed=1)
    fps = settings.video.fps

    # Shots tile the whole video with no gaps and land on frame boundaries.
    assert tl.shots[0].start == 0.0 and abs(tl.shots[-1].end - tl.duration) < 1e-9
    assert all(abs(a.end - b.start) < 1e-9 for a, b in zip(tl.shots, tl.shots[1:]))
    frames = sum(round(s.end * fps) - round(s.start * fps) for s in tl.shots)
    assert frames == tl.n_frames
    # 8 segments: hook, 6 ranks, outro; reveals in countdown order.
    assert [s.kind for s in tl.segments] == ["hook"] + ["rank"] * 6 + ["outro"]
    # Hormozi captions: never more than 3 words on screen.
    assert tl.captions and all(1 <= len(c.words) <= settings.captions.max_words for c in tl.captions)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(tl.captions, tl.captions[1:]))
    # A whoosh before every transition and a pop on every reveal.
    assert sum(e.name == "whoosh" for e in tl.sfx) == 7
    assert sum(e.name == "pop" for e in tl.sfx) == 6
    # Outro recap lights up the list rows 6 → 1.
    recap = [s.highlight_rank for s in tl.shots if s.segment_index == 7]
    assert recap == sorted(recap, reverse=True) and recap[-1] == 1


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    raise SystemExit(1 if failed else 0)
