"""Burned-in captions as ASS subtitles.

Short-form captions are read, not listened to, so they are chunked into a few
words at a time rather than shown as long sentences. Words the director flagged
as emphasis get a colour swap, which is what makes a caption feel edited rather
than auto-generated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .transcribe import Transcript, Word
from .util import ass_time

# ASS uses &HAABBGGRR (alpha, blue, green, red) — not RGB.
STYLE_PRESETS: dict[str, dict[str, str]] = {
    "bold-white": {
        "font": "Inter Display Black", "size": "86", "primary": "&H00FFFFFF",
        "outline": "&H00000000", "back": "&H66000000", "emphasis": "&H0030E0FF",
        "border": "7", "shadow": "2",
    },
    "yellow-pop": {
        "font": "Inter Display Black", "size": "88", "primary": "&H0000E8FF",
        "outline": "&H00101010", "back": "&H66000000", "emphasis": "&H00FFFFFF",
        "border": "7", "shadow": "2",
    },
    "clean": {
        "font": "Inter", "size": "74", "primary": "&H00FFFFFF",
        "outline": "&H00202020", "back": "&H00000000", "emphasis": "&H0080D8FF",
        "border": "4", "shadow": "0",
    },
}


@dataclass
class Chunk:
    start: float
    end: float
    words: list[Word]


def _normalise(text: str) -> str:
    return re.sub(r"[^\w']+", "", (text or "").lower())


def chunk_words(
    words: list[Word],
    *,
    max_words: int = 4,
    max_chars: int = 26,
    max_duration: float = 2.4,
    gap_break: float = 0.42,
) -> list[Chunk]:
    """Group words into caption cards, breaking on pauses and punctuation."""
    chunks: list[Chunk] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            chunks.append(Chunk(current[0].start, current[-1].end, list(current)))
            current.clear()

    for i, word in enumerate(words):
        current.append(word)
        text_len = sum(len(w.text) + 1 for w in current)
        span = current[-1].end - current[0].start
        next_gap = (words[i + 1].start - word.end) if i + 1 < len(words) else 99.0
        ends_sentence = word.text.rstrip().endswith((".", "!", "?", ",", ":", ";"))

        if (len(current) >= max_words or text_len >= max_chars
                or span >= max_duration or next_gap >= gap_break
                or (ends_sentence and len(current) >= 2)):
            flush()
    flush()
    return chunks


def _escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
            .replace("\n", " ").strip())


def build_ass(
    transcript: Transcript,
    *,
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    style: str = "bold-white",
    emphasis: list[str] | None = None,
    margin_v: int | None = None,
    uppercase: bool = True,
) -> str:
    """Render an ASS file covering one clip, timed relative to the clip start."""
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["bold-white"])
    # How many characters actually fit on one line, rather than a number that
    # happened to look right at some other size. A heavy upper-case face runs
    # about 0.62 of its point size per glyph, and the style keeps 60px either
    # side. Cards longer than this used to run off both edges of the frame.
    side = 60
    fits = int((width - 2 * side) / (float(preset["size"]) * 0.62))
    emph = {_normalise(w) for w in (emphasis or []) if _normalise(w)}
    margin = margin_v if margin_v is not None else int(height * 0.17)

    words = [
        Word(w.start - clip_start, w.end - clip_start, w.text)
        for w in transcript.words_between(clip_start, clip_end)
    ]
    words = [w for w in words if w.end > 0 and w.start < (clip_end - clip_start)]
    for w in words:
        w.start = max(0.0, w.start)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{preset['font']},{preset['size']},{preset['primary']},{preset['primary']},{preset['outline']},{preset['back']},-1,0,0,0,100,100,0,0,1,{preset['border']},{preset['shadow']},2,60,60,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: list[str] = []
    for chunk in chunk_words(words, max_chars=max(12, fits)):
        pieces: list[str] = []
        for w in chunk.words:
            text = _escape(w.text)
            if uppercase:
                text = text.upper()
            if _normalise(w.text) in emph:
                pieces.append(f"{{\\c{preset['emphasis']}}}{text}{{\\c{preset['primary']}}}")
            else:
                pieces.append(text)
        body = " ".join(pieces)
        # A small scale-up on entry reads as a "pop" without being distracting.
        body = r"{\fscx92\fscy92\t(0,110,\fscx100\fscy100)}" + body
        lines.append(
            f"Dialogue: 0,{ass_time(chunk.start)},{ass_time(chunk.end + 0.06)},"
            f"Cap,,0,0,0,,{body}"
        )

    return header + "\n".join(lines) + "\n"


def has_any(transcript: Transcript, clip_start: float, clip_end: float) -> bool:
    return bool(transcript.words_between(clip_start, clip_end))
