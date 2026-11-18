"""Speech-to-text with word-level timestamps (faster-whisper, local, free).

Word timestamps are not a nicety here — they are what lets the cue placer drop
a sound effect into a gap between words instead of on top of a syllable, and
what makes word-accurate burned captions possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import ToolError, key_hash, read_json, require, run, write_json


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def wps(self) -> float:
        """Words per second — a decent proxy for pace and excitement."""
        return len(self.words) / self.duration if self.duration > 0.2 else 0.0


@dataclass
class Transcript:
    language: str
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def words(self) -> list[Word]:
        return [w for u in self.utterances for w in u.words]

    def text_between(self, start: float, end: float) -> str:
        parts = [u.text.strip() for u in self.utterances
                 if u.end > start and u.start < end]
        return " ".join(p for p in parts if p).strip()

    def words_between(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.end > start and w.start < end]

    def is_empty(self) -> bool:
        return not any(u.words for u in self.utterances)


def _to_dict(t: Transcript) -> dict[str, Any]:
    return {
        "language": t.language,
        "utterances": [
            {"start": u.start, "end": u.end, "text": u.text,
             "words": [{"start": w.start, "end": w.end, "text": w.text} for w in u.words]}
            for u in t.utterances
        ],
    }


def _from_dict(data: dict[str, Any]) -> Transcript:
    return Transcript(
        language=data.get("language", "en"),
        utterances=[
            Utterance(
                start=float(u["start"]), end=float(u["end"]), text=u.get("text", ""),
                words=[Word(float(w["start"]), float(w["end"]), w["text"])
                       for w in u.get("words", [])],
            )
            for u in data.get("utterances", [])
        ],
    )


def extract_audio(video: Path | str, dest: Path) -> Path:
    """16 kHz mono wav — what whisper wants, and small enough to cache."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run([
        require("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-v", "error",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(dest),
    ])
    return dest


def transcribe(
    video: Path | str,
    work_dir: Path,
    *,
    model_size: str = "small",
    language: str | None = None,
    compute_type: str = "int8",
    device: str = "cpu",
    vad: bool = True,
) -> Transcript:
    """Transcribe with word timestamps, caching the result in work_dir.

    Raises ToolError with install guidance if faster-whisper is unavailable —
    the caller can catch that and fall back to the audio-only path.
    """
    work_dir = Path(work_dir)
    cache_key = key_hash({
        "file": str(Path(video).resolve()),
        "size": Path(video).stat().st_size,
        "model": model_size, "lang": language, "vad": vad,
    })
    cache_path = work_dir / f"transcript_{cache_key}.json"
    cached = read_json(cache_path)
    if cached:
        return _from_dict(cached)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ToolError(
            "faster-whisper is not installed, so there is no transcript to work from.\n"
            "  pip install faster-whisper\n"
            "Or re-run with --no-asr to select clips from audio energy and scene "
            "cuts alone (no captions, weaker clip choices)."
        ) from exc

    audio = extract_audio(video, work_dir / f"audio_{cache_key}.wav")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        vad_filter=vad,
        condition_on_previous_text=False,   # avoids runaway repetition on long files
    )

    utterances: list[Utterance] = []
    for seg in segments:
        words = [
            Word(float(w.start), float(w.end), (w.word or "").strip())
            for w in (seg.words or [])
            if w.start is not None and w.end is not None and (w.word or "").strip()
        ]
        text = (seg.text or "").strip()
        if not text and not words:
            continue
        utterances.append(Utterance(float(seg.start), float(seg.end), text, words))

    transcript = Transcript(language=getattr(info, "language", language or "en"),
                            utterances=utterances)
    write_json(cache_path, _to_dict(transcript))
    return transcript


_WORD_MAX = 1.2           # a 'word' longer than this is a caption artefact
_UTTERANCE_GAP = 0.7      # a pause longer than this starts a new utterance
_UTTERANCE_MAX = 6.0      # ... and so does running on for this long


def _spread(texts: list[str], stamps: list[float], end: float) -> list[float]:
    """Give words even timings across a line that was stamped as one block."""
    span = max(end - stamps[0], 0.25 * len(texts))
    return [stamps[0] + i * span / len(texts) for i in range(len(texts))]


def _json3_words(events: list[dict[str, Any]]) -> list[Word]:
    words: list[Word] = []
    for ev in events:
        if ev.get("aAppend"):
            continue        # rolling duplicate of the line before it
        start = float(ev.get("tStartMs", 0)) / 1000.0
        dur = float(ev.get("dDurationMs", 0)) / 1000.0
        end = start + dur if dur > 0 else start

        texts: list[str] = []
        stamps: list[float] = []
        for seg in ev.get("segs") or []:
            text = (seg.get("utf8") or "").strip()
            if not text:
                continue        # newline-only filler
            texts.append(text)
            stamps.append(start + float(seg.get("tOffsetMs", 0)) / 1000.0)
        if not texts:
            continue

        # Hand-written tracks time whole lines, not words, and auto-generated
        # ones sometimes stamp every segment at the line start. Both get spread.
        if len(texts) == 1 and " " in texts[0]:
            texts = texts[0].split()
            stamps = _spread(texts, stamps, end)
        elif len(texts) > 1 and stamps[-1] - stamps[0] < 0.05:
            stamps = _spread(texts, stamps, end)

        for i, (w_start, text) in enumerate(zip(stamps, texts)):
            if i + 1 < len(stamps):
                w_end = stamps[i + 1]
            else:
                # A line's own duration is the only clue for its last word, and
                # auto-captions routinely overrun well past where speech stopped.
                w_end = min(max(end, w_start + 0.12), w_start + _WORD_MAX)
            words.append(Word(w_start, max(w_end, w_start + 0.08), text))

    words.sort(key=lambda w: w.start)
    # Caption events overlap, so a line's last word can outlive the first word
    # of the line after it. Left alone, that swallows the speech gaps the cue
    # placer aims for and holds captions on screen too long.
    for earlier, later in zip(words, words[1:]):
        if earlier.end > later.start:
            earlier.end = max(later.start, earlier.start + 0.08)
    return words


def _flush(bucket: list[Word]) -> Utterance:
    return Utterance(bucket[0].start, bucket[-1].end,
                     " ".join(w.text for w in bucket), list(bucket))


def from_json3(path: Path | str, *, language: str = "") -> Transcript:
    """Build a transcript from a YouTube json3 caption track.

    Auto-generated captions carry a per-word offset, which is precisely what
    the cue placer and the caption burner need — so a fetched video usually
    needs no Whisper at all, saving a 500 MB model download and minutes of CPU
    on every run.
    """
    path = Path(path)
    if not path.is_file():
        raise ToolError(f"no such caption file: {path}")
    data = read_json(path)
    if not isinstance(data, dict) or "events" not in data:
        raise ToolError(f"{path.name} is not a json3 caption file")

    words = _json3_words(data.get("events") or [])
    if not words:
        raise ToolError(f"{path.name} holds no timed words")

    utterances: list[Utterance] = []
    bucket: list[Word] = []
    for word in words:
        too_late = bucket and word.start - bucket[-1].end > _UTTERANCE_GAP
        too_long = bucket and word.end - bucket[0].start > _UTTERANCE_MAX
        if too_late or too_long:
            utterances.append(_flush(bucket))
            bucket = []
        bucket.append(word)
    if bucket:
        utterances.append(_flush(bucket))

    if not language and path.name.count(".") >= 2:
        language = path.name.split(".")[-2]
    return Transcript(language=language or "en", utterances=utterances)


def empty_transcript() -> Transcript:
    return Transcript(language="", utterances=[])
