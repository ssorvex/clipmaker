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


def empty_transcript() -> Transcript:
    return Transcript(language="", utterances=[])
