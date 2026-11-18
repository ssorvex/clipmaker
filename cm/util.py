"""Shared helpers: subprocess wrapping, ffprobe, caching, small math."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


class ToolError(RuntimeError):
    pass


def require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise ToolError(
            f"`{binary}` not found on PATH. Install ffmpeg "
            "(macOS: `brew install ffmpeg`, Debian/Ubuntu: `apt install ffmpeg`)."
        )
    return path


def run(args: Sequence[str], *, quiet: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, capturing output. Raises ToolError with trimmed stderr."""
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        head = " ".join(str(a) for a in args[:10])
        raise ToolError(
            f"command failed ({proc.returncode}): {head} ...\n"
            f"{(proc.stderr or '').strip()[-3000:]}"
        )
    return proc


def ffmpeg(args: Sequence[str], **kw) -> subprocess.CompletedProcess:
    return run([require("ffmpeg"), "-hide_banner", "-nostdin", "-y", *args], **kw)


def ffprobe_json(path: Path | str, *extra: str) -> dict[str, Any]:
    proc = run([
        require("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", *extra, str(path),
    ])
    return json.loads(proc.stdout or "{}")


def media_info(path: Path | str) -> dict[str, Any]:
    """Duration, dimensions, fps, and whether there is an audio stream."""
    data = ffprobe_json(path)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    fps = 0.0
    if video:
        raw = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        try:
            num, _, den = raw.partition("/")
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    duration = 0.0
    for candidate in (fmt.get("duration"), (video or {}).get("duration"), (audio or {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    return {
        "duration": duration,
        "width": int((video or {}).get("width") or 0),
        "height": int((video or {}).get("height") or 0),
        "fps": fps,
        "has_video": video is not None,
        "has_audio": audio is not None,
    }


def volume_peak_db(path: Path | str) -> float:
    """Peak level in dBFS via volumedetect. Returns -inf-ish for silence."""
    proc = run([
        require("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(path),
        "-af", "volumedetect", "-f", "null", "-",
    ], check=False)
    for line in (proc.stderr or "").splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].strip().split()[0])
            except (IndexError, ValueError):
                break
    return -90.0


def key_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def slugify(text: str, limit: int = 48) -> str:
    out = []
    for ch in (text or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return ("".join(out).strip("-") or "clip")[:limit].strip("-")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def zscores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(var)
    if sd < 1e-9:
        return [0.0] * len(values)
    return [(v - mean) / sd for v in values]


def fmt_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def ass_time(seconds: float) -> str:
    """ASS subtitle timestamp: H:MM:SS.cc"""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):d}:{int(minutes):02d}:{secs:05.2f}"
