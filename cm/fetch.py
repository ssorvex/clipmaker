"""Fetch a long video from a URL, so a pasted link can be the input.

Downloading is the easy half. The half that pays for itself is the subtitles:
YouTube already holds a word-timed transcript for most videos, and lifting it
costs one second and no model. Whisper, for the same result, wants a 500 MB
model download and minutes of CPU. So the fetch asks for captions in the same
pass and lets the caller prefer them, leaving Whisper for videos that have
none.

Everything is cached by video id, so re-running on the same link re-uses the
file on disk instead of pulling it again.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .util import ToolError

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

# Which caption track to take when a video offers several, best first. The
# video's own language is tried ahead of these.
SUB_PREFERENCE = ("en", "tr")


def is_url(value: str) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


@dataclass
class Fetched:
    video: Path
    subtitles: Path | None
    sub_language: str
    video_id: str
    title: str
    uploader: str
    duration: float
    url: str
    cached: bool


def _ytdlp() -> list[str]:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise ToolError(
            "`yt-dlp` not found. Install it with `pip install yt-dlp` "
            "(it is in requirements.txt)."
        ) from None
    return [sys.executable, "-m", "yt_dlp"]


def _auth_hint(stderr: str) -> str:
    """yt-dlp's failures are mostly one of a few known walls — name them."""
    low = stderr.lower()
    if "sign in to confirm" in low or "not a bot" in low:
        return ("\nYouTube is asking for a sign-in. Re-run with "
                "`--cookies-from-browser chrome` (or edge/firefox) to borrow "
                "your browser's session.")
    if "age" in low and "confirm" in low:
        return "\nAge-gated video: `--cookies-from-browser chrome` gets past it."
    if "private video" in low or "members-only" in low:
        return "\nThis video is private or members-only."
    if "unavailable" in low and "country" in low:
        return "\nBlocked in this region."
    return ""


def _info(cmd: list[str], url: str, cookies: str) -> dict:
    args = [*cmd, "--no-playlist", "--skip-download", "--dump-single-json"]
    if cookies:
        args += ["--cookies-from-browser", cookies]
    args.append(url)
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-1200:]
        raise ToolError(f"could not read {url}\n{tail}{_auth_hint(tail)}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ToolError(f"yt-dlp returned something that is not JSON: {exc}") from None


def _pick_sub_language(info: dict, language: str | None) -> tuple[str, bool]:
    """Choose exactly one caption track: its name, and whether it is automatic.

    One track, never a list. Every extra language is another HTTP request, and
    YouTube answers the third one with a 429 — which, asked for in the same
    call as the video, used to take the whole download down with it.

    Hand-written captions are preferred over automatic ones at the same
    language, being properly punctuated and not guessed.
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    original = str(info.get("language") or "").split("-")[0]

    seen: list[str] = []
    for want in [language, original, *SUB_PREFERENCE]:
        if not want or want in seen:
            continue
        seen.append(want)
        for table, is_auto in ((manual, False), (auto, True)):
            for have in table:
                if have == want or have.split("-")[0] == want:
                    return have, is_auto
    return "", False


def _id_from_url(url: str) -> str:
    """Best-effort video id without asking the network.

    Only needed when the metadata call fails: a source already on disk should
    not become unusable because YouTube decided today is a sign-in day.
    """
    parsed = urlparse(str(url))
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/")[0]
    if "youtube" in host:
        for part in parsed.query.split("&"):
            if part.startswith("v="):
                return part[2:]
        bits = [b for b in parsed.path.split("/") if b]
        if len(bits) >= 2 and bits[0] in ("shorts", "live", "embed"):
            return bits[1]
    return ""


def _find_video(cache_dir: Path, video_id: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        hit = cache_dir / f"{video_id}{suffix}"
        if hit.is_file() and hit.stat().st_size > 0:
            return hit
    return None


def _find_subs(cache_dir: Path, video_id: str) -> tuple[Path | None, str]:
    found: dict[str, Path] = {}
    for path in cache_dir.glob(f"{video_id}.*.json3"):
        # <id>.<lang>.json3 — the language is the second-to-last part.
        lang = path.name[len(video_id) + 1:-len(".json3")]
        found[lang] = path
    if not found:
        return None, ""
    for want in SUB_PREFERENCE:
        for lang, path in found.items():
            if lang == want or lang.startswith(want + "-"):
                return path, lang
    lang, path = next(iter(found.items()))
    return path, lang


def _stream(args: list[str], *, echo: bool = True) -> tuple[int, str]:
    """Run yt-dlp, showing progress as it goes. Returns (exit code, last lines).

    Streamed rather than captured because on a one-hour source this is the one
    part of a run that takes minutes with nothing to show for itself.
    """
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        del tail[:-40]
        if echo and (line.startswith("[download]") or line.startswith("[Merger]")
                     or "ERROR" in line):
            print(f"      {line}", flush=True)
    return proc.wait(), "\n".join(tail)


def _download_video(cmd: list[str], url: str, cache_dir: Path, *,
                    max_height: int, cookies: str) -> None:
    fmt = (f"bv*[height<={max_height}][vcodec^=avc1]+ba[acodec^=mp4a]/"
           f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b")
    args = [
        *cmd, "--no-playlist", "--no-warnings",
        "-f", fmt, "--merge-output-format", "mp4",
        "-o", str(cache_dir / "%(id)s.%(ext)s"),
        "--retries", "5", "--fragment-retries", "10",
        "--concurrent-fragments", "4",
        "--progress", "--newline",
    ]
    if cookies:
        args += ["--cookies-from-browser", cookies]
    args.append(url)

    code, tail = _stream(args)
    if code != 0:
        raise ToolError(f"download failed\n{tail}{_auth_hint(tail)}")


def _download_subs(cmd: list[str], url: str, cache_dir: Path, *,
                   language: str, auto: bool, cookies: str) -> None:
    """Best effort, in its own call: no caption track costs a Whisper pass,
    where a failure folded into the video download costs the whole run."""
    args = [
        *cmd, "--no-playlist", "--no-warnings", "--skip-download",
        "--write-auto-subs" if auto else "--write-subs",
        "--sub-format", "json3", "--sub-langs", language,
        "-o", str(cache_dir / "%(id)s.%(ext)s"),
    ]
    if cookies:
        args += ["--cookies-from-browser", cookies]
    args.append(url)

    code, tail = _stream(args, echo=False)
    if code != 0:
        reason = "rate-limited" if "429" in tail else "unavailable"
        print(f"      ! captions {reason} ({language}) — transcribing locally instead",
              flush=True)


def fetch(url: str, cache_dir: Path, *, max_height: int = 1080,
          want_subs: bool = True, language: str | None = None,
          cookies_from_browser: str = "") -> Fetched:
    """Download `url` into `cache_dir` (once) and report what landed."""
    cmd = _ytdlp()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        info = _info(cmd, url, cookies_from_browser)
    except ToolError:
        # A source already on disk should not become unusable because the site
        # decided today is a sign-in day. If the id is recoverable from the URL
        # and the file is here, the network's opinion does not matter.
        guess = _id_from_url(url)
        if guess and _find_video(cache_dir, guess) is not None:
            print(f"      ! metadata unavailable; using the cached copy of {guess}",
                  flush=True)
            info = {"id": guess, "title": guess}
        else:
            raise

    if info.get("_type") == "playlist":
        raise ToolError(
            "that link is a playlist or channel, not one video — "
            "paste the link to a single video"
        )
    video_id = str(info.get("id") or "video")
    title = str(info.get("title") or video_id)
    uploader = str(info.get("uploader") or info.get("channel") or "")
    duration = float(info.get("duration") or 0.0)

    existing = _find_video(cache_dir, video_id)
    subs, sub_lang = _find_subs(cache_dir, video_id)
    cached = existing is not None

    if existing is None:
        _download_video(cmd, url, cache_dir, max_height=max_height,
                        cookies=cookies_from_browser)
        existing = _find_video(cache_dir, video_id)

    if want_subs and subs is None:
        want_lang, is_auto = _pick_sub_language(info, language)
        if want_lang:
            _download_subs(cmd, url, cache_dir, language=want_lang,
                           auto=is_auto, cookies=cookies_from_browser)
            subs, sub_lang = _find_subs(cache_dir, video_id)

    if existing is None:
        raise ToolError(
            f"yt-dlp finished but no video file for {video_id} is in {cache_dir}"
        )

    return Fetched(
        video=existing, subtitles=subs, sub_language=sub_lang, video_id=video_id,
        title=title, uploader=uploader, duration=duration, url=url, cached=cached,
    )
