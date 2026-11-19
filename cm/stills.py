"""Pull thumbnail candidates out of a finished clip.

A thumbnail built on a real frame of the clip beats one an image model
imagines, for a reason that has nothing to do with quality: it is the thing
the viewer is about to watch. The promise and the video match.

The clip has captions burned into it, so most frames are unusable — a
thumbnail with half a sentence across it looks like a mistake. The caption
file knows exactly when text is on screen, so we sample the silences
instead of guessing, then throw away the frames that are too dark or too
flat to read at the size of a phone's grid.

    python -m cm.stills clips/2026-11-18_1715/01-gta-6-plane --count 6

Writes `thumb-1.jpg` ... into the clip folder at full resolution, plus
`thumbs.jpg`, a numbered sheet of them side by side to choose from.
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from .util import media_info, require, run

# Below this the frame is a night shot or a fade and nothing reads on it;
# above it the frame is a white flash. Both are common and both are useless.
DARK = 38.0
BLOWN = 228.0

# A caption's own timing is not the whole truth: the style animates a scale
# over the first 110 ms, and a frame grabbed on the boundary catches the tail
# of one word. Stay clear of the edges.
EDGE = 0.25

# Shorter than this and the gap is a breath between words, not a silence.
MIN_GAP = 0.45

# Inside a long silence, take a frame about this often. Two seconds apart is
# enough for the shot to have changed; closer than that returns near-copies.
SPACING = 1.6


def _ass_times(path: Path) -> list[tuple[float, float]]:
    """Every interval where a caption is on screen, merged where they overlap."""
    spans: list[tuple[float, float]] = []
    pattern = re.compile(r"^Dialogue:\s*[^,]*,([^,]+),([^,]+),")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = pattern.match(line)
        if not found:
            continue
        try:
            start, end = (_clock(found.group(1)), _clock(found.group(2)))
        except ValueError:
            continue
        if end > start:
            spans.append((start, end))

    spans.sort()
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _clock(text: str) -> float:
    hours, minutes, seconds = text.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def gaps(spans: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """The stretches with no caption on screen, trimmed away from the edges."""
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in spans + [(duration, duration)]:
        if start - cursor >= MIN_GAP:
            out.append((cursor + EDGE, start - EDGE))
        cursor = max(cursor, end)
    return [(a, b) for a, b in out if b > a]


def sample_times(windows: list[tuple[float, float]], duration: float) -> list[float]:
    """Candidate timestamps: the middle of short silences, several in long ones."""
    times: list[float] = []
    for start, end in windows:
        span = end - start
        count = max(1, int(span / SPACING))
        step = span / count
        times.extend(start + step * (i + 0.5) for i in range(count))

    if not times:
        # Captions covered everything. Fall back to an even spread and accept
        # that the chosen frame will need the text cropped or covered.
        times = [duration * (i + 0.5) / 8 for i in range(8)]
    return times


def _mean_luma(frame: Path, scratch: Path) -> float:
    """Average brightness, read by asking ffmpeg to scale the frame to one pixel.

    Written to a file rather than read from the pipe: `run` decodes stdout as
    text, which would turn the one byte we want into a replacement character.
    """
    raw = scratch / (frame.stem + ".gray")
    run([require("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-v", "error",
         "-i", str(frame), "-vf", "scale=1:1,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", str(raw)])
    data = raw.read_bytes() if raw.is_file() else b""
    return float(data[0]) if data else 0.0


def band_crop(band: str) -> str:
    """Turn `TOP:BOTTOM` (fractions of the height) into a `crop=` filter.

    Streamers compose their own frame: a facecam box at the top, the game
    below it, the blurred pad of our own render around both. A thumbnail
    wants the game and nothing else, and where the game sits is a property
    of the source, not something worth detecting — one look at the sheet
    gives the two numbers.
    """
    if not band:
        return ""
    top, _, bottom = band.partition(":")
    first, last = float(top), float(bottom or 1.0)
    if not 0.0 <= first < last <= 1.0:
        raise SystemExit("--band wants TOP:BOTTOM between 0 and 1, e.g. 0.32:0.91")
    # Even dimensions: an odd height makes some encoders refuse the frame.
    return (f"crop=iw:floor(ih*{last - first:.4f}/2)*2:"
            f"0:floor(ih*{first:.4f}/2)*2")


def grab(video: Path, when: float, dest: Path, crop: str = "") -> Path:
    args = [require("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-ss", f"{when:.3f}", "-i", str(video)]
    if crop:
        args += ["-vf", crop]
    args += ["-frames:v", "1", "-q:v", "2", str(dest)]
    run(args)
    return dest


def pick(video: Path, captions: Path | None, duration: float,
         count: int, work: Path, crop: str = "") -> list[tuple[float, Path, float]]:
    """Grab every candidate, then keep the best `count` that are spread out.

    Detail is scored by the size of the JPEG: a busy, high-contrast frame
    compresses worse than a smear of sky, and at thumbnail size busy and
    high-contrast is exactly what survives. It is a crude measure and it is
    the right one here — the alternative is a quality model for a decision a
    person is about to make by eye anyway.
    """
    spans = _ass_times(captions) if captions and captions.is_file() else []
    times = sample_times(gaps(spans, duration), duration)

    scored: list[tuple[float, float, Path]] = []
    for i, when in enumerate(times):
        frame = work / ("cand-%03d.jpg" % i)
        try:
            grab(video, when, frame, crop)
        except Exception:
            continue
        if not frame.is_file() or not frame.stat().st_size:
            continue
        luma = _mean_luma(frame, work)
        if luma < DARK or luma > BLOWN:
            continue
        scored.append((float(frame.stat().st_size), when, frame))

    scored.sort(reverse=True)

    # Spread: two frames from the same shot tell a person nothing new, so a
    # candidate has to sit clear of everything already kept.
    apart = max(1.5, duration / (count * 2.0))
    kept: list[tuple[float, Path, float]] = []
    for size, when, frame in scored:
        if any(abs(when - k[0]) < apart for k in kept):
            continue
        kept.append((when, frame, size))
        if len(kept) == count:
            break

    kept.sort()
    return kept


def sheet(frames: list[Path], dest: Path, columns: int = 3,
          cell_width: int = 360) -> Path:
    """One image of the chosen frames, in order, to pick from."""
    rows = max(1, (len(frames) + columns - 1) // columns)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for i, frame in enumerate(frames, 1):
            shutil.copyfile(frame, staged / ("%03d.jpg" % i))
        run([require("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-v", "error",
             "-i", str(staged / "%03d.jpg"),
             "-vf", (f"scale={cell_width}:-2,"
                     f"tile={columns}x{rows}:margin=6:padding=6"),
             "-frames:v", "1", "-q:v", "3", str(dest)])
    return dest


def stills(clip_dir: Path | str, count: int = 6,
           band: str = "") -> tuple[list[Path], Path]:
    clip_dir = Path(clip_dir)
    video = clip_dir if clip_dir.is_file() else clip_dir / "clip.mp4"
    folder = video.parent
    captions = folder / "captions.ass"

    duration = float(media_info(video)["duration"])
    if duration <= 0:
        raise SystemExit("could not read a duration from %s" % video)

    with tempfile.TemporaryDirectory() as tmp:
        chosen = pick(video, captions, duration, count, Path(tmp),
                      band_crop(band))
        if not chosen:
            raise SystemExit("no usable frame found in %s" % video)

        for old in folder.glob("thumb-*.jpg"):
            old.unlink()

        out: list[Path] = []
        for i, (when, frame, _size) in enumerate(chosen, 1):
            dest = folder / ("thumb-%d.jpg" % i)
            shutil.copyfile(frame, dest)
            out.append(dest)
            print("  %d  %5.1f sn  ->  %s" % (i, when, dest.name))

        grid = sheet(out, folder / "thumbs.jpg")

    print("  sheet: %s" % grid)
    return out, grid


def main() -> int:
    ap = argparse.ArgumentParser(description="thumbnail candidates from a clip")
    ap.add_argument("clip", help="a clip folder, or the mp4 itself")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--band", default="",
                    help="keep only this vertical slice, e.g. 0.32:0.91 to "
                         "drop a facecam at the top and the blurred pad below")
    args = ap.parse_args()
    stills(args.clip, count=max(1, args.count), band=args.band)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
