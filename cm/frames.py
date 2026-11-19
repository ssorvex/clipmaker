"""Lay a video's frames out as one contact sheet, so a person can see it.

Choosing b-roll means knowing which stretches of a video show the presenter
and which show the game, and no cheap signal separates those: scene cuts fire
on both, loudness says nothing about the picture. Looking is the reliable
method, and a single tiled image is how you look at twenty-five minutes
without opening twenty-five minutes.

    python -m cm.frames video.mp4 --out sheet.jpg --count 36

The grid reads left to right, top to bottom. The printed table gives the
timestamp of each cell, since burning them into the image needs a font
configuration that is not reliably present on Windows.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .util import require, run


def contact_sheet(video: Path | str, dest: Path | str, *, count: int = 36,
                  columns: int = 0, cell_width: int = 320,
                  start: float = 0.0, end: float = 0.0) -> tuple[Path, list[float]]:
    """Write one tiled image sampling `count` frames, and return their times."""
    from .probe import probe

    video = Path(video)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    duration = probe(video).duration
    first = max(0.0, start)
    last = end if end > first else duration
    span = max(0.1, last - first)

    count = max(1, count)
    columns = columns or max(1, int(math.ceil(math.sqrt(count) * 1.2)))
    rows = int(math.ceil(count / columns))

    # Sample at the centre of each equal slice rather than at its edge: a cut
    # landing on a boundary would otherwise give a frame of the wrong shot.
    step = span / count
    times = [first + step * (i + 0.5) for i in range(count)]

    args = [require("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-ss", f"{first:.3f}", "-t", f"{span:.3f}", "-i", str(video),
            "-vf", (f"fps=1/{step:.4f},scale={cell_width}:-2,"
                    f"tile={columns}x{rows}:margin=4:padding=4"),
            "-frames:v", "1", "-q:v", "4", str(dest)]
    run(args)
    return dest, times


def main() -> int:
    ap = argparse.ArgumentParser(description="tile a video's frames into one image")
    ap.add_argument("video")
    ap.add_argument("--out", default="sheet.jpg")
    ap.add_argument("--count", type=int, default=36)
    ap.add_argument("--columns", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    args = ap.parse_args()

    dest, times = contact_sheet(args.video, args.out, count=args.count,
                                columns=args.columns, start=args.start,
                                end=args.end)
    cols = args.columns or max(1, int(math.ceil(math.sqrt(args.count) * 1.2)))
    print(f"  {dest}  ({len(times)} frames, {cols} per row)")
    for i, t in enumerate(times):
        end = "\n" if (i + 1) % cols == 0 else "  "
        print("%3d:%02d" % (int(t) // 60, int(t) % 60), end=end)
    if len(times) % cols:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
