"""Write the day's posting sheet into the Obsidian vault.

One file per day, one block per video, each field in its own code block so
Obsidian's copy button puts exactly the right text on the clipboard. The point
is a sheet you work down without thinking: open today's note, copy, paste,
schedule, tick.

YouTube and Instagram get different shapes on purpose. YouTube has a separate
tags field that takes plain keywords and is worth filling; Instagram has no
title at all, so its caption opens with what would have been one.

    python daysheet.py <clips_dir> <dest_dir>
    python daysheet.py <clips_dir> <dest_dir> --times 10:00 14:00 18:00
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

VAULT = Path.home() / "clipping-brain"
TIMES = ("10:00", "14:00", "18:00")

# YouTube's tags field wants plain keywords, not hashtags, so each tag we use
# has a written-out form. Kept explicit rather than derived: "GTA6Review" is
# not something a regex turns into "GTA 6 review" without guessing.
KEYWORD = {
    "GTA6": "GTA 6",
    "GTAVI": "GTA VI",
    "GTA5": "GTA 5",
    "RockstarGames": "Rockstar Games",
    "GTA6Review": "GTA 6 review",
    "GTA6Tips": "GTA 6 tips",
    "GTA6Police": "GTA 6 police",
    "GTA6Money": "GTA 6 money",
    "GTA6Secrets": "GTA 6 secrets",
    "GTA6Gameplay": "GTA 6 gameplay",
    "Leonida": "Leonida",
}

# Always included: the searches nobody thinks to tag for but everybody types.
BASE_KEYWORDS = ("Grand Theft Auto VI", "GTA 6 Leonida", "GTA 6 PS5")

# YouTube counts the whole tags field, commas included.
TAGS_LIMIT = 500


def _keyword(tag: str) -> str:
    """A hashtag as a person would type it into a search box."""
    tag = tag.lstrip("#")
    if tag in KEYWORD:
        return KEYWORD[tag]
    # Unknown tag: split CamelCase and letter/digit runs, then lower-case the
    # words that are not already an acronym.
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|\d+", tag) or [tag]
    out = []
    for part in parts:
        out.append(part if part.isupper() or part.isdigit() else part.lower())
    return " ".join(out)


def youtube_tags(hashtags: list[str]) -> str:
    seen: list[str] = []
    for word in [_keyword(t) for t in hashtags] + list(BASE_KEYWORDS):
        if word and word.lower() not in [s.lower() for s in seen]:
            seen.append(word)

    kept: list[str] = []
    for word in seen:
        candidate = ", ".join(kept + [word])
        if len(candidate) > TAGS_LIMIT:
            break
        kept.append(word)
    return ", ".join(kept)


def hashtag_line(hashtags: list[str]) -> str:
    return " ".join("#" + str(t).lstrip("#") for t in hashtags)


def read_clips(clips_dir: Path) -> list[dict]:
    """Number only the folders that hold a clip.

    The batch also contains `.work`, which sorts ahead of everything and would
    otherwise take number 1 and push every video one place along, so the count
    advances on a kept clip rather than on a directory. publish.ps1 numbers the
    copied files the same way, and the two have to agree.
    """
    found = []
    n = 0
    for folder in sorted(p for p in clips_dir.iterdir() if p.is_dir()):
        meta = folder / "meta.json"
        if not (folder / "clip.mp4").is_file() or not meta.is_file():
            continue
        n += 1
        plan = (json.loads(meta.read_text(encoding="utf-8")) or {}).get("plan") or {}
        found.append({
            "n": n,
            "title": plan.get("title") or folder.name,
            "description": plan.get("description") or "",
            "hashtags": [str(t).lstrip("#") for t in (plan.get("hashtags") or [])],
            "duration": round(float(plan.get("duration") or 0)),
            "start": float(plan.get("start") or 0),
        })
    return found


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return "%d:%02d" % (seconds // 60, seconds % 60)


def block(clip: dict, dest_dir: Path, when: str) -> str:
    video = dest_dir / ("%d.mp4" % clip["n"])
    link = video.as_uri()
    tags = hashtag_line(clip["hashtags"])
    description = clip["description"].strip() + "\n\n" + tags
    caption = clip["title"].strip() + "\n\n" + clip["description"].strip() + "\n\n" + tags

    return "\n".join([
        "### VIDEO %d" % clip["n"],
        "",
        "**Link :** [%s](%s)" % (video.name, link),
        "**Posting Time :** %s" % when,
        "**Süre :** %d sn  ·  kaynakta %s" % (clip["duration"], _clock(clip["start"])),
        "",
        "**YouTube — Title**",
        "```",
        clip["title"].strip(),
        "```",
        "",
        "**YouTube — Description + tags**",
        "```",
        description,
        "```",
        "",
        "**YouTube — Tags alanı**",
        "```",
        youtube_tags(clip["hashtags"]),
        "```",
        "",
        "**Instagram — Caption** (başlık yok, ilk satır başlığın yerine geçiyor)",
        "```",
        caption,
        "```",
        "",
        "- [ ] YouTube",
        "- [ ] Instagram",
        "",
    ])


def build(clips_dir: Path, dest_dir: Path, times: list[str]) -> tuple[Path, int]:
    source = {}
    source_file = clips_dir / "source.json"
    if source_file.is_file():
        source = json.loads(source_file.read_text(encoding="utf-8")) or {}

    clips = read_clips(clips_dir)
    if not clips:
        raise SystemExit("no clips with a meta.json in %s" % clips_dir)

    # The folder is named <date>_<time>; fall back to today if it is not.
    stamp = clips_dir.name
    try:
        day = datetime.strptime(stamp[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        day = datetime.now().strftime("%Y-%m-%d")
    clock = stamp[11:13] + ":" + stamp[13:15] if len(stamp) >= 15 else ""

    out = VAULT / "posts" / (day + ".md")
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = out.read_text(encoding="utf-8") if out.is_file() else ""
    batch_no = existing.count("\n## Parti ") + 1
    head = "# %s\n\nO günün yüklenecekleri. Kod bloklarının sağ üstündeki kopyala\ndüğmesi tam olarak yapıştırılacak metni veriyor.\n" % day

    parts = [
        "",
        "## Parti %d%s" % (batch_no, (" — " + clock) if clock else ""),
        "",
        "**Kaynak :** %s" % (source.get("source") or "?"),
        "**Video :** %s%s" % (source.get("title") or "?",
                              (" — " + source["uploader"]) if source.get("uploader") else ""),
        "**Klasör :** `%s`" % dest_dir,
        "",
        "---",
        "",
    ]
    for i, clip in enumerate(clips):
        parts.append(block(clip, dest_dir, times[i] if i < len(times) else ""))

    text = (existing if existing else head) + "\n".join(parts)
    out.write_text(text, encoding="utf-8")
    return out, len(clips)


def main() -> int:
    global VAULT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips_dir", help="the batch folder under clips/")
    ap.add_argument("dest_dir", help="where 1.mp4, 2.mp4 ... were copied")
    ap.add_argument("--times", nargs="*", default=list(TIMES))
    ap.add_argument("--vault", default=str(VAULT))
    args = ap.parse_args()

    VAULT = Path(args.vault)

    out, count = build(Path(args.clips_dir).resolve(),
                       Path(args.dest_dir).resolve(), args.times)
    print("  %d video -> %s" % (count, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
