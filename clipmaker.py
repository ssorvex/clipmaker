#!/usr/bin/env python3
"""clipmaker — turn one long video into a handful of posting-ready short clips,
with sound design.

    python clipmaker.py make podcast.mp4 --clips 5 --out ./out

Pipeline:
    probe        loudness envelope, speech gaps, scene cuts        (ffmpeg)
    transcribe   word-level timestamps                             (faster-whisper, local)
    shortlist    score passages, propose ~12 candidates            (local)
    direct       pick 5, trim, title, design the SFX cue sheet     (Claude, or --no-llm)
    place        snap cues to speech gaps, thin, cap density       (local)
    render       reframe, caption, mix, normalise to -14 LUFS      (ffmpeg)

Run `python clipmaker.py sfx-preview` to hear the built-in effects.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cm import candidates as cand_mod
from cm import captions as cap_mod
from cm import fetch as fetch_mod
from cm import cueplan, director, probe as probe_mod, render as render_mod, sfx as sfx_mod
from cm import transcribe as asr_mod
from cm.util import ToolError, fmt_timecode, require, slugify, write_json


def log(msg: str = "") -> None:
    print(msg, flush=True)


def step(n: int, total: int, msg: str) -> None:
    log(f"[{n}/{total}] {msg}")


# --- helpers --------------------------------------------------------------

def _keyframe_times(cands: list[cand_mod.Candidate], per_clip: int = 2) -> list[float]:
    times: list[float] = []
    for c in cands:
        span = c.duration
        for i in range(per_clip):
            frac = (i + 1) / (per_clip + 1)
            times.append(c.start + span * frac)
    return times


def _write_youtube_sheet(dest: Path, results: list[dict], source: str) -> None:
    lines = [
        f"# Clips from `{Path(source).name}`",
        "",
        "Paste-ready. Check every title against the clip before posting.",
        "",
    ]
    for i, r in enumerate(results, 1):
        plan = r["plan"]
        lines += [
            f"## {i}. {plan['title']}",
            "",
            f"- **File**: `{r['file']}`",
            f"- **Source timecode**: {fmt_timecode(plan['start'])} – {fmt_timecode(plan['end'])}"
            f"  ({plan['duration']:.1f}s)",
            f"- **Why this moment**: {plan['why'] or '—'}",
            "",
            "**Description**",
            "",
            (plan["description"] or "_(write one)_"),
            "",
            "**Tags**: " + (" ".join(f"#{h}" for h in plan["hashtags"]) or "_(none)_"),
            "",
            "**Sound design**",
            "",
        ]
        if plan["cues"]:
            lines.append("| t | effect | gain | why |")
            lines.append("|---|---|---|---|")
            for c in plan["cues"]:
                moved = f" (moved from {c['moved_from']}s)" if "moved_from" in c else ""
                lines.append(
                    f"| {c['t']:.2f}s | `{c['kind']}` | {c['gain_db']:.1f} dB | "
                    f"{c['reason']}{moved} |"
                )
        else:
            lines.append("_no cues placed_")
        lines += ["", f"**Measured loudness**: {r.get('loudness', {})}", "", "---", ""]
    dest.write_text("\n".join(lines))


# --- the main command -----------------------------------------------------

@dataclass
class Prepared:
    """Everything mechanical about a source, before any judgement is applied."""
    source: Path
    probe: Any
    transcript: Any
    candidates: list
    fetched: Any
    work_dir: Path


def prepare(args: argparse.Namespace, work_dir: Path, *, total: int) -> Prepared:
    """Fetch, probe, transcribe, shortlist — the half with no taste in it.

    Split out because the judgement that follows can come from a model, from a
    file, or from a person, and because watching a channel wants this half on
    its own: run it the moment a video appears, leave the picking for later.
    """
    # 0. fetch, when the source is a link rather than a file on disk
    fetched = None
    if fetch_mod.is_url(args.source):
        download_dir = Path(args.download_dir).expanduser().resolve() \
            if args.download_dir else Path(__file__).parent / ".downloads"
        log(f"[fetch] {args.source}")
        fetched = fetch_mod.fetch(
            args.source, download_dir, max_height=args.max_height,
            want_subs=not args.no_yt_subs, language=args.language,
            cookies_from_browser=args.cookies_from_browser,
        )
        source = fetched.video
        log(f"      {fetched.title}")
        log(f"      {fetched.duration / 60:.1f} min"
            + (f", {fetched.uploader}" if fetched.uploader else "")
            + ("  (already on disk)" if fetched.cached else ""))
        if fetched.subtitles:
            log(f"      captions available: {fetched.sub_language}")
        else:
            log("      no captions published for this one — will transcribe locally")
    else:
        source = Path(args.source).expanduser().resolve()
        if not source.is_file():
            raise ToolError(f"no such file: {source}")

    # 1. probe
    step(1, total, "analysing audio and picture ...")
    pr = probe_mod.probe(source, scene_threshold=args.scene_threshold)
    log(f"      {pr.duration:.1f}s, {pr.width}x{pr.height} @ {pr.fps:.2f}fps, "
        f"audio={'yes' if pr.has_audio else 'NO'}")
    log(f"      {len(pr.gaps)} speech gaps, {len(pr.scene_cuts)} scene cuts")
    if not pr.has_audio:
        log("      ! no audio track: captions and speech-aware cue placement are off")

    # 2. transcribe
    step(2, total, "transcribing ...")
    transcript = asr_mod.empty_transcript()
    yt_subs = fetched.subtitles if (fetched is not None and not args.no_yt_subs) else None
    if yt_subs is not None:
        try:
            transcript = asr_mod.from_json3(yt_subs, language=fetched.sub_language)
            log(f"      {len(transcript.utterances)} utterances, "
                f"{len(transcript.words)} words from the site's own captions "
                f"(lang={transcript.language}) — no Whisper needed")
        except ToolError as exc:
            log(f"      ! {exc}")
            log("      falling back to local transcription")
            yt_subs = None
    if yt_subs is None and (args.no_asr or not pr.has_audio):
        log("      skipped (--no-asr)" if args.no_asr else "      skipped (no audio)")
    elif yt_subs is None:
        try:
            t0 = time.time()
            transcript = asr_mod.transcribe(
                source, work_dir, model_size=args.whisper_model,
                language=args.language, device=args.whisper_device,
                compute_type=args.whisper_compute,
            )
            log(f"      {len(transcript.utterances)} utterances, "
                f"{len(transcript.words)} words, lang={transcript.language} "
                f"({time.time() - t0:.0f}s)")
        except ToolError as exc:
            log(f"      ! {exc}")
            log("      continuing without a transcript")

    # 3. shortlist
    step(3, total, "shortlisting candidate moments ...")
    cands = cand_mod.build(
        transcript, pr,
        target_min=args.min_len, target_max=args.max_len, top_k=args.shortlist,
    )
    if not cands:
        raise ToolError("found no usable candidate moments; try --no-asr or a longer source")
    log(f"      {len(cands)} candidates")
    for c in cands:
        log(f"        [{c.id:2d}] {fmt_timecode(c.start)}–{fmt_timecode(c.end)} "
            f"score={c.score:+.2f} {c.text[:66]}")
    write_json(work_dir / "candidates.json", [c.to_dict() for c in cands])

    return Prepared(source=source, probe=pr, transcript=transcript,
                    candidates=cands, fetched=fetched, work_dir=work_dir)


def _write_dossier(prep: Prepared, out_dir: Path, source: str) -> int:
    """Write what is needed to pick clips, and stop before rendering any.

    One file to read rather than three queries: every utterance with the
    second it starts at. That is precise enough to choose a cut point without
    going back to the word timings, which is the slow part of picking.
    """
    lines = [f"{u.start:8.2f}  {u.text}" for u in prep.transcript.utterances]
    (out_dir / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    f = prep.fetched
    write_json(out_dir / "source.json", {
        "source": source,
        "title": f.title if f else Path(source).name,
        "uploader": f.uploader if f else "",
        "duration": round(prep.probe.duration, 2),
        "language": prep.transcript.language,
        "utterances": len(prep.transcript.utterances),
        "words": len(prep.transcript.words),
    })
    write_json(out_dir / "candidates.json",
               [c.to_dict() for c in prep.candidates])

    log("")
    log(f"dossier ready in {out_dir}")
    log(f"  transcript.txt   {len(prep.transcript.utterances)} lines, "
        f"{len(prep.transcript.words)} words")
    log(f"  candidates.json  {len(prep.candidates)} shortlisted moments")
    log("")
    log("write a plan, then cut it:")
    log(f'  python clipmaker.py make "{source}" --plan <plan.json> --out <dir>')
    return 0


def cmd_make(args: argparse.Namespace) -> int:
    require("ffmpeg")
    require("ffprobe")

    out_dir = Path(args.out).expanduser().resolve()
    work_dir = out_dir / ".work"
    cache_dir = Path(args.sfx_cache).expanduser().resolve() if args.sfx_cache \
        else work_dir / "sfx_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    total = 6
    t_start = time.time()

    prep = prepare(args, work_dir, total=total)
    source, pr = prep.source, prep.probe
    transcript, cands, fetched = prep.transcript, prep.candidates, prep.fetched

    if args.prep_only:
        return _write_dossier(prep, out_dir, args.source)

    # 4. direct
    step(4, total, "choosing and designing clips ...")
    usage: dict = {}
    if args.plan:
        plans = director.plan_from_file(
            args.plan, cands, transcript,
            min_len=args.min_len, max_len=args.max_len, source_duration=pr.duration,
        )
        log(f"      {len(plans)} clips read from {Path(args.plan).name} (no model call)")
    elif args.no_llm:
        plans = director.heuristic_plan(
            cands, transcript, n_clips=args.clips,
            min_len=args.min_len, max_len=args.max_len,
        )
        log(f"      heuristic planner: {len(plans)} clips (no model call)")
    else:
        frames: list[Path] = []
        if args.vision:
            frames = probe_mod.extract_keyframes(
                source, _keyframe_times(cands), work_dir / "keyframes"
            )
            log(f"      sampled {len(frames)} keyframes for the visual pass")
        try:
            plans, usage = director.direct(
                cands, n_clips=args.clips, source_duration=pr.duration,
                min_len=args.min_len, max_len=args.max_len,
                model=args.model, effort=args.effort, frames=frames,
                extra_direction=args.direction or "",
            )
            log(f"      {usage.get('model')} chose {len(plans)} clips "
                f"(in={usage.get('input_tokens')} out={usage.get('output_tokens')} tokens)")
        except ToolError as exc:
            log(f"      ! {exc}")
            log("      falling back to the heuristic planner")
            plans = director.heuristic_plan(
                cands, transcript, n_clips=args.clips,
                min_len=args.min_len, max_len=args.max_len,
            )

    if not plans:
        raise ToolError("the planner returned no clips")

    # 5. place cues
    step(5, total, "placing sound effects ...")
    sfx_requests: list[dict] = []
    resolver = sfx_mod.SfxResolver(
        cache_dir,
        library_dir=Path(args.sfx_library).expanduser() if args.sfx_library else None,
        provider=args.sfx_provider,
        request_log=sfx_requests,
    )

    placements: list[cueplan.PlacementReport] = []
    for i, plan in enumerate(plans, 1):
        cues = plan.cues
        if args.sfx_mode == "off":
            cues = []
        elif args.sfx_mode == "structural" or not cues:
            near_cut = any(abs(c - plan.end) < 2.0 for c in pr.scene_cuts)
            first_word, breaks = cueplan.structural_context(
                transcript, plan.start, plan.end)
            cues = cueplan.default_cues(
                plan.duration, has_scene_cut_at_end=near_cut,
                first_word_at=first_word, sentence_breaks=breaks,
            )
        report = cueplan.place(
            cues, clip_start=plan.start, clip_end=plan.end,
            probe=pr, transcript=transcript,
            min_spacing=args.sfx_spacing, per_seconds=args.sfx_density,
            hard_max=args.sfx_max,
        )
        plan.cues = report.kept
        placements.append(report)
        log(f"      clip {i}: {report.summary()}")

    # 6. render
    step(6, total, "rendering ...")
    spec = render_mod.RenderSpec(
        width=args.width, height=args.height, fps=args.fps,
        crop_mode=args.crop, crf=args.crf, preset=args.x264_preset,
    )
    music = Path(args.music).expanduser().resolve() if args.music else None
    results: list[dict] = []

    for i, plan in enumerate(plans, 1):
        name = f"{i:02d}-{slugify(plan.title) or 'clip'}"
        clip_dir = out_dir / name
        clip_dir.mkdir(parents=True, exist_ok=True)

        cues_with_files: list[tuple[cueplan.Cue, Path]] = []
        for cue in plan.cues:
            try:
                path = resolver.resolve(cue.kind, variant=cue.variant, prompt=cue.prompt)
                cues_with_files.append((cue, path))
            except ToolError as exc:
                log(f"      ! cue {cue.kind} skipped: {exc}")

        ass_path = None
        if not args.no_captions and cap_mod.has_any(transcript, plan.start, plan.end):
            ass_path = clip_dir / "captions.ass"
            ass_path.write_text(cap_mod.build_ass(
                transcript, clip_start=plan.start, clip_end=plan.end,
                width=spec.width, height=spec.height, style=args.caption_style,
                emphasis=plan.emphasis, uppercase=not args.caption_mixed_case,
            ))

        dest = clip_dir / "clip.mp4"
        render_mod.render_clip(
            source, dest, start=plan.start, end=plan.end,
            cues_with_files=cues_with_files, spec=spec,
            ass_path=ass_path, music=music, has_audio=pr.has_audio,
        )
        loud = render_mod.measure_loudness(dest)
        info = render_mod.probe_output(dest)
        record = {
            "file": str(dest.relative_to(out_dir)),
            "plan": plan.to_dict(),
            "loudness": loud,
            "output": {"duration": round(info["duration"], 2),
                       "width": info["width"], "height": info["height"]},
        }
        write_json(clip_dir / "meta.json", record)
        results.append(record)
        log(f"      {i}. {plan.title[:58]!r}")
        log(f"         {dest.relative_to(out_dir)}  {info['duration']:.1f}s  "
            f"{info['width']}x{info['height']}  "
            f"{loud.get('lufs', '?')} LUFS  {len(plan.cues)} cues")

    write_json(out_dir / "plan.json", {
        "source": str(source),
        "source_duration": pr.duration,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "clips": args.clips, "min_len": args.min_len, "max_len": args.max_len,
            "sfx_mode": args.sfx_mode, "sfx_provider": args.sfx_provider,
            "model": None if args.no_llm else args.model,
            "crop": args.crop, "size": f"{args.width}x{args.height}",
        },
        "model_usage": usage,
        "clips_out": results,
    })
    _write_youtube_sheet(out_dir / "youtube.md", results, str(source))

    if sfx_requests:
        write_json(out_dir / "sfx_requests.json", sfx_requests)
        log("")
        log(f"  {len(sfx_requests)} sound(s) were requested from a generative "
            "provider that this script cannot call directly.")
        log(f"  They are listed in {out_dir / 'sfx_requests.json'} — generate them, "
            "drop them in the library, and re-run.")

    log("")
    log(f"done in {time.time() - t_start:.0f}s — {len(results)} clips in {out_dir}")
    log(f"  sound sources: {resolver.summary()}")
    log(f"  paste-ready titles and tags: {out_dir / 'youtube.md'}")
    if not args.keep_work:
        shutil.rmtree(work_dir / "keyframes", ignore_errors=True)
    return 0


def cmd_sfx_preview(args: argparse.Namespace) -> int:
    """Render every built-in effect, plus one strip with all of them in a row."""
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"rendering {len(sfx_mod.SFX_KINDS)} effects to {out_dir}")
    paths: list[Path] = []
    for kind in sfx_mod.SFX_KINDS:
        dest = out_dir / f"{kind}.wav"
        sfx_mod.synth(kind, dest)
        r = sfx_mod.RECIPES[kind]
        log(f"  {kind:16s} {r.duration:5.2f}s  {r.role:11s} {r.description}")
        paths.append(dest)

    # One file with all effects spaced a second apart, for a quick listen.
    args_list: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    cursor = 0.0
    for i, p in enumerate(paths):
        args_list += ["-i", str(p)]
        delay = int(cursor * 1000)
        chains.append(f"[{i}:a]adelay={delay}|{delay}[d{i}]")
        labels.append(f"[d{i}]")
        cursor += sfx_mod.RECIPES[sfx_mod.SFX_KINDS[i]].duration + 0.55
    graph = (";".join(chains)
             + f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0[out]")
    strip = out_dir / "_all_effects.wav"
    from cm.util import ffmpeg
    ffmpeg([*args_list, "-filter_complex", graph, "-map", "[out]",
            "-c:a", "pcm_s16le", str(strip)])
    log(f"\nall of them in one file: {strip}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clipmaker",
        description="Cut a long video into short clips with sound design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("make", help="make clips from a video",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    m.add_argument("source", help="the long video: a file path, or a URL to fetch")
    m.add_argument("--out", "-o", default="./clips", help="output directory")
    m.add_argument("--clips", "-n", type=int, default=5, help="how many clips to make")
    m.add_argument("--min-len", type=float, default=18.0, help="minimum clip seconds")
    m.add_argument("--max-len", type=float, default=58.0, help="maximum clip seconds")
    m.add_argument("--shortlist", type=int, default=12,
                   help="candidates to shortlist before the model picks")

    g = m.add_argument_group("fetching a URL")
    g.add_argument("--max-height", type=int, default=1080,
                   help="cap the resolution of a downloaded source")
    g.add_argument("--no-yt-subs", action="store_true",
                   help="ignore the site's own captions and transcribe locally")
    g.add_argument("--cookies-from-browser", default="",
                   help="chrome|edge|firefox — borrow a signed-in session for "
                        "age-gated or sign-in-walled videos")
    g.add_argument("--download-dir", default="",
                   help="where to keep fetched sources "
                        "(default: .downloads beside this script)")

    g = m.add_argument_group("model")
    g.add_argument("--model", default=director.DEFAULT_MODEL, help="Claude model id")
    g.add_argument("--effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--prep-only", action="store_true",
                   help="stop after shortlisting and write a dossier to pick "
                        "from; renders nothing")
    g.add_argument("--plan", default="",
                   help="cut the clips named in a JSON plan instead of choosing "
                        "them: the editorial pass done elsewhere, no model call")
    g.add_argument("--no-llm", action="store_true",
                   help="skip the model entirely; use the heuristic planner")
    g.add_argument("--vision", action="store_true", default=True,
                   help="send keyframes so the model can see the candidates")
    g.add_argument("--no-vision", dest="vision", action="store_false")
    g.add_argument("--direction", default="",
                   help="extra direction for the editor, e.g. 'favour grooming "
                        "transformations, skip anything about pricing'")

    g = m.add_argument_group("transcription")
    g.add_argument("--whisper-model", default="small",
                   help="tiny|base|small|medium|large-v3 (bigger = better, slower)")
    g.add_argument("--whisper-device", default="auto",
                   help="auto|cpu|cuda — auto uses the GPU when there is one")
    g.add_argument("--whisper-compute", default="auto",
                   help="auto|int8|float16|float32 — auto picks per device")
    g.add_argument("--language", default=None, help="force a language code, e.g. en")
    g.add_argument("--no-asr", action="store_true",
                   help="skip transcription (no captions, weaker clip choices)")

    g = m.add_argument_group("sound effects")
    g.add_argument("--sfx-mode", default="full",
                   choices=["full", "structural", "off"],
                   help="full = model-directed cues; structural = build+hit only")
    g.add_argument("--sfx-provider", default="synth",
                   choices=["synth", "library", "elevenlabs", "higgsfield"],
                   help="where sounds come from; synth needs no network or money")
    g.add_argument("--sfx-library", default=str(Path(__file__).parent / "sfx_library"),
                   help="directory of your own sounds, one subfolder per effect name")
    g.add_argument("--sfx-cache", default="", help="where to cache rendered sounds")
    g.add_argument("--sfx-spacing", type=float, default=1.1,
                   help="minimum seconds between two cues")
    g.add_argument("--sfx-density", type=float, default=4.0,
                   help="roughly one cue per this many seconds")
    g.add_argument("--sfx-max", type=int, default=10, help="hard cap on cues per clip")
    g.add_argument("--music", default="", help="optional music bed, ducked under speech")

    g = m.add_argument_group("output")
    g.add_argument("--width", type=int, default=1080)
    g.add_argument("--height", type=int, default=1920)
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--crop", default="blur", choices=["blur", "fill", "pad"],
                   help="how to fit a wide source into a tall frame")
    g.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    g.add_argument("--x264-preset", default="medium")
    g.add_argument("--caption-style", default="bold-white",
                   choices=sorted(cap_mod.STYLE_PRESETS))
    g.add_argument("--caption-mixed-case", action="store_true",
                   help="keep captions as transcribed instead of UPPERCASE")
    g.add_argument("--no-captions", action="store_true")
    g.add_argument("--scene-threshold", type=float, default=8.0,
                   help="scdet sensitivity, 0-100; lower finds more cuts")
    g.add_argument("--keep-work", action="store_true", help="keep intermediate files")
    m.set_defaults(func=cmd_make)

    s = sub.add_parser("sfx-preview", help="render the built-in sound effects")
    s.add_argument("--out", "-o", default="./sfx_preview")
    s.set_defaults(func=cmd_sfx_preview)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ToolError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
