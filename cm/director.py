"""The judgement pass: which moments become clips, and what they sound like.

The local signals in `candidates.py` find passages that are loud, fast and
quotable. They cannot tell whether a passage is *interesting*. That is what
this module asks Claude, in a single structured-output call: pick the best
windows from the shortlist, trim them, title them, and write the sound-effect
cue sheet.

Running without an API key is a first-class path, not a degraded one — see
`heuristic_plan`. It produces the same shape so everything downstream is
identical; you just get mechanical titles and the structural SFX bed.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import Candidate
from .cueplan import Cue, default_cues, default_gain, structural_context
from .sfx import SFX_KINDS, kind_catalogue
from .transcribe import Transcript
from .util import ToolError, clamp

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class ClipPlan:
    start: float
    end: float
    title: str
    hook: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    why: str = ""
    emphasis: list[str] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    source_candidate: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "title": self.title,
            "hook": self.hook,
            "description": self.description,
            "hashtags": self.hashtags,
            "why": self.why,
            "emphasis": self.emphasis,
            "source_candidate": self.source_candidate,
            "cues": [c.to_dict() for c in self.cues],
        }


# --- schema ---------------------------------------------------------------

def _schema(n_clips: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "minItems": 1,
                "maxItems": n_clips,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "integer"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "title": {"type": "string"},
                        "hook": {"type": "string"},
                        "description": {"type": "string"},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                        "why": {"type": "string"},
                        "emphasis_words": {"type": "array", "items": {"type": "string"}},
                        "sfx_cues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "t": {"type": "number"},
                                    "kind": {"type": "string", "enum": SFX_KINDS},
                                    "reason": {"type": "string"},
                                    "intensity": {
                                        "type": "string",
                                        "enum": ["subtle", "normal", "strong"],
                                    },
                                    "prompt": {"type": "string"},
                                },
                                "required": ["t", "kind", "reason", "intensity", "prompt"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "candidate_id", "start", "end", "title", "hook",
                        "description", "hashtags", "why", "emphasis_words", "sfx_cues",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["clips"],
        "additionalProperties": False,
    }


INTENSITY_TRIM = {"subtle": -5.0, "normal": 0.0, "strong": +3.0}

SYSTEM = """\
You are a short-form video editor. You are given candidate passages from one \
long video, already pre-filtered by audio energy, pace and scene changes. Your \
job is to choose the passages that would actually stop someone scrolling, trim \
them to a postable length, and design their sound design.

How to choose:
- A clip must make sense cold, with no setup. If a passage only works because \
of something said earlier, either extend it to include that or skip it.
- Prefer a complete thought: a claim and its payoff. Ending mid-sentence is a \
failure.
- One idea per clip. Do not pick two passages that make the same point.
- Skip housekeeping: intros, sign-offs, subscribe requests, sponsor reads.
- Trim `start` to land just before the first word of the hook, and `end` just \
after the last word of the payoff. You may move both edges within the \
candidate's bounds, and may extend up to 3 seconds beyond them if a thought \
needs it.

Titles and copy:
- `title`: a YouTube Shorts title, under 70 characters, no clickbait the clip \
does not pay off, no emoji unless it earns its place.
- `hook`: the exact spoken words that open the clip, so the caption can match.
- `description`: two sentences max, plus nothing the clip does not contain.
- `hashtags`: 4-8, specific before broad.
- `emphasis_words`: a handful of words from the clip that captions should \
highlight. Use the exact spoken word, as transcribed.
- `why`: one line on why this passage earns a clip. Be concrete and honest; if \
it is the weakest of your picks, say so.

Sound design — the part that makes or breaks the clip:
- Cue times (`t`) are seconds from the start of YOUR trimmed clip, not the \
source video.
- Be sparing. A clip under 20 seconds rarely wants more than 3 cues. Silence \
has more impact than another whoosh.
- Place cues on *structure*: the hook landing, a turn in the argument, a \
reveal, a number, a cut. Never on a random word.
- Do not place cues on top of speech — aim for the breath between sentences. \
Cue times are snapped to the nearest speech gap afterwards, so get within \
about half a second and the placement will be fixed for you.
- `intensity`: `subtle` under speech, `normal` in a gap, `strong` only for a \
genuine reveal. Most cues are subtle or normal.
- `prompt`: a short text description of the sound, used only if the run is \
configured to generate audio from an API. Describe the sound, not the moment \
("low soft wooden thud, dry, no reverb").
- A build with no payoff is worse than no build. If you use `riser`, land it.

Return only the structured object.\
"""


def _prompt(
    candidates: list[Candidate],
    *,
    n_clips: int,
    source_duration: float,
    min_len: float,
    max_len: float,
    extra: str = "",
) -> str:
    lines = [
        f"Source video: {source_duration:.0f} seconds long.",
        f"Pick the best {n_clips} clips. Each must be between {min_len:.0f} and "
        f"{max_len:.0f} seconds after your trimming.",
        "",
        "Available sound effects (use these names exactly):",
        kind_catalogue(),
        "",
        f"{len(candidates)} candidate passages, with the local signals that "
        "flagged them (hook = phrasing score, energy_z / peak_z = loudness "
        "versus the rest of the video, pace_z = words per second, visual = a "
        "scene change nearby):",
        "",
    ]
    for cand in candidates:
        sig = cand.signals
        lines.append(
            f"[candidate {cand.id}] {cand.start:.2f}s - {cand.end:.2f}s "
            f"({cand.duration:.1f}s) | hook={sig.get('hook', 0):.1f} "
            f"energy_z={sig.get('energy_z', 0):+.2f} peak_z={sig.get('peak_z', 0):+.2f} "
            f"pace_z={sig.get('pace_z', 0):+.2f} visual={sig.get('visual', 0):.0f}"
        )
        lines.append(f"  transcript: {cand.text or '(no speech detected)'}")
        lines.append("")
    if extra:
        lines += ["Additional direction from the channel owner:", extra, ""]
    return "\n".join(lines)


def _image_blocks(frames: list[Path]) -> list[dict]:
    blocks = []
    for path in frames:
        try:
            data = base64.standard_b64encode(Path(path).read_bytes()).decode()
        except OSError:
            continue
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        })
    return blocks


def direct(
    candidates: list[Candidate],
    *,
    n_clips: int,
    source_duration: float,
    min_len: float,
    max_len: float,
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    frames: list[Path] | None = None,
    extra_direction: str = "",
    max_tokens: int = 16000,
) -> tuple[list[ClipPlan], dict[str, Any]]:
    """Ask Claude to pick and design the clips. Returns (plans, usage info)."""
    try:
        import anthropic
    except ImportError as exc:
        raise ToolError(
            "the anthropic SDK is not installed:  pip install anthropic\n"
            "Or re-run with --no-llm to use the heuristic planner."
        ) from exc

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise ToolError(
            "no Anthropic credentials found. Set ANTHROPIC_API_KEY, or run "
            "`ant auth login`, or re-run with --no-llm."
        )

    client = anthropic.Anthropic()
    content: list[dict] = [{
        "type": "text",
        "text": _prompt(candidates, n_clips=n_clips, source_duration=source_duration,
                        min_len=min_len, max_len=max_len, extra=extra_direction),
    }]
    if frames:
        content.append({"type": "text",
                        "text": "Keyframes sampled from the candidates, in order:"})
        content.extend(_image_blocks(frames))

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": _schema(n_clips)},
        },
    }

    # Opt into server-side fallback so a safety decline doesn't kill the run.
    try:
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
    except Exception as exc:                      # noqa: BLE001
        # Older gateways / self-hosted proxies may reject the beta; the call
        # itself is fine without it.
        if "fallback" not in str(exc).lower():
            raise
        response = client.messages.create(**kwargs)

    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise ToolError(
            "the model declined to plan this video"
            + (f" ({getattr(details, 'category', '?')})" if details else "")
            + ". Re-run with --no-llm to use the heuristic planner."
        )

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise ToolError("the model returned no text block to parse")
    data = json.loads(text)

    plans = _plans_from_payload(data, candidates, min_len=min_len, max_len=max_len,
                               source_duration=source_duration)
    usage = {
        "model": getattr(response, "model", model),
        "input_tokens": getattr(response.usage, "input_tokens", None),
        "output_tokens": getattr(response.usage, "output_tokens", None),
    }
    return plans, usage


def _plans_from_payload(
    data: dict,
    candidates: list[Candidate],
    *,
    min_len: float,
    max_len: float,
    source_duration: float,
) -> list[ClipPlan]:
    by_id = {c.id: c for c in candidates}
    plans: list[ClipPlan] = []

    for raw in data.get("clips", []):
        cand = by_id.get(raw.get("candidate_id"))
        start = float(raw.get("start", cand.start if cand else 0.0))
        end = float(raw.get("end", cand.end if cand else min_len))

        # Trust but verify: keep the model's trim inside the source and inside
        # the configured length bounds.
        start = clamp(start, 0.0, max(0.0, source_duration - 2.0))
        end = clamp(end, start + 2.0, source_duration)
        if end - start < min_len * 0.6:
            end = clamp(start + min_len, start + 2.0, source_duration)
        if end - start > max_len:
            end = start + max_len

        cues: list[Cue] = []
        for rc in raw.get("sfx_cues", []):
            kind = rc.get("kind")
            if kind not in SFX_KINDS:
                continue
            trim = INTENSITY_TRIM.get(rc.get("intensity", "normal"), 0.0)
            cues.append(Cue(
                t=float(rc.get("t", 0.0)),
                kind=kind,
                gain_db=default_gain(kind) + trim,
                reason=rc.get("reason", ""),
                prompt=rc.get("prompt", ""),
                priority=1.5 if rc.get("intensity") == "strong" else 1.0,
            ))

        plans.append(ClipPlan(
            start=start, end=end,
            title=(raw.get("title") or "Untitled clip").strip(),
            hook=(raw.get("hook") or "").strip(),
            description=(raw.get("description") or "").strip(),
            hashtags=[h.lstrip("#") for h in raw.get("hashtags", []) if h],
            why=(raw.get("why") or "").strip(),
            emphasis=[w for w in raw.get("emphasis_words", []) if w],
            cues=cues,
            source_candidate=raw.get("candidate_id"),
        ))

    return plans


# --- no-API path ----------------------------------------------------------

def heuristic_plan(
    candidates: list[Candidate],
    transcript: Transcript,
    *,
    n_clips: int,
    min_len: float,
    max_len: float,
) -> list[ClipPlan]:
    """Pick the top-scoring candidates and apply the structural SFX bed.

    Deliberately conservative: it does not try to write good titles, because a
    bad title is worse than an obviously placeholder one.
    """
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:n_clips]
    plans: list[ClipPlan] = []
    for cand in sorted(ranked, key=lambda c: c.start):
        end = cand.start + min(max(cand.duration, min_len), max_len)
        text = cand.text or ""
        first_sentence = text.split(".")[0].strip()
        title = (first_sentence[:67] + "...") if len(first_sentence) > 70 else first_sentence
        first_word, breaks = structural_context(transcript, cand.start, end)
        plans.append(ClipPlan(
            start=cand.start, end=end,
            title=title or f"Clip at {cand.start:.0f}s",
            hook=" ".join(text.split()[:12]),
            description="",
            hashtags=[],
            why=f"heuristic pick (score {cand.score:.2f}, "
                f"hook={cand.signals.get('hook', 0):.1f})",
            emphasis=[],
            cues=default_cues(end - cand.start, first_word_at=first_word,
                              sentence_breaks=breaks),
            source_candidate=cand.id,
        ))
    return plans
