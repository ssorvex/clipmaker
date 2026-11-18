"""Shortlist the moments worth looking at.

A 30 minute video is far too much to hand a model raw. This module uses cheap
local signals to propose ~12 candidate windows; only those go to the director.
That keeps the model's job (judgement) separate from the mechanical job
(finding loud, fast, quotable, visually-punctuated passages) and keeps token
cost roughly flat regardless of source length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .probe import Probe
from .transcribe import Transcript, Utterance
from .util import clamp, zscores

# Phrases that reliably mark a moment someone would stop scrolling for.
# Weighted: a direct promise of value beats a generic intensifier.
HOOK_PATTERNS: list[tuple[str, float]] = [
    (r"\bhere'?s the (thing|trick|secret|problem)\b", 3.0),
    (r"\bnobody (tells|talks about|knows)\b", 3.0),
    (r"\b(the )?(biggest|most common|number one) mistake\b", 3.0),
    (r"\bwatch what happens\b", 2.5),
    (r"\b(this|that) is (the reason|why)\b", 2.0),
    (r"\bthe trick is\b", 2.5),
    (r"\bi was wrong\b", 2.5),
    (r"\bmistake\b", 1.5),
    (r"\b(never|always) (do|use|brush|let|feed)\b", 2.0),
    (r"\byou (will|'ll) (hurt|lose|break|regret)\b", 2.0),
    (r"\b(secret|truth|myth)\b", 1.5),
    (r"\blook at (that|this|his|her)\b", 1.8),
    (r"\bthe reveal\b", 2.0),
    (r"\bhow to\b", 1.2),
    (r"\bwhat (i|we) learned\b", 1.2),
    (r"\b(don'?t|stop) \w+ing\b", 1.5),
    (r"\bmost people\b", 1.8),
    (r"\bturns out\b", 1.5),
    (r"\bevery single time\b", 1.8),
    (r"\b(crazy|insane|unbelievable|shocking)\b", 1.5),
    (r"\?", 0.8),                       # a question invites an answer
    (r"\b\d+\b", 0.5),                  # concrete numbers
]
_COMPILED = [(re.compile(p, re.I), w) for p, w in HOOK_PATTERNS]

# Openers that mean "housekeeping", not content.
FILLER_PATTERNS = [
    r"\b(welcome back|thanks for watching|subscribe|like and subscribe)\b",
    r"\b(before we (start|begin)|quick word|a quick note)\b",
    r"\bsee you in the next one\b",
    r"\blink in (the )?(bio|description)\b",
]
_FILLER = [re.compile(p, re.I) for p in FILLER_PATTERNS]


def hook_score(text: str) -> float:
    return sum(w for rx, w in _COMPILED if rx.search(text or ""))


def filler_score(text: str) -> float:
    return sum(1.0 for rx in _FILLER if rx.search(text or ""))


@dataclass
class Candidate:
    id: int
    start: float
    end: float
    text: str
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "score": round(self.score, 3),
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "text": self.text,
        }


def _snap_edge(t: float, probe: Probe, *, duration: float, max_shift: float = 1.2) -> float:
    """Pull a cut point into the nearest speech gap so clips don't start mid-word."""
    snapped = probe.nearest_gap_center(t, max_shift=max_shift)
    if snapped is None:
        return clamp(t, 0.0, duration)
    return clamp(snapped, 0.0, duration)


def _overlap(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def from_transcript(
    transcript: Transcript,
    probe: Probe,
    *,
    target_min: float = 18.0,
    target_max: float = 58.0,
    top_k: int = 12,
    max_overlap: float = 0.35,
) -> list[Candidate]:
    """Seed on high-scoring utterances, grow each to a postable length."""
    utts = [u for u in transcript.utterances if u.duration > 0.4 and u.words]
    if not utts:
        return []

    energies = [probe.envelope.speech_mean_db(u.start, u.end) for u in utts]
    peaks = [probe.envelope.peak_db(u.start, u.end) for u in utts]
    paces = [u.wps for u in utts]
    e_z, p_z, w_z = zscores(energies), zscores(peaks), zscores(paces)

    duration = probe.duration
    scored: list[tuple[float, int, dict[str, float]]] = []
    for i, u in enumerate(utts):
        hook = hook_score(u.text)
        filler = filler_score(u.text)
        # A scene change near the line means there is something to look at.
        visual = 1.0 if any(abs(c - u.start) < 2.5 for c in probe.scene_cuts) else 0.0
        # Intros and outros are rarely the best clip.
        rel = u.start / duration if duration else 0.0
        edge_penalty = 1.0 if (rel < 0.04 or rel > 0.95) else 0.0

        total = (
            1.6 * hook
            + 1.0 * e_z[i]
            + 0.7 * p_z[i]
            + 0.6 * w_z[i]
            + 0.5 * visual
            - 2.2 * filler
            - 1.5 * edge_penalty
        )
        scored.append((total, i, {
            "hook": hook, "energy_z": e_z[i], "peak_z": p_z[i],
            "pace_z": w_z[i], "visual": visual, "filler": filler,
        }))

    scored.sort(key=lambda row: row[0], reverse=True)

    candidates: list[Candidate] = []
    next_id = 0
    for total, idx, signals in scored:
        if len(candidates) >= top_k * 3:
            break
        seed = utts[idx]
        start, end = seed.start, seed.end

        # Grow outward one utterance at a time, preferring to extend forward so
        # the hook lands early in the clip.
        lo, hi = idx, idx
        while (end - start) < target_min:
            grew = False
            if hi + 1 < len(utts) and (utts[hi + 1].end - start) <= target_max:
                hi += 1
                end = utts[hi].end
                grew = True
            if (end - start) < target_min and lo - 1 >= 0 and (end - utts[lo - 1].start) <= target_max:
                lo -= 1
                start = utts[lo].start
                grew = True
            if not grew:
                break

        # A little lead-in so the first word isn't clipped, then snap to gaps.
        start = _snap_edge(max(0.0, start - 0.35), probe, duration=duration)
        end = _snap_edge(min(duration, end + 0.45), probe, duration=duration)
        if end - start < max(6.0, target_min * 0.5):
            continue

        cand = Candidate(
            id=next_id, start=start, end=end,
            text=transcript.text_between(start, end) or seed.text,
            score=total, signals=signals,
        )
        if any(_overlap(cand, existing) > max_overlap for existing in candidates):
            continue
        candidates.append(cand)
        next_id += 1

    candidates.sort(key=lambda c: c.score, reverse=True)
    shortlist = candidates[:top_k]
    for new_id, cand in enumerate(sorted(shortlist, key=lambda c: c.start)):
        cand.id = new_id
    return sorted(shortlist, key=lambda c: c.start)


def from_audio_only(
    probe: Probe,
    *,
    target: float = 32.0,
    top_k: int = 12,
) -> list[Candidate]:
    """Fallback when there is no transcript: rank fixed windows by loudness."""
    duration = probe.duration
    if duration <= target:
        return [Candidate(0, 0.0, duration, "", 0.0, {})]

    step = target / 2.0
    rows: list[tuple[float, float, float]] = []
    t = 0.0
    while t + target <= duration:
        rows.append((probe.envelope.speech_mean_db(t, t + target),
                     probe.envelope.peak_db(t, t + target), t))
        t += step

    means = zscores([r[0] for r in rows])
    peaks = zscores([r[1] for r in rows])
    ranked = sorted(
        ((1.0 * means[i] + 0.8 * peaks[i], rows[i][2]) for i in range(len(rows))),
        key=lambda row: row[0], reverse=True,
    )

    out: list[Candidate] = []
    for score, start in ranked:
        end = min(duration, start + target)
        start = _snap_edge(start, probe, duration=duration)
        end = _snap_edge(end, probe, duration=duration)
        cand = Candidate(len(out), start, end, "", score,
                         {"energy_z": score, "hook": 0.0})
        if any(_overlap(cand, e) > 0.2 for e in out):
            continue
        out.append(cand)
        if len(out) >= top_k:
            break

    for new_id, cand in enumerate(sorted(out, key=lambda c: c.start)):
        cand.id = new_id
    return sorted(out, key=lambda c: c.start)


def build(
    transcript: Transcript,
    probe: Probe,
    *,
    target_min: float = 18.0,
    target_max: float = 58.0,
    top_k: int = 12,
) -> list[Candidate]:
    if transcript.is_empty():
        return from_audio_only(probe, target=(target_min + target_max) / 2, top_k=top_k)
    return from_transcript(transcript, probe, target_min=target_min,
                           target_max=target_max, top_k=top_k)
