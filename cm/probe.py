"""Signal analysis of the source video: loudness envelope, speech gaps, scene cuts.

Everything here is ffmpeg + numpy only, so it runs free and offline. These are
the cheap signals that narrow a 30 minute video down to a handful of candidate
windows before any model is asked to judge anything.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .util import ToolError, media_info, require, run

ANALYSIS_SR = 16000


@dataclass
class Envelope:
    """Frame-wise loudness of the source audio."""
    hop: float
    times: np.ndarray          # frame start times, seconds
    rms_db: np.ndarray         # dBFS per frame
    duration: float
    speech_floor_db: float = -60.0   # frames below this are treated as silence

    def db_at(self, t: float) -> float:
        if len(self.times) == 0:
            return -90.0
        idx = int(np.clip(t / self.hop, 0, len(self.rms_db) - 1))
        return float(self.rms_db[idx])

    def mean_db(self, start: float, end: float) -> float:
        if len(self.rms_db) == 0 or end <= start:
            return -90.0
        lo = int(np.clip(start / self.hop, 0, len(self.rms_db) - 1))
        hi = int(np.clip(end / self.hop, lo + 1, len(self.rms_db)))
        window = self.rms_db[lo:hi]
        return float(window.mean()) if len(window) else -90.0

    def peak_db(self, start: float, end: float) -> float:
        if len(self.rms_db) == 0 or end <= start:
            return -90.0
        lo = int(np.clip(start / self.hop, 0, len(self.rms_db) - 1))
        hi = int(np.clip(end / self.hop, lo + 1, len(self.rms_db)))
        window = self.rms_db[lo:hi]
        return float(window.max()) if len(window) else -90.0

    def speech_mean_db(self, start: float, end: float) -> float:
        """Average loudness across the *talking* frames only.

        A plain mean is dominated by the silence between sentences, which makes
        an energetic passage look identical to a dead one. Averaging only the
        frames above the speech floor is what actually separates them.
        """
        if len(self.rms_db) == 0 or end <= start:
            return -90.0
        lo = int(np.clip(start / self.hop, 0, len(self.rms_db) - 1))
        hi = int(np.clip(end / self.hop, lo + 1, len(self.rms_db)))
        window = self.rms_db[lo:hi]
        if not len(window):
            return -90.0
        voiced = window[window > self.speech_floor_db]
        return float(voiced.mean()) if len(voiced) else float(window.max())


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    envelope: Envelope
    gaps: list[tuple[float, float]] = field(default_factory=list)
    scene_cuts: list[float] = field(default_factory=list)

    def nearest_gap_center(self, t: float, max_shift: float = 1.0) -> float | None:
        """Centre of the closest speech gap within max_shift seconds of t."""
        best, best_dist = None, max_shift
        for start, end in self.gaps:
            centre = (start + end) / 2.0
            # A cut anywhere inside the gap is clean; prefer the closest edge.
            if start <= t <= end:
                return t
            dist = min(abs(t - start), abs(t - end), abs(t - centre))
            if dist < best_dist:
                best, best_dist = centre, dist
        return best

    def in_speech(self, t: float, pad: float = 0.0) -> bool:
        """True when t falls outside every detected gap (i.e. someone is talking)."""
        for start, end in self.gaps:
            if start - pad <= t <= end + pad:
                return False
        return True


def decode_mono(path: Path | str, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Decode the whole audio track to a mono float32 array in [-1, 1]."""
    cmd = [
        require("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error",
        "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise ToolError(f"audio decode failed: {(proc.stderr or b'').decode()[-2000:]}")
    raw = np.frombuffer(proc.stdout, dtype="<i2")
    return raw.astype(np.float32) / 32768.0


def envelope(samples: np.ndarray, sr: int = ANALYSIS_SR, hop: float = 0.05) -> Envelope:
    hop_n = max(1, int(sr * hop))
    usable = (len(samples) // hop_n) * hop_n
    if usable == 0:
        return Envelope(hop, np.zeros(0), np.zeros(0), 0.0)
    frames = samples[:usable].reshape(-1, hop_n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    rms_db = (20.0 * np.log10(np.maximum(rms, 1e-7))).astype(np.float32)
    times = (np.arange(len(rms_db), dtype=np.float32) * hop)
    # Speech floor sits a fixed distance under the track's own speech level, so
    # it adapts to quiet recordings instead of assuming a mastering level.
    loud_ref = float(np.percentile(rms_db, 95)) if len(rms_db) else -60.0
    return Envelope(hop, times, rms_db, len(samples) / sr,
                    speech_floor_db=max(loud_ref - 25.0, -70.0))


def find_gaps(
    env: Envelope,
    *,
    rel_threshold_db: float = 18.0,
    floor_db: float = -45.0,
    min_duration: float = 0.18,
) -> list[tuple[float, float]]:
    """Locate quiet stretches — the only safe places to cut or drop a sound effect.

    The threshold is relative to the track's own speech level (95th percentile),
    so it adapts to quiet recordings instead of using a fixed dBFS number.
    """
    if len(env.rms_db) == 0:
        return []
    loud_ref = float(np.percentile(env.rms_db, 95))
    threshold = max(loud_ref - rel_threshold_db, floor_db)
    quiet = env.rms_db < threshold

    gaps: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start_idx is None:
            start_idx = i
        elif not is_quiet and start_idx is not None:
            dur = (i - start_idx) * env.hop
            if dur >= min_duration:
                gaps.append((start_idx * env.hop, i * env.hop))
            start_idx = None
    if start_idx is not None:
        dur = (len(quiet) - start_idx) * env.hop
        if dur >= min_duration:
            gaps.append((start_idx * env.hop, len(quiet) * env.hop))
    return gaps


_PTS_RE = re.compile(r"pts_time:([0-9.]+)")
_SCD_RE = re.compile(r"lavfi\.scd\.time=([0-9.]+)")


def scene_cuts(path: Path | str, *, threshold: float = 8.0, fps: int = 4) -> list[float]:
    """Timestamps where the picture changes a lot — natural places to cut.

    Decoded at low fps and small size so a 30 minute video analyses in seconds
    rather than minutes. `threshold` is on scdet's 0-100 scale (lower = more
    cuts detected).

    Note: `select='gt(scene,N)'` does NOT export a metadata key, so
    `metadata=print` after it prints nothing. scdet is the filter that actually
    publishes a score; showinfo is the fallback for builds without it.
    """
    vf = f"fps={fps},scale=320:-2,scdet=threshold={threshold},metadata=print:file=-"
    proc = run([
        require("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error",
        "-i", str(path), "-an", "-sn", "-vf", vf, "-f", "null", "-",
    ], check=False)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    cuts = sorted({round(float(m), 3) for m in _SCD_RE.findall(out)})
    if cuts:
        return cuts

    # Fallback: select + showinfo, parsing pts_time out of the log.
    sel = max(0.05, min(0.95, threshold / 40.0))
    vf2 = f"fps={fps},scale=320:-2,select='gt(scene,{sel})',showinfo"
    proc2 = run([
        require("ffmpeg"), "-hide_banner", "-nostdin", "-v", "info",
        "-i", str(path), "-an", "-sn", "-vf", vf2, "-f", "null", "-",
    ], check=False)
    out2 = (proc2.stdout or "") + "\n" + (proc2.stderr or "")
    return sorted({round(float(m), 3) for m in _PTS_RE.findall(out2)})


def probe(path: Path | str, *, scene_threshold: float = 8.0) -> Probe:
    info = media_info(path)
    if info["duration"] <= 0:
        raise ToolError(f"could not read a duration from {path}; is it a video file?")

    if info["has_audio"]:
        samples = decode_mono(path)
        env = envelope(samples)
        gaps = find_gaps(env)
    else:
        env = Envelope(0.05, np.zeros(0), np.zeros(0), info["duration"])
        gaps = []

    cuts = scene_cuts(path, threshold=scene_threshold) if info["has_video"] else []

    return Probe(
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        has_audio=info["has_audio"],
        envelope=env,
        gaps=gaps,
        scene_cuts=cuts,
    )


def extract_keyframes(path: Path | str, times: list[float], out_dir: Path, *, width: int = 512) -> list[Path]:
    """Grab one JPEG per timestamp, for the model's visual pass."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, t in enumerate(times):
        dest = out_dir / f"kf_{i:03d}_{t:.2f}.jpg"
        if not dest.exists():
            run([
                require("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-v", "error",
                "-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1",
                "-vf", f"scale={width}:-2", "-q:v", "4", str(dest),
            ], check=False)
        if dest.exists():
            written.append(dest)
    return written
