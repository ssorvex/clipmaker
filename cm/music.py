"""Compose an instrumental bed for one clip, to that clip's exact length.

Every other route to music is a licence problem: a track from a streaming
service earns a Content ID claim on YouTube and is muted outright on
Instagram, and the free libraries still mean choosing files by hand. A bed
written here is original by construction and never has to be cleared.

Three ideas carry the sound.

**It is composed, not looped.** A loop under a 51-second clip plays the same
eight bars twice and goes nowhere, which is exactly what a bed should not do.
This writes to the clip's own length and gives it an arc: a sparse opening
behind the hook, the filter opening through the middle, a lift over the last
quarter where the payoff lands.

**The filter moves.** Brightness is the whole expressive range of this genre,
so every harmonic carries its own envelope and the cutoff rises and falls
across the piece. Additive synthesis makes that exact rather than approximate:
there is no filter, only harmonics that are or are not present, so nothing
aliases and the sweep is whatever curve we ask for.

**It leaves room for the voice.** Speech lives between roughly 300 Hz and
4 kHz. Rather than a hard ceiling, which sounds like music in the next room,
the spectrum is carved: a broad dip centred at 1.6 kHz, full weight below it,
and some air above 5 kHz where consonants are brief enough not to collide.
The renderer then side-chains the whole thing under the voice as well.

    python -m cm.music out.wav --seconds 51 --mood synthwave
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SR = 48000
MAX_HARMONIC = 7000.0        # nothing above this is synthesised at all


# --------------------------------------------------------------------------- #
# voicings
# --------------------------------------------------------------------------- #

MOODS: dict[str, dict] = {
    # i - VI - III - VII with ninths and sevenths. The progression the genre is
    # built on, voiced with colour so it does not sound like a chord chart.
    "synthwave": {
        "bpm": 84,
        "chords": [[57, 60, 64, 71], [53, 57, 60, 64],
                   [55, 60, 64, 67], [55, 59, 62, 69]],
        "roots": [45, 41, 43, 43],
        "arp": True, "hats": True, "detune": 8.0,
        "cut": (700.0, 3400.0, 5200.0),
    },
    # Slower, no percussion, for clips that talk the whole way through.
    "calm": {
        "bpm": 72,
        "chords": [[57, 60, 64, 67], [53, 57, 60, 64],
                   [50, 53, 57, 60], [52, 55, 59, 62]],
        "roots": [45, 41, 38, 40],
        "arp": False, "hats": False, "detune": 5.0,
        "cut": (600.0, 2100.0, 2900.0),
    },
    # i - bII - VI - VII. The flattened second is what makes it feel wrong on
    # purpose; for reveals and anything with a twist in it.
    "tense": {
        "bpm": 92,
        "chords": [[57, 60, 64, 69], [58, 62, 65, 70],
                   [53, 57, 60, 65], [55, 59, 62, 67]],
        "roots": [45, 46, 41, 43],
        "arp": True, "hats": True, "detune": 11.0,
        "cut": (650.0, 3800.0, 5600.0),
    },
}
MOOD_NAMES = sorted(MOODS)


def _freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# --------------------------------------------------------------------------- #
# the shape of the spectrum, and of the piece
# --------------------------------------------------------------------------- #

def _carve(freq: float) -> float:
    """Per-harmonic gain: a broad dip where speech lives, air left on top.

    A hard low-pass is the obvious way to keep music out of the voice's way
    and it sounds muffled, because it removes the top as well as the middle.
    This takes about 11 dB out of an octave either side of 1.6 kHz and leaves
    the rest, so the bed keeps its weight underneath and its shimmer above
    while the band carrying consonants stays comparatively empty.
    """
    dip_db = -11.0 * float(np.exp(-0.5 * (np.log2(freq / 1600.0) / 1.05) ** 2))
    top = 1.0 / (1.0 + (freq / 7000.0) ** 2)
    return (10.0 ** (dip_db / 20.0)) * top


def _arc(t: np.ndarray, total: float, cut: tuple[float, float, float]) -> np.ndarray:
    """Cutoff over time: sparse open, body through the middle, lift at the end.

    The lift lands in the last quarter because that is where a clip's payoff
    is -- the director puts its closing hit there, and the music should be
    arriving at the same moment rather than holding one level throughout.
    """
    lo, mid, high = cut
    x = np.clip(t / max(total, 0.001), 0.0, 1.0)
    rise = lo + (mid - lo) * np.clip(x / 0.35, 0.0, 1.0) ** 0.7
    lift = (high - mid) * np.clip((x - 0.72) / 0.28, 0.0, 1.0) ** 1.4
    return rise + lift


def _level(t: np.ndarray, total: float) -> np.ndarray:
    """Fade in under the hook, hold, duck away before the clip ends."""
    fade_in = np.clip(t / 1.9, 0.0, 1.0) ** 1.8
    tail = np.clip((total - t) / 0.9, 0.0, 1.0) ** 1.2
    swell = 0.86 + 0.14 * np.clip((t / max(total, 0.001) - 0.4) / 0.5, 0.0, 1.0)
    return fade_in * tail * swell


def _pump(t: np.ndarray, bpm: float, depth: float = 0.26) -> np.ndarray:
    """A dip on every beat that recovers over ~180 ms.

    The genre's one unmistakable gesture, and it does real work here: it keeps
    a sustained pad from sitting as a flat block of energy.
    """
    beat = 60.0 / bpm
    phase = np.mod(t, beat)
    return 1.0 - depth * np.exp(-phase / 0.18)


# --------------------------------------------------------------------------- #
# voices
# --------------------------------------------------------------------------- #

def _pad(notes: list[int], t: np.ndarray, cutoff: np.ndarray, detune: float,
         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Detuned saw stack, each harmonic following the moving cutoff."""
    left = np.zeros_like(t)
    right = np.zeros_like(t)
    for note in notes:
        base = _freq(note)
        for side, cents in enumerate((-detune, detune)):
            f0 = base * (2.0 ** (cents / 1200.0))
            phase0 = rng.uniform(0.0, 6.283)
            voice = np.zeros_like(t)
            n = 1
            while f0 * n <= MAX_HARMONIC and n <= 28:
                f = f0 * n
                # Present only as far as the cutoff has opened, and shaped by
                # the carve so the speech band never fills up.
                gain = (1.0 / n) * _carve(f) / (1.0 + (f / cutoff) ** 4)
                voice += np.sin(2.0 * np.pi * f * t + phase0 * n) * gain
                n += 1
            if side == 0:
                left += voice * 0.95
                right += voice * 0.45
            else:
                left += voice * 0.45
                right += voice * 0.95
    scale = 1.0 / max(1, len(notes) * 2)
    return left * scale, right * scale


def _bass(root: int, t: np.ndarray, bpm: float) -> np.ndarray:
    """Eighth-note root pulses with a sub underneath, well below the voice."""
    out = np.zeros_like(t)
    step = 30.0 / bpm                      # an eighth note
    f = _freq(root)
    phase = np.mod(t, step)
    env = np.exp(-phase / (step * 0.34))
    # Accent the downbeat of each beat so it drives rather than machine-guns.
    accent = np.where(np.mod(np.floor(t / step), 2) < 1, 1.0, 0.62)
    out += np.sin(2.0 * np.pi * f * t) * env * accent
    out += 0.30 * np.sin(4.0 * np.pi * f * t) * env * accent
    # The sub, an octave down, breathing rather than droning. It is deliberately
    # the only thing under 80 Hz: a continuous tone down there is inaudible on a
    # phone and still spends the headroom that everything audible needs.
    if f * 0.5 >= 40.0:
        sub_env = np.exp(-phase / (step * 0.8))
        out += 0.38 * np.sin(np.pi * f * t) * sub_env
    return out * 0.40


def _arp(notes: list[int], t: np.ndarray, bpm: float, cutoff: np.ndarray,
         ) -> tuple[np.ndarray, np.ndarray]:
    """Sixteenth-note arpeggio over the chord tones: the forward motion."""
    step = 15.0 / bpm
    idx = np.floor(t / step).astype(int)
    order = [0, 2, 1, 3] if len(notes) >= 4 else list(range(len(notes)))
    phase = np.mod(t, step)
    env = np.exp(-phase / (step * 0.42))

    left = np.zeros_like(t)
    right = np.zeros_like(t)
    for slot, note_i in enumerate(order):
        here = (np.mod(idx, len(order)) == slot)
        if not here.any():
            continue
        f0 = _freq(notes[note_i] + 12)      # an octave above the pad
        voice = np.zeros_like(t)
        n = 1
        while f0 * n <= MAX_HARMONIC and n <= 10:
            f = f0 * n
            voice += np.sin(2.0 * np.pi * f * t) * (1.0 / (n * n)) * _carve(f) \
                / (1.0 + (f / cutoff) ** 4)
            n += 1
        voice = voice * env * here
        pan = 0.5 + 0.35 * (1 if slot % 2 else -1)
        left += voice * (1.0 - pan + 0.5)
        right += voice * (pan + 0.0)
    return left * 0.17, right * 0.17


def _hats(t: np.ndarray, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Short noise ticks on the offbeat.

    Above the speech band, and brief: masking is about energy at the same
    instant, and a 25 ms tick between syllables costs nothing in clarity
    while giving the bed a pulse that pads alone cannot.
    """
    step = 30.0 / bpm
    phase = np.mod(t - step * 0.5, step)
    env = np.exp(-phase / 0.012) * (phase < 0.05)
    noise = rng.standard_normal(len(t))
    # Crude high-pass: difference of the noise keeps only the top end.
    noise = np.diff(noise, prepend=noise[0])
    return noise * env * 0.055


# --------------------------------------------------------------------------- #
# the piece
# --------------------------------------------------------------------------- #

def compose(seconds: float, *, mood: str = "synthwave", seed: int = 0,
            ) -> np.ndarray:
    """Return a stereo float array of exactly `seconds`."""
    if mood not in MOODS:
        raise ValueError("mood must be one of: " + ", ".join(MOOD_NAMES))
    spec = MOODS[mood]
    bpm = float(spec["bpm"])
    rng = np.random.default_rng(seed)

    total = max(float(seconds), 2.0)
    n_samples = int(total * SR)
    t_all = np.arange(n_samples) / SR

    bar = 4.0 * 60.0 / bpm
    chord_len = 2.0 * bar
    left = np.zeros(n_samples)
    right = np.zeros(n_samples)

    i = 0
    start = 0.0
    while start < total:
        length = min(chord_len, total - start)
        if length < 0.15:
            break
        lo = int(start * SR)
        hi = min(lo + int(length * SR), n_samples)
        t_local = np.arange(hi - lo) / SR
        t_here = t_all[lo:hi]

        cutoff = _arc(t_here, total, spec["cut"])
        notes = spec["chords"][i % len(spec["chords"])]
        root = spec["roots"][i % len(spec["roots"])]

        pad_l, pad_r = _pad(notes, t_local, cutoff, spec["detune"], rng)
        low = _bass(root, t_local, bpm)
        seg_l = pad_l + low
        seg_r = pad_r + low

        if spec["arp"]:
            arp_l, arp_r = _arp(notes, t_local, bpm, cutoff)
            seg_l += arp_l
            seg_r += arp_r
        if spec["hats"]:
            tick = _hats(t_local, bpm, rng)
            seg_l += tick
            seg_r += tick * 0.85

        left[lo:hi] += seg_l
        right[lo:hi] += seg_r
        start += chord_len
        i += 1

    shape = _level(t_all, total) * _pump(t_all, bpm)
    left *= shape
    right *= shape

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-9)
    gain = 0.72 / peak                      # the renderer sets the real level
    return np.stack([left * gain, right * gain], axis=1)


def render_bed(dest: Path | str, seconds: float, *, mood: str = "synthwave",
               seed: int = 0) -> Path:
    stereo = compose(seconds, mood=mood, seed=seed)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((np.clip(stereo, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="compose an instrumental bed")
    ap.add_argument("dest")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--mood", default="synthwave", choices=MOOD_NAMES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = render_bed(args.dest, args.seconds, mood=args.mood, seed=args.seed)
    print("  %s  %.1fs, %s" % (out, args.seconds, args.mood))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
