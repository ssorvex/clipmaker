"""Sound-effect sourcing.

Three ways to get a sound, tried in this order:

  1. `library/<kind>/*.wav`  — your own licensed packs, highest quality, free
  2. procedural synthesis    — built from ffmpeg oscillators and noise on the
                               spot; costs nothing, needs no network, and is
                               copyright-clean because nothing is sampled
  3. a generative API        — ElevenLabs / Freesound, opt-in per run

Everything resolved is cached by content hash under `cache/`, so a sound is
only ever built or paid for once.

The synthesis layer is the reason this tool works out of the box: you do not
have to buy a sound pack or wire up an API before you can post a clip.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from .util import ToolError, ffmpeg, key_hash, slugify, volume_peak_db

SR = 48000


@dataclass
class Layer:
    """One lavfi generator plus the filter chain applied to it."""
    src: str
    post: str = ""


@dataclass
class Recipe:
    kind: str
    duration: float
    layers: list[Layer]
    peak_db: float = -3.0          # normalised asset peak; per-cue gain comes later
    post: str = ""                 # applied to the mixed result
    reverse: bool = False
    description: str = ""
    role: str = "accent"           # accent | transition | emphasis | bed
    tags: list[str] = field(default_factory=list)


def _noise(colour: str = "pink", amp: float = 0.9, seed: int = 0) -> str:
    return f"anoisesrc=d={{dur}}:c={colour}:a={amp}:r={SR}:seed={seed}"


def _eval(expr: str) -> str:
    return f"aevalsrc='{expr}':d={{dur}}:s={SR}"


# --- the catalogue ----------------------------------------------------------
# Envelopes are written explicitly so each sound has a deliberate shape:
# exp(-k*t) for a decaying hit, pow(t/d, n) for a build, mod(t, p) to repeat.

RECIPES: dict[str, Recipe] = {
    "riser": Recipe(
        kind="riser", duration=1.20, role="transition",
        description="tension build that leads into the hook",
        layers=[
            Layer(_eval("0.55*sin(2*PI*(180+1250*pow(t/1.2,2))*t)"),
                  "volume='pow(t/1.2,1.4)':eval=frame"),
            Layer(_noise("white", 0.5, seed=11),
                  "highpass=f=1200,volume='pow(t/1.2,2.2)*0.7':eval=frame"),
        ],
        post="afade=t=out:st=1.12:d=0.08",
        tags=["build", "intro"],
    ),
    "whoosh": Recipe(
        kind="whoosh", duration=0.65, role="transition",
        description="air movement for a cut or a text swipe",
        layers=[
            Layer(_noise("pink", 0.95, seed=3),
                  "highpass=f=320,lowpass=f=5200,"
                  "volume='min(1,t*9)*exp(-3.4*t)':eval=frame"),
            Layer(_eval("0.18*sin(2*PI*(420-260*t/0.65)*t)"),
                  "volume='exp(-4.5*t)':eval=frame"),
        ],
        tags=["cut", "transition"],
    ),
    "whoosh_reverse": Recipe(
        kind="whoosh_reverse", duration=0.60, role="transition",
        description="sucking-in whoosh that anticipates a cut",
        layers=[
            Layer(_noise("pink", 0.9, seed=7),
                  "highpass=f=380,lowpass=f=6000,"
                  "volume='min(1,t*9)*exp(-3.8*t)':eval=frame"),
        ],
        reverse=True,
        tags=["anticipation"],
    ),
    "impact": Recipe(
        kind="impact", duration=0.85, role="accent",
        description="punch landing under the hook line",
        layers=[
            Layer(_eval("sin(2*PI*62*t)*exp(-6.8*t)"), ""),
            Layer(_eval("0.38*sin(2*PI*142*t)*exp(-15*t)"), ""),
            Layer(_noise("white", 0.6, seed=5),
                  "highpass=f=900,volume='exp(-42*t)':eval=frame"),
        ],
        tags=["hit", "hook"],
    ),
    "boom": Recipe(
        kind="boom", duration=1.60, role="accent",
        description="big cinematic sub hit for a reveal",
        layers=[
            Layer(_eval("sin(2*PI*44*t)*exp(-2.6*t)"), ""),
            Layer(_eval("0.3*sin(2*PI*88*t)*exp(-5.0*t)"), ""),
            Layer(_noise("brown", 0.5, seed=13),
                  "lowpass=f=260,volume='exp(-3.2*t)':eval=frame"),
        ],
        tags=["hit", "reveal"],
    ),
    "thud": Recipe(
        kind="thud", duration=0.40, role="accent",
        description="dull low knock, understated",
        layers=[Layer(_eval("0.85*sin(2*PI*78*t)*exp(-13*t)"), "")],
        tags=["hit", "subtle"],
    ),
    "pop": Recipe(
        kind="pop", duration=0.14, role="emphasis",
        description="cartoon pop for a word or a caption appearing",
        layers=[Layer(_eval("sin(2*PI*(920-520*t/0.09)*t)*exp(-36*t)"), "")],
        tags=["caption", "word"],
    ),
    "click": Recipe(
        kind="click", duration=0.06, role="emphasis",
        description="dry tick, for list items and quick cuts",
        layers=[
            Layer(_noise("white", 0.9, seed=17),
                  "highpass=f=1800,volume='exp(-120*t)':eval=frame"),
        ],
        tags=["ui", "list"],
    ),
    "ding": Recipe(
        kind="ding", duration=0.95, role="emphasis",
        description="bright bell for a correct answer or a tip landing",
        layers=[
            Layer(_eval("(0.6*sin(2*PI*1318*t)+0.28*sin(2*PI*2637*t)"
                        "+0.12*sin(2*PI*3955*t))*exp(-5.2*t)"), ""),
        ],
        tags=["positive", "tip"],
    ),
    "sparkle": Recipe(
        kind="sparkle", duration=1.10, role="emphasis",
        description="glittery ascending pings for a before/after reveal",
        layers=[
            Layer(_eval("0.5*sin(2*PI*(2100+900*t)*t)*exp(-26*mod(t,0.13))"),
                  "volume='exp(-1.6*t)':eval=frame"),
            Layer(_eval("0.3*sin(2*PI*(3300+1200*t)*t)*exp(-30*mod(t,0.19))"),
                  "volume='exp(-1.9*t)':eval=frame"),
        ],
        tags=["positive", "reveal", "cute"],
    ),
    "swish": Recipe(
        kind="swish", duration=0.30, role="transition",
        description="short brush of air for a fast cut",
        layers=[
            Layer(_noise("white", 0.8, seed=23),
                  "bandpass=f=2600:width_type=h:w=1800,"
                  "volume='min(1,t*18)*exp(-9*t)':eval=frame"),
        ],
        tags=["cut"],
    ),
    "rewind": Recipe(
        kind="rewind", duration=0.70, role="transition",
        description="descending tape rewind, for a callback or a replay",
        layers=[
            Layer(_eval("0.55*sin(2*PI*(1500-1150*t/0.7)*t)"),
                  "volume='exp(-1.8*t)':eval=frame"),
            Layer(_noise("white", 0.4, seed=29),
                  "highpass=f=800,volume='exp(-5*t)*0.6':eval=frame"),
        ],
        tags=["replay"],
    ),
    "record_scratch": Recipe(
        kind="record_scratch", duration=0.55, role="accent",
        description="hold on — stop the tape",
        layers=[
            Layer(_eval("0.6*sin(2*PI*(620-420*t/0.55)*t+7*sin(2*PI*13*t))"),
                  "volume='exp(-2.6*t)':eval=frame"),
            Layer(_noise("pink", 0.5, seed=31),
                  "bandpass=f=1400:width_type=h:w=1200,"
                  "volume='exp(-4*t)*0.8':eval=frame"),
        ],
        tags=["comedy", "stop"],
    ),
    "suspense": Recipe(
        kind="suspense", duration=2.20, role="bed",
        description="low pulsing drone under a slow build",
        layers=[
            Layer(_eval("0.5*sin(2*PI*58*t)*(0.65+0.35*sin(2*PI*2.4*t))"), ""),
            Layer(_eval("0.18*sin(2*PI*116.5*t)"), ""),
        ],
        post="afade=t=in:d=0.25,afade=t=out:st=1.95:d=0.25",
        peak_db=-9.0,
        tags=["bed", "tension"],
    ),
    "heartbeat": Recipe(
        kind="heartbeat", duration=1.30, role="bed",
        description="two-thump pulse for nerves or anticipation",
        layers=[
            Layer(_eval("0.9*sin(2*PI*52*t)*exp(-16*mod(t,0.62))"
                        "*(1+0.5*exp(-16*mod(t+0.42,0.62)))"), ""),
        ],
        peak_db=-6.0,
        tags=["bed", "tension"],
    ),
    "air": Recipe(
        kind="air", duration=1.00, role="transition",
        description="soft airy wash, the least intrusive transition",
        layers=[
            Layer(_noise("pink", 0.6, seed=37),
                  "highpass=f=2200,volume='sin(PI*t/1.0)*0.8':eval=frame"),
        ],
        peak_db=-8.0,
        tags=["subtle", "transition"],
    ),
    "tick_up": Recipe(
        kind="tick_up", duration=0.75, role="emphasis",
        description="counting ticks for a number or a countdown",
        layers=[
            Layer(_eval("0.7*sin(2*PI*1500*t)*exp(-60*mod(t,0.125))"), ""),
        ],
        tags=["number", "list"],
    ),
    "reveal": Recipe(
        kind="reveal", duration=1.60, role="accent",
        description="riser into a bell — the before/after money shot",
        layers=[
            # ffmpeg's expression evaluator has no < or > operators — gating
            # must use lt()/gt(), or the filtergraph fails to parse.
            Layer(_eval("0.45*sin(2*PI*(200+900*pow(t/1.1,2))*t)"
                        "*lt(t,1.1)*pow(min(t/1.1,1),1.3)"), ""),
            Layer(_eval("(0.55*sin(2*PI*1318*t)+0.25*sin(2*PI*2637*t))"
                        "*gt(t,1.05)*exp(-4.2*max(t-1.05,0))"), ""),
            Layer(_eval("0.6*sin(2*PI*58*t)*gt(t,1.05)*exp(-5*max(t-1.05,0))"), ""),
        ],
        tags=["reveal", "hook"],
    ),
}

#: What the director is allowed to ask for. Keeping this a closed set means a
#: cue can always be satisfied locally, with or without a generative API.
SFX_KINDS = sorted(RECIPES)


def kind_catalogue() -> str:
    """Human-readable catalogue, injected into the director prompt."""
    lines = []
    for kind in SFX_KINDS:
        r = RECIPES[kind]
        lines.append(f"- {kind} ({r.role}, {r.duration:.2f}s): {r.description}")
    return "\n".join(lines)


# --- synthesis -------------------------------------------------------------

def synth(kind: str, dest: Path, *, variant: int = 0) -> Path:
    """Render one effect to a normalised 48 kHz stereo wav."""
    recipe = RECIPES.get(kind)
    if recipe is None:
        raise ToolError(f"unknown sfx kind {kind!r}; known: {', '.join(SFX_KINDS)}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{kind}:{variant}")

    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    for i, layer in enumerate(recipe.layers):
        src = layer.src.replace("{dur}", f"{recipe.duration:.3f}")
        # Vary the noise seed per variant so a repeated kind isn't identical.
        if variant and "seed=" in src:
            src = src.replace(
                f"seed={src.split('seed=')[1].split(':')[0]}",
                f"seed={rng.randint(1, 99999)}",
            )
        inputs += ["-f", "lavfi", "-i", src]
        post = layer.post or "anull"
        chains.append(f"[{i}:a]{post}[L{i}]")
        labels.append(f"[L{i}]")

    graph = ";".join(chains)
    if len(labels) > 1:
        graph += f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0[mx]"
    else:
        graph += f";{labels[0]}anull[mx]"

    tail = ["aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"]
    if recipe.reverse:
        tail.append("areverse")
    if recipe.post:
        tail.append(recipe.post)
    # Hard de-click at both ends: a generator that starts mid-cycle pops.
    tail.append("afade=t=in:d=0.006")
    tail.append(f"afade=t=out:st={max(recipe.duration - 0.012, 0.0):.3f}:d=0.012")
    graph += f";[mx]{','.join(tail)}[out]"

    raw = dest.with_suffix(".raw.wav")
    ffmpeg([*inputs, "-filter_complex", graph, "-map", "[out]",
            "-c:a", "pcm_s16le", str(raw)])

    # Normalise to the recipe's target peak so per-cue gains mean the same
    # thing for every effect.
    peak = volume_peak_db(raw)
    gain = recipe.peak_db - peak if peak > -80 else 0.0
    ffmpeg(["-i", str(raw), "-af", f"volume={gain:.2f}dB",
            "-c:a", "pcm_s16le", str(dest)])
    raw.unlink(missing_ok=True)
    return dest


# --- resolution ------------------------------------------------------------

class SfxResolver:
    """Finds (or makes) the audio file for a cue, and remembers it."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        library_dir: Path | None = None,
        provider: str = "synth",
        request_log: list[dict] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir = Path(library_dir) if library_dir else None
        self.provider = provider
        self.request_log = request_log if request_log is not None else []
        self.stats = {"library": 0, "cache": 0, "synth": 0, "generated": 0, "failed": 0}

    # -- library -----------------------------------------------------------
    def _from_library(self, kind: str, variant: int) -> Path | None:
        if not self.library_dir:
            return None
        folder = self.library_dir / kind
        if not folder.is_dir():
            return None
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg", ".m4a"}
        )
        if not files:
            return None
        return files[variant % len(files)]

    # -- generative --------------------------------------------------------
    def _generate(self, kind: str, prompt: str, dest: Path) -> Path | None:
        """Text-to-SFX via a hosted API. Only called when --sfx-provider asks."""
        if self.provider == "elevenlabs":
            return self._elevenlabs(prompt, dest)
        if self.provider == "higgsfield":
            # Higgsfield's audio generation is an MCP tool, not a REST call we
            # can make from here. Record the request so an agent run with that
            # server attached can fulfil it into the library, then re-run.
            self.request_log.append({"kind": kind, "prompt": prompt,
                                     "suggested_path": str(dest)})
            return None
        return None

    def _elevenlabs(self, prompt: str, dest: Path) -> Path | None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ToolError(
                "--sfx-provider elevenlabs needs ELEVENLABS_API_KEY in the environment."
            )
        try:
            import urllib.error
            import urllib.request
            import json as _json

            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/sound-generation",
                data=_json.dumps({"text": prompt, "duration_seconds": None,
                                  "prompt_influence": 0.4}).encode(),
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = resp.read()
        except Exception as exc:                      # noqa: BLE001 - report, don't crash a render
            print(f"  ! sfx generation failed ({exc}); falling back to synthesis")
            return None

        tmp = dest.with_suffix(".gen.mp3")
        tmp.write_bytes(payload)
        ffmpeg(["-i", str(tmp), "-ar", str(SR), "-ac", "2",
                "-c:a", "pcm_s16le", str(dest)])
        tmp.unlink(missing_ok=True)
        return dest

    # -- public ------------------------------------------------------------
    def resolve(self, kind: str, *, variant: int = 0, prompt: str = "") -> Path:
        """Return a path to audio for this cue, making it if necessary."""
        if kind not in RECIPES:
            kind = "pop"

        from_lib = self._from_library(kind, variant)
        if from_lib is not None:
            self.stats["library"] += 1
            return from_lib

        token = key_hash({"kind": kind, "variant": variant,
                          "provider": self.provider, "prompt": prompt})
        cached = self.cache_dir / f"{slugify(kind)}_{token}.wav"
        if cached.exists():
            self.stats["cache"] += 1
            return cached

        if self.provider not in {"synth", "library"} and prompt:
            made = self._generate(kind, prompt, cached)
            if made is not None and made.exists():
                self.stats["generated"] += 1
                return made

        synth(kind, cached, variant=variant)
        self.stats["synth"] += 1
        return cached

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in self.stats.items() if v]
        return ", ".join(parts) or "none"
