"""Turn a plan into a finished file.

The audio graph is the interesting part. Sound effects are mixed on their own
bus so the voice can stay dominant:

    voice  -> highpass -> speechnorm ----------------\\
                                                      amix -> loudnorm -> out
    sfx_1  -> adelay -> volume --\\                   /
    sfx_n  -> adelay -> volume ---- amix(normalize=0)/

`normalize=0` on amix matters: the default divides every input by the number of
inputs, so adding a fifth sound effect would quietly drop the voice by 7 dB.
An optional music bed is side-chained against the voice so it ducks under
speech automatically.

Output is normalised to -14 LUFS / -1 dBTP, which is what YouTube targets, so
clips don't get turned down on upload.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cueplan import Cue
from .util import ffmpeg, media_info

LUFS_TARGET = -14.0
TRUE_PEAK = -1.0


@dataclass
class RenderSpec:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crop_mode: str = "blur"        # blur | fill | pad
    blur_sigma: int = 26
    video_bitrate: str = ""        # empty = CRF mode
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "192k"
    music_db: float = -26.0

    @property
    def aspect(self) -> str:
        return f"{self.width}:{self.height}"


def _video_chain(spec: RenderSpec, *, ass_path: Path | None) -> str:
    w, h = spec.width, spec.height

    if spec.crop_mode == "fill":
        chain = (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                 f"crop={w}:{h},setsar=1")
    elif spec.crop_mode == "pad":
        chain = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                 f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    else:
        # Blurred fill: the source sits centred over a zoomed, blurred copy of
        # itself. Keeps faces intact on a 16:9 source, unlike a hard crop.
        chain = (
            f"split=2[bg][fg];"
            f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},gblur=sigma={spec.blur_sigma},eq=brightness=-0.06[bgb];"
            f"[fg]scale={w}:-2:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )

    chain += f",fps={spec.fps},format=yuv420p"
    if ass_path is not None:
        escaped = str(ass_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        chain += f",ass='{escaped}'"
    return chain


def _audio_graph(
    cues_with_files: list[tuple[Cue, Path]],
    *,
    has_audio: bool,
    music: Path | None,
    spec: RenderSpec,
    first_input_index: int,
    voice_input: int = 0,
) -> tuple[str, str]:
    """Build the audio half of the filter_complex. Returns (graph, out_label).

    `voice_input` is which input carries the speech. It is input 0 normally and
    input 1 when a b-roll clip has taken the front of the input list to supply
    the picture.
    """
    parts: list[str] = []
    mix_labels: list[str] = []

    if has_audio:
        # highpass clears rumble; speechnorm evens out a presenter who drifts
        # off-mic without the pumping a compressor would add.
        parts.append(f"[{voice_input}:a]highpass=f=85,"
                     "speechnorm=e=6.25:r=0.0005:l=1[voice]")
        voice_label = "[voice]"
    else:
        parts.append(f"anullsrc=channel_layout=stereo:sample_rate=48000[voice]")
        voice_label = "[voice]"

    idx = first_input_index
    sfx_labels: list[str] = []
    for i, (cue, _path) in enumerate(cues_with_files):
        delay_ms = int(max(0.0, cue.t) * 1000)
        label = f"[s{i}]"
        parts.append(
            f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,"
            f"volume={cue.gain_db:.2f}dB,adelay={delay_ms}|{delay_ms}{label}"
        )
        sfx_labels.append(label)
        idx += 1

    if sfx_labels:
        if len(sfx_labels) == 1:
            parts.append(f"{sfx_labels[0]}anull[sfxbus]")
        else:
            parts.append(
                f"{''.join(sfx_labels)}amix=inputs={len(sfx_labels)}:"
                f"normalize=0:dropout_transition=0[sfxbus]"
            )
        mix_labels.append("[sfxbus]")

    if music is not None:
        # The voice is wanted twice: once in the mix and once as the key that
        # ducks the music. A filter label can only be consumed once, so it has
        # to be split explicitly. Without this ffmpeg silently produces a
        # different mix -- which it did, several decibels quiet, as soon as a
        # b-roll input changed the order the graph was resolved in.
        parts.append(f"{voice_label}asplit=2[voicemix][voicekey]")
        voice_label = "[voicemix]"
        parts.append(
            f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,volume={spec.music_db:.1f}dB,"
            f"aloop=loop=-1:size=2e9[musicraw]"
        )
        # Duck the bed under the voice rather than riding a fixed level.
        parts.append(
            "[musicraw][voicekey]sidechaincompress=threshold=0.03:ratio=12:"
            "attack=15:release=320:makeup=1[musicduck]"
        )
        mix_labels.append("[musicduck]")
        idx += 1

    all_labels = [voice_label] + mix_labels
    if len(all_labels) == 1:
        parts.append(f"{voice_label}anull[premaster]")
    else:
        parts.append(
            f"{''.join(all_labels)}amix=inputs={len(all_labels)}:"
            f"normalize=0:dropout_transition=0[premaster]"
        )

    # alimiter catches transients before loudnorm so a strong hit can't clip.
    parts.append("[premaster]alimiter=limit=0.97:attack=5:release=60[mastered]")
    return ";".join(parts), "[mastered]"


def _loudnorm_args(measured: dict[str, str] | None) -> str:
    """Build the loudnorm stage.

    Single-pass loudnorm applies a *dynamic*, time-varying gain. Feeding it
    measured values with `linear=true` applies one constant gain to the whole
    clip instead, which is the right behaviour for a short file: no pumping,
    and the relative balance the mix was designed with is preserved exactly.

    On the test fixture both modes landed within ~0.2 dB of each other, so
    this is insurance rather than a dramatic fix — but the second pass is
    audio-only and cheap, and short clips are where dynamic gain has the
    least material to settle on.
    """
    base = f"loudnorm=I={LUFS_TARGET}:TP={TRUE_PEAK}:LRA=11"
    if not measured:
        return base
    return (
        f"{base}:measured_I={measured['input_i']}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_TP={measured['input_tp']}"
        f":measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def _measure_pass(
    source: Path | str,
    *,
    start: float,
    duration: float,
    cues_with_files: list[tuple[Cue, Path]],
    music: Path | None,
    spec: RenderSpec,
    has_audio: bool,
) -> dict[str, str] | None:
    """Run the audio graph once with `print_format=json` to get real levels.

    Audio-only (`-vn`), so this costs a fraction of the video render.
    """
    import json as _json
    import re

    args: list[str] = ["-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}",
                       "-accurate_seek", "-i", str(source)]
    for _cue, path in cues_with_files:
        args += ["-i", str(path)]
    if music is not None:
        args += ["-stream_loop", "-1", "-i", str(music)]

    graph, label = _audio_graph(cues_with_files, has_audio=has_audio, music=music,
                               spec=spec, first_input_index=1)
    graph += f";{label}{_loudnorm_args(None)}:print_format=json[aout]"

    proc = ffmpeg([*args, "-vn", "-filter_complex", graph, "-map", "[aout]",
                   "-f", "null", "-"], check=False)
    err = proc.stderr or ""
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", err, re.S)
    if not match:
        return None
    try:
        data = _json.loads(match.group(0))
    except ValueError:
        return None
    required = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
    if not all(k in data for k in required):
        return None
    # A fully silent clip measures as -inf and would poison the second pass.
    try:
        if float(data["input_i"]) < -70.0:
            return None
    except ValueError:
        return None
    return {k: str(data[k]) for k in required}


def render_clip(
    source: Path | str,
    dest: Path,
    *,
    start: float,
    end: float,
    cues_with_files: list[tuple[Cue, Path]],
    spec: RenderSpec,
    ass_path: Path | None = None,
    music: Path | None = None,
    has_audio: bool = True,
    broll: Path | None = None,
    broll_start: float = 0.0,
) -> Path:
    """Cut, reframe, caption, mix and normalise one clip.

    With `broll`, the picture comes from that file and only the sound comes
    from `source` -- which is how a piece of commentary becomes a watchable
    clip rather than a person talking at a camera. The b-roll is looped, so a
    source shorter than the clip repeats instead of running out.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.5, end - start)

    # Pass 1: measure the finished mix so pass 2 can normalise linearly.
    measured = _measure_pass(
        source, start=start, duration=duration, cues_with_files=cues_with_files,
        music=music, spec=spec, has_audio=has_audio,
    )

    args: list[str] = []
    voice_input = 0
    if broll is not None:
        # The picture, first in the list so it stays [0:v]. Looped, and not
        # given -t: the output duration and -shortest end the clip.
        args += ["-stream_loop", "-1",
                 "-ss", f"{max(0.0, broll_start):.3f}", "-accurate_seek",
                 "-i", str(broll)]
        voice_input = 1

    # Seek before -i for speed, but keep it accurate; re-encoding anyway.
    args += ["-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}",
             "-accurate_seek", "-i", str(source)]

    for _cue, path in cues_with_files:
        args += ["-i", str(path)]
    if music is not None:
        args += ["-stream_loop", "-1", "-i", str(music)]

    video_chain = _video_chain(spec, ass_path=ass_path)
    audio_graph, mastered = _audio_graph(
        cues_with_files, has_audio=has_audio, music=music, spec=spec,
        first_input_index=voice_input + 1, voice_input=voice_input,
    )
    audio_graph += f";{mastered}{_loudnorm_args(measured)},aresample=48000[aout]"
    graph = f"[0:v]{video_chain}[vout];{audio_graph}"

    args += [
        "-filter_complex", graph,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", spec.preset, "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-g", str(spec.fps * 2),
        "-c:a", "aac", "-b:a", spec.audio_bitrate, "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-t", f"{duration:.3f}",
        "-shortest",
    ]
    if spec.video_bitrate:
        args += ["-b:v", spec.video_bitrate]
    else:
        args += ["-crf", str(spec.crf)]
    args.append(str(dest))

    ffmpeg(args)
    return dest


def measure_loudness(path: Path | str) -> dict[str, float]:
    """Read back integrated loudness and true peak, to prove normalisation."""
    from .util import require, run
    proc = run([
        require("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(path),
        "-af", "ebur128=peak=true", "-f", "null", "-",
    ], check=False)
    out = (proc.stderr or "")
    result: dict[str, float] = {}
    tail = out[out.rfind("Summary"):] if "Summary" in out else out
    for key, label in (("I:", "lufs"), ("Peak:", "peak_dbfs"), ("LRA:", "lra")):
        idx = tail.find(key)
        if idx == -1:
            continue
        chunk = tail[idx + len(key):].strip().split()
        if chunk:
            try:
                result[label] = float(chunk[0])
            except ValueError:
                pass
    return result


def probe_output(path: Path | str) -> dict:
    return media_info(path)
