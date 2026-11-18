# clipmaker

Turn one long video into a handful of posting-ready vertical clips, with sound
design.

```bash
pip install -r requirements.txt          # ffmpeg must already be on PATH
export ANTHROPIC_API_KEY=sk-ant-...      # or run with --no-llm
python clipmaker.py make long-video.mp4 --clips 5 --out ./clips
```

You get, per clip: a 1080×1920 MP4 with burned captions and placed sound
effects, normalised to −14 LUFS (YouTube's target), plus a `meta.json`, and one
`youtube.md` with paste-ready titles, descriptions and tags.

---

## How it works

| Stage | What it does | Cost |
|---|---|---|
| **probe** | loudness envelope, speech gaps, scene cuts | free, ffmpeg |
| **transcribe** | word-level timestamps | free, local (faster-whisper) |
| **shortlist** | score passages, propose ~12 candidates | free, local |
| **direct** | pick 5, trim, title, design the SFX cue sheet | one Claude call |
| **place** | snap cues to speech gaps, thin, cap density | free, local |
| **render** | reframe, caption, mix, normalise | free, ffmpeg |

The split matters. The local stages do the mechanical work — finding passages
that are loud, fast, quotable and visually punctuated — so the model only ever
sees a shortlist of about twelve candidates with their transcripts. That means
**one model call per video regardless of length**: a 10-minute and a 90-minute
source cost roughly the same.

---

## Sound effects

This was the hard part, and it splits into three separate problems.

### 1. Where do the sounds come from

Three sources, tried in order:

1. **`sfx_library/<kind>/*.wav`** — your own licensed packs. Highest quality.
2. **Procedural synthesis** — built from ffmpeg oscillators and filtered noise
   at render time. This is the default, and it is why the tool works with no
   network, no API key, and no sound pack. Nothing is sampled, so there is
   nothing to clear for a monetised channel.
3. **A generative API** — `--sfx-provider elevenlabs` (needs
   `ELEVENLABS_API_KEY`) for text-to-SFX.

Everything resolved is cached by content hash, so a sound is built or paid for
exactly once and reused across every future video.

Hear the built-in set:

```bash
python clipmaker.py sfx-preview --out ./sfx_preview
# writes one wav per effect plus _all_effects.wav
```

18 effects, each with a role: `riser` `whoosh` `whoosh_reverse` `impact`
`boom` `thud` `pop` `click` `ding` `sparkle` `swish` `rewind`
`record_scratch` `suspense` `heartbeat` `air` `tick_up` `reveal`.

### 2. Which sound goes where

Claude gets the transcript, the signal scores and (with `--vision`) sampled
keyframes, and returns a cue sheet — a time, an effect name, an intensity and
a reason, per cue. It is told to place cues on *structure* (the hook landing, a
turn in the argument, a reveal, a number, a cut) and never on a random word.

With `--no-llm`, or if the model call fails, every clip still gets the
structural bed: a build into the first word and a hit when it lands.

### 3. Making it not sound amateur

This is the part that is pure craft, and it is enforced in code rather than
left to the model (`cm/cueplan.py`):

- **Snap to speech gaps.** Every cue is pulled to the nearest gap between
  words. Word-level timestamps make this precise.
- **Word onsets are legal, mid-word is not.** A transient landing exactly on a
  word onset reads as deliberate accenting. A cue stranded *inside* a word gets
  pulled down 4.5 dB so the words win.
- **Builds and their payoffs are exempt from spacing.** A `riser` 1.1 s before
  an `impact` is the single most reliable pattern in short-form. A naive
  minimum-spacing rule deletes the riser and leaves a hit with no build — worse
  than neither.
- **Don't start a build you can't land.** If there isn't room for at least 55%
  of the riser before its payoff, the riser is dropped rather than truncated.
- **Density cap** of roughly one cue per 4 seconds, and a minimum 1.1 s
  between cues, so nothing machine-guns.
- **No duplicate effect within 2.5 s**, and repeated effects get different
  synthesis variants so they aren't bit-identical.
- **Nothing in the last 0.3 s**, where it would be cut off.

Dropped cues are recorded with a reason in `meta.json` rather than silently
discarded.

### 4. The mix

```
voice  -> highpass -> speechnorm ---------------\
                                                 amix -> limiter -> loudnorm -> out
sfx_1  -> adelay -> volume --\                  /
sfx_n  -> adelay -> volume ---- amix(normalize=0)
music  -> volume -> sidechaincompress(key=voice)
```

- `normalize=0` on amix is load-bearing. The default divides every input by the
  number of inputs, so adding a fifth sound effect would quietly drop the voice
  by 7 dB.
- A music bed (`--music`) is side-chained against the voice, so it ducks under
  speech automatically instead of riding a fixed level.
- Per-cue gains are relative to a normalised asset peak, so `-11 dB` means the
  same thing for every effect.
- Two-pass `loudnorm` to −14 LUFS / −1 dBTP: a constant gain across the clip
  rather than single-pass dynamic gain.

---

## Options worth knowing

```bash
# steer the editor
--direction "favour grooming transformations, skip anything about pricing"

# sound
--sfx-mode full|structural|off     # structural = build+hit only
--sfx-provider synth|library|elevenlabs
--sfx-density 4.0                  # roughly one cue per N seconds
--sfx-spacing 1.1                  # minimum seconds between cues
--music bed.mp3                    # ducked under speech

# framing and captions
--crop blur|fill|pad               # blur keeps faces intact on a 16:9 source
--caption-style bold-white|yellow-pop|clean
--no-captions

# cost and speed
--no-llm                           # no model call at all
--effort low|medium|high|xhigh|max
--whisper-model tiny|base|small|medium|large-v3
--no-asr                           # skip transcription entirely
```

Run with `--no-llm` first to see which moments the local signals pick. If the
shortlist is wrong, the model pass won't save it — tune `--shortlist` and the
hook patterns in `cm/candidates.py` instead.

---

## Which model

Default is `claude-opus-5-5`. The director pass is one structured-output call
over a shortlist — roughly 4–8k tokens in, 2–3k out — so it costs a few cents
per video and the model choice barely moves the bill.

`--model claude-fable-5-2` works and is more capable, but this task does not
need it: the heavy lifting (finding candidates, placing cues, mixing) is
deterministic local code, and the model is only exercising editorial taste over
a pre-filtered list. Spend the effort budget instead — `--effort xhigh` on
Opus 5.5 is a better trade than switching model at default effort.

---

## Using your own sounds

Drop files into `sfx_library/<effect-name>/`:

```
sfx_library/
  whoosh/   whoosh-01.wav  whoosh-02.wav
  impact/   hit-soft.wav   hit-hard.wav
```

Folder names must match the effect names above. Multiple files per folder are
used as variants. Anything found there wins over synthesis.

### Higgsfield

`--sfx-provider higgsfield` does not generate anything directly — Higgsfield's
audio generation is an MCP tool, not a REST endpoint this script can call. It
writes `sfx_requests.json` listing the prompts instead, so an agent run with
that server attached can generate them into `sfx_library/` deliberately. This
keeps the tool from spending credits on its own.

---

## Limits

- Clip selection is only as good as the transcript. On noisy or multi-speaker
  audio use `--whisper-model medium` or `large-v3`.
- There is no face or subject tracking, so `--crop fill` on a 16:9 source can
  cut off someone standing off-centre. `--crop blur` (the default) never does.
- The first run downloads a Whisper model (~500 MB for `small`).
- Procedural effects are clean and usable but they are synthetic. For a channel
  with a signature sound, buy a pack and point `--sfx-library` at it.
