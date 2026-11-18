# Your own sound effects

Drop audio files here in a folder named after the effect they replace:

    sfx_library/
      whoosh/   whoosh-01.wav   whoosh-02.wav
      impact/   hit-soft.wav    hit-hard.wav
      ding/     bell.wav

Folder names must match the built-in effect names (run
`python clipmaker.py sfx-preview` to list them). Several files in one folder
are used as variants, so a repeated effect doesn't sound identical.

Anything found here is used in preference to the synthesised version.
Accepted: .wav .mp3 .aiff .flac .ogg .m4a

Keep licence documentation for anything you add — a monetised channel needs it.
