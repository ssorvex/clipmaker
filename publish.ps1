<#
  Put the finished clips where they are easy to upload from.

  Uploading happens at this PC rather than on the phone, because TikTok only
  offers scheduling on its desktop site -- so the clips land in a plain local
  folder next to a copy-paste sheet, and each one is uploaded by hand and
  given its posting time in the platform's own scheduler. No API, no account,
  no monthly fee, and nothing has to stay running afterwards.

  Old batches are pruned: three clips a day is about 100 MB a day, and a
  folder that quietly fills up is worse than one that forgets.
#>
param(
    [string]$From = "",
    [string]$To   = "$env:USERPROFILE\Shorts",
    [int]$Keep    = 5,
    # One posting time per clip, in the order the clips come out. Scheduling
    # is done in each app at upload time, so these are a reminder rather than
    # anything this script enforces.
    [string[]]$Times = @("10:00", "14:00", "18:00")
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $From) {
    $newest = Get-ChildItem (Join-Path $root "clips") -Directory -ErrorAction SilentlyContinue |
              Sort-Object Name -Descending | Select-Object -First 1
    if (-not $newest) { throw "No clips folder found under $root\clips" }
    $From = $newest.FullName
}
if (-not (Test-Path $From)) { throw "No such folder: $From" }

$name = Split-Path $From -Leaf
$dest = Join-Path $To $name
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("SHORTS  $name")
$lines.Add("")
$lines.Add("Tick each one off as you post it.")
$lines.Add("")

$i = 0
foreach ($d in (Get-ChildItem $From -Directory | Sort-Object Name)) {
    $clip = Join-Path $d.FullName "clip.mp4"
    $meta = Join-Path $d.FullName "meta.json"
    if (-not (Test-Path $clip)) { continue }

    $i++
    Copy-Item $clip (Join-Path $dest "$i.mp4") -Force

    $p = (Get-Content $meta -Raw -Encoding UTF8 | ConvertFrom-Json).plan
    $tags = ($p.hashtags | ForEach-Object { "#$_" }) -join " "
    $secs = [math]::Round([double]$p.duration, 0)

    $when = if ($i -le $Times.Count) { $Times[$i - 1] } else { "" }
    $lines.Add("================================================")
    $lines.Add("  $i.mp4   -   $secs seconds")
    if ($when) { $lines.Add("  PLANLA:  $when") }
    $lines.Add("================================================")
    $lines.Add("")
    $lines.Add("TITLE")
    $lines.Add($p.title)
    $lines.Add("")
    $lines.Add("DESCRIPTION")
    $lines.Add($p.description)
    $lines.Add("")
    $lines.Add("TAGS")
    $lines.Add($tags)
    $lines.Add("")
    $lines.Add("POSTED:   [ ] YouTube     [ ] TikTok     [ ] Instagram")
    $lines.Add("")
    $lines.Add("")
}

if ($i -eq 0) { throw "No clip.mp4 found in any subfolder of $From" }

# No byte-order mark: PowerShell's own UTF8 writes one, and phone text viewers
# show it as a stray character on the first line.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines((Join-Path $dest "POST.txt"), $lines, $utf8NoBom)

Write-Host ""
Write-Host "  $i clip(s) + POST.txt ready to upload:" -ForegroundColor Green
Write-Host "  $dest"

# The Obsidian sheet: the same three clips written out as something to work
# down and paste. Best effort -- a vault that is missing or moved must not
# cost us the copied clips.
$sheet = Join-Path $root "daysheet.py"
if (Test-Path $sheet) {
    try {
        & python $sheet $From $dest --times @Times
    } catch {
        Write-Host "  (could not write the Obsidian sheet: $_)" -ForegroundColor Yellow
    }
}

if ($Keep -gt 0) {
    $old = Get-ChildItem $To -Directory | Sort-Object Name -Descending | Select-Object -Skip $Keep
    foreach ($o in $old) {
        Remove-Item $o.FullName -Recurse -Force -Confirm:$false
        Write-Host "  pruned old batch: $($o.Name)" -ForegroundColor DarkGray
    }
}
Write-Host ""
