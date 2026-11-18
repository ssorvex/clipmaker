<#
    clip.ps1 — paste a link, get posting-ready shorts.

        .\clip.ps1 https://www.youtube.com/watch?v=XXXXXXXX
        .\clip.ps1 https://youtu.be/XXXXXXXX -Clips 3 -MaxLen 45
        .\clip.ps1 <url> -Direction "skip the intro, favour the car chases"

    Defaults to 5 clips of 30-60 seconds with captions and sound design.
    Output lands in clips\<date_time>\ unless -Out says otherwise.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Url,
    [int]$Clips = 5,
    [double]$MinLen = 30,
    [double]$MaxLen = 60,
    [string]$Out = "",
    [string]$Direction = "",
    [string]$Cookies = "",
    [switch]$NoSound,
    [switch]$NoCaptions
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($Out)) {
    $stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
    $Out = Join-Path $here "clips\$stamp"
}

$cmd = @(
    "make", $Url,
    "--out", $Out,
    "--clips", $Clips,
    "--min-len", $MinLen,
    "--max-len", $MaxLen
)

if (-not $env:ANTHROPIC_API_KEY -and -not $env:ANTHROPIC_AUTH_TOKEN) {
    Write-Host "No ANTHROPIC_API_KEY set - running the local planner (--no-llm)." -ForegroundColor Yellow
    Write-Host "Clips still get captions and the build-and-hit sound bed." -ForegroundColor Yellow
    $cmd += "--no-llm"
}

if ($Direction)  { $cmd += @("--direction", $Direction) }
if ($Cookies)    { $cmd += @("--cookies-from-browser", $Cookies) }
if ($NoSound)    { $cmd += @("--sfx-mode", "off") }
if ($NoCaptions) { $cmd += "--no-captions" }

Write-Host ""
Write-Host "clipmaker -> $Out" -ForegroundColor Cyan
Write-Host ""

& python (Join-Path $here "clipmaker.py") @cmd
$code = $LASTEXITCODE

if ($code -eq 0) {
    Write-Host ""
    Write-Host "Done. Clips, meta.json and youtube.md are in:" -ForegroundColor Green
    Write-Host "  $Out" -ForegroundColor Green
}
exit $code
