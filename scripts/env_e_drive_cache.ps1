# env_e_drive_cache.ps1
# ---------------------------------------------------------------------------
# HARD RULE OF THIS PROJECT: nothing related to this project may be written to
# the C: drive (no dependencies, no pip/HF/matplotlib/torch caches, no temp
# files).  See docs/ENVIRONMENT_SETUP.md 2.6 and docs/PROJECT_CONTEXT.md 0.1.
#
# This script points every cache/temp env var used by our toolchain at E:\cache.
#
# Usage (note the leading dot -- it must be DOT-SOURCED to affect your shell):
#   . .\scripts\env_e_drive_cache.ps1              # this shell only
#   . .\scripts\env_e_drive_cache.ps1 -Persist     # also write User-level vars
#   . .\scripts\env_e_drive_cache.ps1 -Check       # print state, do not change
# ---------------------------------------------------------------------------
param(
    [switch]$Persist,
    [switch]$Check
)

$root = 'E:\cache'

$map = [ordered]@{
    'PIP_CACHE_DIR'  = "$root\pip"
    'MPLCONFIGDIR'   = "$root\matplotlib"
    'HF_HOME'        = "$root\huggingface"
    'TORCH_HOME'     = "$root\torch"
    'XDG_CACHE_HOME' = "$root\xdg"
    'TEMP'           = "$root\tmp"
    'TMP'            = "$root\tmp"
}

function New-CacheDir([string]$path) {
    if (Test-Path $path) { return }
    try {
        New-Item -ItemType Directory -Force -ErrorAction Stop -Path $path | Out-Null
    } catch {
        # E: root ACE has no (OI)(CI) inherit flag -- known issue on this
        # machine (docs/ENVIRONMENT_SETUP.md 5).  Re-grant Modify for the
        # current SID, then retry once.
        $sid = (New-Object System.Security.Principal.NTAccount($env:USERNAME)).Translate(
            [System.Security.Principal.SecurityIdentifier]).Value
        Write-Host "[FIX ] grant ACL on $root for $sid (E: drive inherit issue)"
        & icacls $root /grant "*${sid}:(OI)(CI)(M)" /T | Out-Null
        New-Item -ItemType Directory -Force -ErrorAction Stop -Path $path | Out-Null
    }
}

Write-Host "=== cache/temp redirection (C: drive must stay clean) ==="

if (-not $Check) {
    New-CacheDir $root
    foreach ($d in $map.Values) { New-CacheDir $d }
}

$bad = 0
foreach ($k in $map.Keys) {
    $want = $map[$k]
    $now = [Environment]::GetEnvironmentVariable($k, 'Process')
    if ($Check) {
        $state = if ($now) { $now } else { '(unset -> may fall back to C:)' }
    } else {
        Set-Item -Path "Env:$k" -Value $want
        $now = $want
        if ($Persist) { [Environment]::SetEnvironmentVariable($k, $want, 'User') }
        $state = $now
    }
    $mark = '[OK  ]'
    if (-not $state -or $state -like 'C:*') { $mark = '[WARN]'; $bad++ }
    Write-Host ("{0} {1,-16} = {2}" -f $mark, $k, $state)
}

if (-not $Check) {
    $persistNote = if ($Persist) { 'written to User env (new terminals inherit)' } else { 'this shell only' }
    Write-Host "mode: $persistNote"
}

$py = 'E:\python3.11.7\python.exe'
if (Test-Path $py) {
    Write-Host "pip cache dir -> $(& $py -m pip cache dir 2>&1)"
}

if ($bad -gt 0) {
    Write-Host "[WARN] $bad var(s) may still write to C: -- run without -Check"
    exit 1
}
Write-Host "[OK] all cache/temp vars point off the C: drive"
