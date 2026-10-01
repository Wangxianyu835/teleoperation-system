# preflight.ps1
# ---------------------------------------------------------------------------
# One-command enforcement of docs/CONVENTIONS.md (the single source of truth
# for this project's rules) -- see CONVENTIONS.md section 3 for the check list.
#
# NOTE: every literal in this file is ASCII on purpose.  PowerShell 5.1 reads
# .ps1 files using the machine's ANSI/OEM code page, so non-ASCII literals can
# turn into mojibake on a Chinese (GBK) console.  Chinese explanations live in
# docs/CONVENTIONS.md; the checkers (Python) print their own Chinese messages.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -Full
#   powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -Base 3e4e763
#   powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -Fix
#
# Switches:
#   -Full       also run pytest + the retargeting pipeline (rule 5 baseline)
#   -Base <sha> baseline commit that holds the teammate's committed code
#               (rule 3); default 3e4e763 = the merge of the retargeting branch
#   -Fix        dot-source scripts\env_e_drive_cache.ps1 in this process before
#               checking the C: drive rule (process only -- NEVER writes the
#               User-level environment, that is the user's call)
#
# Exit code: 0 = no FAIL (WARNs are allowed); 1 = at least one FAIL
# ---------------------------------------------------------------------------
param(
    [switch]$Full,
    [switch]$Fix,
    [string]$Base = '3e4e763'
)

$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$repo = Split-Path -Parent $here
Set-Location $repo

$py = 'E:\python3.11.7\python.exe'

# Paths owned by the teammate (rule 3).  Only files that already exist in the
# baseline commit count as violations -- my own new files are fine.
$teamPaths = @('retargeting', 'config', 'input_adapters', 'tests',
               'inspect_angle_h5.py', 'main_offline_dual_teleop.py')

# Things that must never be committed (rule 7).
$forbiddenPrefix = @('lib/', 'linkerhand_sdk/', '.venv/', 'data/', 'outputs/', 'logs/')
$bigExt = @('.pt', '.pth', '.ckpt', '.onnx', '.mp4', '.avi', '.h5', '.hdf5', '.zip')
$maxNewFileMB = 5

$fail = 0
$warn = 0

function Head([string]$t) { Write-Host ''; Write-Host "=== $t ===" }
function Say([string]$m, [string]$t) { Write-Host ("{0} {1}" -f $m, $t) }
function Fail([string]$t) { Say '[FAIL]' $t; $script:fail++ }
function Warn([string]$t) { Say '[WARN]' $t; $script:warn++ }
function Pass([string]$t) { Say '[OK  ]' $t }

Write-Host ('=' * 78)
Write-Host 'preflight -- project conventions check (docs/CONVENTIONS.md)'
Write-Host ('=' * 78)
Write-Host "repo     : $repo"
Write-Host "baseline : $Base"

# --- 1. interpreter (rule 2) ----------------------------------------------
Head '1) interpreter: E:\python3.11.7\python.exe only (rule 2)'
if (Test-Path $py) {
    $ver = (& $py -c "import sys;print(sys.version.split()[0])" 2>&1) | Select-Object -Last 1
    Pass "$py  (python $ver)"
} else {
    Fail "$py not found -- fix the interpreter before anything else"
}
$cmd = Get-Command python -ErrorAction SilentlyContinue
if ($cmd) {
    if ($cmd.Source -like '*\.venv\*') {
        Fail "PATH 'python' resolves to $($cmd.Source) -- that .venv is an EMPTY shell, use the full path"
    } else {
        Say '[INFO]' "PATH 'python' resolves to $($cmd.Source)"
    }
}

# --- 2. GBK safety (rule 4) ----------------------------------------------
Head '2) GBK safety of .py sources (rule 4)'
if (Test-Path $py) {
    $gbkLog = Join-Path $env:TEMP "preflight_gbk_$PID.log"
    & cmd /c "`"$py`" scripts\check_gbk_safe.py --strict > `"$gbkLog`" 2>&1"
    $gbkExit = $LASTEXITCODE
    if ($gbkExit -eq 0) {
        Pass 'no non-GBK characters in .py'
    } else {
        Fail "check_gbk_safe.py --strict -> exit $gbkExit"
        Get-Content $gbkLog -Encoding UTF8 | Select-String 'U\+|FAIL' |
            Select-Object -First 8 | ForEach-Object { Write-Host "        $($_.Line)" }
        Write-Host "        full log: $gbkLog"
    }
}

# --- 3. C: drive zero-write (rule 1) -------------------------------------
Head '3) C: drive zero-write (rule 1)'
if ($Fix) {
    . (Join-Path $here 'env_e_drive_cache.ps1') | Out-Null
    Say '[INFO]' '-Fix: cache/temp vars redirected in this process only'
}
if (Test-Path $py) {
    $cdLog = Join-Path $env:TEMP "preflight_cdrive_$PID.log"
    & cmd /c "`"$py`" scripts\check_no_c_drive.py --strict > `"$cdLog`" 2>&1"
    $cdExit = $LASTEXITCODE
    $cdText = Get-Content $cdLog -Encoding UTF8
    if ($cdExit -eq 0) {
        $line = $cdText | Select-Object -Last 1
        Pass $(if ($line) { $line.Trim() } else { 'no C: drive write risk detected' })
    } else {
        Fail "check_no_c_drive.py --strict -> exit $cdExit (something would write to C:)"
        $cdText | Select-String 'FAIL' | Select-Object -First 8 |
            ForEach-Object { Write-Host "        $($_.Line)" }
        Write-Host '        fix: . .\scripts\env_e_drive_cache.ps1 -Persist'
    }
}

# --- 4. teammate code untouched (rule 3) ---------------------------------
Head "4) teammate's committed code untouched (rule 3, baseline $Base)"
& git rev-parse --verify --quiet "$Base^{commit}" > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Fail "baseline commit '$Base' not found -- pass -Base <sha> (or fetch the repo)"
} else {
    $baseFiles = @(& git ls-tree -r --name-only $Base -- $teamPaths)
    $diff = @(& git diff --name-status $Base -- $teamPaths)
    $viol = @()
    foreach ($ln in $diff) {
        if (-not $ln) { continue }
        $parts = $ln -split "`t"
        $status = $parts[0]
        $path = $parts[-1]
        if ($status -like 'A*') { continue }              # my own new file: allowed
        if ($status -like 'R*') { $viol += "$status $($parts[1]) -> $path"; continue }
        if ($baseFiles -contains $path) { $viol += "$status $path" }
    }
    if ($viol.Count -eq 0) {
        Pass "$($baseFiles.Count) teammate-owned files under baseline: 0 modified/deleted"
    } else {
        Fail "$($viol.Count) teammate-owned file(s) changed (rule 3 forbids this):"
        $viol | Select-Object -First 12 | ForEach-Object { Write-Host "        $_" }
        Write-Host '        fix: git checkout <baseline> -- <file>  and move the logic into your own new file'
    }
}

# --- 5. no big assets staged (rule 7) ------------------------------------
Head '5) no big assets staged for commit (rule 7)'
$st = @(& git status --porcelain)
$bad = @()
foreach ($ln in $st) {
    if ($ln.Length -lt 4) { continue }
    $path = $ln.Substring(3).Trim('"')
    if ($path -match ' -> ') { $path = ($path -split ' -> ')[-1].Trim('"') }
    $blocked = $null
    foreach ($p in $forbiddenPrefix) { if ($path.StartsWith($p)) { $blocked = $p; break } }
    if ($blocked) { $bad += "$path  (inside $blocked -- must stay out of git)"; continue }
    if (-not (Test-Path $path -PathType Leaf)) { continue }
    $ext = [IO.Path]::GetExtension($path).ToLower()
    $isBig = ((Get-Item $path).Length / 1MB) -gt $maxNewFileMB
    $kept = $path.StartsWith('robots/') -or $path.StartsWith('datasets/')
    if (-not $kept -and ($isBig -or ($bigExt -contains $ext))) {
        $mb = '{0:N1}' -f ((Get-Item $path).Length / 1MB)
        $bad += "$path  ($mb MB / $ext -- big asset, keep it out of git)"
    }
}
if ($bad.Count -eq 0) {
    Pass 'nothing forbidden and no big asset staged'
} else {
    Fail "$($bad.Count) item(s) should not be committed:"
    $bad | Select-Object -First 12 | ForEach-Object { Write-Host "        $_" }
}

# --- 6. rule files present (rule 12) -------------------------------------
Head '6) rule files present and in sync (rule 12)'
foreach ($f in @('.clinerules', 'docs\CONVENTIONS.md')) {
    if (Test-Path $f) { Pass "$f present" }
    else { Fail "$f missing -- conventions must live in docs/CONVENTIONS.md + .clinerules" }
}

# --- 7. full baseline (-Full, rule 5) ------------------------------------
if ($Full) {
    Head '7) full baseline: pytest + retargeting pipeline (rule 5)'
    $pytestLog = Join-Path $env:TEMP "preflight_pytest_$PID.log"
    & cmd /c "`"$py`" -m pytest tests -q > `"$pytestLog`" 2>&1"
    $pytestExit = $LASTEXITCODE
    $pytestTail = (Get-Content $pytestLog -Encoding UTF8 | Select-String 'passed|failed|error') |
        Select-Object -Last 1
    if ($pytestExit -eq 0) { Pass "pytest tests -q -> exit 0  [$($pytestTail.Line.Trim())]" }
    else { Fail "pytest tests -q -> exit $pytestExit  [$($pytestTail.Line.Trim())]"; Write-Host "        log: $pytestLog" }

    $pipeLog = Join-Path $env:TEMP "preflight_pipeline_$PID.log"
    & cmd /c "`"$py`" scripts\run_retargeting_pipeline.py > `"$pipeLog`" 2>&1"
    $pipeExit = $LASTEXITCODE
    $pass = (Get-Content $pipeLog -Encoding UTF8 | Select-String '\[PASS\]').Count
    $bad2 = (Get-Content $pipeLog -Encoding UTF8 | Select-String '\[FAIL\]').Count
    if ($pipeExit -eq 0 -and $bad2 -eq 0) { Pass "run_retargeting_pipeline.py -> exit 0  ($pass/7 PASS)" }
    else { Fail "run_retargeting_pipeline.py -> exit $pipeExit  (PASS=$pass FAIL=$bad2)"; Write-Host "        log: $pipeLog" }
} else {
    Head '7) full baseline (rule 5) -- SKIPPED'
    Say '[INFO]' 'run with -Full before and after any change (pytest + retargeting pipeline)'
}

# --- summary --------------------------------------------------------------
Write-Host ''
Write-Host ('-' * 78)
if ($fail -eq 0) {
    Write-Host "RESULT: [OK] all checks passed ($warn warning(s)) -- see docs/CONVENTIONS.md"
    exit 0
}
Write-Host "RESULT: [FAIL] $fail check(s) failed ($warn warning(s)) -- fix them first, see docs/CONVENTIONS.md"
exit 1

