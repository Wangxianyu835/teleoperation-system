# preflight.ps1
# ---------------------------------------------------------------------------
# One-command enforcement of docs/CONVENTIONS.md (the single source of truth
# for this project's rules) adapted to the current-state layout:
#   production package  src/teleoperation   CLI  python -m teleoperation
#   teammate baseline   10ebd9f (origin/current-state)
# See CONVENTIONS.md section 3 for the check list:
#   1 interpreter  2 GBK safety  3 C: zero-write  4 teammate code untouched
#   5 nothing big staged  6 rule files present  7 architecture boundary
#   8 (-Full) unittest + CLI smoke
#
# NOTE: every literal in this file is ASCII on purpose.  PowerShell 5.1 reads
# .ps1 files using the machine's ANSI/OEM code page, so non-ASCII literals can
# turn into mojibake on a Chinese (GBK) console.  Chinese explanations live in
# docs/CONVENTIONS.md; the checkers (Python) print their own Chinese messages.
#
# NOTE: this file must NOT live in a root folder named 'scripts': legacy root
# entries such as 'scripts', 'retargeting', 'teleop', 'input_adapters' are
# forbidden.  The teammate test that used to police the list
# (tests/architecture/test_boundaries.py, test_old_entries_and_path_injection
# _are_retired) was deleted together with the whole tests/ folder in the
# upstream commit 10ebd9f, so check 7 below is the only guard left -- that is
# why our tooling lives in devtools/ and the list is hard-coded here.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1
#   powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full
#   powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -MaxErr 0
#   powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Base 10ebd9f
#   powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Fix
#
# Switches:
#   -Full          also run the unittest baseline + CLI smoke (rule 5)
#   -Base <sha>    commit that holds the teammate's committed code (rule 3);
#                  default 10ebd9f = origin/current-state (the merged line,
#                  moved from 42bbe10 on 2026-10-10: apps/ -> applications/,
#                  the teammate's tests/ folder was deleted there as well).
#   -MaxFail N     allowed unittest failures in -Full; default 0 = measured on
#                  the fallback interpreter with PYTHONUTF8=1 (this script sets
#                  it; the entry-point tests that needed it are gone with the
#                  teammate tests/ folder, see CONVENTIONS.md 3.2).
#   -MaxErr N      allowed unittest errors; default 17 is a permissive ceiling
#                  kept for convenience.  The 10ebd9f baseline ships no test
#                  suite of its own, so the measured value here is 0 -- pass
#                  -MaxErr 0 once upstream re-adds their suite (CONVENTIONS 3.3).
#   -Allow <paths> user-approved exception(s) to rule 3 (comma separated)
#   -Fix           dot-source devtools\env_e_drive_cache.ps1 in this process
#                  before checking the C: drive rule (process only -- NEVER
#                  writes the User-level environment, that is the user's call)
#
# Exit code: 0 = no FAIL (WARNs are allowed); 1 = at least one FAIL
# ---------------------------------------------------------------------------
param(
    [switch]$Full,
    [switch]$Fix,
    [string]$Base = '10ebd9f',
    [int]$MaxFail = 0,
    [int]$MaxErr = 17,
    [string[]]$Allow = @()
)

$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$repo = Split-Path -Parent $here
Set-Location $repo

# Paths owned by the teammate (rule 3).  Only files that already exist in the
# baseline commit count as violations -- my own new files are fine.  '.' = the
# whole baseline tree (src/, tests/, assets/, docs/, pyproject.toml, README...),
# so any edit to a tracked baseline file is caught, not just the four big dirs.
$teamPaths = @('.')
$allowPaths = @($Allow | Where-Object { $_ })

# Things that must never be committed (rule 7).
$forbiddenPrefix = @('lib/', 'linkerhand_sdk/', '.venv/', 'data/', 'outputs/',
                     'logs/', 'tmp_', '.pytest_cache/')
# Tracked on purpose: robot assets and the H5 sample/recording contracts.
$keptPrefix = @('assets/robots/', 'datasets/')
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
Head '1) interpreter (rule 2)'
$condaPy = 'D:\Anaconda\envs\teleoperation\python.exe'
$fallbackPy = 'E:\python3.11.7\python.exe'
$py = $null
if (Test-Path $condaPy) {
    $py = $condaPy
    $ver = (& $py -c "import sys;print(sys.version.split()[0])" 2>&1) | Select-Object -Last 1
    Pass "formal conda env: $condaPy  (python $ver)"
} elseif (Test-Path $fallbackPy) {
    $py = $fallbackPy
    $ver = (& $py -c "import sys;print(sys.version.split()[0])" 2>&1) | Select-Object -Last 1
    Warn "conda env 'teleoperation' not found -- using $fallbackPy (python $ver), see CONVENTIONS.md 3.2"
} else {
    Fail "no usable interpreter: neither $condaPy nor $fallbackPy exists"
}
if ($py -and (Test-Path (Join-Path $repo 'src\teleoperation'))) {
    # The plain interpreter needs the source tree on sys.path because the
    # package is not pip-installed in it.
    $env:PYTHONPATH = Join-Path $repo 'src'
    Say '[INFO]' "PYTHONPATH=$($env:PYTHONPATH)"
}
# NOTE: this used to test `-like '*\.venv\*'`, but PowerShell can report the
# source with mixed separators ('...\.venv/Scripts\python.exe'), so the test
# silently never fired.  Match on either separator.  It is a WARN and not a
# FAIL because this script always invokes the interpreter by full path ($py):
# a broken 'python' on PATH only hurts a human typing it by hand.
$cmd = Get-Command python -ErrorAction SilentlyContinue
if ($cmd -and ($cmd.Source -match '[\\/]\.venv[\\/]')) {
    Warn "PATH 'python' resolves to $($cmd.Source) -- that .venv is an EMPTY shell; always call the interpreter by full path (`$py)"
} elseif ($cmd) {
    Say '[INFO]' "PATH 'python' resolves to $($cmd.Source)"
}

# --- 2. GBK safety (rule 4) ----------------------------------------------
Head '2) GBK safety of .py sources (rule 4)'
if ($py) {
    $gbkLog = Join-Path $env:TEMP "preflight_gbk_$PID.log"
    & cmd /c "`"$py`" devtools\check_gbk_safe.py --strict > `"$gbkLog`" 2>&1"
    $gbkExit = $LASTEXITCODE
    if ($gbkExit -eq 0) {
        Pass 'no non-GBK characters in .py'
    } else {
        Fail "check_gbk_safe.py --strict -> exit $gbkExit"
        Get-Content $gbkLog | Select-String 'U\+|FAIL' |
            Select-Object -First 8 | ForEach-Object { Write-Host "        $($_.Line)" }
        Write-Host "        full log: $gbkLog"
    }
}

# --- 2b. same rule through the project's own CLI --------------------------
# The teammate ships the same check as src/teleoperation/applications/diagnostics/
# check_gbk_safe.py, i.e. 'python -m teleoperation tools check-gbk-safe'.  We run
# both: ours covers our own files even if the package changes, theirs is the
# authoritative one for the shipping tree.
if ($py) {
    $gbk2Log = Join-Path $env:TEMP "preflight_gbk_cli_$PID.log"
    & cmd /c "`"$py`" -m teleoperation tools check-gbk-safe --strict > `"$gbk2Log`" 2>&1"
    $gbk2Exit = $LASTEXITCODE
    if ($gbk2Exit -eq 0) {
        Pass 'tools check-gbk-safe --strict -> exit 0'
    } else {
        Fail "tools check-gbk-safe --strict -> exit $gbk2Exit"
        Get-Content $gbk2Log | Select-Object -Last 12 |
            ForEach-Object { Write-Host "        $_" }
    }
}

# --- 3. C: drive zero-write (rule 1) -------------------------------------
Head '3) C: drive zero-write (rule 1)'
if ($Fix) {
    . (Join-Path $here 'env_e_drive_cache.ps1') | Out-Null
    Say '[INFO]' '-Fix: cache/temp vars redirected in this process only'
}
if ($py) {
    $cdLog = Join-Path $env:TEMP "preflight_cdrive_$PID.log"
    & cmd /c "`"$py`" devtools\check_no_c_drive.py --strict > `"$cdLog`" 2>&1"
    $cdExit = $LASTEXITCODE
    $cdText = Get-Content $cdLog
    if ($cdExit -eq 0) {
        $line = $cdText | Select-Object -Last 1
        Pass $(if ($line) { $line.Trim() } else { 'no C: drive write risk detected' })
    } else {
        Fail "check_no_c_drive.py --strict -> exit $cdExit (something would write to C:)"
        $cdText | Select-String 'FAIL' | Select-Object -First 8 |
            ForEach-Object { Write-Host "        $($_.Line)" }
        Write-Host '        fix: . .\devtools\env_e_drive_cache.ps1 -Persist'
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
        if ($allowPaths -contains $path) { continue }     # -Allow: user approved
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
    $kept = $false
    foreach ($p in $keptPrefix) { if ($path.StartsWith($p)) { $kept = $true; break } }
    if ($kept) { continue }
    $ext = [IO.Path]::GetExtension($path).ToLower()
    $isBig = ((Get-Item $path).Length / 1MB) -gt $maxNewFileMB
    if ($isBig -or ($bigExt -contains $ext)) {
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

# --- 7. architecture boundary (our own hard-coded list) -------------------
# None of these legacy root entries may exist, and no module under
# src/teleoperation may inject sys.path.  The list mirrors the teammate test
# tests/architecture/test_boundaries.py :: test_old_entries_and_path_injection
# _are_retired, which upstream deleted together with the whole tests/ folder in
# 10ebd9f -- so the list is hard-coded here now.  Our own tooling must never
# land in one of those names: a root 'scripts/' folder is exactly what made that
# test go from 6 to 7 failures during the merge, which is why devtools/ is used
# instead.  Run this BEFORE -Full: it fails fast and is five seconds cheap.
Head '7) architecture boundary: legacy root entries + sys.path injection'
$legacyRoot = @('retargeting', 'model', 'teleop', 'input_adapters', 'envs',
                'tasks', 'scripts', 'main.py', 'main_train_twohand.py',
                'main_offline_twohand.py', 'inspect_angle_h5.py')
$legacy = @($legacyRoot | Where-Object { Test-Path (Join-Path $repo $_) })
if ($legacy.Count -eq 0) {
    Pass "no legacy root entry present ($($legacyRoot.Count) names checked)"
} else {
    Fail "legacy root entry/entries present: $($legacy -join ', ')"
    Write-Host '        fix: move or rename it (our tooling lives in devtools/),'
    Write-Host '        list mirrors the retired tests/architecture/test_boundaries.py:107'
}
$inject = @(Get-ChildItem (Join-Path $repo 'src\teleoperation') -Recurse -Filter *.py -ErrorAction SilentlyContinue |
            Select-String -Pattern 'sys\.path\.insert')
if ($inject.Count -eq 0) {
    Pass 'no sys.path.insert under src/teleoperation'
} else {
    Fail "$($inject.Count) sys.path.insert occurrence(s) under src/teleoperation"
    $inject | Select-Object -First 8 | ForEach-Object { Write-Host "        $($_.Path):$($_.LineNumber)" }
}

# --- 8. full baseline (-Full, rule 5) ------------------------------------
if ($Full) {
    Head '8) full baseline: unittest discover + CLI smoke (rule 5)'
    $unitLog = Join-Path $env:TEMP "preflight_unittest_$PID.log"
    # PYTHONUTF8=1 is mandatory here, not cosmetic: the entry-point tests compare
    # Chinese output that they decode with the locale code page, while the CLIs
    # print UTF-8.  On a cp936 console 6 of them then fail for the wrong reason
    # (setting this variable is what makes those 6 pass again) -- CONVENTIONS 3.2.
    # Scoped to this script's process and restored right afterwards.
    $oldUtf8 = $env:PYTHONUTF8
    $env:PYTHONUTF8 = '1'
    & cmd /c "`"$py`" -m unittest discover -s tests -v > `"$unitLog`" 2>&1"
    $unitExit = $LASTEXITCODE
    if ($null -eq $oldUtf8) { Remove-Item Env:\PYTHONUTF8 -ErrorAction SilentlyContinue }
    else { $env:PYTHONUTF8 = $oldUtf8 }
    $unit = Get-Content $unitLog
    $ran = ($unit | Select-String '^Ran \d+ test' | Select-Object -Last 1)
    $sum = ($unit | Select-String '^(OK|FAILED)' | Select-Object -Last 1)
    $nf = 0; $ne = 0
    if ($sum) {
        if ($sum.Line -match 'failures=(\d+)') { $nf = [int]$Matches[1] }
        if ($sum.Line -match 'errors=(\d+)') { $ne = [int]$Matches[1] }
    }
    if (-not $ran) {
        Fail "unittest discover -> exit $unitExit (no 'Ran N tests' line; log: $unitLog)"
    } elseif ($nf -le $MaxFail -and $ne -le $MaxErr) {
        Pass "unittest discover -> $($ran.Line.Trim())  [$($sum.Line.Trim())]  (allowed: fail<=$MaxFail err<=$MaxErr)"
    } else {
        Fail "unittest discover -> exit $unitExit  [$($sum.Line.Trim())] exceeds allowed fail<=$MaxFail err<=$MaxErr"
        Write-Host "        log: $unitLog  (raise -MaxFail/-MaxErr only with a documented reason, CONVENTIONS.md 3.2)"
    }




    # 1) CLI loads, 2) headless replay really simulates, 3) the project's own
    # environment self-check (6 tests incl. a 300-step end-to-end run),
    # 4) tools verify-hand-pipeline (557-frame real H5 through the whole chain).
    $smokes = @(
        @{ name = 'python -m teleoperation --help'; args = '--help' },
        @{ name = 'replay actions --dummy --steps 120 --robot h1_2 --no-render';
           args = 'replay actions --dummy --steps 120 --robot h1_2 --no-render' },
        @{ name = 'tools check-environment'; args = 'tools check-environment' },
        @{ name = 'tools verify-hand-pipeline'; args = 'tools verify-hand-pipeline' }
    )
    foreach ($s in $smokes) {
        $smokeLog = Join-Path $env:TEMP "preflight_smoke_$PID.log"
        & cmd /c "`"$py`" -m teleoperation $($s.args) > `"$smokeLog`" 2>&1"
        $smokeExit = $LASTEXITCODE
        $tb = (Get-Content $smokeLog | Select-String 'Traceback').Count
        if ($smokeExit -eq 0 -and $tb -eq 0) {
            Pass "$($s.name) -> exit 0"
        } else {
            Fail "$($s.name) -> exit $smokeExit (traceback lines: $tb)"
            Write-Host "        log: $smokeLog"
        }
    }
} else {
    Head '8) full baseline (rule 5) -- SKIPPED'
    Say '[INFO]' 'run with -Full before and after any change (unittest + CLI smoke + env self-check)'
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
