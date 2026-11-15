# Current repository checks. Run in an activated environment, or pass -PythonExe.
param(
    [switch]$Full,
    [string]$PythonExe = 'python'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$runId = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$logRoot = Join-Path $repo "outputs/preflight/$runId"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$failed = 0
$checked = 0
function Invoke-Check([string]$Name, [string[]]$Arguments) {
    $script:checked++
    $log = Join-Path $logRoot "$Name.log"
    # Windows PowerShell treats native stderr as an ErrorRecord. Preserve it in
    # the log and use the native exit code to decide whether the check passed.
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $PythonExe -B @Arguments *> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($code -eq 0) {
        Write-Host "[OK] $Name (log: $log)"
    } else {
        $script:failed++
        Write-Host "[FAIL] $Name (exit $code, log: $log)"
        Get-Content -LiteralPath $log -Tail 20
    }
}
$savedPath = $env:PYTHONPATH
$savedBytecode = $env:PYTHONDONTWRITEBYTECODE
$savedUtf8 = $env:PYTHONUTF8
Push-Location $repo
try {
    $env:PYTHONPATH = (@((Join-Path $repo 'src'), $repo, $savedPath) | Where-Object { $_ }) -join [IO.Path]::PathSeparator
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUTF8 = '1'
    Invoke-Check 'dependencies' @('-c', 'import numpy, pybullet, pinocchio, pink')
    Invoke-Check 'cli-help' @('-m', 'teleoperation', '--help')
    if ($Full) {
        Invoke-Check 'unittest' @('-m', 'unittest', 'discover', '-s', 'tests', '-q')
        Invoke-Check 'simulation-audit' @('devtools/audit_task_criteria.py', '--json', (Join-Path $logRoot 'task-audit.json'))
    } else {
        Write-Host '[SKIP] full unittest suite and 30-task simulation audit; pass -Full to run both'
    }
} finally {
    Pop-Location
    $env:PYTHONPATH = $savedPath
    $env:PYTHONDONTWRITEBYTECODE = $savedBytecode
    $env:PYTHONUTF8 = $savedUtf8
}
Write-Host "Checks: $checked; failures: $failed; logs: $logRoot"
if ($failed) { exit 1 }
exit 0
