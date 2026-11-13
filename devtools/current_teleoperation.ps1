#requires -Version 5.1
<#
Operation examples for runtime revision 44908ee; no retargeting algorithms here.
See docs/current-teleoperation-usage-report-2026-11-13.md for current limitations.
Use -DryRun to print the exact argument array without opening devices or writing.
#>
[CmdletBinding()]
param(
    [ValidateSet('Check','Fit','RecordBody','CameraArms','CameraBoth','CameraHands',
                 'VisionProHands','VisionProBoth','BodyReplay','CommandReplay')]
    [string]$Mode = 'Check',
    [ValidateSet('h1_2','gr1_t2','g1','tron2a_l21')]
    [string[]]$Robots = @('g1'),
    [ValidateSet('left','right','both')][string]$Side = 'both',
    [ValidateSet('dex','nn')][string]$Backend = 'dex',
    [hashtable]$HandCheckpoints = @{},
    [string]$InputPath,
    [string]$VisionProIp,
    [ValidateSet('arms','both')][string]$BodyParts = 'arms',
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe,
    [string]$FittingDirectory,
    [string]$OutputDirectory,
    [ValidateRange(0,2147483647)][int]$Frames = 300,
    [ValidateRange(0,2147483647)][int]$CameraIndex = 0,
    [ValidateRange(50,240)][double]$ControlHz = 100,
    [ValidateSet('full','simplified')][string]$Geometry = 'full',
    [switch]$Headless,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RepositoryRoot)
if (-not $PythonExe) { $PythonExe = Join-Path $RepositoryRoot '.venv/dex-native/Scripts/python.exe' }
if (-not $FittingDirectory) {
    $FittingDirectory = Join-Path $RepositoryRoot 'outputs/acceptance/unified-smpl-pink/fittings-v2'
}
if (-not $OutputDirectory) {
    $operationName = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + ([guid]::NewGuid().ToString('N').Substring(0,6))
    $OutputDirectory = Join-Path $RepositoryRoot ('outputs/operation-runs/' + $operationName)
}
# Resolve caller-relative paths before temporarily entering the repository.
$OutputDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
$PythonExe = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PythonExe)
$FittingDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($FittingDirectory)
if ($InputPath) { $InputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InputPath) }
if (-not $Robots.Count -or ($Robots | Select-Object -Unique).Count -ne $Robots.Count) {
    throw 'Select distinct robot names.'
}
$selector = if ($Robots.Count -eq 1) { @('--robot', $Robots[0]) } else { @('--robots') + $Robots }
$workerOptions = @(
    '--smplerx-worker-python', (Join-Path $RepositoryRoot '.venv/smplerx-worker/Scripts/python.exe'),
    '--smplerx-root', (Join-Path $RepositoryRoot '.venv/smplerx-assets/SMPLer-X'),
    '--smplerx-checkpoint', (Join-Path $RepositoryRoot '.venv/smplerx-assets/smpler_x_s32.pth.tar'),
    '--smplx-model-dir', (Join-Path $RepositoryRoot 'datasets/smpl_models')
)
$handAsset = Join-Path $RepositoryRoot 'datasets/hand_landmarker.task'
$teleopArgs = @('-m','teleoperation')

switch ($Mode) {
    'Check' { $teleopArgs += @('hand','list-models') }
    'Fit' {
        $teleopArgs += @('arm','fit','--robots') + $Robots + @(
            '--body-model-path', (Join-Path $RepositoryRoot 'datasets/smpl_models'),
            '--body-model-type','smplx','--gender','neutral','--num-betas','20',
            '--fit-steps','300','--device','cpu',
            '--output-dir', (Join-Path $OutputDirectory 'fittings'),
            '--report', (Join-Path $OutputDirectory 'fit-report.json'))
    }
    'RecordBody' {
        if ($Frames -eq 0) { throw 'RecordBody requires a positive frame count; use q/Ctrl+C to stop early.' }
        $teleopArgs += @('hand','record-world','--frames',"$Frames",'--max-fps','0',
            '--fps','30','--width','640','--height','480','--camera-index',"$CameraIndex",
            '--model-asset-path',$handAsset,'--startup-timeout','120',
            '--output', (Join-Path $OutputDirectory 'body.h5')) + $workerOptions
        if ($Headless) { $teleopArgs += '--no-preview' }
    }
    'CommandReplay' {
        if (-not $InputPath) { throw 'CommandReplay requires -InputPath pointing to accepted command H5.' }
        $teleopArgs += @('robot','replay') + $selector + @('--input',$InputPath,
            '--geometry',$Geometry,'--report',(Join-Path $OutputDirectory 'replay-report.json'))
        if ($Headless) { $teleopArgs += '--headless' }
    }
    default {
        $parts = switch ($Mode) {
            'CameraArms' { 'arms' }
            'CameraBoth' { 'both' }
            'CameraHands' { 'hands' }
            'VisionProHands' { 'hands' }
            'VisionProBoth' { 'both' }
            'BodyReplay' { $BodyParts }
        }
        $teleopArgs += @('robot','control') + $selector + @('--parts',$parts,'--side',$Side,
            '--backend',$Backend,'--frames',"$Frames",'--geometry',$Geometry,
            '--record-results',(Join-Path $OutputDirectory 'commands.h5'),
            '--report',(Join-Path $OutputDirectory 'control-report.json'))
        if ($Mode -like 'VisionPro*') {
            if ([bool]$InputPath -eq [bool]$VisionProIp) { throw 'Choose exactly one of -InputPath or -VisionProIp.' }
            if ($Mode -eq 'VisionProBoth' -and $Robots -contains 'tron2a_l21') {
                throw 'TRON2A VR arms are unsupported; use VisionProHands for that robot.'
            }
            $teleopArgs += @('--source','visionpro')
            if ($InputPath) { $teleopArgs += @('--input-h5',$InputPath) }
            else { $teleopArgs += @('--ip',$VisionProIp) }
        } else {
            $teleopArgs += @('--source','camera')
            if ($Mode -eq 'BodyReplay') {
                if (-not $InputPath) { throw 'BodyReplay requires -InputPath pointing to raw SMPL-X observation H5.' }
                $teleopArgs += @('--input-body',$InputPath)
            } else {
                $teleopArgs += @('--camera-index',"$CameraIndex",'--model-asset-path',$handAsset)
                if ($parts -ne 'hands') { $teleopArgs += $workerOptions }
            }
            if ($parts -ne 'hands') {
                $teleopArgs += @('--arm-fitting-dir',$FittingDirectory,'--body-device','cpu',
                    '--control-hz',$ControlHz.ToString([Globalization.CultureInfo]::InvariantCulture),
                    '--target-max-age','0.3','--body-startup-timeout','120')
            }
        }
        foreach ($model in ($HandCheckpoints.Keys | Sort-Object)) {
            $weightPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath([string]$HandCheckpoints[$model])
            $teleopArgs += @('--hand-checkpoint',"$model",$weightPath)
        }
        if ($Headless) { $teleopArgs += '--headless' }
    }
}

if ($DryRun) {
    [pscustomobject]@{python=$PythonExe;working_directory=$RepositoryRoot;
        arguments=$teleopArgs;output_directory=$OutputDirectory} | ConvertTo-Json -Depth 5
    return
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot 'src/teleoperation/cli'))) {
    throw "Repository source not found: $RepositoryRoot"
}
if ($Mode -eq 'Check') {
    @($PythonExe,$handAsset,$workerOptions[1],$workerOptions[3],$workerOptions[5],$workerOptions[7],$FittingDirectory) |
        ForEach-Object { [pscustomobject]@{exists=(Test-Path -LiteralPath $_);path=$_} } | Format-Table -AutoSize
} else {
    if (Test-Path -LiteralPath $OutputDirectory) { throw "Choose a fresh output directory: $OutputDirectory" }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    Write-Host "Run output: $OutputDirectory"
}
$savedPythonPath = $env:PYTHONPATH
$savedBytecode = $env:PYTHONDONTWRITEBYTECODE
$runtimePaths = @('src','.', 'outputs/unified-arm-runtime',
    'outputs/acceptance/teleopbench-gr1-20261111/dependencies') | ForEach-Object { Join-Path $RepositoryRoot $_ }
$env:PYTHONPATH = ($runtimePaths + @($savedPythonPath) | Where-Object { $_ }) -join [IO.Path]::PathSeparator
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location $RepositoryRoot
try {
    & $PythonExe @teleopArgs
    if ($LASTEXITCODE -ne 0) { throw "Teleoperation exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:PYTHONPATH = $savedPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $savedBytecode
}
