# Process-only D: caches and the isolated dex-native interpreter.
$dexRepo = Split-Path -Parent $PSScriptRoot
$dexOutput = Join-Path $dexRepo 'outputs/tmp_dex_native'
$env:PYTHONPATH = Join-Path $dexRepo 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = $dexRepo.Replace('\', '/')
foreach ($dexEntry in @{TEMP='tmp/lib'; TMP='tmp/lib'; MPLCONFIGDIR='matplotlib'; TORCH_HOME='torch'; HF_HOME='huggingface'; XDG_CACHE_HOME='xdg'; PIP_CACHE_DIR='pip/lib'}.GetEnumerator()) {
    $dexPath = Join-Path $dexOutput ('cache/' + $dexEntry.Value)
    New-Item -ItemType Directory -Force -Path $dexPath | Out-Null
    Set-Item -Path ('Env:' + $dexEntry.Key) -Value $dexPath
}
$dexPython = Join-Path $dexRepo '.venv/dex-native/Scripts/python.exe'
