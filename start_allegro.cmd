@echo off
pushd "%~dp0"
"%~dp0.conda-env\python.exe" -I -B "%~dp0scripts\launch_demo.py" %*
set "dex_demo_exit=%errorlevel%"
popd
exit /b %dex_demo_exit%
