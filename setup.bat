@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem setup.bat always uses the repository bootstrap, so cwd and Python package
rem state stay deterministic when it is launched from Explorer or another dir.
where py >nul 2>&1
if errorlevel 1 (
  echo [setup:error] Python Launcher ^(py.exe^) was not found.
  echo [setup:hint] Install Python 3.12 for Windows, then rerun setup.bat.
  exit /b 2
)

py -3.12 scripts\bootstrap.py %*
set "setup_exit=%ERRORLEVEL%"
if not "%setup_exit%"=="0" (
  echo [setup:error] Setup failed with exit code %setup_exit%.
  echo [setup:hint] See .cache\bootstrap.log for the complete command output.
)
exit /b %setup_exit%
