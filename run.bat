@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "project_python=%~dp0.venv\Scripts\python.exe"
if not exist "%project_python%" (
  echo [run:error] The isolated environment is missing: "%project_python%"
  echo [run:hint] Run setup.bat first. Use setup.bat --skip-models for a base install.
  exit /b 2
)

echo [run] Local server: http://127.0.0.1:7866
echo [run] Press Ctrl+C in this window to stop it.
"%project_python%" -m anime_picker serve --port 7866 %*
set "run_exit=%ERRORLEVEL%"
if not "%run_exit%"=="0" (
  echo [run:error] Server exited with code %run_exit%.
  echo [run:hint] Run .venv\Scripts\python.exe scripts\doctor.py --skip-models for runtime diagnostics.
)
exit /b %run_exit%
