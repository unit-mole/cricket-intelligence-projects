@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo ERROR: Run 00_SETUP_ENVIRONMENT.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pytest -q
pause
