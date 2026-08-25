@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo ERROR: .venv not found. Run 00_SETUP_ENVIRONMENT.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\check_environment.py
pause
