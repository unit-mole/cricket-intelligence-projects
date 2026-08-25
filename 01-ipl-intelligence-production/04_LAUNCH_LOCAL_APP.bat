@echo off
setlocal
cd /d "%~dp0"
if not exist artifacts\PRODUCTION_READY.flag (
  echo ============================================================
  echo APP LOCKED
  echo Production assets have not passed verification.
  echo Run 01_IMPORT_V2_CHAMPION.bat and 02_VERIFY_PRODUCTION_PACKAGE.bat.
  echo ============================================================
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python app.py
