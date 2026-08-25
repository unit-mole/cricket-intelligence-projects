@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo IPL INTELLIGENCE PRODUCTION - ENVIRONMENT SETUP
echo Frozen V2 champion; no model retraining in this repository.
echo ============================================================
where py >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python launcher ^(py^) was not found.
  pause
  exit /b 1
)
if not exist .venv (
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail
python -m pip install -e .
if errorlevel 1 goto :fail
echo.
echo ============================================================
echo PRODUCTION ENVIRONMENT READY.
echo Next: run CHECK_SYSTEM.bat, then 01_IMPORT_V2_CHAMPION.bat.
echo ============================================================
pause
exit /b 0
:fail
echo.
echo ERROR: Environment setup failed.
pause
exit /b 1
