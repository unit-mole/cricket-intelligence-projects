@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo ERROR: Run 00_SETUP_ENVIRONMENT.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo ============================================================
echo VERIFYING FROZEN V2 PRODUCTION ASSETS
echo Checks metrics, strict rows, checksums, bundle metadata,
echo runtime predictions, and probability symmetry.
echo ============================================================
python scripts\verify_production.py
if errorlevel 1 goto :fail
echo.
echo Production verification PASS. Next: 03_RUN_TESTS.bat
pause
exit /b 0
:fail
echo.
echo PRODUCTION VERIFICATION FAILED. App remains locked.
pause
exit /b 1
