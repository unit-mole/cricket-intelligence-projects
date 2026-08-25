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
echo IMPORTING THE ACCEPTED V2 CHAMPION
echo This does NOT retrain or modify IPL_Intelligence_Lab_V2.
echo ============================================================
python scripts\import_v2_champion.py
if errorlevel 1 goto :fail
echo.
echo Next: run 02_VERIFY_PRODUCTION_PACKAGE.bat
pause
exit /b 0
:fail
echo.
echo ERROR: V2 champion import failed. Read the message above.
pause
exit /b 1
