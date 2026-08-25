@echo off
setlocal
cd /d "%~dp0"
if not exist artifacts\PRODUCTION_READY.flag (
  echo ERROR: Production is not verified. Run 02_VERIFY_PRODUCTION_PACKAGE.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\prepare_github_release.py
if errorlevel 1 goto :fail
echo.
echo GitHub release audit complete. See dist\github_release\.
pause
exit /b 0
:fail
echo ERROR: GitHub release preparation failed.
pause
exit /b 1
