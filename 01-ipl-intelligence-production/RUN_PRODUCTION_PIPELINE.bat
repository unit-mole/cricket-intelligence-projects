@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo IPL INTELLIGENCE PRODUCTION - CONTROLLED PIPELINE
echo ============================================================
call 00_SETUP_ENVIRONMENT.bat || exit /b 1
call CHECK_SYSTEM.bat || exit /b 1
call 01_IMPORT_V2_CHAMPION.bat || exit /b 1
call 02_VERIFY_PRODUCTION_PACKAGE.bat || exit /b 1
call 03_RUN_TESTS.bat || exit /b 1
echo.
echo Core production pipeline is complete.
echo Launch manually with 04_LAUNCH_LOCAL_APP.bat.
echo Deployment preparation is intentionally separate.
pause
