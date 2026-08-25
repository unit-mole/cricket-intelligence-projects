@echo off
setlocal
cd /d "%~dp0"
if not exist artifacts\PRODUCTION_READY.flag (
  echo ERROR: Production is not verified. Run 02_VERIFY_PRODUCTION_PACKAGE.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\build_hf_bundle.py
if errorlevel 1 goto :fail
echo.
echo Hugging Face Space bundle ready under dist\huggingface_space\
echo Follow DEPLOYMENT_GUIDE.md when we are ready to publish.
pause
exit /b 0
:fail
echo ERROR: Hugging Face bundle creation failed.
pause
exit /b 1
