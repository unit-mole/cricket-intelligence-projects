@echo off
cd /d %~dp0
if not exist artifacts\PRODUCTION_READY.flag (
 echo Production package is not verified. Run 01_IMPORT_FINAL_COMPONENTS.bat and 02_VERIFY_PRODUCTION_PACKAGE.bat first.
 pause
 exit /b 1
)
call .venv\Scripts\activate
python app.py
