@echo off
cd /d %~dp0
call .venv\Scripts\activate
python scripts\import_final_components.py
pause
