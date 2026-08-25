@echo off
cd /d %~dp0
call .venv\Scripts\activate
python scripts\check_environment.py
pause
