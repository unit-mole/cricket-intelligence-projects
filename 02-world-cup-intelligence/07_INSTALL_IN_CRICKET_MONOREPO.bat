@echo off
cd /d %~dp0
call .venv\Scripts\activate
python scripts\install_monorepo.py
pause
