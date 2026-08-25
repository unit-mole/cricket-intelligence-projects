@echo off
cd /d %~dp0
call .venv\Scripts\activate
python scripts\verify_production.py
pause
