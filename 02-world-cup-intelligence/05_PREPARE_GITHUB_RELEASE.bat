@echo off
cd /d %~dp0
call .venv\Scripts\activate
python scripts\prepare_github_release.py
pause
