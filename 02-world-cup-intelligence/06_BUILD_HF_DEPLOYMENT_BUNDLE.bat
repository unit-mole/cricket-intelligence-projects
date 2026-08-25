@echo off
cd /d %~dp0
call .venv\Scripts\activate
python scripts\build_hf_bundle.py
pause
