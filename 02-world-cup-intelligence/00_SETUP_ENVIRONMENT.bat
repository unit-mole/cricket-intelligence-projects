@echo off
cd /d %~dp0
if not exist .venv py -3.12 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
echo.
echo ============================================================
echo ICC World Cup Intelligence Production environment ready.
echo ============================================================
pause
