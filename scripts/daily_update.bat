@echo off
REM Daily price update — point Windows Task Scheduler at this file.
cd /d "%~dp0\.."
python scripts\daily_update.py
