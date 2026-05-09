@echo off
REM Launch the Streamlit dashboard in your browser.
cd /d "%~dp0\.."
streamlit run src\app.py
