@echo off
set PYTHONIOENCODING=utf-8
call .\.venv\Scripts\activate.bat
streamlit run gui/app.py
pause
