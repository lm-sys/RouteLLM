@echo off
setlocal
cd /d "%~dp0"

set VENV_PYTHON=%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe
"%VENV_PYTHON%" start.py %*
