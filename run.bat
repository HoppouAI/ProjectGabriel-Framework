@echo off
title Project Gabriel Supervisor
echo Starting Project Gabriel...

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found! Please run setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" supervisor.py

echo.
echo Application stopped.
pause
