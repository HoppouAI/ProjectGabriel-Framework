@echo off
setlocal
title Project Gabriel Setup

echo ===================================================
echo      Project Gabriel Framework Setup Wizard
echo ===================================================
echo.

:: Create bin directory for local uv installation
if not exist "bin" mkdir "bin"

:: Check if uv is already installed locally
if exist "bin\uv.exe" (
    echo [INFO] uv is already installed in bin\
) else (
    echo [INFO] Installing uv to local bin folder...
    :: Set install directory to local bin folder
    set "UV_INSTALL_DIR=%~dp0bin"
    :: Run the installer
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)

:: Add local bin to PATH for this session
set "PATH=%~dp0bin;%PATH%"

echo.
echo [INFO] Creating virtual environment with Python 3.13.3...
uv venv --python 3.13.3
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo                Hardware Selection
echo ===================================================
echo.
echo 1. NVIDIA GPU (Recommended for Vision) - Installs CUDA support
echo 2. CPU Only (Slower Vision)
echo.

:: Use choice command for robust input handling
choice /C 12 /M "Enter your choice"
set "user_choice=%errorlevel%"

echo.
echo [INFO] Installing base dependencies from requirements.txt...
uv pip install -r requirements.txt

:: Branch based on choice
if "%user_choice%"=="1" goto install_cuda
if "%user_choice%"=="2" goto install_cpu

:install_cuda
echo.
echo [INFO] Configuring for NVIDIA GPU (CUDA)...
echo [INFO] Uninstalling default torch...
uv pip uninstall torch torchvision torchaudio
echo [INFO] Installing CUDA-enabled torch...
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
goto finish

:install_cpu
echo.
echo [INFO] Keeping default CPU torch installation.
goto finish

:finish
echo.
echo ===================================================
echo               Setup Complete!
echo ===================================================
echo.
echo You can now run the AI using 'run.bat'.
echo.
pause
