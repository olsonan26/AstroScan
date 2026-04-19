@echo off
echo ============================================
echo   AstroScan Installer for Windows
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

echo [3/4] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

echo [4/4] Setting up config...
if not exist config.yaml (
    copy config.example.yaml config.yaml
    echo Created config.yaml - please edit it with your OpenRouter API key!
)

:: Create directories
if not exist input mkdir input
if not exist output mkdir output
if not exist knowledge_base mkdir knowledge_base

echo.
echo ============================================
echo   Installation complete!
echo.
echo   Next steps:
echo   1. Edit config.yaml with your API key
echo   2. Drop book page photos in the input/ folder
echo   3. Run: python -m astroscan process
echo ============================================
pause
