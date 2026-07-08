@echo off
REM ============================================================
REM  AquaShield AI - Chay app web (Windows, 1 cham)
REM  Lan dau se tu tao moi truong ao va cai thu vien (~5-10 phut).
REM ============================================================
setlocal
cd /d "%~dp0"

echo [AquaShield] Kiem tra Python...
python --version >nul 2>&1
if errorlevel 1 (
  echo [LOI] Chua cai Python. Tai tai https://www.python.org/downloads/ ^(nho tick "Add to PATH"^).
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [AquaShield] Tao moi truong ao .venv ...
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  echo [AquaShield] Cai PyTorch ban CPU ...
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  echo [AquaShield] Cai thu vien con lai ...
  python -m pip install gradio pillow numpy matplotlib
) else (
  call ".venv\Scripts\activate.bat"
)

echo [AquaShield] Khoi dong app... Trinh duyet se mo http://127.0.0.1:7860
python app.py
pause
