@echo off
REM ============================================================
REM  AquaShield AI - Dong goi thanh file .exe (Windows)
REM  Tao ra: dist\AquaShieldAI\AquaShieldAI.exe
REM  Luu y: exe se lon (~1.5-2.5 GB) vi kem PyTorch. Can ~6 GB trong khi build.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [Buoc 1] Tao moi truong ao va cai thu vien...
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  python -m pip install pillow numpy matplotlib pyinstaller
) else (
  call ".venv\Scripts\activate.bat"
  python -m pip install pyinstaller >nul 2>&1
)

echo [Buoc 2] Dong goi bang PyInstaller...
pyinstaller aquashield.spec --noconfirm

echo.
echo [XONG] File chay: dist\AquaShieldAI\AquaShieldAI.exe
echo Ban co the nen ca thu muc dist\AquaShieldAI de chia se.
pause
