@echo off
REM ============================================================
REM  AquaShield AI - Mobile: mo tren dien thoai (cung mang Wi-Fi)
REM  Chay file nay tren may tinh, roi mo dia chi hien ra bang
REM  trinh duyet tren dien thoai (Chrome/Safari).
REM ============================================================
setlocal
cd /d "%~dp0"

echo [AquaShield Mobile] Dia chi IP LAN cua may nay:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo    http://%%a:8000/aquashield_mobile.html
echo.
echo Mo mot trong cac dia chi tren bang trinh duyet DIEN THOAI (cung Wi-Fi).
echo Nhan Ctrl+C de dung server.
echo.
python -m http.server 8000
pause
