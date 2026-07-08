#!/usr/bin/env bash
# AquaShield AI — chạy app web (Linux/macOS)
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[AquaShield] Tạo môi trường ảo .venv ..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  echo "[AquaShield] Cài PyTorch bản CPU ..."
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  python -m pip install gradio pillow numpy matplotlib
else
  source .venv/bin/activate
fi

echo "[AquaShield] Khởi động app tại http://127.0.0.1:7860"
# Đặt SHARE=1 để tạo link công khai tạm thời: SHARE=1 ./run_app.sh
python app.py
