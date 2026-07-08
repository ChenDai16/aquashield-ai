# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec cho AquaShield AI (bản desktop Tkinter).
# Build:  pyinstaller aquashield.spec   (chay tren Windows co Python + torch CPU)
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("model/resnet18_transfer.pt", "model"),
         ("sample_images", "sample_images")]
binaries = []
hiddenimports = collect_submodules("torchvision.models")

for pkg in ("torch", "torchvision"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# matplotlib (dung cho Grad-CAM colormap)
for pkg in ("matplotlib",):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

block_cipher = None

a = Analysis(
    ["app_desktop.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["gradio", "onnx", "onnxruntime"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AquaShieldAI",
    debug=False, strip=False, upx=True, console=False,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, name="AquaShieldAI",
)
