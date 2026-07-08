# -*- coding: utf-8 -*-
"""
export_onnx.py — Xuất mô hình sang ONNX cho bản mobile / triển khai on-device.
Tạo hai file trong ./mobile:
    resnet18_shrimp.onnx        (FP32, ~45 MB)
    resnet18_shrimp_int8.onnx   (lượng tử hoá động, ~11 MB — dùng cho mobile)

CÁCH CHẠY:
    pip install onnx onnxruntime
    python export_onnx.py

Bản int8 đã kèm sẵn trong ./mobile — chỉ chạy lại khi bạn train lại model.
"""
import os
import torch, torch.nn as nn
import torchvision

WEIGHTS = os.path.join("model", "resnet18_transfer.pt")
OUT_DIR = "mobile"
os.makedirs(OUT_DIR, exist_ok=True)
FP32 = os.path.join(OUT_DIR, "resnet18_shrimp.onnx")
INT8 = os.path.join(OUT_DIR, "resnet18_shrimp_int8.onnx")


class Wrapped(nn.Module):
    """ResNet-18 + Softmax để ONNX xuất trực tiếp xác suất (output tên 'probs')."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


def main():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    ck = torch.load(WEIGHTS, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    m.load_state_dict(state, strict=False)
    m.eval()

    model = Wrapped(m).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, FP32,
        input_names=["input"], output_names=["probs"],
        dynamic_axes={"input": {0: "N"}, "probs": {0: "N"}},
        opset_version=13,
    )
    print("Đã xuất FP32:", FP32, f"({os.path.getsize(FP32)/1e6:.1f} MB)")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(FP32, INT8, weight_type=QuantType.QUInt8)
        print("Đã xuất INT8:", INT8, f"({os.path.getsize(INT8)/1e6:.1f} MB)")
    except Exception as e:
        print("Bỏ qua lượng tử hoá (cài onnxruntime để bật):", e)


if __name__ == "__main__":
    main()
