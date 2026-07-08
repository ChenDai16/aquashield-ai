# -*- coding: utf-8 -*-
"""
AquaShield AI — Phát hiện sớm bệnh trên tôm Sú (Penaeus monodon) qua ảnh
=========================================================================
Ứng dụng web (Gradio) phân loại 1 trong 4 tình trạng từ ảnh chụp điện thoại:
    • BG      — Đen mang (Black Gill)
    • BG+WSSV — Đồng nhiễm Đen mang + Đốm trắng
    • HS      — Tôm khoẻ (Healthy Shrimp)
    • WSSV    — Hội chứng đốm trắng (White Spot Syndrome Virus)

Mô hình: ResNet-18 (transfer learning) — resnet18_transfer.pt
Độ chính xác kiểm định trên tập Test chính thức (256 ảnh): 98.05%.

CÁCH CHẠY (local, khuyến nghị)
------------------------------
    pip install -r requirements.txt
    python app.py                 # mở http://127.0.0.1:7860

Bản dùng thử online (chia sẻ link công khai tạm thời):
    SHARE=1 python app.py

Trỏ tới trọng số ở nơi khác:
    WEIGHTS=/duong/dan/resnet18_transfer.pt python app.py

LƯU Ý: Đây là công cụ HỖ TRỢ RA QUYẾT ĐỊNH cho người nuôi, KHÔNG thay thế
xét nghiệm PCR/mô bệnh học của cơ quan thú y thuỷ sản.
"""

import os
import sys
import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import gradio as gr

# ---------------------------------------------------------------------------
# 0) Cấu hình chung
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
IMG_SIZE = 224
CONF_LOW = 0.55            # ngưỡng cảnh báo "độ tin cậy thấp"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Thứ tự lớp = ImageFolder (alphabet) khi train — TUYỆT ĐỐI không đổi:
#   0 "Black Gill - BG"
#   1 "Black Gill_White Spot Syndrome Virus - BG_WSSV"
#   2 "Healthy Shrimp - HS"
#   3 "White Spot Syndrome Virus - WSSV"
CLASS_KEYS = ["BG", "BG_WSSV", "HS", "WSSV"]
CLASS_NAMES = [
    "BG — Đen mang",
    "BG+WSSV — Đen mang & Đốm trắng",
    "HS — Tôm khoẻ",
    "WSSV — Đốm trắng",
]
HS_IDX = 2  # chỉ số lớp "Tôm khoẻ"

# Thông tin & khuyến nghị theo từng lớp (nội dung tham khảo cho người nuôi)
DISEASE_INFO = {
    "BG": {
        "title": "🟤 Đen mang (Black Gill)",
        "level": "Cảnh báo",
        "cause": (
            "Mang chuyển nâu/đen do lắng đọng melanin khi mô mang bị tổn thương. "
            "Nguyên nhân thường gặp: nấm *Fusarium*, vi khuẩn, nguyên sinh động vật "
            "bám mang, hoặc chất lượng nước kém (nhiều hữu cơ, khí độc NH₃/NO₂⁻/H₂S, "
            "thiếu oxy hoà tan)."
        ),
        "action": (
            "• Kiểm tra và cải thiện chất lượng nước: tăng sục khí/oxy, siphon đáy, "
            "thay nước, kiểm soát NH₃/NO₂⁻/H₂S.\n"
            "• Giảm mật độ, giảm cho ăn tạm thời để hạ tải hữu cơ.\n"
            "• Soi tươi mang dưới kính hiển vi để xác định tác nhân (nấm/vi khuẩn/ký sinh).\n"
            "• Cân nhắc chế phẩm vi sinh xử lý đáy; dùng thuốc theo hướng dẫn cán bộ thú y."
        ),
    },
    "WSSV": {
        "title": "⚪ Hội chứng đốm trắng (WSSV)",
        "level": "NGUY HIỂM CAO",
        "cause": (
            "Bệnh do virus White Spot Syndrome Virus — lây lan cực nhanh, tỷ lệ chết "
            "có thể tới 100% trong 3–10 ngày. Dấu hiệu: đốm trắng tròn 0.5–2 mm dưới "
            "vỏ giáp đầu ngực, tôm bỏ ăn, tấp mé, đỏ thân. Thuộc danh mục bệnh phải "
            "khai báo (WOAH/OIE)."
        ),
        "action": (
            "• CÁCH LY ngay ao nghi nhiễm; KHÔNG xả nước ra kênh chung.\n"
            "• Báo cán bộ thú y thuỷ sản; lấy mẫu xét nghiệm PCR để khẳng định.\n"
            "• Không có thuốc đặc trị — ưu tiên an toàn sinh học, thu hoạch khẩn cấp nếu tôm đạt cỡ.\n"
            "• Khử trùng ao, dụng cụ, diệt vật chủ trung gian (giáp xác hoang dã) trước khi thả lại."
        ),
    },
    "BG_WSSV": {
        "title": "🔴 Đồng nhiễm Đen mang + Đốm trắng",
        "level": "NGUY HIỂM RẤT CAO",
        "cause": (
            "Xuất hiện đồng thời dấu hiệu đen mang và đốm trắng — tổ hợp nặng: nền "
            "sức khoẻ/nước suy giảm (đen mang) cộng nhiễm virus WSSV. Diễn tiến xấu "
            "nhanh và rủi ro mất trắng rất lớn."
        ),
        "action": (
            "• Xử lý theo mức WSSV: cách ly, báo thú y, xét nghiệm PCR khẩn.\n"
            "• Đồng thời cải thiện nước & đáy để giảm áp lực đen mang.\n"
            "• Ưu tiên quyết định thu hoạch sớm/an toàn sinh học thay vì điều trị kéo dài."
        ),
    },
    "HS": {
        "title": "🟢 Tôm khoẻ (Healthy)",
        "level": "Bình thường",
        "cause": (
            "Không phát hiện dấu hiệu bất thường trên ảnh: mang sạch, vỏ giáp không "
            "đốm trắng, màu sắc và hình thái bình thường."
        ),
        "action": (
            "• Duy trì quản lý tốt: theo dõi oxy, pH, độ kiềm, khí độc định kỳ.\n"
            "• Chụp kiểm tra định kỳ để phát hiện sớm bất thường.\n"
            "• Giữ an toàn sinh học: kiểm soát nguồn nước, con giống, dụng cụ."
        ),
    },
}

# ---------------------------------------------------------------------------
# 1) Tìm & nạp trọng số mô hình
# ---------------------------------------------------------------------------
def find_weights():
    candidates = [
        os.environ.get("WEIGHTS", ""),
        os.path.join(APP_DIR, "model", "resnet18_transfer.pt"),
        os.path.join(APP_DIR, "resnet18_transfer.pt"),
        "./resnet18_transfer.pt",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(APP_DIR, "**", "resnet18_transfer.pt"), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(
        "Không tìm thấy 'resnet18_transfer.pt'. Đặt file vào thư mục ./model/ "
        "hoặc set biến môi trường WEIGHTS trỏ tới đường dẫn file."
    )


def load_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    wp = find_weights()
    print(f"[AquaShield] Nạp trọng số: {wp}")
    ckpt = torch.load(wp, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # tương thích checkpoint có tiền tố "model."
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[AquaShield] ⚠️ Thiếu khoá: {missing}")
    if unexpected:
        print(f"[AquaShield] ⚠️ Khoá dư: {unexpected}")
    model.eval().to(DEVICE)
    print(f"[AquaShield] Thiết bị: {DEVICE.upper()} — sẵn sàng.")
    return model


MODEL = load_model()

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# 2) Grad-CAM trên layer4 (giải thích vùng AI tập trung)
# ---------------------------------------------------------------------------
_feat = {}
MODEL.layer4.register_forward_hook(
    lambda m, i, o: _feat.__setitem__("act", o.detach()))
MODEL.layer4.register_full_backward_hook(
    lambda m, gi, go: _feat.__setitem__("grad", go[0].detach()))


def grad_cam(x, cls):
    MODEL.zero_grad(set_to_none=True)
    out = MODEL(x)
    out[0, cls].backward()
    grad = _feat["grad"][0]           # (C,h,w)
    act = _feat["act"][0]             # (C,h,w)
    weights = grad.mean(dim=(1, 2))   # (C,)
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    return cam, out.detach()


def overlay_cam(pil, cam):
    import matplotlib.cm as cm
    base = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE)), dtype=float) / 255.0
    cam_up = np.asarray(
        Image.fromarray((cam * 255).astype("uint8")).resize((IMG_SIZE, IMG_SIZE))
    ) / 255.0
    heat = cm.jet(cam_up)[..., :3]
    blend = 0.55 * base + 0.45 * heat
    return Image.fromarray((blend * 255).astype("uint8"))


# ---------------------------------------------------------------------------
# 3) Hàm dự đoán chính
# ---------------------------------------------------------------------------
def predict(pil):
    if pil is None:
        return {}, None, "Hãy tải lên hoặc chụp một ảnh tôm Sú để phân tích."
    pil = pil.convert("RGB")
    x = PREPROCESS(pil).unsqueeze(0).to(DEVICE)

    # dự đoán (không cần grad)
    with torch.no_grad():
        probs = MODEL(x).softmax(1)[0].cpu().numpy()
    cls = int(probs.argmax())
    conf = float(probs[cls])

    # Grad-CAM (cần grad)
    x_cam = PREPROCESS(pil).unsqueeze(0).to(DEVICE)
    x_cam.requires_grad_(True)
    try:
        cam, _ = grad_cam(x_cam, cls)
        overlay = overlay_cam(pil, cam)
    except Exception as e:
        print("[AquaShield] Grad-CAM lỗi:", e)
        overlay = pil.resize((IMG_SIZE, IMG_SIZE))

    label_scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    key = CLASS_KEYS[cls]
    info = DISEASE_INFO[key]
    healthy = (cls == HS_IDX)
    banner = ("✅ Tôm có dấu hiệu **khoẻ mạnh**."
              if healthy else
              "⚠️ **NGHI NHIỄM BỆNH** — nên cách ly & xét nghiệm khẳng định.")
    low_conf = ("\n\n> ⚠️ *Độ tin cậy thấp (<%.0f%%). Ảnh có thể mờ/thiếu sáng hoặc "
                "không rõ vùng mang & vỏ giáp — hãy chụp lại rõ hơn.*" % (CONF_LOW * 100)
                if conf < CONF_LOW else "")

    note = (
        f"## {info['title']}  —  {conf*100:.1f}%\n"
        f"**Mức độ:** {info['level']}\n\n"
        f"{banner}{low_conf}\n\n"
        f"**Nhận định:** {info['cause']}\n\n"
        f"**Khuyến nghị xử lý:**\n{info['action']}\n\n"
        f"---\n*Công cụ hỗ trợ ra quyết định — không thay thế xét nghiệm PCR/thú y thuỷ sản.*"
    )
    return label_scores, overlay, note


# ---------------------------------------------------------------------------
# 4) Giao diện Gradio (mobile-friendly, có camera)
# ---------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1080px !important; margin: auto;}
#title-md h1 {margin-bottom: 0;}
footer {visibility: hidden;}
"""

def build_ui():
    with gr.Blocks(title="AquaShield AI — Phát hiện sớm bệnh tôm Sú",
                   theme=gr.themes.Soft(primary_hue="teal"), css=CSS) as demo:
        gr.Markdown(
            "# 🦐 AquaShield AI\n"
            "### Phát hiện sớm bệnh trên tôm Sú qua ảnh điện thoại — ResNet-18 (98% test acc.)",
            elem_id="title-md",
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="📷 Ảnh tôm Sú (tải lên hoặc chụp bằng camera)",
                    height=340,
                )
                btn = gr.Button("🔍 Phân tích", variant="primary", size="lg")
                gr.Markdown(
                    "*Mẹo chụp: đủ sáng, nét, thấy rõ **mang** và **vỏ giáp đầu ngực**; "
                    "chụp cận 1 con trên nền tương phản.*"
                )
            with gr.Column(scale=1):
                lbl = gr.Label(num_top_classes=4, label="Xác suất theo lớp")
                cam_out = gr.Image(label="🔥 Grad-CAM — vùng AI tập trung", height=260)
        md = gr.Markdown()

        # ví dụ mẫu (nếu có thư mục sample_images)
        sample_dir = os.path.join(APP_DIR, "sample_images")
        if os.path.isdir(sample_dir):
            samples = sorted(glob.glob(os.path.join(sample_dir, "*")))
            if samples:
                gr.Examples(examples=[[s] for s in samples[:8]], inputs=inp,
                            label="Ảnh mẫu để thử nhanh")

        btn.click(predict, inputs=inp, outputs=[lbl, cam_out, md])

        gr.Markdown(
            "---\n"
            "**AquaShield AI** • Đề tài AGRITECH 2026 — Trần Quang Đại (Đại học Nha Trang). "
            "Mô hình ResNet-18 transfer learning, 4 lớp. "
            "*Bản demo hỗ trợ ra quyết định — không thay thế chẩn đoán thú y thuỷ sản.*"
        )
    return demo


if __name__ == "__main__":
    share = os.environ.get("SHARE", "0") == "1"
    server_name = os.environ.get("HOST", "127.0.0.1")
    server_port = int(os.environ.get("PORT", "7860"))
    demo = build_ui()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
