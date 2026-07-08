# -*- coding: utf-8 -*-
"""
AquaShield AI — Ứng dụng DESKTOP (Tkinter) phân loại bệnh tôm Sú.
Phiên bản này KHÔNG cần trình duyệt và đóng gói .exe rất gọn bằng PyInstaller.

CHẠY THỬ:  python app_desktop.py
ĐÓNG GÓI:  xem build_exe.bat  (tạo AquaShieldAI.exe trong thư mục dist/)
"""
import os, sys, glob, threading
import numpy as np
from PIL import Image, ImageTk

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms

import tkinter as tk
from tkinter import filedialog, ttk


def resource_dir():
    """Thư mục chứa tài nguyên (hỗ trợ cả khi chạy từ .exe PyInstaller)."""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = resource_dir()
RUN_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CONF_LOW = 0.55
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

CLASS_KEYS = ["BG", "BG_WSSV", "HS", "WSSV"]
CLASS_NAMES = ["BG — Đen mang", "BG+WSSV — Đen mang & Đốm trắng",
               "HS — Tôm khoẻ", "WSSV — Đốm trắng"]
HS_IDX = 2
COLORS = {"BG": "#d97706", "WSSV": "#dc2626", "BG_WSSV": "#b91c1c", "HS": "#16a34a"}

INFO = {
 "BG": ("🟤 Đen mang (Black Gill) — Cảnh báo",
        "Mang chuyển nâu/đen do lắng đọng melanin khi mô mang tổn thương. Nguyên nhân: "
        "nấm Fusarium, vi khuẩn, nguyên sinh động vật bám mang, hoặc nước kém "
        "(nhiều hữu cơ, khí độc NH3/NO2/H2S, thiếu oxy).",
        "• Cải thiện nước: tăng sục khí/oxy, siphon đáy, thay nước, kiểm soát khí độc.\n"
        "• Giảm mật độ, giảm cho ăn để hạ tải hữu cơ.\n"
        "• Soi tươi mang xác định tác nhân.\n"
        "• Dùng chế phẩm vi sinh/thuốc theo hướng dẫn cán bộ thú y."),
 "WSSV": ("⚪ Hội chứng đốm trắng (WSSV) — NGUY HIỂM CAO",
        "Do virus WSSV — lây lan cực nhanh, tỷ lệ chết tới 100% trong 3–10 ngày. "
        "Dấu hiệu: đốm trắng tròn 0.5–2 mm dưới vỏ giáp đầu ngực, bỏ ăn, tấp mé, đỏ thân. "
        "Bệnh phải khai báo (WOAH/OIE).",
        "• CÁCH LY ngay ao nghi nhiễm; KHÔNG xả nước ra kênh chung.\n"
        "• Báo cán bộ thú y; lấy mẫu PCR để khẳng định.\n"
        "• Không có thuốc đặc trị — ưu tiên an toàn sinh học, thu hoạch khẩn nếu đạt cỡ.\n"
        "• Khử trùng ao, dụng cụ; diệt giáp xác hoang dã trước khi thả lại."),
 "BG_WSSV": ("🔴 Đồng nhiễm Đen mang + Đốm trắng — NGUY HIỂM RẤT CAO",
        "Xuất hiện đồng thời đen mang và đốm trắng — tổ hợp nặng: nền sức khoẻ/nước "
        "suy giảm cộng nhiễm virus WSSV. Diễn tiến xấu rất nhanh, rủi ro mất trắng lớn.",
        "• Xử lý theo mức WSSV: cách ly, báo thú y, xét nghiệm PCR khẩn.\n"
        "• Đồng thời cải thiện nước & đáy để giảm áp lực đen mang.\n"
        "• Ưu tiên thu hoạch sớm/an toàn sinh học thay vì điều trị kéo dài."),
 "HS": ("🟢 Tôm khoẻ (Healthy) — Bình thường",
        "Không phát hiện dấu hiệu bất thường: mang sạch, vỏ giáp không đốm trắng, "
        "màu sắc và hình thái bình thường.",
        "• Duy trì quản lý tốt: theo dõi oxy, pH, độ kiềm, khí độc định kỳ.\n"
        "• Chụp kiểm tra định kỳ để phát hiện sớm bất thường.\n"
        "• Giữ an toàn sinh học: kiểm soát nước, con giống, dụng cụ."),
}


def find_weights():
    for p in [os.environ.get("WEIGHTS", ""),
              os.path.join(APP_DIR, "model", "resnet18_transfer.pt"),
              os.path.join(RUN_DIR, "model", "resnet18_transfer.pt"),
              os.path.join(APP_DIR, "resnet18_transfer.pt")]:
        if p and os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(APP_DIR, "**", "resnet18_transfer.pt"), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("Không tìm thấy model/resnet18_transfer.pt")


def load_model():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    ck = torch.load(find_weights(), map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    m.load_state_dict(state, strict=False)
    return m.eval().to(DEVICE)


PREP = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                           transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

MODEL = load_model()
_feat = {}
MODEL.layer4.register_forward_hook(lambda m, i, o: _feat.__setitem__("act", o.detach()))
MODEL.layer4.register_full_backward_hook(lambda m, gi, go: _feat.__setitem__("grad", go[0].detach()))


def grad_cam(x, cls):
    MODEL.zero_grad(set_to_none=True)
    out = MODEL(x)
    out[0, cls].backward()
    g, a = _feat["grad"][0], _feat["act"][0]
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    return (cam / (cam.max() + 1e-8)).cpu().numpy()


def overlay(pil, cam):
    import matplotlib.cm as cm
    base = np.asarray(pil.resize((IMG_SIZE, IMG_SIZE)), float) / 255.0
    cu = np.asarray(Image.fromarray((cam*255).astype("uint8")).resize((IMG_SIZE, IMG_SIZE)))/255.0
    heat = cm.jet(cu)[..., :3]
    return Image.fromarray(((0.55*base + 0.45*heat)*255).astype("uint8"))


def infer(pil):
    pil = pil.convert("RGB")
    x = PREP(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = MODEL(x).softmax(1)[0].cpu().numpy()
    cls = int(probs.argmax())
    xc = PREP(pil).unsqueeze(0).to(DEVICE); xc.requires_grad_(True)
    try:
        ov = overlay(pil, grad_cam(xc, cls))
    except Exception:
        ov = pil.resize((IMG_SIZE, IMG_SIZE))
    return probs, cls, ov


# --------------------------------------------------------------------------
# Giao diện Tkinter
# --------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AquaShield AI — Phát hiện sớm bệnh tôm Sú")
        self.geometry("940x620")
        self.configure(bg="#f0fdfa")
        self._img_refs = []

        top = tk.Frame(self, bg="#0f766e"); top.pack(fill="x")
        tk.Label(top, text="🦐 AquaShield AI", bg="#0f766e", fg="white",
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(top, text="ResNet-18 • 98% test acc. • AGRITECH 2026",
                 bg="#0f766e", fg="#c7fdf3", font=("Segoe UI", 10)).pack(side="left", pady=10)

        bar = tk.Frame(self, bg="#f0fdfa"); bar.pack(fill="x", padx=16, pady=10)
        tk.Button(bar, text="📁  Chọn ảnh tôm…", command=self.pick,
                  bg="#0f766e", fg="white", font=("Segoe UI", 12, "bold"),
                  relief="flat", padx=16, pady=8, cursor="hand2").pack(side="left")
        self.status = tk.Label(bar, text=f"Sẵn sàng ({DEVICE.upper()}).",
                               bg="#f0fdfa", fg="#475569", font=("Segoe UI", 10))
        self.status.pack(side="left", padx=14)

        body = tk.Frame(self, bg="#f0fdfa"); body.pack(fill="both", expand=True, padx=16, pady=6)

        left = tk.Frame(body, bg="#f0fdfa"); left.pack(side="left", fill="y")
        self.img_lbl = tk.Label(left, bg="#e2e8f0", width=300, height=300)
        self.img_lbl.pack(pady=(0, 8))
        self.cam_lbl = tk.Label(left, bg="#e2e8f0")
        self.cam_lbl.pack()
        tk.Label(left, text="Grad-CAM — vùng AI tập trung", bg="#f0fdfa",
                 fg="#475569", font=("Segoe UI", 9)).pack()

        right = tk.Frame(body, bg="#f0fdfa"); right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self.title_lbl = tk.Label(right, text="Hãy chọn một ảnh để phân tích", bg="#f0fdfa",
                                  fg="#0f172a", font=("Segoe UI", 15, "bold"),
                                  wraplength=520, justify="left", anchor="w")
        self.title_lbl.pack(fill="x")
        self.bars = tk.Frame(right, bg="#f0fdfa"); self.bars.pack(fill="x", pady=8)
        self.cause = tk.Label(right, text="", bg="#f0fdfa", fg="#0f172a",
                              font=("Segoe UI", 10), wraplength=520, justify="left", anchor="w")
        self.cause.pack(fill="x", pady=(4, 8))
        self.action = tk.Label(right, text="", bg="#ffffff", fg="#0f172a",
                               font=("Segoe UI", 10), wraplength=500, justify="left",
                               anchor="w", padx=12, pady=10, relief="flat")
        self.action.pack(fill="x")
        tk.Label(right, text="Công cụ hỗ trợ ra quyết định — không thay thế xét nghiệm PCR/thú y.",
                 bg="#f0fdfa", fg="#94a3b8", font=("Segoe UI", 9)).pack(anchor="w", pady=6)

    def pick(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh tôm Sú",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")])
        if not path:
            return
        self.status.config(text="Đang phân tích…")
        self.update_idletasks()
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path):
        try:
            pil = Image.open(path)
            probs, cls, ov = infer(pil)
            self.after(0, lambda: self._show(pil, probs, cls, ov))
        except Exception as e:
            self.after(0, lambda: self.status.config(text=f"Lỗi: {e}"))

    def _show(self, pil, probs, cls, ov):
        self._img_refs.clear()
        disp = pil.convert("RGB").copy(); disp.thumbnail((300, 300))
        ph = ImageTk.PhotoImage(disp); self._img_refs.append(ph)
        self.img_lbl.config(image=ph, width=disp.width, height=disp.height)
        camimg = ov.resize((200, 200)); ph2 = ImageTk.PhotoImage(camimg); self._img_refs.append(ph2)
        self.cam_lbl.config(image=ph2)

        key = CLASS_KEYS[cls]; title, cause, action = INFO[key]; conf = probs[cls]
        self.title_lbl.config(text=f"{title}\n{('✅ Tôm khoẻ mạnh.' if cls==HS_IDX else '⚠️ NGHI NHIỄM BỆNH — nên cách ly & xét nghiệm.')}  ({conf*100:.1f}%)",
                              fg=COLORS[key])
        for w in self.bars.winfo_children():
            w.destroy()
        order = sorted(range(4), key=lambda i: -probs[i])
        for i in order:
            row = tk.Frame(self.bars, bg="#f0fdfa"); row.pack(fill="x", pady=2)
            tk.Label(row, text=CLASS_NAMES[i], width=26, anchor="w", bg="#f0fdfa",
                     fg="#475569", font=("Segoe UI", 9)).pack(side="left")
            track = tk.Frame(row, bg="#eef2f6", height=16, width=240); track.pack(side="left", padx=6)
            track.pack_propagate(False)
            fill = tk.Frame(track, bg="#14b8a6" if i == cls else "#94d3cc",
                            height=16, width=int(240*probs[i]))
            fill.pack(side="left")
            tk.Label(row, text=f"{probs[i]*100:5.1f}%", bg="#f0fdfa", fg="#0f172a",
                     font=("Segoe UI", 9, "bold")).pack(side="left")
        pre = ""
        if conf < CONF_LOW:
            pre = "⚠️ Độ tin cậy thấp — ảnh có thể mờ/thiếu sáng. Hãy chụp lại rõ mang & vỏ giáp.\n\n"
        self.cause.config(text=pre + cause)
        self.action.config(text=action)
        self.status.config(text="Xong.")


if __name__ == "__main__":
    App().mainloop()
