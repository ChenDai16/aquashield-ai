# 🦐 AquaShield AI — Phát hiện sớm bệnh trên tôm Sú qua ảnh

Ứng dụng AI phân loại **4 tình trạng** của tôm Sú (*Penaeus monodon*) từ một tấm ảnh
chụp bằng điện thoại, kèm **bản đồ vùng chú ý (Grad-CAM)** và **khuyến nghị xử lý**.

| Lớp | Mã | Ý nghĩa |
|---|---|---|
| 0 | **BG** | Đen mang (Black Gill) |
| 1 | **BG+WSSV** | Đồng nhiễm Đen mang + Đốm trắng |
| 2 | **HS** | Tôm khoẻ (Healthy) |
| 3 | **WSSV** | Hội chứng đốm trắng (White Spot Syndrome Virus) |

**Mô hình:** ResNet-18 (transfer learning) · **Độ chính xác kiểm định: 98.05%** trên
tập Test chính thức (256 ảnh — 64 ảnh × 4 lớp).

---

## 📊 Kết quả kiểm định (đã đánh giá lại trên toàn bộ tập Test)

```
Test accuracy: 98.05%  (n=256)

class                 precision   recall     f1   support
BG (Đen mang)            0.969     0.984   0.977      64
BG+WSSV                  0.954     0.969   0.961      64
HS (Khỏe)                1.000     1.000   1.000      64
WSSV (Đốm trắng)         1.000     0.969   0.984      64
------------------------------------------------------------
accuracy                                  0.980     256
macro avg                0.981     0.980   0.981     256
```

Xem chi tiết trong `results/` (báo cáo + ma trận nhầm lẫn). Các lỗi hiếm chỉ xảy ra
giữa những lớp có biểu hiện gần giống nhau (BG ↔ BG+WSSV; WSSV → BG+WSSV) — hợp lý
về mặt sinh học. Lớp **HS** và **WSSV** đạt precision 100%.

> Con số 93.75% trong hồ sơ trước là kết quả trên tập con 96 ảnh. Bản đánh giá này
> chạy trên **toàn bộ 256 ảnh Test** và cho **98.05%**.

---

## 📁 Cấu trúc thư mục

```
AquaShieldAI_App/
├── app.py                     # ⭐ App web Gradio (bản chính, khuyến nghị)
├── app_desktop.py             # App desktop Tkinter (để đóng gói .exe)
├── evaluate.py                # Chạy lại đánh giá trên tập Test
├── export_onnx.py             # Xuất lại ONNX từ .pt (khi train lại)
├── requirements.txt
├── run_app.bat / run_app.sh   # 1-chạm khởi động app web
├── build_exe.bat + aquashield.spec   # Đóng gói .exe
├── model/
│   ├── resnet18_transfer.pt   # Trọng số PyTorch (44 MB)
│   ├── resnet18_shrimp.onnx   # ONNX FP32
│   └── labels.json            # Nhãn + cấu hình tiền xử lý
├── mobile/
│   ├── aquashield_mobile.html # ⭐ App mobile chạy on-device (offline)
│   ├── resnet18_shrimp_int8.onnx  # Model nhẹ 11 MB cho điện thoại
│   ├── resnet18_shrimp.onnx       # Model FP32 dự phòng
│   └── serve_mobile.bat       # Mở app trên điện thoại cùng Wi-Fi
├── sample_images/             # 8 ảnh mẫu (2/lớp) để thử nhanh
└── results/                   # Báo cáo + confusion matrix
```

---

## 🚀 Cách chạy — 4 nền tảng

### 1) Web app (local) — khuyến nghị cho demo cuộc thi
Cách dễ nhất trên Windows: **nháy đúp `run_app.bat`**. Lần đầu sẽ tự tạo môi trường
ảo và cài thư viện (~5–10 phút), sau đó tự mở trình duyệt.

Thủ công:
```bash
pip install -r requirements.txt
python app.py            # mở http://127.0.0.1:7860
```

### 2) Bản dùng thử online (chia sẻ link tạm thời)
```bash
SHARE=1 python app.py    # Windows: set SHARE=1 && python app.py
```
Gradio tạo một link công khai `*.gradio.live` (hết hạn sau ~72 giờ) để demo từ xa.
Ảnh vẫn xử lý trên máy của bạn.

### 3) App mobile (chạy THỰC TẾ trên điện thoại, on-device, offline)
App trong `mobile/aquashield_mobile.html` chạy suy luận **ngay trên trình duyệt điện
thoại** bằng ONNX Runtime Web — **không gửi ảnh lên máy chủ**.

Cách mở nhanh (điện thoại cùng Wi-Fi với máy tính):
1. Vào thư mục `mobile/`, nháy đúp **`serve_mobile.bat`**.
2. Trên điện thoại, mở địa chỉ hiện ra, ví dụ `http://192.168.1.10:8000/aquashield_mobile.html`.
3. Bấm **📷 Chụp ảnh** → chụp con tôm → nhận kết quả tức thì.

> Cần mạng ở **lần đầu** để tải thư viện + model (~11 MB); sau đó trình duyệt cache lại.
> Có thể đưa cả thư mục `mobile/` lên GitHub Pages / Netlify để có link https dùng mọi nơi.

### 4) Desktop .exe (Windows, không cần cài Python cho người dùng cuối)
```bat
build_exe.bat
```
Kết quả: `dist\AquaShieldAI\AquaShieldAI.exe`. Nén cả thư mục `dist\AquaShieldAI\`
để chia sẻ. (File lớn ~1.5–2.5 GB vì kèm PyTorch; cần ~6 GB trống khi build.)

Chạy thử bản desktop mà không đóng gói: `python app_desktop.py`.

---

## 🔬 Chạy lại đánh giá / xuất lại ONNX
```bash
python evaluate.py     # tạo lại report + confusion matrix trong results/
python export_onnx.py  # tạo lại mobile/resnet18_shrimp(_int8).onnx
```

---

## ⚙️ Chi tiết kỹ thuật
- **Kiến trúc:** ResNet-18, lớp `fc` thay bằng `Linear(512 → 4)`.
- **Tiền xử lý:** Resize 224×224 → chuẩn hoá ImageNet (mean `[.485,.456,.406]`,
  std `[.229,.224,.225]`).
- **Thứ tự lớp:** theo `ImageFolder` alphabet — **không được đổi** để nhãn khớp model.
- **Grad-CAM:** trên `layer4` để minh hoạ vùng ảnh AI dựa vào.
- **Ngưỡng tin cậy:** dưới 55% app cảnh báo "độ tin cậy thấp, chụp lại rõ hơn".

## ⚠️ Giới hạn & lưu ý trung thực
AquaShield AI là **công cụ hỗ trợ ra quyết định** cho người nuôi để **sàng lọc và
cảnh báo sớm**, **không thay thế** xét nghiệm PCR/mô bệnh học của cơ quan thú y thuỷ
sản. Kết quả phụ thuộc chất lượng ảnh (ánh sáng, độ nét, góc chụp thấy rõ mang và vỏ
giáp). Với ca nghi ngờ, luôn cách ly và lấy mẫu xét nghiệm khẳng định.

---
*AGRITECH 2026 — Trần Quang Đại, Nuôi trồng Thuỷ sản, Đại học Nha Trang.*
